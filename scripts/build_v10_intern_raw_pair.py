#!/usr/bin/env python3
"""Build v10 raw-pair Parquets: validation/test molecule membership defined by TxAgent's own
authoritative heldout identity, replacing v8/v9's "let touched molecules define the split" design.

Thin wrapper around ``scripts/build_v8_intern_raw_pair.py`` (ranking-anchor construction,
``_ranking_span``, degree caps, and template rendering). V10 calibrates soft targets independently
inside every pair bucket: the bucket's pairwise-distance 5th/95th percentiles map to transfer
probabilities 0.95/0.05 through an unrestricted sigmoid. Ordinary-pair selection uses conservative
``target_a > 0.6`` / ``target_a < 0.4`` strata and additionally forbids ranking pairs from the same
release subset.

**What changes, and why**: v8/v9 assign validation/test molecule membership *emergently* -- whatever
ranking-anchor/ordinary-pair construction happens to touch defines the split, and every record of a
touched molecule (not just the touched instance) moves out of train. In the v9 build this meant only
~3-5% of the records swept into validation/test were ever used by a final eval row; a handful of
extremely record-rich "hub" molecules (up to ~2,000 records for one compound) dragged their entire
record set out of train the moment construction sampled even one of their records, costing train
~41% of the eligible pool for a benchmark that only needed ~5% of that volume.

TxAgent's ``starling_normalized_v6`` artifact now ships a dedicated heldout-identity file per split
methodology -- ``data/processed_starling/Bioavailability_Ma/{random,scaffold}/
heldout_molecule_labels.jsonl``, each a pre-combined union of that methodology's own valid+test
molecule identities (209+209=418 apiece). Per user decision: union random's and scaffold's heldout
identities into one pool (``starling_txagent_eligible_v6._heldout_union_molecules`` -- 852 molecules
/ 40,848 records against the current eligible pool). This makes eval's molecule footprint a hard,
small, known-upfront ceiling (852, restricted-candidate-pool) instead of an emergent, unbounded one
-- train recovers to (roughly) ``eligible - reserved`` (~157,700 records / ~17,858 molecules),
better than either v9 variant.

Architecture, concretely:

1. ``reserved_molecules = _heldout_union_molecules(rows)`` (~852) -- the supplied candidate pool
   for both splits. Ordinary pairs draw from its union with ranking-touched records, preserving the
   same reserved boundary today while remaining correct if ranking's source broadens later.
2. **No partition of the reserved pool.** An earlier version of this script split the 852 molecules
   60/40 into disjoint test/validation subsets before construction ran -- this turned out to be an
   unnecessary constraint (v8/v9 already treat validation/test overlap as an accepted
   simplification; the only invariant that ever mattered is train staying disjoint from both) and it
   actively starved thin concepts: halving Fg's already-tiny ~50-molecule pool left too few
   same-bucket candidates for even one 20-wide ranking anchor. Both splits now draw from the *full*
   852-molecule pool.
3. Construction runs with **shared degree-cap state across all of it** -- one
   ``anchor_molecules_used``/``ranking_record_degree``/``ordinary_query_degree``/
   ``ordinary_retrieval_degree`` set of dicts threaded through every call below, same
   cross-split-sharing mechanism ``scripts/build_v8_intern_raw_pair.py::_make_split_eval_v8`` uses.
   ``overlap_molecules`` is passed as the reserved pool itself (every molecule here already is a
   Starling-heldout identity by definition, so the two-phase overlap/new logic in
   ``_ranking_anchors_v8`` degenerates harmlessly to one effective phase). Per user decision,
   ``RECORD_DEGREE_CAP``/``ORDINARY_DEGREE_CAP`` are raised (monkeypatched on the ``v8`` module,
   same mechanism as the ``_candidate_label`` override) to let the same limited pool of records be
   reused more before a bucket gives up.
4. **Construction is a general (split, concepts-subset) primitive (`run_construction`), manually
   sequenced rather than hardcoded to "everything, split by split".** Running test's or
   validation's full 5-concept construction back to back (as an earlier version of this script did)
   meant common concepts (oral_bioavailability/oral_exposure/Fa, plentiful) and rare concepts
   (Fg/Fh, thin) competed for the *same* shared degree-cap state and the *same* one-time-ever
   ``anchor_molecules_used`` -- a molecule touched incidentally by common-concept construction
   becomes permanently unavailable to Fg/Fh even if it also carries Fg/Fh records. Per user
   decision, ``RARE_CONCEPTS`` (Fg, Fh) are built for *both* splits before ``COMMON_CONCEPTS``
   touches anything at all: test-rare, then validation-rare, then test-common, then
   validation-common -- protecting the thin concepts' scarce molecule pool from common-concept
   competition, while still giving test first access within each phase (matching the existing
   test-gets-priority convention).
5. Final split membership is **touched-based**, same semantics as v8 (a molecule can end up in both
   validation and test if both splits' construction happen to touch it -- allowed, matching v8's own
   stated position). The one absolute rule: a reserved molecule can never fall back to train. If
   neither split's construction ends up touching it, it is dropped from the dataset entirely (not
   trained on, not evaluated on) rather than leaking into train.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

import pipeline.pair_core as pair_core
import scripts.build_raw_pair as pair_driver
import scripts.build_v7_intern_raw_pair as v7
import scripts.build_v8_intern_raw_pair as v8
from pipeline.pair_core import record_key
from pipeline.source_normalization.starling_txagent_eligible_v6 import _heldout_union_molecules
from pipeline.v3_policy import file_sha256
from scripts.build_raw_pair import build
from scripts.build_v7_intern_raw_pair import _iter_list_groups_capped, _pair_bucket_concentration, _train_target_diagnostics, _write_train_flat_variable

# Raised from v8/v9's 8, per user decision: when a concept's reserved pool is thin (Fg, Fh), let
# the same limited records be reused more before a bucket gives up on forming a full-width anchor
# or a labeled ordinary pair. Monkeypatched onto the v8 module below (same mechanism as the
# _candidate_label override) since _ranking_span/_ordinary_pairs_v8 reference these as bare globals.
RECORD_DEGREE_CAP = 24
ORDINARY_DEGREE_CAP = 24

# Fg/Fh have far thinner reserved pools than the other three concepts (7 and 124 pair buckets vs.
# 4/54/100, and ~50/~200 reserved molecules vs. hundreds) -- see run_construction's caller for why
# they're scheduled before COMMON_CONCEPTS.
RARE_CONCEPTS = {"Fg", "Fh"}
COMMON_CONCEPTS = {"oral_bioavailability", "oral_exposure", "Fa"}
RANKING_ANCHORS_BY_SPLIT = {
    "validation": dict(v8.RANKING_ANCHORS_PER_CONCEPT_VALIDATION),
    "test": {**v8.RANKING_ANCHORS_PER_CONCEPT_TEST, "Fg": 20},
}
HUB_DATASET_ID = "jiosephlee/assay-transfer-raw-pair-v10-intern"
ELIGIBLE_SOURCE_SHA256 = "5e520b2564e92d4c7707d2a7bdc9f0acccfe7a803a2b2703e3c3008656310be9"
CALIBRATION_PATH = Path("configs/assay_transfer/v10/pairwise_distance_quantiles.json")
CALIBRATION_SHA256 = "860082eb4af851ea013542c8b6219eb5d412ffb15b26aac312cd637e7a14cdf7"
TARGET_PROBABILITY_AT_P05 = 0.95
TARGET_PROBABILITY_AT_P95 = 0.05
ORDINARY_TRANSFER_MIN_PROBABILITY = 0.6
ORDINARY_NONTRANSFER_MAX_PROBABILITY = 0.4
_TARGET_ANCHOR_LOGIT = math.log(TARGET_PROBABILITY_AT_P05 / TARGET_PROBABILITY_AT_P95)
_CALIBRATION_BY_BUCKET: dict[str, tuple[float, float]] = {}


def _load_calibration(path: Path) -> tuple[dict, dict[str, tuple[float, float]]]:
    if path == CALIBRATION_PATH and file_sha256(path) != CALIBRATION_SHA256:
        raise ValueError("v10 calibration artifact hash mismatch")
    document = json.loads(path.read_text())
    if document.get("schema_version") != "pairwise_distance_quantiles_v1":
        raise ValueError("unexpected pairwise-distance calibration schema")
    if document.get("lower_percentile") != 5 or document.get("upper_percentile") != 95:
        raise ValueError("v10 calibration must use the 5th and 95th percentiles")
    if document.get("target_probability_at_lower") != TARGET_PROBABILITY_AT_P05:
        raise ValueError("unexpected lower-anchor target probability")
    if document.get("target_probability_at_upper") != TARGET_PROBABILITY_AT_P95:
        raise ValueError("unexpected upper-anchor target probability")
    anchors = {}
    for bucket, values in document.get("buckets", {}).items():
        lower, upper = float(values["distance_p05"]), float(values["distance_p95"])
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise ValueError(f"invalid calibration anchors for bucket {bucket}")
        anchors[str(bucket)] = (lower, upper)
    if len(anchors) != 289:
        raise ValueError(f"expected 289 calibrated buckets, found {len(anchors)}")
    return document, anchors


def _target_for_v10(query: dict, retrieval: dict) -> dict[str, float]:
    bucket = str(query["pair_bucket_key"])
    try:
        lower, upper = _CALIBRATION_BY_BUCKET[bucket]
    except KeyError as error:
        raise ValueError(f"no v10 calibration for pair bucket {bucket}") from error
    distance = abs(float(retrieval["comparison_value"]) - float(query["comparison_value"]))
    target_z = _TARGET_ANCHOR_LOGIT * (lower + upper - 2.0 * distance) / (upper - lower)
    if target_z >= 0.0:
        target_a = 1.0 / (1.0 + math.exp(-min(target_z, 700.0)))
    else:
        exponential = math.exp(max(target_z, -700.0))
        target_a = exponential / (1.0 + exponential)
    return {
        "distance": distance,
        "target_z": target_z,
        "target_a": target_a,
        "target_b": 1.0 - target_a,
        "calibration_distance_p05": lower,
        "calibration_distance_p95": upper,
    }


def _candidate_label_v10(query: dict, retrieval: dict) -> str | None:
    target_a = _target_for_v10(query, retrieval)["target_a"]
    if target_a > ORDINARY_TRANSFER_MIN_PROBABILITY:
        return "transfer"
    if target_a < ORDINARY_NONTRANSFER_MAX_PROBABILITY:
        return "nontransfer"
    return None


def _is_decisive_v10(row: dict, _query: dict) -> bool:
    target_a = float(row["target_a"])
    return (target_a > ORDINARY_TRANSFER_MIN_PROBABILITY or
            target_a < ORDINARY_NONTRANSFER_MAX_PROBABILITY)


def _ordinary_pairs_v10(records: list[dict], split: str, allowed_records: set[str],
                        per_concept_target: dict[str, int], query_degree: dict[str, int],
                        retrieval_degree: dict[str, int], initial_forbidden: set[str]) -> list[dict]:
    """V8 ordinary selection with ranking pairs pre-forbidden for release-subset isolation."""
    pool = [row for row in records if record_key(row) in allowed_records]
    concepts = sorted({str(row["assay_concept"]) for row in pool
                       if per_concept_target.get(str(row["assay_concept"]), 0) > 0})
    indexed, selected = v8._condition_index_by_bucket(pool), defaultdict(list)
    forbidden = set(initial_forbidden)
    queries = sorted(pool, key=lambda row: v8.stable_hash(record_key(row)))
    for pass_index in range(32):
        for query in queries:
            concept = str(query["assay_concept"])
            per_label = per_concept_target.get(concept, 0)
            if per_label <= 0 or query_degree[record_key(query)] >= v8.ORDINARY_DEGREE_CAP:
                continue
            candidates = v8.propose_records(
                query, indexed[(query["canonical_endpoint_key"], query["pair_bucket_key"])], 128,
                forbidden, f"{split}:ordinary:{pass_index}")
            labels = sorted(("transfer", "nontransfer"),
                            key=lambda label: len(selected[concept, label]))
            for wanted_label in labels:
                eligible = [item for item in candidates if v8._candidate_label(query, item) == wanted_label
                            and retrieval_degree[record_key(item)] < v8.ORDINARY_DEGREE_CAP]
                retrieval = next((item for item in eligible if retrieval_degree[record_key(item)] == 0), None)
                if retrieval is None and eligible:
                    retrieval = eligible[0]
                if retrieval is None or len(selected[concept, wanted_label]) >= per_label:
                    continue
                row = v8.row_for(query, retrieval, split)
                row["binary_label"] = 1 if wanted_label == "transfer" else 0
                row["is_decisive"] = v8._is_decisive(row, query)
                v8._fix_metadata(row)
                selected[concept, wanted_label].append(row)
                forbidden.add(v8.unordered_pair_id(query, retrieval))
                query_degree[record_key(query)] += 1
                retrieval_degree[record_key(retrieval)] += 1
                break
        if all(len(selected[concept, label]) >= per_concept_target.get(concept, 0)
               for concept in concepts for label in ("transfer", "nontransfer")):
            break
    return [row for concept in concepts for label in ("transfer", "nontransfer")
            for row in selected[concept, label]]


def run_construction(records: list[dict], split: str, concepts: set[str], overlap_molecules: set[str],
                     anchor_molecules_used: set[str], ranking_record_degree: dict[str, int],
                     ordinary_query_degree: dict[str, int], ordinary_retrieval_degree: dict[str, int]) -> dict:
    """General, reusable construction primitive: build ranking anchors + ordinary pairs for one
    (split, concepts-subset) at a time, against whatever degree-cap state the caller threads
    through. Concepts not in `concepts` are left completely untouched by this call -- a full
    split's construction is composed by calling this repeatedly with different concept subsets, in
    whatever order the caller chooses (see _make_split_eval_v10 for this build's specific
    ordering). Reuses scripts/build_v8_intern_raw_pair.py's `_ranking_anchors_v8` and
    `_ranking_rows_from_anchors`; `_ordinary_pairs_v10` adds ranking-pair exclusion. Restricting to `concepts` is just
    a matter of only including those concepts' keys in the per-concept target dicts passed in
    (both functions already only activate concepts present with positive budget in their target
    dict, so no separate record-filtering by concept is needed). Ordinary selection receives the
    union of ranking-touched records and every record supplied to this call."""
    ranking_full = RANKING_ANCHORS_BY_SPLIT[split]
    ranking_target = {concept: target for concept, target in ranking_full.items() if concept in concepts}
    ordinary_target = {concept: target for concept, target in v8.ORDINARY_PER_CONCEPT_LABEL.items() if concept in concepts}

    anchors, ranking_touched_molecules = v8._ranking_anchors_v8(
        records, split, overlap_molecules, ranking_target, anchor_molecules_used, ranking_record_degree)
    ranking_rows = v8._ranking_rows_from_anchors(anchors, split)
    touched_records = {record_key(query) for query, _ in anchors}
    touched_records.update(record_key(candidate) for _, candidates in anchors for candidate in candidates)
    ordinary_pool_records = touched_records | {record_key(row) for row in records}
    ranking_pairs = {v8.unordered_pair_id(query, candidate)
                     for query, candidates in anchors for candidate in candidates}

    ordinary_rows = _ordinary_pairs_v10(
        records, split, ordinary_pool_records, ordinary_target,
        ordinary_query_degree, ordinary_retrieval_degree, ranking_pairs)

    touched_molecules = set(ranking_touched_molecules)
    for row in ordinary_rows:
        touched_molecules.add(row["query_smiles"])
        touched_molecules.add(row["retrieval_smiles"])

    return {
        "ordinary": ordinary_rows, "ranking": ranking_rows, "molecules": touched_molecules,
        "ranking_anchor_concepts": Counter(str(query["assay_concept"]) for query, _ in anchors),
    }


def _renumber_ranking_rows(rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        member_index = index % v8.RANKING_ANCHOR_WIDTH
        if row["ranking_member_index"] != member_index:
            raise RuntimeError("ranking rows are not contiguous complete anchors")
        anchor_index = index // v8.RANKING_ANCHOR_WIDTH
        row["ranking_query_id"] = f"{row['split']}-rank-{anchor_index:04d}"


def _merge_results(*results: dict) -> dict:
    merged = {"ordinary": [], "ranking": [], "molecules": set(), "ranking_anchor_concepts": Counter()}
    for result in results:
        merged["ordinary"] += result["ordinary"]
        merged["ranking"] += result["ranking"]
        merged["molecules"] |= result["molecules"]
        merged["ranking_anchor_concepts"] += result["ranking_anchor_concepts"]
    _renumber_ranking_rows(merged["ranking"])
    return merged


def _v10_diagnostics(results: dict[str, dict], reserved_molecules: set[str]) -> dict:
    def _pair_ids(split_result: dict) -> set[str]:
        return ({row["pair_id"] for row in split_result["ordinary"]} |
                {row["pair_id"] for row in split_result["ranking"]})

    def _ordinary_achieved(rows: list[dict]) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"transfer": 0, "nontransfer": 0})
        for row in rows:
            label = "transfer" if row["binary_label"] == 1 else "nontransfer"
            counts[str(row["assay_concept"])][label] += 1
        return {concept: dict(labels) for concept, labels in counts.items()}

    touched_union = results["validation"]["molecules"] | results["test"]["molecules"]
    return {
        "reserved_pool_molecules": len(reserved_molecules),
        "touched_pool_molecules": {split: len(results[split]["molecules"]) for split in ("validation", "test")},
        "reserved_but_never_touched_molecules": len(reserved_molecules - touched_union),
        "ranking_anchors_achieved": {
            split: dict(results[split]["ranking_anchor_concepts"]) for split in ("validation", "test")
        },
        "ranking_anchors_target": {
            split: dict(RANKING_ANCHORS_BY_SPLIT[split]) for split in ("validation", "test")
        },
        "ordinary_pairs_achieved": {
            split: _ordinary_achieved(results[split]["ordinary"]) for split in ("validation", "test")
        },
        "ordinary_pairs_target_per_label": v8.ORDINARY_PER_CONCEPT_LABEL,
        "validation_test_molecule_overlap": len(results["validation"]["molecules"] & results["test"]["molecules"]),
        "validation_test_pair_id_overlap": len(_pair_ids(results["validation"]) & _pair_ids(results["test"])),
    }


def _make_split_eval_v10():
    """Memoizing closure, same rationale as v8's _make_split_eval_v8: build() calls
    _split_records once but main() also needs the resulting cache for diagnostics/readers."""
    cache: dict = {}

    def _split_records_v10(rows: list[dict]) -> dict[str, list[dict]]:
        if not cache:
            reserved_molecules = _heldout_union_molecules(rows)
            restricted_rows = [row for row in rows if str(row["canonical_smiles"]) in reserved_molecules]

            anchor_molecules_used: set[str] = set()
            ranking_record_degree: dict[str, int] = defaultdict(int)
            ordinary_query_degree: dict[str, int] = defaultdict(int)
            ordinary_retrieval_degree: dict[str, int] = defaultdict(int)
            state = (reserved_molecules, anchor_molecules_used, ranking_record_degree,
                     ordinary_query_degree, ordinary_retrieval_degree)

            # Manually ordered per user decision: rare concepts (Fg, Fh) built for both splits
            # before either split's common-concept construction touches anything, so common
            # concepts can never consume a molecule/record that rare concepts also needed via the
            # shared, one-time-ever anchor_molecules_used / degree-cap state. Test still gets
            # first access within each phase, matching the existing test-priority convention.
            test_rare = run_construction(restricted_rows, "test", RARE_CONCEPTS, *state)
            validation_rare = run_construction(restricted_rows, "validation", RARE_CONCEPTS, *state)
            test_common = run_construction(restricted_rows, "test", COMMON_CONCEPTS, *state)
            validation_common = run_construction(restricted_rows, "validation", COMMON_CONCEPTS, *state)

            results = {
                "test": _merge_results(test_rare, test_common),
                "validation": _merge_results(validation_rare, validation_common),
            }
            cache["results"], cache["reserved"] = results, reserved_molecules

            validation_molecules, test_molecules = results["validation"]["molecules"], results["test"]["molecules"]
            split_rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
            for row in rows:
                smiles = str(row["canonical_smiles"])
                in_validation, in_test = smiles in validation_molecules, smiles in test_molecules
                if in_validation:
                    split_rows["validation"].append(row)
                if in_test:
                    split_rows["test"].append(row)
                if not in_validation and not in_test and smiles not in reserved_molecules:
                    split_rows["train"].append(row)
            cache["split_rows"] = {name: sorted(values, key=record_key) for name, values in split_rows.items()}
        return cache["split_rows"]
    return _split_records_v10, cache


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("datasets/eligible/assay_transfer_starling_txagent_v9/records.parquet"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("datasets/hf_parquet/assay_transfer_raw_pair_v10_intern"),
    )
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--listnet", action="store_true",
                        help="build indivisible 4-member ListNet groups; default is flat per-pair (v6_5-style)")
    parser.add_argument("--train-groups", type=int, default=250000)
    parser.add_argument("--train-pairs", type=int, default=1250000)
    return parser.parse_args()


def _configure_v10_globals(calibration_path: Path = CALIBRATION_PATH) -> dict:
    global _CALIBRATION_BY_BUCKET
    calibration, _CALIBRATION_BY_BUCKET = _load_calibration(calibration_path)
    pair_core.target_for = _target_for_v10
    pair_driver.target_for = _target_for_v10
    v7.target_for = _target_for_v10
    v8.target_for = _target_for_v10
    v7._is_decisive = _is_decisive_v10
    v8._is_decisive = _is_decisive_v10
    v8._candidate_label = _candidate_label_v10
    v8.RECORD_DEGREE_CAP = RECORD_DEGREE_CAP
    v8.ORDINARY_DEGREE_CAP = ORDINARY_DEGREE_CAP
    pair_core.render_prompt = v8._render_prompt_v8
    return calibration


def _configure_pair_driver(cache: dict, split_records_fn, train_group_iterator) -> None:
    ordinary_fn, ranking_fn = v8._make_eval_readers(cache["results"])
    pair_core.eligible = v8._eligible_with_pair_bucket
    pair_core.iter_list_groups = train_group_iterator
    pair_driver._ordinary = ordinary_fn
    pair_driver._ranking = ranking_fn
    pair_driver.iter_list_groups = train_group_iterator
    pair_driver._write_train_flat = _write_train_flat_variable
    pair_driver._split_records = split_records_fn


def _persist_manifest(output: Path, manifest: dict) -> None:
    text = json.dumps(manifest, indent=2)
    (output / "manifest.json").write_text(text + "\n")
    print(text)


def _verified_source_hash(path: Path) -> str:
    actual = file_sha256(path)
    if actual != ELIGIBLE_SOURCE_SHA256:
        raise ValueError(
            f"v10 eligible-source hash mismatch for {path}: "
            f"expected {ELIGIBLE_SOURCE_SHA256}, found {actual}"
        )
    return actual


def _prepare_build(args: argparse.Namespace) -> tuple:
    source_hash = _verified_source_hash(args.source)
    train_group_iterator = (functools.partial(_iter_list_groups_capped, min_group_size=4)
                            if args.listnet else _iter_list_groups_capped)
    calibration = _configure_v10_globals(args.calibration)
    split_records_fn, cache = _make_split_eval_v10()
    rows = pq.read_table(args.source).to_pylist()
    expected_records = len(rows)
    split_rows = split_records_fn(rows)
    expected_split = {name: len(values) for name, values in split_rows.items()}
    validation_molecules = {str(row["canonical_smiles"]) for row in split_rows["validation"]}
    test_molecules = {str(row["canonical_smiles"]) for row in split_rows["test"]}
    train_molecules = {str(row["canonical_smiles"]) for row in split_rows["train"]}
    assert not (train_molecules & (validation_molecules | test_molecules)), \
        "train leaks molecules already used in validation/test"
    assert not (train_molecules & cache["reserved"]), \
        "train contains a reserved (TxAgent heldout-identity) molecule"
    split_quotas = {
        "molecules": {name: len({r["canonical_smiles"] for r in values})
                     for name, values in split_rows.items()},
        "records": expected_split,
    }
    _configure_pair_driver(cache, split_records_fn, train_group_iterator)
    return (source_hash, calibration, split_records_fn, cache, train_group_iterator,
            expected_records, expected_split, split_quotas)


def _calibration_manifest(calibration: dict, path: Path) -> dict:
    return {
        "schema_version": calibration["schema_version"],
        "artifact": "pairwise_distance_quantiles.json",
        "artifact_sha256": file_sha256(path),
        "source_policy_sha256": calibration["source_policy_sha256"],
        "percentiles": [calibration["lower_percentile"], calibration["upper_percentile"]],
        "anchor_probabilities": [
            calibration["target_probability_at_lower"],
            calibration["target_probability_at_upper"],
        ],
        "sigmoid_tails_clipped": False,
        "bucket_count": len(calibration["buckets"]),
    }


def _enrich_manifest(manifest: dict, args: argparse.Namespace, source_hash: str,
                     calibration: dict, split_quotas: dict, cache: dict) -> None:
    manifest["version"] = "v10-txagent-raw-pair"
    manifest["hub_dataset_id"] = HUB_DATASET_ID
    manifest["eligible_source_sha256"] = source_hash
    manifest["target_calibration"] = _calibration_manifest(calibration, args.calibration)
    manifest["split_quotas"] = split_quotas
    manifest["notes"] = {
        "heldout_identity_driven_reserved_pool": True,
        "validation_test_overlap_allowed": True,
        "fg_in_eval": True,
        "record_degree_cap": RECORD_DEGREE_CAP,
        "ordinary_degree_cap": ORDINARY_DEGREE_CAP,
        "ordinary_candidate_target_a_thresholds": {
            "transfer_min_exclusive": ORDINARY_TRANSFER_MIN_PROBABILITY,
            "nontransfer_max_exclusive": ORDINARY_NONTRANSFER_MAX_PROBABILITY,
        },
    }
    manifest["diagnostics"] = {
        **_v10_diagnostics(cache["results"], cache["reserved"]),
        "pair_bucket_concentration_by_concept": _pair_bucket_concentration(args.source),
        "train_target_distribution": _train_target_diagnostics(Path(manifest["paths"]["train"])),
    }


def main() -> None:
    args = _parse_args()
    prepared = _prepare_build(args)
    (source_hash, calibration, _split_fn, cache, _iterator,
     expected_records, expected_split, split_quotas) = prepared

    volume = args.train_groups if args.listnet else args.train_pairs
    manifest = build(
        args.source, args.output, volume, listnet=args.listnet,
        expected_records=expected_records,
        expected_split=expected_split,
    )
    _enrich_manifest(manifest, args, source_hash, calibration, split_quotas, cache)
    shutil.copyfile(args.calibration, args.output / "pairwise_distance_quantiles.json")
    _persist_manifest(args.output, manifest)


if __name__ == "__main__":
    main()
