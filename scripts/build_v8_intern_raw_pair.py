#!/usr/bin/env python3
"""Build v8 raw-pair Parquets: pairs-first validation/test construction, replacing v7/v7.1's
molecule-first reservation.

Thin wrapper around ``scripts/build_v7_intern_raw_pair.py`` -- reuses its pairing/rendering/train
logic directly (imported, not copied). Validation/test construction is new:

1. **Eligible-records source has no heldout exclusion.** v7/v7.1 dropped any record whose molecule
   overlaps Starling's own Bioavailability_Ma held-out test set (random + scaffold splits) entirely
   (``rejected_heldout_excluded`` in ``starling_txagent_eligible.py``) -- 26,714 records / 715
   distinct molecules, discarding real signal instead of routing it anywhere useful. v8's
   eligible-records artifact (``datasets/eligible/assay_transfer_starling_txagent_v8``) is built
   with ``--disable-heldout-exclusion``, so those 715 molecules' records (all 5 concepts, Fg
   included) are present and available.

2. **Validation/test are built pairs-first, not molecule-first.** The old approach (reserve N
   molecules via SHA-256 priority, then hope there's enough same-``pair_bucket_key`` density to
   sample benchmark pairs from them) had no formula relating the reserved molecule count to what
   was actually needed -- the real binding constraint is ranking's requirement of 20 same-bucket
   candidates per anchor, which varies by orders of magnitude in feasibility across concepts
   (``oral_bioavailability`` has only 6 distinct buckets; ``Fa`` has 1,842; ``Fg`` has 161 across
   just 1,056 records). And the old ``_ranking_variable_size`` had no per-concept fairness at all,
   so bucket-dense concepts could dominate the ranking benchmark while fragmented ones contributed
   nothing -- the same fairness bug already fixed for train's round-robin, never applied to eval.

   v8 instead generates exactly what's needed via the same concept-then-bucket round-robin
   mechanism already validated for train (``_iter_list_groups_capped`` in
   ``build_v7_intern_raw_pair.py``), and lets whichever molecules get touched define the split.
   Validation and test are each built independently, starting from the full eligible pool, and are
   explicitly allowed to overlap each other (an accepted simplification -- train staying disjoint
   from their union is the invariant that matters). For each split: ranking anchors are built
   first (the harder constraint), Starling-overlap molecules exhausted before new ones (a literal
   two-phase process -- see ``_ranking_anchors_v8``), then ordinary (binary transfer/non-transfer)
   pairs are drawn from the exact same records ranking already touched, not a separate reservation.
   Fg's per-concept targets are halved (its pool is known thin), and the whole construction is
   best-effort -- shortfall is fine, not a ``RuntimeError`` like the old benchmarks raised. Finally
   ``train = eligible - (validation_molecules | test_molecules)``.

   Ranking candidates have no hard reuse cap across anchors, per user decision -- a molecule may
   appear as a candidate for as many different anchors as the pool allows, but ``_ranking_span``
   *prefers* molecules the split's construction has already touched, so the eval molecule universe
   doesn't grow unboundedly larger than necessary. That preference is tolerance-bounded, not
   unconditional: candidates are still spread by *value*, not rank position (for `width`
   evenly-spaced target_z points between the pool's min and max, pick the closest candidate to
   each), and a touched molecule only gets to fill a slot if it's within half the inter-point
   spacing of that slot's target -- otherwise the plain nearest-value match wins, same as if there
   were no preference at all. So reuse never comes at the cost of a slot's spread quality. Still at
   most twice within any one anchor's own 20 candidates (in-group distinctness, local to that
   anchor only, not persisted between anchors). Ordinary's query/retrieval caps stay record-level
   and symmetric (8/8), matching train's existing convention (just at eval's own cap value).

   Because ``scripts.build_raw_pair.build()`` calls ``_split_records`` once up front and
   then ``_ordinary`` before ``_ranking`` per split -- backwards from "ranking first, ordinary
   reuses ranking's pool" -- the real construction (both phases, both benchmark types, both splits)
   happens entirely inside ``_split_records`` (cached), and the ``_ordinary``/``_ranking``
   monkeypatch targets become pure cache reads. See ``_make_split_eval_v8``/``_make_eval_readers``.

3. **Prompts render through the standalone ``pipeline.prompt_rendering`` module** and
   ``templates/assay_transfer_v8_intern/`` -- see that module/directory for the placeholder-token
   cleaning and template fixes (unrelated to this split redesign, done earlier).

Everything else -- ``TRAIN_RETRIEVAL_DEGREE_CAP``/``TRAIN_QUERY_DEGREE_CAP`` (6/6), the train
concept/bucket/record round-robin, variable-size (1-4) flat groups, ``pair_bucket_key`` pairing
gate, ``is_decisive`` column, sigmoid target -- is unchanged from v7/v7.1.
"""

from __future__ import annotations

import argparse
import functools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

import pipeline.pair_core as pair_core
import pipeline.source_normalization.starling_txagent_eligible as ste
import scripts.build_raw_pair as pair_driver
from pipeline.pair_core import propose_records, record_key, row_for, stable_hash, target_for, unordered_pair_id
from pipeline.prompt_rendering import render_prompt as _render_prompt_v8_impl
from scripts.build_raw_pair import build
from scripts.build_v7_intern_raw_pair import (
    _candidate_label, _condition_index_by_bucket, _eligible_with_pair_bucket, _fix_metadata,
    _is_decisive, _iter_list_groups_capped, _pair_bucket_concentration, _train_target_diagnostics,
    _write_train_flat_variable,
)

V8_TEMPLATE_DIR = str(Path(__file__).resolve().parents[1] / "templates" / "assay_transfer_v8_intern")

RANKING_ANCHOR_WIDTH = 20
ORDINARY_DEGREE_CAP = 8     # record-level query_degree / retrieval_degree cap for ordinary pairs
# Cross-split reusable-but-capped limit: a record may be used up to this many times *combined*
# across validation AND test (as a ranking candidate, or as an ordinary query/retrieval), not
# per-split. Validation is built first and consumes this budget; test inherits whatever's left,
# which is what actually forces the two splits to diverge instead of independently recomputing
# near-identical constructions. Anchor molecules are separately hard-deduplicated (not capped) --
# see anchor_molecules_used.
RECORD_DEGREE_CAP = 8

# Fg halved -- its overlap pool (38 molecules) and bucket density (161 buckets / 1,056 records) are
# both known thin. These are ceilings, not guarantees: the whole construction is best-effort.
# Ranking anchor targets are asymmetric between splits, per user decision: test is built first
# (see _make_split_eval_v8) and gets the larger target below; validation gets roughly half of
# test's per-concept numbers (so test is the split more weighted towards Starling-overlap coverage,
# since it gets first access to both the shared cross-split budget and the overlap-molecule pool).
RANKING_ANCHORS_PER_CONCEPT_TEST = {   # 4*50 + 25 = 225 anchors -> <=4,500 rows
    "oral_bioavailability": 50, "oral_exposure": 50, "Fa": 50, "Fh": 50, "Fg": 25,
}
RANKING_ANCHORS_PER_CONCEPT_VALIDATION = {   # ~half of the test targets (Fg: 10, not exactly half)
    "oral_bioavailability": 25, "oral_exposure": 25, "Fa": 25, "Fh": 25, "Fg": 10,
}
ORDINARY_PER_CONCEPT_LABEL = {
    "oral_bioavailability": 100, "oral_exposure": 100, "Fa": 100, "Fh": 100, "Fg": 50,
}


def _render_prompt_v8(query: dict, retrieval: dict) -> str:
    return _render_prompt_v8_impl(query, retrieval, template_dir=V8_TEMPLATE_DIR)


def _overlap_molecules(rows: list[dict]) -> set[str]:
    """Molecules whose parent structure identity matches Starling's own Bioavailability_Ma
    held-out test set (random + scaffold splits) -- 715 distinct molecules confirmed against the
    v8 eligible-records artifact."""
    heldout_keys = ste._heldout_parent_keys([ste.DEFAULT_RANDOM_HELDOUT, ste.DEFAULT_SCAFFOLD_HELDOUT])
    parent_key_by_molecule: dict[str, str] = {}
    for row in rows:
        smiles = str(row["canonical_smiles"])
        if smiles not in parent_key_by_molecule:
            parent_key_by_molecule[smiles] = ste._record_parent_key(row)
    return {smiles for smiles, key in parent_key_by_molecule.items() if key in heldout_keys}


def _ranking_span(query: dict, proposed: list[dict], width: int, touched_molecules: set[str],
                  record_degree: dict[str, int]) -> list[dict]:
    """Select a value-spread candidate list without mutating shared degree state.

    Slots are evenly spaced from the pool's minimum to maximum target_z. A previously touched
    molecule wins a slot when it is within 0.75 slot widths of the target; otherwise the nearest
    available value wins. Each molecule may appear twice per anchor, while each distinct record
    must remain below the caller's shared cross-split degree cap. The caller commits degrees only
    after accepting a complete list, so an incomplete attempt has no persistent state effects."""
    if not proposed:
        return []
    scored = sorted(((row, target_for(query, row)["target_z"]) for row in proposed), key=lambda item: item[1])
    z_min, z_max = scored[0][1], scored[-1][1]
    slot_width = (z_max - z_min) / (width - 1) if width > 1 else 0.0
    tolerance = slot_width * 0.75   # loosened slightly from 0.5, per user decision
    desired_values = [z_min + i * slot_width for i in range(width)]
    chosen, molecule_counts = [], defaultdict(int)
    for target_value in desired_values:
        def _open(item: tuple[dict, float]) -> bool:
            row, _ = item
            return (row not in chosen and molecule_counts[row["canonical_smiles"]] < 2
                    and record_degree.get(record_key(row), 0) < RECORD_DEGREE_CAP)

        preferred = min(
            (item for item in scored if _open(item) and item[0]["canonical_smiles"] in touched_molecules
             and abs(item[1] - target_value) <= tolerance),
            key=lambda item: abs(item[1] - target_value),
            default=None,
        )
        candidate = preferred if preferred is not None else min(
            (item for item in scored if _open(item)),
            key=lambda item: abs(item[1] - target_value),
            default=None,
        )
        if candidate is not None:
            row, _ = candidate
            chosen.append(row)
            molecule_counts[row["canonical_smiles"]] += 1
    return chosen


def _ranking_anchors_v8(records: list[dict], split: str, overlap_molecules: set[str],
                        per_concept_target: dict[str, int], anchor_molecules_used: set[str],
                        record_degree: dict[str, int]) -> tuple[list[tuple[dict, list[dict]]], set[str]]:
    """Two-phase concept->bucket->record round-robin (structural fork of
    _iter_list_groups_capped) building ranking anchors: exactly RANKING_ANCHOR_WIDTH same-bucket
    candidates per anchor, value-spread via _ranking_span (candidates reusable up to
    RECORD_DEGREE_CAP, preferring already-touched molecules within a per-slot tolerance -- see
    _ranking_span), per-concept budgeted. A molecule can only become an anchor once *ever*, not
    just once per split -- `anchor_molecules_used` and `record_degree` are passed in by the caller
    and shared across both validation's and test's construction (validation runs first and
    consumes the budget; test inherits whatever's left), which is what forces the two splits to
    diverge instead of independently recomputing the same deterministic construction. Phase 1
    exhausts the Starling-overlap pool; Phase 2 tops up from everything else. Returns the anchor
    list and the union of every molecule touched (as anchor or candidate) across both phases --
    also the live set _ranking_span reads its "already touched" preference from (this one *is*
    reset per split -- see _v8_split_eval -- so the preference doesn't fight the divergence goal by
    pulling test back towards validation's molecules)."""
    used: set[str] = set()
    remaining_budget = dict(per_concept_target)
    all_anchors: list[tuple[dict, list[dict]]] = []
    touched_molecules: set[str] = set()

    def _run_phase(pool: list[dict], phase_name: str) -> None:
        indexed = _condition_index_by_bucket(pool)
        concept_buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for row in sorted(pool, key=lambda item: stable_hash(record_key(item))):
            concept_buckets[str(row["assay_concept"])][row["pair_bucket_key"]].append(row)
        for concept in list(concept_buckets):
            for bucket in list(concept_buckets[concept]):
                if len(concept_buckets[concept][bucket]) < 2:
                    del concept_buckets[concept][bucket]
            if not concept_buckets[concept]:
                del concept_buckets[concept]

        bucket_lists = {c: sorted(b, key=stable_hash) for c, b in concept_buckets.items()}
        concept_bucket_offset: dict[str, int] = defaultdict(int)
        record_offset: dict[tuple[str, str], int] = defaultdict(int)
        bucket_failures: dict[tuple[str, str], int] = defaultdict(int)
        active_buckets = {c: set(b) for c, b in bucket_lists.items()}
        active_concepts = {c for c in active_buckets if remaining_budget.get(c, 0) > 0}
        group = 0
        while active_concepts:
            for concept in sorted(tuple(active_concepts)):
                buckets, live = bucket_lists[concept], active_buckets[concept]
                offset, bucket = concept_bucket_offset[concept], None
                for step in range(len(buckets)):
                    candidate_bucket = buckets[(offset + step) % len(buckets)]
                    if candidate_bucket in live:
                        bucket = candidate_bucket
                        concept_bucket_offset[concept] = (offset + step + 1) % len(buckets)
                        break
                if bucket is None:
                    active_concepts.discard(concept)
                    continue

                bucket_records = concept_buckets[concept][bucket]
                bucket_key = (concept, bucket)
                query = bucket_records[record_offset[bucket_key] % len(bucket_records)]
                record_offset[bucket_key] += 1

                chosen: list[dict] = []
                if query["canonical_smiles"] not in anchor_molecules_used:
                    bucket_pool = indexed[(query["canonical_endpoint_key"], query["pair_bucket_key"])]
                    # Request the whole bucket, not a small multiple of the width: propose_records'
                    # early-stop-at-count is a performance optimization for train's hot path (millions
                    # of calls), which doesn't apply here (only hundreds of calls total for the whole
                    # ranking benchmark) -- capping low would let _ranking_span's value-based spread
                    # only see a random pseudo-random subsample of a large bucket, biasing its min/max
                    # away from the bucket's true extremes.
                    proposed = propose_records(
                        query, bucket_pool, len(bucket_pool), used,
                        f"{split}:{phase_name}:{group}:{record_offset[bucket_key]}")
                    chosen = _ranking_span(query, proposed, RANKING_ANCHOR_WIDTH, touched_molecules,
                                           record_degree)
                if len(chosen) < RANKING_ANCHOR_WIDTH:
                    bucket_failures[bucket_key] += 1
                    if bucket_failures[bucket_key] >= len(bucket_records):
                        live.discard(bucket)
                        if not live:
                            active_concepts.discard(concept)
                    continue

                bucket_failures[bucket_key] = 0
                anchor_molecules_used.add(query["canonical_smiles"])
                record_degree[record_key(query)] += 1
                for retrieval in chosen:
                    record_degree[record_key(retrieval)] += 1
                    used.add(unordered_pair_id(query, retrieval))
                all_anchors.append((query, chosen))
                touched_molecules.add(query["canonical_smiles"])
                touched_molecules.update(r["canonical_smiles"] for r in chosen)
                remaining_budget[concept] -= 1
                group += 1
                if remaining_budget[concept] <= 0:
                    active_concepts.discard(concept)

    overlap_pool = [r for r in records if r["canonical_smiles"] in overlap_molecules]
    _run_phase(overlap_pool, "overlap")

    new_pool = [r for r in records if r["canonical_smiles"] not in touched_molecules]
    _run_phase(new_pool, "new")

    return all_anchors, touched_molecules


def _ranking_rows_from_anchors(anchors: list[tuple[dict, list[dict]]], split: str) -> list[dict]:
    output = []
    for number, (query, candidates) in enumerate(anchors):
        for index, retrieval in enumerate(candidates):
            row = row_for(query, retrieval, split)
            row["is_decisive"] = _is_decisive(row, query)
            _fix_metadata(row)
            row.update({"ranking_query_id": f"{split}-rank-{number:04d}", "ranking_member_index": index})
            output.append(row)
    return output


def _ordinary_pairs_v8(records: list[dict], split: str, touched_records: set[str],
                       per_concept_target: dict[str, int], query_degree: dict[str, int],
                       retrieval_degree: dict[str, int]) -> list[dict]:
    """Structural fork of _ordinary_variable_concepts: pool restricted to the exact records
    ranking already touched (not a separate reservation), record-level 8/8 degree caps,
    per-concept targets (Fg halved), best-effort -- returns whatever was achieved instead of
    raising on shortfall. `query_degree`/`retrieval_degree` are passed in by the caller and shared
    across both splits (validation first, test inherits the remaining budget), same cross-split
    divergence reasoning as _ranking_anchors_v8's `record_degree`.

    Retrieval selection prefers a record *not yet* used by ordinary anywhere (`retrieval_degree
    [record] == 0`, the same shared dict the degree cap reads) over one that's already been picked,
    still subject to the cap -- so ordinary pairs spread out across the touched-records pool rather
    than concentrating repeat usage on whichever records `propose_records` happens to return first.
    Note this can only affect *which specific records* get used within ranking's already-fixed
    touched-molecule set -- ordinary's pool is a strict subset of ranking's own selected records
    (see `touched_records` above), so no choice made here can add or remove a molecule from the
    split's final molecule universe; it only reshuffles usage within it."""
    pool = [r for r in records if record_key(r) in touched_records]
    concepts = sorted({str(r["assay_concept"]) for r in pool
                       if per_concept_target.get(str(r["assay_concept"]), 0) > 0})
    indexed, selected = _condition_index_by_bucket(pool), defaultdict(list)
    forbidden: set[str] = set()
    queries = sorted(pool, key=lambda row: stable_hash(record_key(row)))
    for pass_index in range(32):
        for query in queries:
            concept = str(query["assay_concept"])
            per_label = per_concept_target.get(concept, 0)
            if per_label <= 0 or query_degree[record_key(query)] >= ORDINARY_DEGREE_CAP:
                continue
            candidates = propose_records(
                query, indexed[(query["canonical_endpoint_key"], query["pair_bucket_key"])], 128,
                forbidden, f"{split}:ordinary:{pass_index}")
            labels = sorted(("transfer", "nontransfer"), key=lambda label: len(selected[concept, label]))
            for wanted_label in labels:
                eligible = [item for item in candidates if _candidate_label(query, item) == wanted_label
                           and retrieval_degree[record_key(item)] < ORDINARY_DEGREE_CAP]
                retrieval = next((item for item in eligible if retrieval_degree[record_key(item)] == 0), None)
                if retrieval is None and eligible:
                    retrieval = eligible[0]
                if retrieval is None or len(selected[concept, wanted_label]) >= per_label:
                    continue
                row = row_for(query, retrieval, split)
                row["binary_label"] = 1 if wanted_label == "transfer" else 0
                row["is_decisive"] = _is_decisive(row, query)
                _fix_metadata(row)
                selected[concept, wanted_label].append(row)
                forbidden.add(unordered_pair_id(query, retrieval))
                query_degree[record_key(query)] += 1
                retrieval_degree[record_key(retrieval)] += 1
                break
        if all(len(selected[concept, label]) >= per_concept_target.get(concept, 0)
               for concept in concepts for label in ("transfer", "nontransfer")):
            break
    return [row for concept in concepts for label in ("transfer", "nontransfer")
            for row in selected[concept, label]]


def _v8_split_eval(records: list[dict], split: str, overlap_molecules: set[str],
                   anchor_molecules_used: set[str], ranking_record_degree: dict[str, int],
                   ordinary_query_degree: dict[str, int],
                   ordinary_retrieval_degree: dict[str, int]) -> dict[str, Any]:
    """Ranking anchors first, then ordinary pairs drawn from ranking's touched-records pool. The
    four degree/dedup structures are shared with the caller across both splits -- see
    _make_split_eval_v8. See module docstring."""
    ranking_target = (RANKING_ANCHORS_PER_CONCEPT_TEST if split == "test"
                      else RANKING_ANCHORS_PER_CONCEPT_VALIDATION)
    anchors, ranking_touched_molecules = _ranking_anchors_v8(
        records, split, overlap_molecules, ranking_target,
        anchor_molecules_used, ranking_record_degree)
    ranking_rows = _ranking_rows_from_anchors(anchors, split)
    touched_records = {record_key(query) for query, _ in anchors}
    touched_records.update(record_key(c) for _, candidates in anchors for c in candidates)

    ordinary_rows = _ordinary_pairs_v8(records, split, touched_records, ORDINARY_PER_CONCEPT_LABEL,
                                       ordinary_query_degree, ordinary_retrieval_degree)

    touched_molecules = set(ranking_touched_molecules)
    for row in ordinary_rows:
        touched_molecules.add(row["query_smiles"])
        touched_molecules.add(row["retrieval_smiles"])

    return {
        "ordinary": ordinary_rows,
        "ranking": ranking_rows,
        "molecules": touched_molecules,
        "ranking_anchor_concepts": Counter(str(query["assay_concept"]) for query, _ in anchors),
    }


def _make_split_eval_v8():
    """Memoizing closure: build() calls _split_records once, but main() also needs the resulting
    row lists/cache up front for build()'s expected_split and for the _ordinary/_ranking reader
    functions below -- caching avoids repeating the (expensive) real construction.

    The four shared degree/dedup structures (anchor_molecules_used, ranking's record_degree,
    ordinary's query_degree/retrieval_degree) are created once here and passed into *both* splits'
    _v8_split_eval calls -- **test runs first**, per user decision, so it gets first access to both
    the shared budget and the Starling-overlap molecule pool (matching its larger per-concept
    targets, RANKING_ANCHORS_PER_CONCEPT_TEST); validation runs second and inherits whatever's
    left, which is also what forces the two splits to diverge instead of independently recomputing
    the same deterministic construction."""
    cache: dict[str, Any] = {}

    def _split_records_v8(rows: list[dict]) -> dict[str, list[dict]]:
        if not cache:
            overlap_molecules = _overlap_molecules(rows)
            anchor_molecules_used: set[str] = set()
            ranking_record_degree: dict[str, int] = defaultdict(int)
            ordinary_query_degree: dict[str, int] = defaultdict(int)
            ordinary_retrieval_degree: dict[str, int] = defaultdict(int)
            cache["test"] = _v8_split_eval(
                rows, "test", overlap_molecules, anchor_molecules_used, ranking_record_degree,
                ordinary_query_degree, ordinary_retrieval_degree)
            cache["validation"] = _v8_split_eval(
                rows, "validation", overlap_molecules, anchor_molecules_used, ranking_record_degree,
                ordinary_query_degree, ordinary_retrieval_degree)
            validation_molecules = cache["validation"]["molecules"]
            test_molecules = cache["test"]["molecules"]
            split_rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
            for row in rows:
                smiles = str(row["canonical_smiles"])
                in_validation, in_test = smiles in validation_molecules, smiles in test_molecules
                if in_validation:
                    split_rows["validation"].append(row)
                if in_test:
                    split_rows["test"].append(row)
                if not in_validation and not in_test:
                    split_rows["train"].append(row)
            cache["split_rows"] = {name: sorted(values, key=record_key) for name, values in split_rows.items()}
        return cache["split_rows"]
    return _split_records_v8, cache


def _make_eval_readers(cache: dict[str, Any]):
    def _ordinary(rows: list[dict], split: str, used: set[str]) -> list[dict]:
        return cache[split]["ordinary"]

    def _ranking(rows: list[dict], split: str, used: set[str]) -> list[dict]:
        return cache[split]["ranking"]
    return _ordinary, _ranking


def _eval_diagnostics(cache: dict[str, Any]) -> dict[str, Any]:
    validation_molecules, test_molecules = cache["validation"]["molecules"], cache["test"]["molecules"]

    def _pair_ids(split_cache: dict[str, Any]) -> set[str]:
        return {row["pair_id"] for row in split_cache["ordinary"]} | {row["pair_id"] for row in split_cache["ranking"]}

    def _ordinary_achieved(rows: list[dict]) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"transfer": 0, "nontransfer": 0})
        for row in rows:
            label = "transfer" if row["binary_label"] == 1 else "nontransfer"
            counts[str(row["assay_concept"])][label] += 1
        return {concept: dict(labels) for concept, labels in counts.items()}

    return {
        "ranking_anchors_achieved": {
            split: dict(cache[split]["ranking_anchor_concepts"]) for split in ("validation", "test")
        },
        "ranking_anchors_target": {
            "validation": RANKING_ANCHORS_PER_CONCEPT_VALIDATION, "test": RANKING_ANCHORS_PER_CONCEPT_TEST,
        },
        "ordinary_pairs_achieved": {
            split: _ordinary_achieved(cache[split]["ordinary"]) for split in ("validation", "test")
        },
        "ordinary_pairs_target_per_label": ORDINARY_PER_CONCEPT_LABEL,
        "validation_test_molecule_overlap": len(validation_molecules & test_molecules),
        "validation_test_pair_id_overlap": len(_pair_ids(cache["validation"]) & _pair_ids(cache["test"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("datasets/eligible/assay_transfer_starling_txagent_v8/records.parquet"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("datasets/hf_parquet/assay_transfer_raw_pair_v8_intern"),
    )
    parser.add_argument("--target-validation", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--target-test", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--listnet", action="store_true",
                        help="build indivisible 4-member ListNet groups; default is flat per-pair (v6_5-style)")
    parser.add_argument("--train-groups", type=int, default=250000)
    parser.add_argument("--train-pairs", type=int, default=1250000)
    args = parser.parse_args()

    if args.target_validation is not None or args.target_test is not None:
        print("warning: --target-validation/--target-test are deprecated no-ops under v8's "
             "pairs-first construction (per-concept targets replace a single split-size number); "
             "ignoring.")

    train_group_iterator = (functools.partial(_iter_list_groups_capped, min_group_size=4)
                            if args.listnet else _iter_list_groups_capped)

    split_records_fn, cache = _make_split_eval_v8()
    rows = pq.read_table(args.source).to_pylist()
    expected_records = len(rows)
    split_rows = split_records_fn(rows)
    expected_split = {name: len(values) for name, values in split_rows.items()}

    validation_molecules, test_molecules = cache["validation"]["molecules"], cache["test"]["molecules"]
    train_molecules = {str(row["canonical_smiles"]) for row in split_rows["train"]}
    assert not (train_molecules & (validation_molecules | test_molecules)), \
        "train leaks molecules already used in validation/test"

    overlap_molecules = _overlap_molecules(rows)
    split_quotas = {
        "molecules": {name: len({r["canonical_smiles"] for r in values})
                     for name, values in split_rows.items()},
        "records": expected_split,
        "overlap_molecule_fraction": {
            "validation": round(len(validation_molecules & overlap_molecules) / max(1, len(validation_molecules)), 4),
            "test": round(len(test_molecules & overlap_molecules) / max(1, len(test_molecules)), 4),
        },
    }

    ordinary_fn, ranking_fn = _make_eval_readers(cache)
    pair_core.eligible = _eligible_with_pair_bucket
    pair_core.render_prompt = _render_prompt_v8
    pair_core.iter_list_groups = train_group_iterator
    pair_driver._ordinary = ordinary_fn
    pair_driver._ranking = ranking_fn
    pair_driver.iter_list_groups = train_group_iterator
    pair_driver._write_train_flat = _write_train_flat_variable
    pair_driver._split_records = split_records_fn

    volume = args.train_groups if args.listnet else args.train_pairs
    manifest = build(
        args.source, args.output, volume, listnet=args.listnet,
        expected_records=expected_records,
        expected_split=expected_split,
    )
    manifest["version"] = "v8-txagent-raw-pair"
    manifest["split_quotas"] = split_quotas
    manifest["notes"] = {
        "starling_overlap_pairs_first_val_test": True,
        "fg_in_eval": True,
    }
    manifest["diagnostics"] = {
        **_eval_diagnostics(cache),
        "pair_bucket_concentration_by_concept": _pair_bucket_concentration(args.source),
        "train_target_distribution": _train_target_diagnostics(Path(manifest["paths"]["train"])),
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
