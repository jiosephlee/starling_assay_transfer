#!/usr/bin/env python3
"""Build v7 raw-pair Parquets from the TxAgent-sourced eligible records.

Thin wrapper around ``pipeline.pair_core`` / ``scripts/build_raw_pair.py`` --
same pairing/labeling/rendering logic, unchanged. The only difference from the v6.5 driver:
this source's eligible-record count and unique-molecule count differ from v6.5's frozen dataset,
so ``pipeline.pair_core.SPLIT_SIZES``/``SPLIT_RECORDS`` (hardcoded to v6.5's exact counts,
not parameterized) are monkey-patched here to v7's own frozen quotas before calling ``build()``.

v7's quotas were derived once (not re-derived per run) by running
``pipeline.pair_core._heldout_subset`` against the real v7 molecule-weight distribution, holding
validation/test at 1250 molecules each (same absolute size as v6.5, per user decision) and
picking the natural achievable record total nearest each pool's proportional average. See
``pipeline/source_normalization/starling_txagent_eligible.py`` for the eligible-records adapter.

Also, ``scripts.build_raw_pair._ordinary``/``_ranking`` hardcode their heldout-benchmark
row targets (``2000 = 5 concepts * 2 labels * 200``, ``1000`` distinct ranking anchors) as
v6.5-dataset-specific assumptions, not parameters. TxAgent's current Fg normalization snapshot has
zero valid records (all 27,671 ``fg`` rows fail with ``missing_canonical_unit``/
``unresolved_structure`` upstream, confirmed not an adapter bug), so v7 only has 4 populated assay
concepts. Local copies below compute/parameterize these targets instead of assuming v6.5's numbers
(125/concept/label = 1,000/split; 500 ranking anchors x 20 candidates = 10,000/split).

``pipeline.pair_core.eligible()`` only checks ``canonical_endpoint_key`` equality (plus different
molecule) -- it ignores ``pair_bucket_key`` (TxAgent's condition key: source_id + canonical_endpoint
+ canonical_unit + source-specific context fields such as assay system, dose bin, species) entirely
when deciding which records may be paired, even though every eligible row already carries it. Left
as-is, this lets e.g. an intrinsic-clearance pair form between a hepatocyte-system value and a
microsomal-system value -- different, non-comparable assay systems, with nothing gating against it.
``eligible`` is monkey-patched below to also require matching ``pair_bucket_key``, applied globally
(all 4 concepts, per user decision).

Train diversity: ``_ordinary`` already caps ``retrieval_degree[...] < 8`` so no single molecule
dominates as a retrieval partner in eval, but the train path (``iter_list_groups``/``_span_four``)
has no equivalent cap at all. Local copies below thread a shared, global ``retrieval_degree``
counter through train generation with the same cap. It also had no cap on the query/anchor side --
``iter_list_groups``'s round-robin simply cycles every record in a concept forever, so a molecule
with many raw eligible records (e.g. a heavily-studied reference compound) could be selected as the
query tens of thousands of times, starving the group budget away from everything else. A second,
independent ``query_degree`` counter (``TRAIN_QUERY_DEGREE_CAP = 6``, same value as the retrieval
cap, tracked separately -- so a given record can appear at most 6 emitted rows as query + 6 as
retrieval = 12 times total) is threaded through ``_iter_list_groups_capped`` below to bound this
too. Note both caps are keyed by raw ``record_key`` (child_id), not ``canonical_smiles`` -- a
molecule with many distinct eligible records is not deduplicated across those records.

Source/bucket diversity: TxAgent's ``source_id`` maps 1:1 onto ``assay_concept``
(``SOURCE_TO_CONCEPT`` in ``starling_txagent_eligible.py`` -- ``fa``/``fg``/``fh``/``oral_exposure``/
``direct_hf`` each own exactly one concept), so the existing concept-level round-robin already
balances source representation for free. Bucket diversity was not free, though: query selection
used to round-robin over a flat, per-concept list of *records*, so a mega ``pair_bucket_key`` with
thousands of records still got proportionally thousands of times more query-attempt volume than a
small bucket, even after both were individually degree-capped. ``_iter_list_groups_capped`` now
round-robins two levels deep -- concept, then ``pair_bucket_key`` within that concept, then record
within that bucket -- so every bucket gets one query attempt per concept pass regardless of how many
records it contains. Buckets with fewer than 2 records (never pairable) are dropped up front. Note
this round-robin is only locally fair (per pass, among currently-live buckets), not globally fair
over the whole run: a bucket's total lifespan is still bounded by how many distinct records it has
(each record capped at ``TRAIN_QUERY_DEGREE_CAP`` query uses before the bucket's per-record budgets
run out), so a concept/bucket with far more distinct records can still end up contributing more
total rows over a long run even though every live bucket gets an equal number of attempts per pass.

Bucket coverage: ``_span_four_capped``'s caller previously discarded any attempt that didn't produce
exactly 4 chosen retrievals. Since a query can only find *other* molecules sharing its own
``pair_bucket_key``, this structurally excluded every bucket with fewer than 5 total records (query
+ 4 distinct partners) -- 2,656 of the 5,017 pair_bucket_keys with >=2 molecules (53%), confirmed
against the frozen eligible-records artifact. ``_iter_list_groups_capped`` now accepts a
``min_group_size`` (default 1) and only discards an attempt that yields zero chosen retrievals,
so small buckets can still contribute partial (1-3 member) groups in the default flat/pairwise
build. ``--listnet`` mode keeps the strict exactly-4 requirement (``min_group_size=4``) since a
ListNet group's ranking loss assumes a fixed width. Because average group size can now drop below 4
in flat mode, ``_write_train_flat_variable`` (a local copy of
``build_raw_pair._write_train_flat``) requests a generous ``wanted_groups = wanted_pairs``
upper bound from the generator instead of assuming every group yields exactly 4 rows, so the flat
train split doesn't under-shoot ``--train-pairs`` merely because more groups are now partial.

Decisiveness: every row (all splits) gets an ``is_decisive`` boolean
(``distance <= transfer_max or distance >= not_transfer_min``, reusing the boundary logic already
written -- but dead -- as ``build_raw_pair._is_decisive``), so the deadband-vs-decisive
split is visible and filterable in the dataset itself instead of invisible as before.

Template: prompts render from ``templates/assay_transfer_v7_intern/`` (new, TxAgent-schema-aware
per-concept templates -- cites ``policy_family``/``endpoint_subtype``/``threshold_display`` from
TxAgent's registry directly), not the reused v6.5 templates.

Metadata contract: ``row_for()``'s default ``metadata`` (``{"target_distribution": {...},
"target_z": ...}``) doesn't match what the SFT/ranking training harness expects --
``metadata.soft_targets`` keyed by the raw completion tokens ("(A)"/"(B)"), and
``metadata.groups.assay_concept`` (not a bare top-level lookup) for grouped/ranking evaluation.
``_fix_metadata()`` reshapes every row's ``metadata`` after each ``row_for()`` call, in all three
row-producing paths (`_ordinary_variable_concepts`, `_ranking_variable_size`,
`_iter_list_groups_capped`).
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
import scripts.build_raw_pair as pair_driver
from pipeline.pair_core import (propose_records, record_key, row_for,
                                stable_hash, target_for, unordered_pair_id)
from pipeline.pair_core import render_prompt as _original_render_prompt
from scripts.build_raw_pair import build, selected_concepts

# Frozen for the TxAgent Bioavailability_Ma eligible-records artifact
# (datasets/eligible/assay_transfer_starling_txagent_v7/records.parquet), re-derived against the
# updated (more granular) TxAgent endpoint-policy registry: 18,406 unique molecules / 201,177
# eligible records total.
SPLIT_SIZES = {"train": 15906, "validation": 1250, "test": 1250}
SPLIT_RECORDS = {"train": 173853, "validation": 13662, "test": 13662}

EVAL_PER_CONCEPT_LABEL = 125       # 125 * 4 concepts * 2 labels = 1,000 per split
RANKING_ANCHORS = 500              # 500 anchors * 20 candidates = 10,000 per split
TRAIN_RETRIEVAL_DEGREE_CAP = 6     # each record can appear as retrieval at most 6x
TRAIN_QUERY_DEGREE_CAP = 6         # separate counter from retrieval_degree; +6 as query = 12x total
TRAIN_WRITE_BATCH_ROWS = 32_768
TRAIN_ZSTD_LEVEL = 3

V7_TEMPLATE_DIR = str(Path(__file__).resolve().parents[1] / "templates" / "assay_transfer_v7_intern")


def _render_prompt_v7(query: dict, retrieval: dict) -> str:
    return _original_render_prompt(query, retrieval, template_dir=V7_TEMPLATE_DIR)


def _eligible_with_pair_bucket(query: dict, retrieval: dict) -> bool:
    """pair_core.eligible() plus a matching pair_bucket_key (condition-key) requirement."""
    return (query["canonical_endpoint_key"] == retrieval["canonical_endpoint_key"] and
            query["canonical_smiles"] != retrieval["canonical_smiles"] and
            query["pair_bucket_key"] == retrieval["pair_bucket_key"])


def _condition_index_by_bucket(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Like pipeline.pair_core.condition_index, but keyed on (canonical_endpoint_key,
    pair_bucket_key) instead of canonical_endpoint_key alone.

    Since eligible() now requires an exact pair_bucket_key match on top of the endpoint key,
    indexing by endpoint key alone means propose_records() has to linearly scan a pool that may
    be mostly bucket-ineligible (e.g. oral_bioavailability has 12,797 molecules under one
    endpoint key, fragmented across many report-type/dose buckets) before finding count matches.
    Pre-splitting the pool by the same key eligible() actually checks removes that wasted scan.
    """
    indexed: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        indexed[(str(row["canonical_endpoint_key"]), row["pair_bucket_key"])].append(row)
    return indexed


def _fix_metadata(row: dict) -> None:
    """Reshape row_for()'s metadata into the contract the SFT/ranking harness requires:
    metadata.soft_targets keyed by the raw completion tokens (not metadata.target_distribution),
    and metadata.groups.assay_concept (not a bare top-level lookup) for grouped/ranking eval."""
    distribution = row["target_distribution"]
    row["metadata"] = {
        "soft_targets": {"(A)": distribution["transfer"], "(B)": distribution["nontransfer"]},
        "groups": {"assay_concept": row["assay_concept"]},
        "target_z": row["target_z"],
    }


def _is_decisive(row: dict, query: dict) -> bool:
    return row["distance"] <= query["transfer_max"] or row["distance"] >= query["not_transfer_min"]


def _candidate_label(query: dict, retrieval: dict) -> str | None:
    distance = target_for(query, retrieval)["distance"]
    if distance <= query["transfer_max"]:
        return "transfer"
    return "nontransfer" if distance >= query["not_transfer_min"] else None


def _ordinary_variable_concepts(records: list[dict], split: str, forbidden: set[str]) -> list[dict]:
    """Copy of build_raw_pair._ordinary: variable concept count, downsized target,
    plus an is_decisive flag on every row (trivially True here, kept for schema uniformity)."""
    concepts = selected_concepts(records)
    per_label = EVAL_PER_CONCEPT_LABEL
    target_total = len(concepts) * 2 * per_label
    indexed, selected = _condition_index_by_bucket(records), defaultdict(list)
    query_degree, retrieval_degree = defaultdict(int), defaultdict(int)
    queries = sorted(records, key=lambda row: stable_hash(record_key(row)))
    for pass_index in range(32):
        for query in queries:
            concept = query["assay_concept"]
            if query_degree[record_key(query)] >= 16:
                continue
            candidates = propose_records(
                query, indexed[(query["canonical_endpoint_key"], query["pair_bucket_key"])], 128,
                forbidden, f"ordinary:{pass_index}")
            labels = sorted(("transfer", "nontransfer"), key=lambda label: len(selected[concept, label]))
            for wanted_label in labels:
                retrieval = next((item for item in candidates if _candidate_label(query, item) == wanted_label
                                  and retrieval_degree[record_key(item)] < 8), None)
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
        if all(len(selected[concept, label]) == per_label for concept in concepts
               for label in ("transfer", "nontransfer")):
            break
    output = [row for concept in concepts for label in ("transfer", "nontransfer")
              for row in selected[concept, label]]
    if len(output) != target_total:
        counts = {f"{concept}:{label}": len(selected[concept, label])
                  for concept in concepts for label in ("transfer", "nontransfer")}
        raise RuntimeError(f"{split} ordinary benchmark requires {target_total} rows: {counts}")
    return output


def _ranking_variable_size(records: list[dict], split: str, forbidden: set[str]) -> list[dict]:
    """Copy of build_raw_pair._ranking with a downsized anchor count + is_decisive."""
    anchors_wanted = RANKING_ANCHORS
    indexed, anchors = _condition_index_by_bucket(records), []
    for query in records:
        candidates = propose_records(
            query, indexed[(query["canonical_endpoint_key"], query["pair_bucket_key"])], 20,
            forbidden, "ranking")
        if len(candidates) == 20:
            anchors.append((query, candidates))
    anchors.sort(key=lambda item: stable_hash(record_key(item[0])))
    selected, used_molecules = [], set()
    for query, candidates in anchors:
        if query["canonical_smiles"] in used_molecules:
            continue
        selected.append((query, candidates))
        used_molecules.add(query["canonical_smiles"])
        if len(selected) == anchors_wanted:
            break
    if len(selected) != anchors_wanted:
        raise RuntimeError(f"{split} ranking benchmark needs {anchors_wanted} distinct-molecule anchors")
    output = []
    for number, (query, candidates) in enumerate(selected):
        for index, retrieval in enumerate(candidates):
            row = row_for(query, retrieval, split)
            row["is_decisive"] = _is_decisive(row, query)
            _fix_metadata(row)
            row.update({"ranking_query_id": f"{split}-rank-{number:04d}", "ranking_member_index": index})
            output.append(row)
    return output


def _span_four_capped(query: dict, proposed: list[dict], retrieval_degree: dict) -> list[dict]:
    """Copy of pipeline.pair_core._span_four with a global retrieval-degree cap added."""
    def score(row: dict) -> float:
        target = target_for(query, row)
        return float(target["target_z"] if "target_z" in target else target["target_a"])

    ranked = sorted(proposed, key=score)
    desired = [0, len(ranked) // 3, 2 * len(ranked) // 3, len(ranked) - 1]
    chosen, molecule_counts = [], defaultdict(int)
    for position in desired:
        order = sorted(range(len(ranked)), key=lambda index: (abs(index - position), index))
        candidate = next(
            (ranked[index] for index in order
             if ranked[index] not in chosen
             and molecule_counts[ranked[index]["canonical_smiles"]] < 2
             and retrieval_degree[record_key(ranked[index])] < TRAIN_RETRIEVAL_DEGREE_CAP),
            None,
        )
        if candidate is not None:
            chosen.append(candidate)
            molecule_counts[candidate["canonical_smiles"]] += 1
    return chosen


def _scheduled_buckets(records: list[dict]) -> tuple[dict, dict[str, list[str]]]:
    concept_buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in sorted(records, key=lambda item: stable_hash(record_key(item))):
        concept_buckets[str(row["assay_concept"])][row["pair_bucket_key"]].append(row)
    for concept in list(concept_buckets):
        for bucket in list(concept_buckets[concept]):
            if len(concept_buckets[concept][bucket]) < 2:
                del concept_buckets[concept][bucket]
        if not concept_buckets[concept]:
            del concept_buckets[concept]
    bucket_lists: dict[str, list[str]] = {
        concept: sorted(buckets, key=stable_hash) for concept, buckets in concept_buckets.items()
    }
    return concept_buckets, bucket_lists


def _next_live_bucket(
    concept: str, buckets: list[str], live: set[str], offsets: dict[str, int],
) -> str | None:
    offset = offsets[concept]
    for step in range(len(buckets)):
        candidate = buckets[(offset + step) % len(buckets)]
        if candidate in live:
            offsets[concept] = (offset + step + 1) % len(buckets)
            return candidate
    return None


def _deactivate_failed_bucket(
    bucket_key: tuple[str, str], bucket_records: list[dict], failures: dict,
    active_buckets: dict[str, set[str]], active_concepts: set[str],
) -> None:
    failures[bucket_key] += 1
    if failures[bucket_key] < len(bucket_records):
        return
    concept, bucket = bucket_key
    active_buckets[concept].discard(bucket)
    if not active_buckets[concept]:
        active_concepts.discard(concept)


def _materialize_group(
    query: dict, chosen: list[dict], split: str, group: int, used: set,
    retrieval_degree: dict[str, int],
) -> list[dict]:
    group_id = f"{split}-list-{group:09d}"
    rows = []
    for index, retrieval in enumerate(chosen):
        used.add(unordered_pair_id(query, retrieval))
        retrieval_degree[record_key(retrieval)] += 1
        item = row_for(query, retrieval, split)
        item["is_decisive"] = _is_decisive(item, query)
        _fix_metadata(item)
        item.update({"listnet_group_id": group_id, "listnet_group_index": group,
                     "listnet_query_group_id": record_key(query), "listnet_member_index": index})
        rows.append(item)
    return rows


def _iter_list_groups_capped(
    records: list[dict], split: str, wanted: int, min_group_size: int = 1,
):
    """Generate bucket-balanced groups under independent query/retrieval degree caps."""
    indexed = _condition_index_by_bucket(records)
    concept_buckets, bucket_lists = _scheduled_buckets(records)
    retrieval_degree: dict[str, int] = defaultdict(int)
    query_degree: dict[str, int] = defaultdict(int)
    concept_bucket_offset: dict[str, int] = defaultdict(int)
    record_offset: dict[tuple[str, str], int] = defaultdict(int)
    bucket_failures: dict[tuple[str, str], int] = defaultdict(int)
    active_buckets: dict[str, set[str]] = {concept: set(buckets) for concept, buckets in bucket_lists.items()}
    active_concepts, used = set(active_buckets), set()
    group = 0
    while group < wanted and active_concepts:
        for concept in sorted(tuple(active_concepts)):
            buckets, live = bucket_lists[concept], active_buckets[concept]
            bucket = _next_live_bucket(concept, buckets, live, concept_bucket_offset)
            if bucket is None:
                active_concepts.discard(concept)
                continue
            bucket_records = concept_buckets[concept][bucket]
            bucket_key = (concept, bucket)
            query = bucket_records[record_offset[bucket_key] % len(bucket_records)]
            record_offset[bucket_key] += 1

            query_id = record_key(query)
            remaining = TRAIN_QUERY_DEGREE_CAP - query_degree[query_id]
            chosen: list[dict] = []
            if remaining > 0:
                proposed = propose_records(
                    query, indexed[(query["canonical_endpoint_key"], query["pair_bucket_key"])], 16,
                    used, f"{group}:{record_offset[bucket_key]}")
                chosen = _span_four_capped(query, proposed, retrieval_degree)[:remaining]
            if len(chosen) < min_group_size:
                _deactivate_failed_bucket(
                    bucket_key, bucket_records, bucket_failures,
                    active_buckets, active_concepts,
                )
                continue
            bucket_failures[bucket_key] = 0
            query_degree[query_id] += len(chosen)
            yield _materialize_group(query, chosen, split, group, used, retrieval_degree)
            group += 1
            if group >= wanted:
                break


def _write_train_flat_variable(output: Path, records: list[dict], wanted_pairs: int,
                               schema=None) -> tuple[str, int]:
    """Copy of build_raw_pair._write_train_flat that requests a generous group budget
    (``wanted_groups = wanted_pairs``, i.e. the worst case of every group yielding a single row)
    instead of assuming every group yields exactly 4 rows -- v7's default groups can be 1-4 wide."""
    path = output / "train" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    wanted_groups = wanted_pairs
    writer, batch, count = None, [], 0
    for group in _iter_list_groups_capped(records, "train", wanted_groups):
        for row in group:
            if count >= wanted_pairs:
                break
            for column in pair_driver._LISTNET_COLUMNS:
                row.pop(column, None)
            batch.append(row)
            count += 1
        if len(batch) >= TRAIN_WRITE_BATCH_ROWS:
            writer = pair_driver._flush(
                path, writer, batch, schema, compression_level=TRAIN_ZSTD_LEVEL
            )
            batch = []
        if count >= wanted_pairs:
            break
    if batch:
        writer = pair_driver._flush(
            path, writer, batch, schema, compression_level=TRAIN_ZSTD_LEVEL
        )
    if writer is None:
        raise RuntimeError("no usable v7 training pairs")
    writer.close()
    return str(path), count


def _pair_bucket_concentration(source_path: Path) -> dict[str, Any]:
    """Diagnostic only: how concentrated is each concept's largest pair_bucket_key."""
    table = pq.read_table(source_path, columns=["assay_concept", "pair_bucket_key"])
    df = table.to_pandas()
    report = {}
    for concept, group in df.groupby("assay_concept"):
        counts = Counter(group["pair_bucket_key"])
        total = len(group)
        largest_key, largest_count = counts.most_common(1)[0]
        report[concept] = {
            "distinct_pair_bucket_keys": len(counts),
            "total_records": total,
            "largest_bucket_fraction": round(largest_count / total, 4),
            "largest_bucket_key": largest_key,
        }
    return report


def _train_target_diagnostics(train_path: Path) -> dict[str, Any]:
    """Diagnostic only: how much of train is decisive vs. deadband."""
    table = pq.read_table(train_path, columns=["target_a", "is_decisive"])
    df = table.to_pandas()
    quantiles = df["target_a"].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).to_dict()
    return {
        "target_a_quantiles": {str(k): round(float(v), 4) for k, v in quantiles.items()},
        "decisive_fraction": round(float(df["is_decisive"].mean()), 4),
        "rows": len(df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("datasets/eligible/assay_transfer_starling_txagent_v7/records.parquet"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("datasets/hf_parquet/assay_transfer_raw_pair_v7_intern"),
    )
    parser.add_argument("--listnet", action="store_true",
                        help="build indivisible 4-member ListNet groups; default is flat per-pair (v6_5-style)")
    parser.add_argument("--train-groups", type=int, default=250000)
    parser.add_argument("--train-pairs", type=int, default=1250000)
    args = parser.parse_args()

    # --listnet keeps the strict exactly-4 requirement (a ListNet group's ranking loss assumes a
    # fixed width); the default flat/pairwise build allows partial (1-3 member) groups so buckets
    # too small to ever fill 4 slots can still contribute pairs. See module docstring.
    train_group_iterator = (functools.partial(_iter_list_groups_capped, min_group_size=4)
                            if args.listnet else _iter_list_groups_capped)

    pair_core.SPLIT_SIZES = SPLIT_SIZES
    pair_core.SPLIT_RECORDS = SPLIT_RECORDS
    pair_core.eligible = _eligible_with_pair_bucket
    pair_core.render_prompt = _render_prompt_v7
    pair_core.iter_list_groups = train_group_iterator
    pair_driver._ordinary = _ordinary_variable_concepts
    pair_driver._ranking = _ranking_variable_size
    pair_driver.iter_list_groups = train_group_iterator
    pair_driver._write_train_flat = _write_train_flat_variable

    volume = args.train_groups if args.listnet else args.train_pairs
    manifest = build(
        args.source, args.output, volume, listnet=args.listnet,
        expected_records=sum(SPLIT_RECORDS.values()),
        expected_split=SPLIT_RECORDS,
    )
    manifest["version"] = "v7-txagent-raw-pair"
    manifest["split_quotas"] = {"molecules": SPLIT_SIZES, "records": SPLIT_RECORDS}
    manifest["diagnostics"] = {
        "pair_bucket_concentration_by_concept": _pair_bucket_concentration(args.source),
        "train_target_distribution": _train_target_diagnostics(Path(manifest["paths"]["train"])),
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
