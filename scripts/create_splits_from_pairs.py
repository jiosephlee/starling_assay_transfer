#!/usr/bin/env python3
"""Create molecule-disjoint train/validation/test splits from generic pair records."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_transfer import (  # noqa: E402
    EVAL_SPLITS,
    SPLITS,
    bucket_for_value,
    compact_float,
    compact_json,
    compute_quantile_thresholds,
    finite_float,
    largest_remainder_allocation,
    normalize_stratum_value,
    prepare_output_dir,
    read_jsonl_gz,
    stable_priority,
    utc_now,
    write_json,
)


DEFAULT_INPUT_DIR = Path("datasets/pairs/generic_transfer_pairs")
DEFAULT_OUTPUT_DIR = Path("datasets/pairs_split/generic_transfer_pair_splits")
DEFAULT_METADATA_COLUMNS = [
    "bioavailability_report_type",
    "species_or_population",
    "dose",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
    "extra_details",
]
SPLIT_SCHEMA_VERSION = "generic_transfer_pair_splits_v1"
SPLIT_VERSION = "single_assay_molecule_disjoint_pair_first_stratified_v2"
BALANCED_SUBSET = "balanced_stratum"
PROPORTIONAL_SUBSET = "proportional_stratum"


def pair_molecules(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row["molecule_a"]["canonical_smiles"]),
        str(row["molecule_b"]["canonical_smiles"]),
    )


def pair_stratum(
    row: dict[str, Any],
    *,
    similarity_thresholds: list[float],
    metadata_columns: list[str],
    metadata_stratification_mode: str,
) -> tuple[Any, ...]:
    similarity = finite_float(row.get("weighted_tanimoto"))
    if similarity is None:
        raise ValueError(f"missing weighted_tanimoto for {row.get('pair_id')}")
    label = row.get("transfer_label")
    if label not in {"transfer", "not_transfer"}:
        raise ValueError(f"invalid transfer_label for {row.get('pair_id')}: {label!r}")
    parts: list[Any] = [label, bucket_for_value(similarity, similarity_thresholds)]
    for column in metadata_columns:
        if metadata_stratification_mode == "presence":
            presence = row.get("stratification_metadata") or {}
            if column in presence:
                left, right = presence[column]
            else:
                left_value = (row["molecule_a"].get("metadata") or {}).get(column)
                right_value = (row["molecule_b"].get("metadata") or {}).get(column)
                left = "not_null" if left_value is not None else "null"
                right = "not_null" if right_value is not None else "null"
        elif metadata_stratification_mode == "value":
            left = normalize_stratum_value((row["molecule_a"].get("metadata") or {}).get(column))
            right = normalize_stratum_value((row["molecule_b"].get("metadata") or {}).get(column))
        else:
            raise ValueError(f"unknown metadata stratification mode: {metadata_stratification_mode}")
        parts.append(tuple(sorted((left, right))))
    return tuple(parts)


def stratum_string(stratum: tuple[Any, ...], metadata_columns: list[str]) -> str:
    label, bucket, *meta = stratum
    parts = [f"transfer_label={label}", f"similarity_bucket={bucket}"]
    for column, values in zip(metadata_columns, meta, strict=True):
        parts.append(f"{column}={values[0]}<>{values[1]}")
    return "|".join(parts)


def serializable_counter(counter: Counter[Any], metadata_columns: list[str]) -> dict[str, int]:
    return {
        stratum_string(key, metadata_columns) if isinstance(key, tuple) else str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: repr(item[0]))
    }


def balanced_allocation(total: int, counts: Counter[Any]) -> dict[Any, int]:
    """Allocate as evenly as possible across strata, capped by available counts."""
    if total <= 0:
        return {}
    active = [key for key, count in sorted(counts.items(), key=lambda item: repr(item[0])) if count > 0]
    allocation: Counter[Any] = Counter()
    remaining = min(total, sum(counts.values()))
    while remaining > 0 and active:
        next_active: list[Any] = []
        progressed = False
        for key in active:
            if remaining <= 0:
                break
            if allocation[key] >= counts[key]:
                continue
            allocation[key] += 1
            remaining -= 1
            progressed = True
            if allocation[key] < counts[key]:
                next_active.append(key)
        if not progressed:
            break
        active = next_active
    return dict(allocation)


def load_pairs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[float]]:
    input_path = args.input_dir / "records.jsonl.gz"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    rows: list[dict[str, Any]] = []
    similarities: list[float] = []
    for row in read_jsonl_gz(input_path, max_rows=args.max_rows):
        if row.get("schema_version") != "generic_transfer_pairs_v1":
            raise ValueError(f"unexpected source schema: {row.get('schema_version')}")
        similarity = finite_float(row.get("weighted_tanimoto"))
        if similarity is None:
            raise ValueError(f"invalid weighted_tanimoto for {row.get('pair_id')}")
        rows.append(row)
        similarities.append(float(similarity))
    if not rows:
        raise RuntimeError("no pair rows loaded")
    return rows, similarities


def build_graph(rows: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, list[int]]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    molecule_edges: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        left, right = pair_molecules(row)
        if left == right:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
        molecule_edges[left].append(idx)
        molecule_edges[right].append(idx)
    return adjacency, molecule_edges


def internal_edges_for_molecules(
    molecules: set[str],
    molecule_edges: dict[str, list[int]],
    rows: list[dict[str, Any]],
    available_edges: set[int],
) -> set[int]:
    candidates: set[int] = set()
    for mol in molecules:
        candidates.update(molecule_edges.get(mol, []))
    out: set[int] = set()
    for idx in candidates:
        if idx not in available_edges:
            continue
        left, right = pair_molecules(rows[idx])
        if left in molecules and right in molecules:
            out.add(idx)
    return out


def pair_reuse_count(row: dict[str, Any], selected_molecules: set[str]) -> int:
    left, right = pair_molecules(row)
    return int(left in selected_molecules) + int(right in selected_molecules)


def choose_stratum(
    *,
    remaining: Counter[tuple[Any, ...]],
    allocation: dict[tuple[Any, ...], int],
    selectable_counts: Counter[tuple[Any, ...]],
    seed: int,
    split: str,
    step: int,
) -> tuple[Any, ...] | None:
    best: tuple[float, int, int, int] | None = None
    best_stratum: tuple[Any, ...] | None = None
    for stratum, need in remaining.items():
        if need <= 0 or selectable_counts.get(stratum, 0) <= 0:
            continue
        quota = max(1, allocation.get(stratum, 1))
        score = (
            need / quota,
            need,
            selectable_counts[stratum],
            -stable_priority(seed, split, "stratum", step, repr(stratum)),
        )
        if best is None or score > best:
            best = score
            best_stratum = stratum
    return best_stratum


def choose_pair_from_stratum(
    *,
    stratum: tuple[Any, ...],
    edges_by_stratum: dict[tuple[Any, ...], list[int]],
    selected_edges: set[int],
    selected_molecules: set[str],
    rows: list[dict[str, Any]],
    seed: int,
    split: str,
    step: int,
) -> int | None:
    best: tuple[int, int, int, int] | None = None
    best_idx: int | None = None
    for idx in edges_by_stratum.get(stratum, []):
        if idx in selected_edges:
            continue
        left, right = pair_molecules(rows[idx])
        reuse = int(left in selected_molecules) + int(right in selected_molecules)
        new_molecules = 2 - reuse
        score = (
            reuse,
            -new_molecules,
            -stable_priority(seed, split, "pair", step, rows[idx].get("pair_id")),
            -idx,
        )
        if best is None or score > best:
            best = score
            best_idx = idx
    return best_idx


def select_split_pairs(
    *,
    rows: list[dict[str, Any]],
    available_edges: set[int],
    excluded_molecules: set[str],
    target_pairs: int,
    allocation_mode: str,
    strata_by_edge: dict[int, tuple[Any, ...]],
    available_strata: Counter[tuple[Any, ...]],
    initial_selected_molecules: set[str] | None,
    seed: int,
    split: str,
) -> tuple[set[int], set[str], dict[str, Any]]:
    if allocation_mode == "balanced":
        allocation = balanced_allocation(target_pairs, available_strata)
    elif allocation_mode == "proportional":
        allocation = largest_remainder_allocation(target_pairs, available_strata)
    else:
        raise ValueError(f"unknown allocation mode: {allocation_mode}")
    selected_edges: set[int] = set()
    selected_molecules: set[str] = set(initial_selected_molecules or set())
    selected_strata: Counter[tuple[Any, ...]] = Counter()
    remaining: Counter[tuple[Any, ...]] = Counter(allocation)
    edges_by_stratum: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    selectable_counts: Counter[tuple[Any, ...]] = Counter()
    skipped = Counter()
    reuse_counts: Counter[str] = Counter()

    for idx in sorted(available_edges):
        left, right = pair_molecules(rows[idx])
        if left in excluded_molecules or right in excluded_molecules:
            skipped["touches_excluded_molecule"] += 1
            continue
        stratum = strata_by_edge[idx]
        if allocation.get(stratum, 0) <= 0:
            skipped["unallocated_stratum"] += 1
            continue
        edges_by_stratum[stratum].append(idx)
        selectable_counts[stratum] += 1

    step = 0
    while len(selected_edges) < target_pairs and remaining:
        stratum = choose_stratum(
            remaining=remaining,
            allocation=allocation,
            selectable_counts=selectable_counts,
            seed=seed,
            split=split,
            step=step,
        )
        if stratum is None:
            skipped["no_selectable_remaining_stratum"] += 1
            break
        idx = choose_pair_from_stratum(
            stratum=stratum,
            edges_by_stratum=edges_by_stratum,
            selected_edges=selected_edges,
            selected_molecules=selected_molecules,
            rows=rows,
            seed=seed,
            split=split,
            step=step,
        )
        if idx is None:
            remaining[stratum] = 0
            selectable_counts[stratum] = 0
            skipped["stratum_exhausted"] += 1
            continue
        reuse = pair_reuse_count(rows[idx], selected_molecules)
        selected_edges.add(idx)
        selected_molecules.update(pair_molecules(rows[idx]))
        selected_strata[stratum] += 1
        remaining[stratum] -= 1
        selectable_counts[stratum] -= 1
        reuse_counts[str(reuse)] += 1
        if remaining[stratum] <= 0:
            del remaining[stratum]
        step += 1

    metadata = {
        "target_pairs": target_pairs,
        "allocation_mode": allocation_mode,
        "selected_pairs": len(selected_edges),
        "selected_molecules": len(selected_molecules),
        "selected_strata": selected_strata,
        "allocation": allocation,
        "unfilled_allocation": dict(sorted((str(k), int(v)) for k, v in remaining.items() if v > 0)),
        "reuse_counts": dict(sorted(reuse_counts.items())),
        "skipped": dict(sorted(skipped.items())),
    }
    return selected_edges, selected_molecules, metadata


def write_splits(
    *,
    rows: list[dict[str, Any]],
    split_edges: dict[str, set[int]],
    eval_subset_by_edge: dict[int, str],
    strata_by_edge: dict[int, tuple[Any, ...]],
    output_dir: Path,
    gzip_compresslevel: int,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows_by_split": Counter(),
        "rows_by_eval_subset": {split: Counter() for split in SPLITS},
        "transfer_label_counts": {split: Counter() for split in SPLITS},
        "stratum_counts": {split: Counter() for split in SPLITS},
        "unique_molecules": {split: set() for split in SPLITS},
        "thrown_out_pairs": Counter(),
        "thrown_out_strata": Counter(),
    }
    edge_to_split = {
        idx: split
        for split, indexes in split_edges.items()
        for idx in indexes
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: gzip.open(output_dir / f"{split}.jsonl.gz", "wt", compresslevel=gzip_compresslevel)
        for split in SPLITS
    }
    try:
        for idx, row in enumerate(rows):
            split = edge_to_split.get(idx)
            if split is None:
                stats["thrown_out_pairs"]["cross_split_or_unused"] += 1
                stats["thrown_out_strata"][strata_by_edge[idx]] += 1
                continue
            out = dict(row)
            out["schema_version"] = SPLIT_SCHEMA_VERSION
            out["source_schema_version"] = row.get("schema_version")
            out["split"] = split
            out["split_version"] = SPLIT_VERSION
            out["eval_subset"] = eval_subset_by_edge.get(idx)
            handles[split].write(compact_json(out) + "\n")
            stats["rows_by_split"][split] += 1
            stats["rows_by_eval_subset"][split][out["eval_subset"] or "none"] += 1
            stats["transfer_label_counts"][split][str(row.get("transfer_label"))] += 1
            stats["stratum_counts"][split][strata_by_edge[idx]] += 1
            stats["unique_molecules"][split].update(pair_molecules(row))
    finally:
        for handle in handles.values():
            handle.close()
    return stats


def summarize_stats(stats: dict[str, Any], metadata_columns: list[str]) -> dict[str, Any]:
    return {
        "rows_by_split": dict(sorted(stats["rows_by_split"].items())),
        "rows_by_eval_subset": {
            split: dict(sorted(stats["rows_by_eval_subset"][split].items())) for split in SPLITS
        },
        "transfer_label_counts": {
            split: dict(sorted(stats["transfer_label_counts"][split].items())) for split in SPLITS
        },
        "stratum_counts": {
            split: serializable_counter(stats["stratum_counts"][split], metadata_columns)
            for split in SPLITS
        },
        "unique_molecules": {
            split: len(stats["unique_molecules"][split]) for split in SPLITS
        },
        "thrown_out_pairs": dict(sorted(stats["thrown_out_pairs"].items())),
        "thrown_out_strata": serializable_counter(stats["thrown_out_strata"], metadata_columns),
    }


def validate(stats: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    molecules = stats["unique_molecules"]
    overlaps: dict[str, int] = {}
    for left_idx, left in enumerate(SPLITS):
        for right in SPLITS[left_idx + 1 :]:
            key = f"{left}_{right}"
            overlaps[key] = len(molecules[left] & molecules[right])
            if overlaps[key]:
                errors.append(f"{key} molecule overlap: {overlaps[key]}")
    return {
        "molecule_overlap": overlaps,
        "n_errors": len(errors),
        "errors": errors[:100],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    prepare_output_dir(
        args.output_dir,
        ["train.jsonl.gz", "validation.jsonl.gz", "test.jsonl.gz", "metadata.json"],
        args.overwrite,
    )
    rows, similarities = load_pairs(args)
    similarity_thresholds = compute_quantile_thresholds(similarities, args.similarity_buckets)
    strata_by_edge = {
        idx: pair_stratum(
            row,
            similarity_thresholds=similarity_thresholds,
            metadata_columns=args.metadata_columns,
            metadata_stratification_mode=args.metadata_stratification_mode,
        )
        for idx, row in enumerate(rows)
    }
    adjacency, molecule_edges = build_graph(rows)
    all_edges = set(range(len(rows)))
    all_strata = Counter(strata_by_edge.values())

    split_edges: dict[str, set[int]] = {split: set() for split in SPLITS}
    split_molecules: dict[str, set[str]] = {split: set() for split in SPLITS}
    selection_metadata: dict[str, Any] = {}
    eval_subset_by_edge: dict[int, str] = {}
    excluded_molecules: set[str] = set()
    available_edges = set(all_edges)

    for split in EVAL_SPLITS:
        selection_metadata[split] = {}
        for subset, target, mode in (
            (BALANCED_SUBSET, args.balanced_pairs_per_eval_split, "balanced"),
            (PROPORTIONAL_SUBSET, args.proportional_pairs_per_eval_split, "proportional"),
        ):
            subset_available_edges = available_edges - split_edges[split]
            available_strata = Counter(strata_by_edge[idx] for idx in subset_available_edges)
            selected_edges, selected_molecules, metadata = select_split_pairs(
                rows=rows,
                available_edges=subset_available_edges,
                excluded_molecules=excluded_molecules,
                target_pairs=target,
                allocation_mode=mode,
                strata_by_edge=strata_by_edge,
                available_strata=available_strata,
                initial_selected_molecules=split_molecules[split],
                seed=args.seed,
                split=f"{split}:{subset}",
            )
            split_edges[split].update(selected_edges)
            split_molecules[split].update(selected_molecules)
            for idx in selected_edges:
                eval_subset_by_edge[idx] = subset
            selection_metadata[split][subset] = metadata
        excluded_molecules.update(split_molecules[split])
        available_edges = {
            idx
            for idx in available_edges
            if not (set(pair_molecules(rows[idx])) & excluded_molecules)
        }

    train_molecules = set(adjacency) - excluded_molecules
    split_molecules["train"] = train_molecules
    split_edges["train"] = internal_edges_for_molecules(
        train_molecules,
        molecule_edges,
        rows,
        available_edges,
    )
    stats = write_splits(
        rows=rows,
        split_edges=split_edges,
        eval_subset_by_edge=eval_subset_by_edge,
        strata_by_edge=strata_by_edge,
        output_dir=args.output_dir,
        gzip_compresslevel=args.gzip_compresslevel,
    )
    validation = validate(stats)
    metadata = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_pair_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "source_rows": len(rows),
        "split_version": SPLIT_VERSION,
        "seed": args.seed,
        "target_pairs": {
            split: {
                BALANCED_SUBSET: args.balanced_pairs_per_eval_split,
                PROPORTIONAL_SUBSET: args.proportional_pairs_per_eval_split,
                "total": args.balanced_pairs_per_eval_split
                + args.proportional_pairs_per_eval_split,
            }
            for split in EVAL_SPLITS
        },
        "selection_policy": {
            "order": "validation_then_test_then_train",
            "eval_unit": "pair_first_stratum_quota_selection",
            "eval_subsets": {
                BALANCED_SUBSET: "equal allocation across available strata, capped by stratum size",
                PROPORTIONAL_SUBSET: "largest-remainder allocation proportional to available strata",
            },
            "eval_pair_selection": "within needed strata, prefer pairs with two reused molecules, then one, then zero",
            "eval_internal_edge_policy": (
                "do not close eval molecule sets over all internal edges; this keeps stratum "
                "quotas primary"
            ),
            "train_policy": "all available sampled pairs internal to remaining molecules",
            "cross_split_pair_policy": "throw_out_any_pair_touching_validation_or_test_molecules",
            "molecule_overlap_allowed": False,
            "stratification_fields": ["transfer_label", "similarity_bucket", *args.metadata_columns],
            "metadata_stratification_mode": args.metadata_stratification_mode,
            "null_token": "__NULL__",
        },
        "metadata_columns": args.metadata_columns,
        "similarity_quantile_thresholds": [compact_float(value) for value in similarity_thresholds],
        "source_strata": serializable_counter(all_strata, args.metadata_columns),
        "selection": {
            split: {
                subset: {
                    **{
                        key: value
                        for key, value in selection_metadata[split][subset].items()
                        if key not in {"selected_strata", "allocation"}
                    },
                    "selected_strata": serializable_counter(
                        selection_metadata[split][subset].get("selected_strata", Counter()),
                        args.metadata_columns,
                    ),
                    "allocation": serializable_counter(
                        Counter(selection_metadata[split][subset].get("allocation", {})),
                        args.metadata_columns,
                    ),
                }
                for subset in (BALANCED_SUBSET, PROPORTIONAL_SUBSET)
            }
            for split in EVAL_SPLITS
        },
        "write_stats": summarize_stats(stats, args.metadata_columns),
        "validation": validation,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-columns", nargs="+", default=DEFAULT_METADATA_COLUMNS)
    parser.add_argument(
        "--metadata-stratification-mode",
        choices=("presence", "value"),
        default="presence",
        help="Use null/not-null flags by default; value mode uses exact metadata values.",
    )
    parser.add_argument("--balanced-pairs-per-eval-split", type=int, default=15_000)
    parser.add_argument("--proportional-pairs-per-eval-split", type=int, default=15_000)
    parser.add_argument("--similarity-buckets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--gzip-compresslevel", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.balanced_pairs_per_eval_split < 1 or args.proportional_pairs_per_eval_split < 1:
        parser.error("balanced/proportional eval subset targets must be positive")
    if args.similarity_buckets < 2:
        parser.error("--similarity-buckets must be at least 2")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    if not 0 <= args.gzip_compresslevel <= 9:
        parser.error("--gzip-compresslevel must be between 0 and 9")
    return args


def main() -> None:
    metadata = build(parse_args())
    print(
        json.dumps(
            {
                "output_dir": metadata["output_dir"],
                "rows_by_split": metadata["write_stats"]["rows_by_split"],
                "unique_molecules": metadata["write_stats"]["unique_molecules"],
                "thrown_out_pairs": metadata["write_stats"]["thrown_out_pairs"],
                "validation": metadata["validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
