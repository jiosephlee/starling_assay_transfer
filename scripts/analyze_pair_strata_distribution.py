#!/usr/bin/env python3
"""Analyze sampled pair strata and target eval allocations without writing pair rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_transfer import (  # noqa: E402
    FingerprintCache,
    bucket_for_value,
    compact_float,
    compute_quantile_thresholds,
    finite_float,
    iter_parquet_rows,
    largest_remainder_allocation,
    normalize_stratum_value,
    stable_priority,
    utc_now,
    write_json,
)


DEFAULT_OUTPUT_DIR = Path("datasets/analysis/pair_strata_distribution")
DEFAULT_METADATA_COLUMNS = [
    "bioavailability_report_type",
    "species_or_population",
    "dose",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
    "extra_details",
]


def load_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = set(args.metadata_columns)
    requested.update([args.smiles_column, args.value_column])
    records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    fp_cache = FingerprintCache(radius=args.fingerprint_radius, nbits=args.fingerprint_nbits)
    for row_index, row in enumerate(
        iter_parquet_rows(
            args.input,
            columns=sorted(requested),
            batch_size=args.read_batch_size,
            max_rows=args.max_rows,
        )
    ):
        value = finite_float(row.get(args.value_column))
        smiles = row.get(args.smiles_column)
        if value is None:
            stats["missing_numeric_value"] += 1
            continue
        if smiles is None or not str(smiles).strip():
            stats["missing_smiles"] += 1
            continue
        canonical_smiles = str(smiles).strip()
        fp = fp_cache.get(canonical_smiles)
        if fp is None:
            stats["invalid_smiles"] += 1
            continue
        records.append(
            {
                "row_index": row_index,
                "canonical_smiles": canonical_smiles,
                "value": float(value),
                "metadata": {column: row.get(column) for column in args.metadata_columns},
                "_fp": fp,
            }
        )
        stats["records_loaded"] += 1
    if len(records) < 2:
        raise RuntimeError("need at least two valid records")
    return records, dict(sorted(stats.items()))


def sampled_pair_indices(
    *,
    n_records: int,
    target_candidates: int,
    seed: int,
    attempt_factor: int,
) -> Iterator[tuple[int, int]]:
    total = n_records * (n_records - 1) // 2
    target = min(target_candidates, total)
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    max_attempts = max(target * attempt_factor, target + 1024)
    attempts = 0
    while len(seen) < target and attempts < max_attempts:
        attempts += 1
        left = rng.randrange(n_records)
        right = rng.randrange(n_records - 1)
        if right >= left:
            right += 1
        if left > right:
            left, right = right, left
        pair = (left, right)
        if pair in seen:
            continue
        seen.add(pair)
        yield pair


def label_pair(diff: float, transfer_threshold: float, not_transfer_threshold: float) -> str | None:
    if diff <= transfer_threshold:
        return "transfer"
    if diff >= not_transfer_threshold:
        return "not_transfer"
    return None


def metadata_pair_value(left: Any, right: Any, mode: str) -> tuple[str, str]:
    if mode == "presence":
        left_text = "not_null" if left is not None else "null"
        right_text = "not_null" if right is not None else "null"
    elif mode == "value":
        left_text = normalize_stratum_value(left)
        right_text = normalize_stratum_value(right)
    else:
        raise ValueError(f"unknown metadata stratification mode: {mode}")
    return tuple(sorted((left_text, right_text)))  # type: ignore[return-value]


def stratum_key(
    *,
    label: str,
    similarity_bucket: int,
    metadata_pairs: tuple[tuple[str, str], ...],
) -> tuple[Any, ...]:
    return (label, similarity_bucket, *metadata_pairs)


def stratum_to_row(
    stratum: tuple[Any, ...],
    *,
    metadata_columns: list[str],
    source_count: int,
    total_source: int,
    proportional_count: int,
    balanced_count: int,
) -> dict[str, Any]:
    label, bucket, *metadata = stratum
    row: dict[str, Any] = {
        "transfer_label": label,
        "similarity_bucket": bucket,
        "source_count": source_count,
        "source_fraction": source_count / total_source if total_source else 0.0,
        "target_30k_proportional_count": proportional_count,
        "target_30k_balanced_count": balanced_count,
    }
    for column, pair_value in zip(metadata_columns, metadata, strict=True):
        row[f"{column}_pair"] = f"{pair_value[0]}<>{pair_value[1]}"
    return row


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, load_stats = load_records(args)
    n = len(records)
    total_possible_pairs = n * (n - 1) // 2
    target_candidates = (
        args.max_candidate_pairs
        if args.max_candidate_pairs is not None
        else math.ceil(total_possible_pairs * args.pair_sample_fraction)
    )
    if target_candidates < 1:
        raise RuntimeError("candidate-pair target is zero")
    fp_cache = FingerprintCache(radius=args.fingerprint_radius, nbits=args.fingerprint_nbits)
    fp_cache.cache.update({record["canonical_smiles"]: record["_fp"] for record in records})

    compact_rows: list[tuple[str, float, tuple[tuple[str, str], ...]]] = []
    drop_reasons: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    similarities: list[float] = []
    candidates_seen = 0
    for left_idx, right_idx in sampled_pair_indices(
        n_records=n,
        target_candidates=target_candidates,
        seed=args.seed,
        attempt_factor=args.attempt_factor,
    ):
        candidates_seen += 1
        left = records[left_idx]
        right = records[right_idx]
        diff = abs(float(left["value"]) - float(right["value"]))
        label = label_pair(diff, args.transfer_threshold, args.not_transfer_threshold)
        if label is None:
            drop_reasons["deadband"] += 1
            continue
        similarity = fp_cache.similarity(left["_fp"], right["_fp"])
        if not math.isfinite(similarity):
            drop_reasons["invalid_similarity"] += 1
            continue
        metadata_pairs = tuple(
            metadata_pair_value(
                left["metadata"].get(column),
                right["metadata"].get(column),
                args.metadata_stratification_mode,
            )
            for column in args.metadata_columns
        )
        compact_rows.append((label, float(similarity), metadata_pairs))
        similarities.append(float(similarity))
        label_counts[label] += 1
        if args.progress_every and candidates_seen % args.progress_every == 0:
            print(
                f"sampled candidates={candidates_seen:,}; labeled={len(compact_rows):,}",
                file=sys.stderr,
                flush=True,
            )

    if not compact_rows:
        raise RuntimeError("no labeled pairs sampled")
    similarity_thresholds = compute_quantile_thresholds(similarities, args.similarity_buckets)
    stratum_counts: Counter[tuple[Any, ...]] = Counter()
    for label, similarity, metadata_pairs in compact_rows:
        stratum_counts[
            stratum_key(
                label=label,
                similarity_bucket=bucket_for_value(similarity, similarity_thresholds),
                metadata_pairs=metadata_pairs,
            )
        ] += 1

    proportional = largest_remainder_allocation(args.target_eval_pairs, stratum_counts)
    balanced = balanced_allocation(args.target_eval_pairs, stratum_counts)
    rows = [
        stratum_to_row(
            stratum,
            metadata_columns=args.metadata_columns,
            source_count=count,
            total_source=len(compact_rows),
            proportional_count=proportional.get(stratum, 0),
            balanced_count=balanced.get(stratum, 0),
        )
        for stratum, count in sorted(stratum_counts.items(), key=lambda item: (-item[1], repr(item[0])))
    ]
    csv_path = args.output_dir / "stratum_distribution.csv"
    with csv_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    top_path = args.output_dir / "top_strata.csv"
    with top_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows[: args.top_n])

    metadata = {
        "schema_version": "pair_strata_distribution_analysis_v1",
        "created_at_utc": utc_now(),
        "input": args.input,
        "output_dir": str(args.output_dir),
        "valid_records": n,
        "load_stats": load_stats,
        "total_possible_pairs": total_possible_pairs,
        "pair_sample_fraction": args.pair_sample_fraction,
        "max_candidate_pairs": args.max_candidate_pairs,
        "candidate_pairs_seen": candidates_seen,
        "labeled_pairs_seen": len(compact_rows),
        "label_counts": dict(sorted(label_counts.items())),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "target_eval_pairs": args.target_eval_pairs,
        "n_strata": len(stratum_counts),
        "nonzero_proportional_strata": sum(1 for value in proportional.values() if value > 0),
        "nonzero_balanced_strata": sum(1 for value in balanced.values() if value > 0),
        "similarity_quantile_thresholds": [compact_float(value) for value in similarity_thresholds],
        "thresholds": {
            "transfer": args.transfer_threshold,
            "not_transfer": args.not_transfer_threshold,
        },
        "metadata_columns": args.metadata_columns,
        "metadata_stratification_mode": args.metadata_stratification_mode,
        "files": {
            "full_distribution": csv_path.name,
            "top_strata": top_path.name,
            "metadata": "metadata.json",
        },
    }
    write_json(args.output_dir / "metadata.json", metadata)
    return metadata


def balanced_allocation(total: int, counts: Counter[Any]) -> dict[Any, int]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="datasets/base/Oral_bioavailability_cleaned")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--value-column", default="oral_bioavailability_value")
    parser.add_argument("--metadata-columns", nargs="+", default=DEFAULT_METADATA_COLUMNS)
    parser.add_argument(
        "--metadata-stratification-mode",
        choices=("presence", "value"),
        default="presence",
    )
    parser.add_argument("--transfer-threshold", type=float, default=10.0)
    parser.add_argument("--not-transfer-threshold", type=float, default=30.0)
    parser.add_argument("--pair-sample-fraction", type=float, default=0.001)
    parser.add_argument("--max-candidate-pairs", type=int, default=1_000_000)
    parser.add_argument("--target-eval-pairs", type=int, default=30_000)
    parser.add_argument("--similarity-buckets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--attempt-factor", type=int, default=10)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-nbits", type=int, default=2048)
    parser.add_argument("--read-batch-size", type=int, default=8192)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()
    if not 0 < args.pair_sample_fraction <= 1:
        parser.error("--pair-sample-fraction must be in (0, 1]")
    if args.max_candidate_pairs is not None and args.max_candidate_pairs < 1:
        parser.error("--max-candidate-pairs must be positive")
    if args.transfer_threshold >= args.not_transfer_threshold:
        parser.error("--transfer-threshold must be less than --not-transfer-threshold")
    for name in ("target_eval_pairs", "similarity_buckets", "attempt_factor", "top_n"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    metadata = analyze(parse_args())
    print(
        json.dumps(
            {
                "output_dir": metadata["output_dir"],
                "valid_records": metadata["valid_records"],
                "total_possible_pairs": metadata["total_possible_pairs"],
                "candidate_pairs_seen": metadata["candidate_pairs_seen"],
                "labeled_pairs_seen": metadata["labeled_pairs_seen"],
                "label_counts": metadata["label_counts"],
                "n_strata": metadata["n_strata"],
                "files": metadata["files"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
