#!/usr/bin/env python3
"""Create generic molecular transfer pairs from one numeric dataset."""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from rdkit import DataStructs

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_transfer import (  # noqa: E402
    FingerprintCache,
    bucket_for_value,
    compact_float,
    compact_json,
    compute_quantile_thresholds,
    finite_float,
    normalize_stratum_value,
    prepare_output_dir,
    stable_hash_text,
    utc_now,
    write_json,
)


DEFAULT_OUTPUT_DIR = Path("datasets/pairs/generic_transfer_pairs")
DEFAULT_METADATA_COLUMNS = [
    "bioavailability_report_type",
    "species_or_population",
    "dose",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
    "extra_details",
]
PAIR_SCHEMA_VERSION = "generic_transfer_pairs_v1"


def encode_fp(fp: Any) -> str:
    return base64.b64encode(DataStructs.BitVectToBinaryText(fp)).decode("ascii")


def decode_fp(text: str) -> Any:
    return DataStructs.CreateFromBinaryText(base64.b64decode(text.encode("ascii")))


def load_fingerprint_cache(path: Path | None, fp_cache: FingerprintCache) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"enabled": path is not None, "path": None if path is None else str(path), "loaded": 0}
    loaded = 0
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            fp_cache.cache[str(row["smiles"])] = (
                decode_fp(str(row["morgan"])),
                decode_fp(str(row["feature"])),
            )
            loaded += 1
    return {"enabled": True, "path": str(path), "loaded": loaded}


def save_fingerprint_cache(path: Path | None, fp_cache: FingerprintCache) -> dict[str, Any]:
    if path is None:
        return {"enabled": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with gzip.open(path, "wt", compresslevel=1) as handle:
        for smiles, fps in sorted(fp_cache.cache.items()):
            if fps is None:
                continue
            handle.write(
                compact_json(
                    {
                        "smiles": smiles,
                        "morgan": encode_fp(fps[0]),
                        "feature": encode_fp(fps[1]),
                    }
                )
                + "\n"
            )
            written += 1
    return {"enabled": True, "path": str(path), "written": written}


def weighted_tanimoto(left_fp: tuple[Any, Any], right_fp: tuple[Any, Any]) -> float:
    morgan = DataStructs.TanimotoSimilarity(left_fp[0], right_fp[0])
    feature = DataStructs.TanimotoSimilarity(left_fp[1], right_fp[1])
    return 0.8 * float(morgan) + 0.2 * float(feature)


def load_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = set(args.metadata_columns)
    requested.update([args.smiles_column, args.value_column])
    if args.record_id_column:
        requested.add(args.record_id_column)
    columns = sorted(requested)
    records: list[dict[str, Any]] = []
    fp_cache = FingerprintCache(radius=args.fingerprint_radius, nbits=args.fingerprint_nbits)
    cache_load_stats = load_fingerprint_cache(args.fingerprint_cache, fp_cache)
    stats = Counter()

    from common_transfer import iter_parquet_rows

    for row_index, row in enumerate(
        iter_parquet_rows(
            args.input,
            columns=columns,
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
        record_id = (
            str(row.get(args.record_id_column))
            if args.record_id_column and row.get(args.record_id_column) is not None
            else str(row_index)
        )
        metadata = {column: row.get(column) for column in args.metadata_columns}
        records.append(
            {
                "record_id": record_id,
                "row_index": row_index,
                "canonical_smiles": canonical_smiles,
                "value": float(value),
                "metadata": metadata,
                "_fp": fp,
            }
        )
        stats["records_loaded"] += 1
    if len(records) < 2:
        raise RuntimeError("need at least two valid records to create pairs")
    cache_save_stats = save_fingerprint_cache(args.fingerprint_cache, fp_cache)
    out_stats = dict(sorted(stats.items()))
    out_stats["fingerprint_cache"] = {
        "load": cache_load_stats,
        "save": cache_save_stats,
        "unique_smiles_cached_in_memory": sum(1 for value in fp_cache.cache.values() if value is not None),
    }
    return records, out_stats


def all_pair_indices(n: int) -> Iterator[tuple[int, int]]:
    for left in range(n):
        for right in range(left + 1, n):
            yield left, right


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


def pair_row(
    *,
    left: dict[str, Any],
    right: dict[str, Any],
    similarity: float,
    transfer_label: str,
    diff: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    left_id = str(left["record_id"])
    right_id = str(right["record_id"])
    if left_id > right_id:
        left, right = right, left
        left_id, right_id = right_id, left_id
    pair_id = stable_hash_text(left_id, right_id)
    metadata_presence = {
        column: (
            "not_null" if left["metadata"].get(column) is not None else "null",
            "not_null" if right["metadata"].get(column) is not None else "null",
        )
        for column in args.metadata_columns
    }
    molecule_a = {
        "record_id": left_id,
        "row_index": int(left["row_index"]),
        "canonical_smiles": left["canonical_smiles"],
    }
    molecule_b = {
        "record_id": right_id,
        "row_index": int(right["row_index"]),
        "canonical_smiles": right["canonical_smiles"],
    }
    if args.pair_payload == "full":
        molecule_a["metadata"] = left["metadata"]
        molecule_b["metadata"] = right["metadata"]
    out = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "pair_id": pair_id,
        "record_id_a": left_id,
        "record_id_b": right_id,
        "row_index_a": int(left["row_index"]),
        "row_index_b": int(right["row_index"]),
        "group_id": args.group_id,
        "molecule_a": molecule_a,
        "molecule_b": molecule_b,
        "transfer_label": transfer_label,
        "value_difference": compact_float(diff),
        "T_transfer": compact_float(args.transfer_threshold),
        "T_not_transfer": compact_float(args.not_transfer_threshold),
        "weighted_tanimoto": round(float(similarity), 4),
        "stratification_metadata": metadata_presence,
    }
    if args.pair_payload == "full":
        out["metadata_strata"] = {
            column: (
                normalize_stratum_value(left["metadata"].get(column)),
                normalize_stratum_value(right["metadata"].get(column)),
            )
            for column in args.metadata_columns
        }
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    prepare_output_dir(
        args.output_dir,
        ["records.jsonl.gz", "pair_summary.csv", "metadata.json"],
        args.overwrite,
    )
    records, load_stats = load_records(args)
    n = len(records)
    total_possible_pairs = n * (n - 1) // 2
    target_candidates = math.ceil(total_possible_pairs * args.pair_sample_fraction)
    if args.max_candidate_pairs is not None:
        target_candidates = min(target_candidates, args.max_candidate_pairs)
    if target_candidates < 1:
        raise RuntimeError("pair sampling target is zero; increase --pair-sample-fraction")

    candidate_iter: Iterator[tuple[int, int]]
    if target_candidates >= total_possible_pairs and total_possible_pairs <= args.max_exhaustive_pairs:
        candidate_iter = all_pair_indices(n)
        sampling_mode = "exhaustive"
    else:
        candidate_iter = sampled_pair_indices(
            n_records=n,
            target_candidates=target_candidates,
            seed=args.seed,
            attempt_factor=args.attempt_factor,
        )
        sampling_mode = "random_unordered_without_replacement"

    rows_written = 0
    candidates_seen = 0
    drop_reasons: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    similarities: list[float] = []
    summary_rows: list[dict[str, Any]] = []

    records_path = args.output_dir / "records.jsonl.gz"
    with gzip.open(records_path, "wt", compresslevel=args.gzip_compresslevel) as handle:
        for left_idx, right_idx in candidate_iter:
            candidates_seen += 1
            left = records[left_idx]
            right = records[right_idx]
            if left["record_id"] == right["record_id"]:
                drop_reasons["same_record"] += 1
                continue
            diff = abs(float(left["value"]) - float(right["value"]))
            label = label_pair(diff, args.transfer_threshold, args.not_transfer_threshold)
            if label is None:
                drop_reasons["deadband"] += 1
                continue
            similarity = weighted_tanimoto(left["_fp"], right["_fp"])
            if not math.isfinite(similarity):
                drop_reasons["invalid_similarity"] += 1
                continue
            row = pair_row(
                left=left,
                right=right,
                similarity=similarity,
                transfer_label=label,
                diff=diff,
                args=args,
            )
            handle.write(compact_json(row) + "\n")
            rows_written += 1
            label_counts[label] += 1
            similarities.append(float(similarity))
            if args.progress_every and rows_written % args.progress_every == 0:
                print(f"wrote pairs: {rows_written:,}", file=sys.stderr, flush=True)

    similarity_thresholds = (
        compute_quantile_thresholds(similarities, args.similarity_buckets)
        if similarities
        else []
    )
    similarity_bucket_counts: Counter[str] = Counter()
    for similarity in similarities:
        similarity_bucket_counts[str(bucket_for_value(similarity, similarity_thresholds))] += 1

    summary_rows.append(
        {
            "group_id": args.group_id,
            "valid_records": n,
            "total_possible_pairs": total_possible_pairs,
            "target_candidate_pairs": target_candidates,
            "candidate_pairs_seen": candidates_seen,
            "written_pairs": rows_written,
            "transfer_pairs": label_counts.get("transfer", 0),
            "not_transfer_pairs": label_counts.get("not_transfer", 0),
            "deadband_dropped_pairs": drop_reasons.get("deadband", 0),
        }
    )
    with (args.output_dir / "pair_summary.csv").open("wt", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "input": args.input,
        "output_dir": str(args.output_dir),
        "group_id": args.group_id,
        "smiles_column": args.smiles_column,
        "value_column": args.value_column,
        "record_id_column": args.record_id_column,
        "metadata_columns": args.metadata_columns,
        "pair_payload": args.pair_payload,
        "stratification_metadata_policy": "null_vs_not_null per metadata column",
        "value_unit": args.value_unit,
        "valid_records": n,
        "load_stats": load_stats,
        "total_possible_pairs": total_possible_pairs,
        "pair_sample_fraction": args.pair_sample_fraction,
        "max_candidate_pairs": args.max_candidate_pairs,
        "target_candidate_pairs": target_candidates,
        "candidate_pairs_seen": candidates_seen,
        "pairs_written": rows_written,
        "pairs_by_transfer_label": dict(sorted(label_counts.items())),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "thresholds": {
            "transfer": args.transfer_threshold,
            "not_transfer": args.not_transfer_threshold,
            "deadband_policy": "drop transfer_threshold < diff < not_transfer_threshold",
        },
        "similarity": {
            "fingerprint_radius": args.fingerprint_radius,
            "fingerprint_nbits": args.fingerprint_nbits,
            "morgan_weight": 0.8,
            "feature_morgan_weight": 0.2,
            "quantile_thresholds": [compact_float(value) for value in similarity_thresholds],
            "bucket_counts": dict(sorted(similarity_bucket_counts.items())),
        },
        "sampling_mode": sampling_mode,
        "files": {
            "records": "records.jsonl.gz",
            "pair_summary": "pair_summary.csv",
            "metadata": "metadata.json",
        },
    }
    write_json(args.output_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local Parquet file/dir or HF dataset repo ID.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--value-column", default="oral_bioavailability_value")
    parser.add_argument("--record-id-column", default=None)
    parser.add_argument("--metadata-columns", nargs="+", default=DEFAULT_METADATA_COLUMNS)
    parser.add_argument("--pair-payload", choices=("minimal", "full"), default="minimal")
    parser.add_argument("--group-id", default="global")
    parser.add_argument("--value-unit", default="percent")
    parser.add_argument("--transfer-threshold", type=float, default=10.0)
    parser.add_argument("--not-transfer-threshold", type=float, default=30.0)
    parser.add_argument("--pair-sample-fraction", type=float, required=True)
    parser.add_argument("--max-candidate-pairs", type=int, default=None)
    parser.add_argument("--max-exhaustive-pairs", type=int, default=10_000_000)
    parser.add_argument("--similarity-buckets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--attempt-factor", type=int, default=10)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-nbits", type=int, default=2048)
    parser.add_argument(
        "--fingerprint-cache",
        type=Path,
        default=None,
        help="Optional gzip JSONL cache of precomputed Morgan and feature-Morgan fingerprints.",
    )
    parser.add_argument("--read-batch-size", type=int, default=8192)
    parser.add_argument("--gzip-compresslevel", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.pair_sample_fraction <= 1:
        parser.error("--pair-sample-fraction must be in (0, 1]")
    if args.max_candidate_pairs is not None and args.max_candidate_pairs < 1:
        parser.error("--max-candidate-pairs must be positive")
    if args.transfer_threshold < 0 or args.not_transfer_threshold < 0:
        parser.error("thresholds must be non-negative")
    if args.transfer_threshold >= args.not_transfer_threshold:
        parser.error("--transfer-threshold must be less than --not-transfer-threshold")
    for name in (
        "similarity_buckets",
        "attempt_factor",
        "fingerprint_radius",
        "fingerprint_nbits",
        "read_batch_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_rows is not None and args.max_rows < 2:
        parser.error("--max-rows must be at least 2")
    if not 0 <= args.gzip_compresslevel <= 9:
        parser.error("--gzip-compresslevel must be between 0 and 9")
    return args


def main() -> None:
    metadata = build(parse_args())
    print(
        json.dumps(
            {
                "output_dir": metadata["output_dir"],
                "valid_records": metadata["valid_records"],
                "candidate_pairs_seen": metadata["candidate_pairs_seen"],
                "pairs_written": metadata["pairs_written"],
                "pairs_by_transfer_label": metadata["pairs_by_transfer_label"],
                "drop_reasons": metadata["drop_reasons"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
