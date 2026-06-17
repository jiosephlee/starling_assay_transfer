#!/usr/bin/env python3
"""Create compact Parquet molecular transfer pairs from one numeric dataset."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_transfer import (  # noqa: E402
    FingerprintCache,
    bucket_for_value,
    compact_float,
    finite_float,
    iter_parquet_rows,
    prepare_output_dir,
    utc_now,
    write_json,
)


DEFAULT_OUTPUT_DIR = Path("datasets/pairs_compact/generic_transfer_pairs")
DEFAULT_METADATA_COLUMNS = [
    "bioavailability_report_type",
    "species_or_population",
    "dose",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
    "extra_details",
]
SCHEMA_VERSION = "generic_transfer_pairs_compact_parquet_v1"
LABEL_CODES = {"transfer": 1, "not_transfer": 0}
PRESENCE_PAIR_CODES = {
    "null<>null": 0,
    "not_null<>null": 1,
    "not_null<>not_null": 2,
}

WORKER_RECORDS: list[dict[str, Any]] = []
WORKER_METADATA_COLUMNS: list[str] = []
WORKER_SIMILARITY_THRESHOLDS: list[float] = []
WORKER_TRANSFER_THRESHOLD = 10.0
WORKER_NOT_TRANSFER_THRESHOLD = 30.0
WORKER_ROW_GROUP_SIZE = 250_000
WORKER_COMPRESSION = "zstd"
WORKER_SCHEMA: pa.Schema | None = None
WORKER_PROGRESS_EVERY = 0


def presence_pair_code(left: Any, right: Any) -> int:
    left_text = "not_null" if left is not None else "null"
    right_text = "not_null" if right is not None else "null"
    key = "<>".join(sorted((left_text, right_text)))
    return PRESENCE_PAIR_CODES[key]


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
        raise RuntimeError("need at least two valid records to create pairs")
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


def all_pair_indices(n_records: int) -> Iterator[tuple[int, int]]:
    for left in range(n_records):
        for right in range(left + 1, n_records):
            yield left, right


def count_pairs_for_left_range(n_records: int, start: int, end: int) -> int:
    if end <= start:
        return 0
    return sum(n_records - left - 1 for left in range(start, end))


def left_ranges_by_pair_count(n_records: int, n_ranges: int) -> list[tuple[int, int]]:
    n_ranges = max(1, n_ranges)
    total = n_records * (n_records - 1) // 2
    target = max(1, math.ceil(total / n_ranges))
    ranges: list[tuple[int, int]] = []
    start = 0
    count = 0
    for left in range(n_records - 1):
        count += n_records - left - 1
        if count >= target and left + 1 > start:
            ranges.append((start, left + 1))
            start = left + 1
            count = 0
    if start < n_records - 1:
        ranges.append((start, n_records - 1))
    return ranges


def label_pair(diff: float, transfer_threshold: float, not_transfer_threshold: float) -> str | None:
    if diff <= transfer_threshold:
        return "transfer"
    if diff >= not_transfer_threshold:
        return "not_transfer"
    return None


def schema(metadata_columns: list[str]) -> pa.Schema:
    fields = [
        pa.field("row_index_a", pa.uint32()),
        pa.field("row_index_b", pa.uint32()),
        pa.field("transfer_label", pa.int8()),
        pa.field("value_difference", pa.float32()),
        pa.field("weighted_tanimoto", pa.float32()),
        pa.field("similarity_bucket", pa.int8()),
    ]
    for column in metadata_columns:
        fields.append(pa.field(f"{column}_presence_pair", pa.int8()))
    return pa.schema(fields)


def write_batch(writer: pq.ParquetWriter, rows: list[dict[str, Any]]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=writer.schema))
        rows.clear()


def weighted_similarity(left_fp: tuple[Any, Any], right_fp: tuple[Any, Any]) -> float:
    from rdkit import DataStructs

    morgan = DataStructs.TanimotoSimilarity(left_fp[0], right_fp[0])
    feature = DataStructs.TanimotoSimilarity(left_fp[1], right_fp[1])
    return 0.8 * float(morgan) + 0.2 * float(feature)


def init_worker(
    records: list[dict[str, Any]],
    metadata_columns: list[str],
    similarity_thresholds: list[float],
    transfer_threshold: float,
    not_transfer_threshold: float,
    row_group_size: int,
    compression: str,
    out_schema: pa.Schema,
    progress_every: int,
) -> None:
    global WORKER_RECORDS
    global WORKER_METADATA_COLUMNS
    global WORKER_SIMILARITY_THRESHOLDS
    global WORKER_TRANSFER_THRESHOLD
    global WORKER_NOT_TRANSFER_THRESHOLD
    global WORKER_ROW_GROUP_SIZE
    global WORKER_COMPRESSION
    global WORKER_SCHEMA
    global WORKER_PROGRESS_EVERY

    WORKER_RECORDS = records
    WORKER_METADATA_COLUMNS = metadata_columns
    WORKER_SIMILARITY_THRESHOLDS = similarity_thresholds
    WORKER_TRANSFER_THRESHOLD = transfer_threshold
    WORKER_NOT_TRANSFER_THRESHOLD = not_transfer_threshold
    WORKER_ROW_GROUP_SIZE = row_group_size
    WORKER_COMPRESSION = compression
    WORKER_SCHEMA = out_schema
    WORKER_PROGRESS_EVERY = progress_every


def process_left_range(
    shard_id: int,
    start: int,
    end: int,
    output_path: str,
) -> dict[str, Any]:
    if WORKER_SCHEMA is None:
        raise RuntimeError("worker schema not initialized")
    records = WORKER_RECORDS
    metadata_columns = WORKER_METADATA_COLUMNS
    similarity_thresholds = WORKER_SIMILARITY_THRESHOLDS
    candidates_seen = 0
    pairs_written = 0
    drop_reasons: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    buffer: list[dict[str, Any]] = []
    path = Path(output_path)
    with pq.ParquetWriter(
        path,
        WORKER_SCHEMA,
        compression=WORKER_COMPRESSION,
        use_dictionary=False,
        write_statistics=True,
    ) as writer:
        for left_idx in range(start, end):
            left = records[left_idx]
            for right_idx in range(left_idx + 1, len(records)):
                candidates_seen += 1
                right = records[right_idx]
                diff = abs(float(left["value"]) - float(right["value"]))
                label = label_pair(diff, WORKER_TRANSFER_THRESHOLD, WORKER_NOT_TRANSFER_THRESHOLD)
                if label is None:
                    drop_reasons["deadband"] += 1
                    continue
                similarity = weighted_similarity(left["_fp"], right["_fp"])
                if not math.isfinite(similarity):
                    drop_reasons["invalid_similarity"] += 1
                    continue
                row = {
                    "row_index_a": int(left["row_index"]),
                    "row_index_b": int(right["row_index"]),
                    "transfer_label": LABEL_CODES[label],
                    "value_difference": diff,
                    "weighted_tanimoto": similarity,
                    "similarity_bucket": (
                        bucket_for_value(similarity, similarity_thresholds)
                        if similarity_thresholds
                        else None
                    ),
                }
                for column in metadata_columns:
                    row[f"{column}_presence_pair"] = presence_pair_code(
                        left["metadata"].get(column),
                        right["metadata"].get(column),
                    )
                buffer.append(row)
                pairs_written += 1
                label_counts[label] += 1
                if len(buffer) >= WORKER_ROW_GROUP_SIZE:
                    write_batch(writer, buffer)
        write_batch(writer, buffer)
    return {
        "shard_id": shard_id,
        "path": path.name,
        "left_start": start,
        "left_end": end,
        "candidate_pairs_seen": candidates_seen,
        "pairs_written": pairs_written,
        "pairs_by_transfer_label": dict(sorted(label_counts.items())),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    prepare_output_dir(args.output_dir, ["records.parquet", "metadata.json"], args.overwrite)
    records, load_stats = load_records(args)
    n = len(records)
    total_possible_pairs = n * (n - 1) // 2
    if args.enumerate_all:
        target_candidates = total_possible_pairs
        pair_iter = all_pair_indices(n)
        sampling_mode = "deterministic_exhaustive_row_index_order"
    else:
        target_candidates = math.ceil(total_possible_pairs * args.pair_sample_fraction)
        if args.max_candidate_pairs is not None:
            target_candidates = min(target_candidates, args.max_candidate_pairs)
        pair_iter = sampled_pair_indices(
            n_records=n,
            target_candidates=target_candidates,
            seed=args.seed,
            attempt_factor=args.attempt_factor,
        )
        sampling_mode = "random_unordered_without_replacement"
    if target_candidates < 1:
        raise RuntimeError("pair sampling target is zero")

    candidates_seen = 0
    pairs_written = 0
    drop_reasons: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    similarity_thresholds = list(args.similarity_thresholds or [])
    if similarity_thresholds and len(similarity_thresholds) != args.similarity_buckets - 1:
        raise ValueError("--similarity-thresholds must provide similarity_buckets - 1 values")
    out_schema = schema(args.metadata_columns)

    shard_metadata: list[dict[str, Any]] = []
    if args.enumerate_all and args.workers > 1:
        records_dir = args.output_dir / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        ranges = left_ranges_by_pair_count(n, max(args.workers * args.tasks_per_worker, 1))
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(
                records,
                args.metadata_columns,
                similarity_thresholds,
                args.transfer_threshold,
                args.not_transfer_threshold,
                args.row_group_size,
                args.parquet_compression,
                out_schema,
                args.progress_every,
            ),
        ) as executor:
            futures = []
            for shard_id, (start, end) in enumerate(ranges):
                futures.append(
                    executor.submit(
                        process_left_range,
                        shard_id,
                        start,
                        end,
                        str(records_dir / f"part-{shard_id:05d}.parquet"),
                    )
                )
            completed = 0
            for future in as_completed(futures):
                shard = future.result()
                shard_metadata.append(shard)
                completed += 1
                candidates_seen += int(shard["candidate_pairs_seen"])
                pairs_written += int(shard["pairs_written"])
                label_counts.update(shard.get("pairs_by_transfer_label") or {})
                drop_reasons.update(shard.get("drop_reasons") or {})
                print(
                    f"completed shard {completed}/{len(futures)}: "
                    f"candidates={candidates_seen:,}; labeled={pairs_written:,}",
                    file=sys.stderr,
                    flush=True,
                )
        shard_metadata.sort(key=lambda row: int(row["shard_id"]))
    else:
        fp_cache = FingerprintCache(radius=args.fingerprint_radius, nbits=args.fingerprint_nbits)
        fp_cache.cache.update({record["canonical_smiles"]: record["_fp"] for record in records})
        buffer: list[dict[str, Any]] = []
        with pq.ParquetWriter(
            args.output_dir / "records.parquet",
            out_schema,
            compression=args.parquet_compression,
            use_dictionary=False,
            write_statistics=True,
        ) as writer:
            for left_idx, right_idx in pair_iter:
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
                presence_codes = tuple(
                    presence_pair_code(left["metadata"].get(column), right["metadata"].get(column))
                    for column in args.metadata_columns
                )
                row = {
                    "row_index_a": int(left["row_index"]),
                    "row_index_b": int(right["row_index"]),
                    "transfer_label": LABEL_CODES[label],
                    "value_difference": diff,
                    "weighted_tanimoto": similarity,
                    "similarity_bucket": (
                        bucket_for_value(similarity, similarity_thresholds)
                        if similarity_thresholds
                        else None
                    ),
                }
                for column, code in zip(args.metadata_columns, presence_codes, strict=True):
                    row[f"{column}_presence_pair"] = code
                buffer.append(row)
                pairs_written += 1
                label_counts[label] += 1
                if len(buffer) >= args.row_group_size:
                    write_batch(writer, buffer)
                if args.progress_every and candidates_seen % args.progress_every == 0:
                    print(
                        f"processed candidates={candidates_seen:,}; labeled={pairs_written:,}",
                        file=sys.stderr,
                        flush=True,
                    )
            write_batch(writer, buffer)
    if pairs_written <= 0:
        raise RuntimeError("no labeled pairs written")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "input": args.input,
        "output_dir": str(args.output_dir),
        "smiles_column": args.smiles_column,
        "value_column": args.value_column,
        "metadata_columns": args.metadata_columns,
        "valid_records": n,
        "load_stats": load_stats,
        "total_possible_pairs": total_possible_pairs,
        "pair_sample_fraction": args.pair_sample_fraction,
        "enumerate_all": args.enumerate_all,
        "max_candidate_pairs": args.max_candidate_pairs,
        "target_candidate_pairs": target_candidates,
        "candidate_pairs_seen": candidates_seen,
        "pairs_written": pairs_written,
        "pairs_by_transfer_label": dict(sorted(label_counts.items())),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "sampling_mode": sampling_mode,
        "workers": args.workers,
        "tasks_per_worker": args.tasks_per_worker,
        "thresholds": {
            "transfer": args.transfer_threshold,
            "not_transfer": args.not_transfer_threshold,
        },
        "similarity": {
            "fingerprint_radius": args.fingerprint_radius,
            "fingerprint_nbits": args.fingerprint_nbits,
            "morgan_weight": 0.8,
            "feature_morgan_weight": 0.2,
            "quantile_thresholds": [compact_float(value) for value in similarity_thresholds],
            "bucket_policy": (
                "provided_cli_thresholds"
                if similarity_thresholds
                else "not_bucketed_in_streaming_output"
            ),
        },
        "encoding": {
            "transfer_label": LABEL_CODES,
            "metadata_presence_pair": PRESENCE_PAIR_CODES,
            "integer_types": {
                "row_index_a": "uint32",
                "row_index_b": "uint32",
                "transfer_label": "int8",
                "similarity_bucket": "int8",
                "*_presence_pair": "int8",
            },
            "float_types": {
                "value_difference": "float32",
                "weighted_tanimoto": "float32",
            },
        },
        "parquet": {
            "compression": args.parquet_compression,
            "row_group_size": args.row_group_size,
            "use_dictionary": False,
        },
        "files": {
            "records": "records/" if shard_metadata else "records.parquet",
            "metadata": "metadata.json",
        },
        "shards": shard_metadata,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local Parquet file/dir or HF dataset repo ID.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--value-column", default="oral_bioavailability_value")
    parser.add_argument("--metadata-columns", nargs="+", default=DEFAULT_METADATA_COLUMNS)
    parser.add_argument("--transfer-threshold", type=float, default=10.0)
    parser.add_argument("--not-transfer-threshold", type=float, default=30.0)
    parser.add_argument("--pair-sample-fraction", type=float, default=0.01)
    parser.add_argument(
        "--enumerate-all",
        action="store_true",
        help="Enumerate every unordered row pair deterministically instead of random sampling.",
    )
    parser.add_argument("--max-candidate-pairs", type=int, default=None)
    parser.add_argument("--similarity-buckets", type=int, default=5)
    parser.add_argument("--similarity-thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--attempt-factor", type=int, default=10)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-nbits", type=int, default=2048)
    parser.add_argument("--read-batch-size", type=int, default=8192)
    parser.add_argument("--row-group-size", type=int, default=250_000)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tasks-per-worker", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.pair_sample_fraction <= 1:
        parser.error("--pair-sample-fraction must be in (0, 1]")
    if args.enumerate_all and args.max_candidate_pairs is not None:
        parser.error("--max-candidate-pairs is not used with --enumerate-all")
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
        "row_group_size",
        "workers",
        "tasks_per_worker",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_rows is not None and args.max_rows < 2:
        parser.error("--max-rows must be at least 2")
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
