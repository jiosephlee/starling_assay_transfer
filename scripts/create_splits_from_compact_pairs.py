#!/usr/bin/env python3
"""Create molecule-disjoint splits from compact Parquet pair shards."""

from __future__ import annotations

import argparse
import heapq
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_transfer import (  # noqa: E402
    EVAL_SPLITS,
    SPLITS,
    bucket_for_value,
    compact_float,
    largest_remainder_allocation,
    stable_priority,
    utc_now,
    write_json,
)


DEFAULT_INPUT_DIR = Path("datasets/pairs_compact/oral_bioavailability_pairs_full")
DEFAULT_OUTPUT_DIR = Path("datasets/pairs_split_compact/oral_bioavailability_pair_splits")
DEFAULT_METADATA_COLUMNS = [
    "bioavailability_report_type",
    "species_or_population",
    "dose",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
    "extra_details",
]
SCHEMA_VERSION = "generic_transfer_pair_splits_compact_v1"
SPLIT_VERSION = "compact_pair_first_proportional_molecule_disjoint_v1"
LABEL_TEXT = {0: "not_transfer", 1: "transfer"}
PRESENCE_TEXT = {0: "null<>null", 1: "not_null<>null", 2: "not_null<>not_null"}


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class ProgressLogger:
    def __init__(self, phase: str, total: int | None, interval_seconds: float) -> None:
        self.phase = phase
        self.total = total if total and total > 0 else None
        self.interval_seconds = interval_seconds
        self.start = time.monotonic()
        self.last = 0.0
        self.update(0, force=True)

    def update(self, current: int, *, force: bool = False, extra: str = "") -> None:
        if self.interval_seconds <= 0 and not force:
            return
        now = time.monotonic()
        if not force and now - self.last < self.interval_seconds:
            return
        elapsed = max(now - self.start, 1e-9)
        rate = current / elapsed
        if self.total:
            pct = 100.0 * current / self.total
            remaining = max(self.total - current, 0)
            eta = remaining / rate if rate > 0 else None
            total_text = f"{current:,}/{self.total:,} ({pct:.2f}%)"
        else:
            eta = None
            total_text = f"{current:,}"
        suffix = f" {extra}" if extra else ""
        print(
            f"[{utc_now()}] {self.phase}: rows={total_text} "
            f"rate={rate:,.0f}/s elapsed={format_duration(elapsed)} eta={format_duration(eta)}{suffix}",
            file=sys.stderr,
            flush=True,
        )
        self.last = now

    def finish(self, current: int, *, extra: str = "") -> None:
        self.update(current, force=True, extra=extra)


def records_path(input_dir: Path) -> Path:
    records_dir = input_dir / "records"
    if records_dir.exists():
        return records_dir
    return input_dir / "records.parquet"


def compact_schema(metadata_columns: list[str]) -> pa.Schema:
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


def output_schema(metadata_columns: list[str]) -> pa.Schema:
    return compact_schema(metadata_columns).append(pa.field("split", pa.string()))


def iter_batches(input_dir: Path, columns: list[str] | None, batch_size: int) -> Any:
    dataset = ds.dataset(records_path(input_dir), format="parquet")
    yield from dataset.to_batches(columns=columns, batch_size=batch_size)


def parquet_files(input_dir: Path) -> list[Path]:
    path = records_path(input_dir)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"no Parquet files found under {path}")
    return files


def parquet_files_for_path(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"no Parquet files found under {path}")
    return files


def parquet_row_count(path: Path) -> int:
    return sum(pq.ParquetFile(file_path).metadata.num_rows for file_path in parquet_files_for_path(path))


def pair_key(row_index_a: int, row_index_b: int) -> tuple[int, int]:
    return (int(row_index_a), int(row_index_b))


def row_molecules(row: dict[str, Any]) -> tuple[int, int]:
    return pair_key(row["row_index_a"], row["row_index_b"])


def stratum_for_row(row: dict[str, Any], metadata_columns: list[str]) -> tuple[Any, ...]:
    return (
        int(row["transfer_label"]),
        int(row["similarity_bucket"]),
        *(int(row[f"{column}_presence_pair"]) for column in metadata_columns),
    )


def stratum_to_string(stratum: tuple[Any, ...], metadata_columns: list[str]) -> str:
    label, bucket, *presence = stratum
    parts = [f"transfer_label={LABEL_TEXT[int(label)]}", f"similarity_bucket={bucket}"]
    for column, code in zip(metadata_columns, presence, strict=True):
        parts.append(f"{column}={PRESENCE_TEXT[int(code)]}")
    return "|".join(parts)


def serializable_counter(counter: Counter[Any], metadata_columns: list[str]) -> dict[str, int]:
    return {
        stratum_to_string(key, metadata_columns) if isinstance(key, tuple) else str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: repr(item[0]))
    }


def checkpoints_dir(output_dir: Path) -> Path:
    return output_dir / "checkpoints"


def checkpoint_path(args: argparse.Namespace, name: str) -> Path:
    return checkpoints_dir(args.output_dir) / name


def stratum_entries(counter: Counter[tuple[Any, ...]]) -> list[dict[str, Any]]:
    return [
        {"stratum": [int(value) for value in key], "count": int(count)}
        for key, count in sorted(counter.items(), key=lambda item: repr(item[0]))
    ]


def counter_from_entries(entries: list[dict[str, Any]]) -> Counter[tuple[Any, ...]]:
    return Counter({tuple(entry["stratum"]): int(entry["count"]) for entry in entries})


def write_counter_checkpoint(
    args: argparse.Namespace,
    name: str,
    counter: Counter[tuple[Any, ...]],
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    path = checkpoint_path(args, name)
    payload = {
        "created_at_utc": utc_now(),
        "entries": stratum_entries(counter),
        "n_strata": len(counter),
        **(extra or {}),
    }
    write_json(path, payload)


def read_counter_checkpoint(args: argparse.Namespace, name: str) -> Counter[tuple[Any, ...]] | None:
    if not args.resume_checkpoints:
        return None
    path = checkpoint_path(args, name)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    print(f"[{utc_now()}] reuse checkpoint: {path}", file=sys.stderr, flush=True)
    return counter_from_entries(payload["entries"])


def serializable_selection_meta(meta: dict[str, Any], metadata_columns: list[str]) -> dict[str, Any]:
    out = {
        key: value
        for key, value in meta.items()
        if key not in {"selected_strata", "allocation"}
    }
    out["selected_strata_entries"] = stratum_entries(meta.get("selected_strata", Counter()))
    out["allocation_entries"] = stratum_entries(Counter(meta.get("allocation", {})))
    out["selected_strata"] = serializable_counter(meta.get("selected_strata", Counter()), metadata_columns)
    out["allocation"] = serializable_counter(Counter(meta.get("allocation", {})), metadata_columns)
    return out


def restore_selection_meta(payload: dict[str, Any]) -> dict[str, Any]:
    meta = {
        key: value
        for key, value in payload["meta"].items()
        if key not in {"selected_strata_entries", "allocation_entries", "selected_strata", "allocation"}
    }
    meta["selected_strata"] = counter_from_entries(payload["meta"].get("selected_strata_entries", []))
    meta["allocation"] = counter_from_entries(payload["meta"].get("allocation_entries", []))
    return meta


def write_selection_checkpoint(
    args: argparse.Namespace,
    split: str,
    pairs: dict[tuple[int, int], str],
    molecules: set[int],
    meta: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    split_pairs = [pair for pair, pair_split in pairs.items() if pair_split == split]
    payload = {
        "created_at_utc": utc_now(),
        "split": split,
        "pairs": [[int(left), int(right)] for left, right in sorted(split_pairs)],
        "molecules": sorted(int(value) for value in molecules),
        "meta": serializable_selection_meta(meta, args.metadata_columns),
        "rows": rows,
    }
    write_json(checkpoint_path(args, f"{split}_selection.json"), payload)


def read_selection_checkpoint(
    args: argparse.Namespace,
    split: str,
) -> tuple[dict[tuple[int, int], str], set[int], dict[str, Any], list[dict[str, Any]]] | None:
    if not args.resume_checkpoints:
        return None
    path = checkpoint_path(args, f"{split}_selection.json")
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    print(f"[{utc_now()}] reuse checkpoint: {path}", file=sys.stderr, flush=True)
    pairs = {(int(left), int(right)): split for left, right in payload["pairs"]}
    molecules = {int(value) for value in payload["molecules"]}
    return pairs, molecules, restore_selection_meta(payload), payload.get("rows", [])


def stratum_code_arrays(batch: pa.RecordBatch, metadata_columns: list[str], similarity_buckets: int) -> np.ndarray:
    code = batch.column("transfer_label").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    multiplier = 2
    code = code + multiplier * batch.column("similarity_bucket").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    multiplier *= similarity_buckets
    for column in metadata_columns:
        values = batch.column(f"{column}_presence_pair").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        code = code + multiplier * values
        multiplier *= 3
    return code


def stratum_from_code(code: int, metadata_columns: list[str], similarity_buckets: int) -> tuple[int, ...]:
    remaining = int(code)
    label = remaining % 2
    remaining //= 2
    bucket = remaining % similarity_buckets
    remaining //= similarity_buckets
    presence: list[int] = []
    for _column in metadata_columns:
        presence.append(remaining % 3)
        remaining //= 3
    return (label, bucket, *presence)


def code_from_stratum(stratum: tuple[Any, ...], similarity_buckets: int) -> int:
    label, bucket, *presence = (int(value) for value in stratum)
    code = label + 2 * bucket
    multiplier = 2 * similarity_buckets
    for value in presence:
        code += multiplier * value
        multiplier *= 3
    return int(code)


def counter_from_code_counts(
    code_counts: Counter[int],
    metadata_columns: list[str],
    similarity_buckets: int,
) -> Counter[tuple[Any, ...]]:
    return Counter(
        {
            stratum_from_code(code, metadata_columns, similarity_buckets): int(count)
            for code, count in code_counts.items()
            if count > 0
        }
    )


def compute_similarity_thresholds(args: argparse.Namespace) -> list[float]:
    if args.similarity_thresholds:
        return list(args.similarity_thresholds)

    files = parquet_files(args.input_dir)
    file_stats: list[tuple[Path, pq.ParquetFile, int]] = []
    file_counts: Counter[Path] = Counter()
    for path in files:
        parquet_file = pq.ParquetFile(path)
        n_rows = parquet_file.metadata.num_rows
        file_stats.append((path, parquet_file, n_rows))
        file_counts[path] = n_rows

    total_rows = sum(count for _path, _parquet_file, count in file_stats)
    target_rows = min(args.max_quantile_values, total_rows)
    if target_rows <= 0:
        raise ValueError("cannot compute similarity thresholds from empty input")

    allocations = largest_remainder_allocation(target_rows, file_counts)
    sample = np.empty(target_rows, dtype=np.float32)
    offset = 0
    row_groups_read = 0
    progress = ProgressLogger(
        "similarity quantile sample",
        target_rows,
        args.progress_every_seconds,
    )
    for path, parquet_file, n_file_rows in file_stats:
        file_quota = allocations.get(path, 0)
        if file_quota <= 0 or n_file_rows <= 0:
            continue
        n_groups = parquet_file.metadata.num_row_groups
        group_rows = [parquet_file.metadata.row_group(index).num_rows for index in range(n_groups)]
        groups_to_read = min(
            n_groups,
            max(1, int(np.ceil(file_quota / max(1.0, n_file_rows / max(1, n_groups))))),
        )
        selected_groups = np.linspace(0, n_groups - 1, groups_to_read, dtype=np.int64)
        selected_groups = sorted(set(int(index) for index in selected_groups))
        selected_counts = Counter(
            {index: group_rows[index] for index in selected_groups}
        )
        group_allocations = largest_remainder_allocation(file_quota, selected_counts)
        for group_index in selected_groups:
            group_quota = group_allocations.get(group_index, 0)
            if group_quota <= 0:
                continue
            values = (
                parquet_file.read_row_group(group_index, columns=["weighted_tanimoto"])
                .column("weighted_tanimoto")
                .combine_chunks()
                .to_numpy(zero_copy_only=False)
            )
            take = min(group_quota, len(values), target_rows - offset)
            if take <= 0:
                break
            if take == len(values):
                sample[offset : offset + take] = values.astype(np.float32, copy=False)
            else:
                positions = np.linspace(0, len(values) - 1, take, dtype=np.int64)
                sample[offset : offset + take] = values[positions].astype(np.float32, copy=False)
            offset += take
            row_groups_read += 1
            progress.update(offset, extra=f"row_groups_read={row_groups_read:,}")
        if offset >= target_rows:
            break

    if offset <= 0:
        raise ValueError("cannot compute similarity thresholds from empty quantile sample")
    sample_view = sample[:offset]
    quantiles = np.quantile(
        sample_view,
        [index / args.similarity_buckets for index in range(1, args.similarity_buckets)],
        method="linear",
    )
    args._quantile_values_seen = int(total_rows)
    args._quantile_values_sampled = int(offset)
    args._quantile_row_groups_read = int(row_groups_read)
    args._quantile_files_read = int(sum(1 for path in allocations if allocations[path] > 0))
    args._quantile_sample_strategy = "proportional_parquet_file_even_row_group_sample"
    progress.finish(offset, extra=f"row_groups_read={row_groups_read:,}")
    return [float(value) for value in quantiles]


def ensure_bucketed_input(args: argparse.Namespace, thresholds: list[float]) -> Path:
    bucketed_dir = args.output_dir / "_bucketed_input"
    if args.reuse_bucketed_input:
        if not bucketed_dir.exists():
            raise FileNotFoundError(f"--reuse-bucketed-input requested but {bucketed_dir} does not exist")
        args._bucketed_rows = parquet_row_count(bucketed_dir)
        write_json(
            checkpoint_path(args, "bucketed_input.json"),
            {
                "created_at_utc": utc_now(),
                "path": str(bucketed_dir),
                "rows": int(args._bucketed_rows),
                "parts": len(parquet_files_for_path(bucketed_dir)),
                "reused": True,
            },
        )
        print(
            f"[{utc_now()}] reuse bucketed input: path={bucketed_dir} rows={args._bucketed_rows:,}",
            file=sys.stderr,
            flush=True,
        )
        return bucketed_dir
    if bucketed_dir.exists():
        shutil.rmtree(bucketed_dir)
    bucketed_dir.mkdir(parents=True, exist_ok=True)
    if not thresholds:
        raise ValueError("similarity thresholds are required when building bucketed input")
    writer_id = 0
    writer: pq.ParquetWriter | None = None
    rows_in_writer = 0
    out_schema = compact_schema(args.metadata_columns)
    columns = [field.name for field in out_schema]
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_files(args.input_dir))
    processed = 0
    progress = ProgressLogger("bucket compact pairs", total_rows, args.progress_every_seconds)
    threshold_array = np.array(thresholds, dtype=np.float32)
    try:
        for batch in iter_batches(args.input_dir, columns, args.batch_size):
            similarities = batch.column("weighted_tanimoto").to_numpy(zero_copy_only=False)
            buckets = np.searchsorted(threshold_array, similarities, side="left").astype(np.int8, copy=False)
            table = pa.Table.from_batches([batch])
            table = table.set_column(
                table.schema.get_field_index("similarity_bucket"),
                "similarity_bucket",
                pa.array(buckets, type=pa.int8()),
            )
            table = table.cast(out_schema)
            if writer is None:
                writer = pq.ParquetWriter(
                    bucketed_dir / f"part-{writer_id:05d}.parquet",
                    out_schema,
                    compression=args.parquet_compression,
                    use_dictionary=False,
                    write_statistics=True,
                )
            writer.write_table(table)
            rows_in_writer += table.num_rows
            processed += table.num_rows
            if rows_in_writer >= args.bucket_file_row_limit:
                writer.close()
                writer = None
                writer_id += 1
                rows_in_writer = 0
            progress.update(processed, extra=f"parts_written={writer_id + int(writer is not None):,}")
    finally:
        if writer is not None:
            writer.close()
            writer_id += 1
    progress.finish(processed, extra=f"parts_written={writer_id:,}")
    args._bucketed_rows = int(processed)
    write_json(
        checkpoint_path(args, "bucketed_input.json"),
        {
            "created_at_utc": utc_now(),
            "path": str(bucketed_dir),
            "rows": int(processed),
            "parts": writer_id,
            "reused": False,
        },
    )
    return bucketed_dir


def write_similarity_quantile_metadata(args: argparse.Namespace, thresholds: list[float]) -> None:
    metadata = {
        "created_at_utc": utc_now(),
        "source_pair_dir": str(args.input_dir),
        "similarity_buckets": args.similarity_buckets,
        "similarity_thresholds": [compact_float(value) for value in thresholds],
        "similarity_bucket_policy": (
            "provided_fixed_thresholds"
            if args.similarity_thresholds
            else "estimated_quantile_thresholds"
        ),
        "similarity_quantile_thresholds": [compact_float(value) for value in thresholds],
        "similarity_quantile_thresholds_available": bool(thresholds),
        "similarity_quantile_estimation": {
            "mode": "provided"
            if args.similarity_thresholds
            else getattr(args, "_quantile_sample_strategy", "proportional_parquet_sample"),
            "values_seen": int(getattr(args, "_quantile_values_seen", 0)),
            "values_sampled": int(getattr(args, "_quantile_values_sampled", 0)),
            "max_quantile_values": args.max_quantile_values,
            "files_read": int(getattr(args, "_quantile_files_read", 0)),
            "row_groups_read": int(getattr(args, "_quantile_row_groups_read", 0)),
        },
    }
    write_json(args.output_dir / "similarity_quantiles.json", metadata)


def read_similarity_quantile_metadata(args: argparse.Namespace) -> list[float] | None:
    path = args.output_dir / "similarity_quantiles.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    estimation = payload.get("similarity_quantile_estimation", {})
    args._quantile_sample_strategy = estimation.get("mode", "reused_similarity_quantile_metadata")
    args._quantile_values_seen = int(estimation.get("values_seen", 0))
    args._quantile_values_sampled = int(estimation.get("values_sampled", 0))
    args._quantile_files_read = int(estimation.get("files_read", 0))
    args._quantile_row_groups_read = int(estimation.get("row_groups_read", 0))
    print(f"[{utc_now()}] reuse similarity quantile metadata: {path}", file=sys.stderr, flush=True)
    return [float(value) for value in payload.get("similarity_quantile_thresholds", [])]


def collect_stratum_counts(
    input_path: Path,
    metadata_columns: list[str],
    batch_size: int,
    excluded_molecules: set[int] | None = None,
    *,
    progress_label: str = "collect stratum counts",
    progress_every_seconds: float = 60.0,
    total_rows: int | None = None,
    similarity_buckets: int = 5,
) -> Counter[tuple[Any, ...]]:
    columns = ["row_index_a", "row_index_b", "transfer_label", "similarity_bucket"] + [
        f"{column}_presence_pair" for column in metadata_columns
    ]
    code_counts: Counter[int] = Counter()
    dataset = ds.dataset(input_path, format="parquet")
    excluded_array = (
        np.fromiter(excluded_molecules, dtype=np.uint32)
        if excluded_molecules
        else np.array([], dtype=np.uint32)
    )
    total_rows = total_rows or parquet_row_count(input_path)
    processed = 0
    progress = ProgressLogger(progress_label, total_rows, progress_every_seconds)
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        codes = stratum_code_arrays(batch, metadata_columns, similarity_buckets=similarity_buckets)
        if excluded_array.size:
            left = batch.column("row_index_a").to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
            right = batch.column("row_index_b").to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
            keep = ~np.isin(left, excluded_array, assume_unique=False) & ~np.isin(
                right,
                excluded_array,
                assume_unique=False,
            )
            codes = codes[keep]
        if codes.size:
            batch_counts = np.bincount(codes)
            for code in np.nonzero(batch_counts)[0]:
                code_counts[int(code)] += int(batch_counts[code])
        processed += batch.num_rows
        progress.update(processed, extra=f"strata={len(code_counts):,}")
    progress.finish(processed, extra=f"strata={len(code_counts):,}")
    return counter_from_code_counts(code_counts, metadata_columns, similarity_buckets=similarity_buckets)


def candidate_pool_capacity(args: argparse.Namespace, quota: int) -> int:
    oversampled = max(args.min_candidates_per_stratum, quota * args.candidate_pool_multiplier)
    return max(quota, min(args.max_candidates_per_stratum, oversampled))


def collect_candidate_pools(
    *,
    input_path: Path,
    metadata_columns: list[str],
    batch_size: int,
    allocation: dict[tuple[Any, ...], int],
    excluded_molecules: set[int],
    seed: int,
    split: str,
    args: argparse.Namespace,
) -> tuple[dict[tuple[Any, ...], list[dict[str, Any]]], Counter[str]]:
    columns = [field.name for field in compact_schema(metadata_columns)]
    capacities = {
        stratum: candidate_pool_capacity(args, quota)
        for stratum, quota in allocation.items()
        if quota > 0
    }
    code_to_stratum = {
        code_from_stratum(stratum, args.similarity_buckets): stratum
        for stratum in capacities
    }
    allocated_codes = np.fromiter(code_to_stratum, dtype=np.int64)
    heaps: dict[tuple[Any, ...], list[tuple[int, int, int, dict[str, Any]]]] = {
        stratum: [] for stratum in capacities
    }
    skipped = Counter()
    dataset = ds.dataset(input_path, format="parquet")
    total_rows = int(getattr(args, "_bucketed_rows", 0)) or parquet_row_count(input_path)
    processed = 0
    progress = ProgressLogger(
        f"collect {split} candidate pools",
        total_rows,
        args.progress_every_seconds,
    )
    if not capacities:
        progress.finish(0, extra="pooled=0 allocated_strata=0")
        return {stratum: [] for stratum in allocation}, skipped

    excluded_array = (
        np.fromiter(excluded_molecules, dtype=np.uint32)
        if excluded_molecules
        else np.array([], dtype=np.uint32)
    )
    pooled_rows = 0
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        codes = stratum_code_arrays(batch, metadata_columns, similarity_buckets=args.similarity_buckets)
        left_array = batch.column("row_index_a").to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
        right_array = batch.column("row_index_b").to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
        if excluded_array.size:
            touches_excluded = np.isin(left_array, excluded_array, assume_unique=False) | np.isin(
                right_array,
                excluded_array,
                assume_unique=False,
            )
        else:
            touches_excluded = np.zeros(batch.num_rows, dtype=bool)
        eligible = ~touches_excluded
        allocated = np.isin(codes, allocated_codes, assume_unique=False)
        candidate_mask = eligible & allocated

        skipped["touches_excluded_molecule"] += int(touches_excluded.sum())
        skipped["unallocated_stratum"] += int((eligible & ~allocated).sum())

        if candidate_mask.any():
            candidate_codes = codes[candidate_mask]
            candidate_rows = batch.filter(pa.array(candidate_mask)).to_pylist()
        else:
            candidate_codes = np.array([], dtype=np.int64)
            candidate_rows = []

        for row, code in zip(candidate_rows, candidate_codes, strict=True):
            left, right = row_molecules(row)
            stratum = code_to_stratum[int(code)]
            capacity = capacities[stratum]
            priority = stable_priority(seed, split, "candidate_pool", repr(stratum), left, right)
            entry = (-priority, left, right, row)
            heap = heaps[stratum]
            if len(heap) < capacity:
                heapq.heappush(heap, entry)
                pooled_rows += 1
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, entry)
        processed += batch.num_rows
        progress.update(processed, extra=f"pooled={pooled_rows:,} allocated_strata={len(capacities):,}")

    pools: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for stratum, heap in heaps.items():
        rows = [entry[3] for entry in heap]
        rows.sort(
            key=lambda row: stable_priority(
                seed,
                split,
                "candidate_pool_order",
                repr(stratum),
                row["row_index_a"],
                row["row_index_b"],
            )
        )
        pools[stratum] = rows
    progress.finish(
        processed,
        extra=f"pooled={sum(len(rows) for rows in pools.values()):,} allocated_strata={len(capacities):,}",
    )
    return pools, skipped


def choose_stratum(
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


def select_eval_split(
    *,
    input_path: Path,
    metadata_columns: list[str],
    batch_size: int,
    target_pairs: int,
    available_strata: Counter[tuple[Any, ...]],
    excluded_molecules: set[int],
    seed: int,
    split: str,
    args: argparse.Namespace,
) -> tuple[dict[tuple[int, int], str], set[int], dict[str, Any], list[dict[str, Any]]]:
    allocation = largest_remainder_allocation(target_pairs, available_strata)
    remaining: Counter[tuple[Any, ...]] = Counter(allocation)
    selected_pairs: dict[tuple[int, int], str] = {}
    selected_molecules: set[int] = set()
    selected_rows: list[dict[str, Any]] = []
    selected_strata: Counter[tuple[Any, ...]] = Counter()
    reuse_counts: Counter[str] = Counter()
    rows_by_stratum, skipped = collect_candidate_pools(
        input_path=input_path,
        metadata_columns=metadata_columns,
        batch_size=batch_size,
        allocation=allocation,
        excluded_molecules=excluded_molecules,
        seed=seed,
        split=split,
        args=args,
    )

    selectable_counts = Counter({key: len(value) for key, value in rows_by_stratum.items()})
    step = 0
    while len(selected_pairs) < target_pairs and remaining:
        stratum = choose_stratum(remaining, allocation, selectable_counts, seed, split, step)
        if stratum is None:
            skipped["no_selectable_remaining_stratum"] += 1
            break
        rows = rows_by_stratum.get(stratum) or []
        best: tuple[int, int, int, int] | None = None
        best_pos: int | None = None
        for pos, row in enumerate(rows):
            left, right = row_molecules(row)
            key = pair_key(left, right)
            if key in selected_pairs:
                continue
            reuse = int(left in selected_molecules) + int(right in selected_molecules)
            score = (
                reuse,
                -stable_priority(seed, split, "pair", step, left, right),
                -left,
                -right,
            )
            if best is None or score > best:
                best = score
                best_pos = pos
        if best_pos is None:
            remaining[stratum] = 0
            selectable_counts[stratum] = 0
            skipped["stratum_exhausted"] += 1
            continue
        row = rows.pop(best_pos)
        left, right = row_molecules(row)
        key = pair_key(left, right)
        reuse = int(left in selected_molecules) + int(right in selected_molecules)
        selected_pairs[key] = split
        selected_rows.append(row)
        selected_molecules.update(key)
        selected_strata[stratum] += 1
        remaining[stratum] -= 1
        selectable_counts[stratum] -= 1
        reuse_counts[str(reuse)] += 1
        if remaining[stratum] <= 0:
            del remaining[stratum]
        step += 1

    return selected_pairs, selected_molecules, {
        "target_pairs": target_pairs,
        "selected_pairs": len(selected_pairs),
        "selected_molecules": len(selected_molecules),
        "selected_strata": selected_strata,
        "allocation": allocation,
        "candidate_pool": {
            "pool_multiplier": args.candidate_pool_multiplier,
            "min_candidates_per_stratum": args.min_candidates_per_stratum,
            "max_candidates_per_stratum": args.max_candidates_per_stratum,
            "pooled_rows": int(sum(len(rows) for rows in rows_by_stratum.values())),
            "allocated_strata": int(len(allocation)),
            "pooled_strata": int(sum(1 for rows in rows_by_stratum.values() if rows)),
        },
        "reuse_counts": dict(sorted(reuse_counts.items())),
        "unfilled_allocation": {
            stratum_to_string(key, metadata_columns): int(value)
            for key, value in remaining.items()
            if value > 0
        },
        "skipped": dict(sorted(skipped.items())),
    }, selected_rows


class SplitWriters:
    def __init__(
        self,
        output_dir: Path,
        metadata_columns: list[str],
        row_group_size: int,
        compression: str,
    ) -> None:
        self.output_dir = output_dir
        self.metadata_columns = metadata_columns
        self.row_group_size = row_group_size
        self.compression = compression
        self.schema = output_schema(metadata_columns)
        self.buffers: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
        self.part_ids: Counter[str] = Counter()
        for split in SPLITS:
            (output_dir / split).mkdir(parents=True, exist_ok=True)

    def write(self, split: str, row: dict[str, Any]) -> None:
        out = dict(row)
        out["split"] = split
        self.buffers[split].append(out)
        if len(self.buffers[split]) >= self.row_group_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        buffer = self.buffers[split]
        if not buffer:
            return
        path = self.output_dir / split / f"part-{self.part_ids[split]:05d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(buffer, schema=self.schema),
            path,
            compression=self.compression,
        )
        self.part_ids[split] += 1
        buffer.clear()

    def close(self) -> None:
        for split in SPLITS:
            self.flush(split)


class TableSplitWriters:
    def __init__(self, output_dir: Path, metadata_columns: list[str], compression: str) -> None:
        self.output_dir = output_dir
        self.schema = output_schema(metadata_columns)
        self.compression = compression
        self.part_ids: Counter[str] = Counter()
        for split in SPLITS:
            split_dir = output_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)

    def write_table(self, split: str, table: pa.Table) -> None:
        if table.num_rows <= 0:
            return
        if "split" not in table.column_names:
            table = table.append_column("split", pa.array([split] * table.num_rows, type=pa.string()))
        table = table.select([field.name for field in self.schema])
        path = self.output_dir / split / f"part-{self.part_ids[split]:05d}.parquet"
        pq.write_table(table, path, compression=self.compression)
        self.part_ids[split] += 1


def pair_code_arrays(left: np.ndarray, right: np.ndarray, multiplier: int) -> np.ndarray:
    return left.astype(np.int64, copy=False) * int(multiplier) + right.astype(np.int64, copy=False)


def update_stats_from_rows(
    stats: dict[str, Any],
    split: str,
    rows: list[dict[str, Any]],
    metadata_columns: list[str],
) -> None:
    for row in rows:
        left, right = row_molecules(row)
        stats["rows_by_split"][split] += 1
        stats["transfer_label_counts"][split][str(int(row["transfer_label"]))] += 1
        stats["stratum_counts"][split][stratum_for_row(row, metadata_columns)] += 1
        stats["unique_molecules"][split].update((left, right))


def update_stats_from_masked_batch(
    stats: dict[str, Any],
    split: str,
    batch: pa.RecordBatch,
    mask: np.ndarray,
    metadata_columns: list[str],
    similarity_buckets: int,
) -> None:
    count = int(mask.sum())
    if count <= 0:
        return
    stats["rows_by_split"][split] += count
    labels = batch.column("transfer_label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    label_counts = np.bincount(labels[mask], minlength=2)
    for label, label_count in enumerate(label_counts):
        if label_count:
            stats["transfer_label_counts"][split][str(label)] += int(label_count)
    codes = stratum_code_arrays(batch, metadata_columns, similarity_buckets=similarity_buckets)[mask]
    if codes.size:
        code_counts = np.bincount(codes)
        stats["stratum_counts"][split].update(
            counter_from_code_counts(
                Counter({int(code): int(code_counts[code]) for code in np.nonzero(code_counts)[0]}),
                metadata_columns,
                similarity_buckets,
            )
        )
    left = batch.column("row_index_a").to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
    right = batch.column("row_index_b").to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
    stats["unique_molecules"][split].update(np.unique(np.concatenate([left[mask], right[mask]])).astype(int).tolist())


def write_split_outputs(
    *,
    input_path: Path,
    output_dir: Path,
    selected_pairs: dict[tuple[int, int], str],
    selected_rows_by_split: dict[str, list[dict[str, Any]]],
    eval_molecules: set[int],
    metadata_columns: list[str],
    batch_size: int,
    compression: str,
    progress_every_seconds: float,
    similarity_buckets: int,
    pair_code_multiplier: int,
    total_rows: int | None = None,
) -> dict[str, Any]:
    writers = TableSplitWriters(output_dir, metadata_columns, compression)
    stats: dict[str, Any] = {
        "rows_by_split": Counter(),
        "transfer_label_counts": {split: Counter() for split in SPLITS},
        "stratum_counts": {split: Counter() for split in SPLITS},
        "unique_molecules": {split: set() for split in SPLITS},
        "thrown_out_pairs": Counter(),
        "thrown_out_strata": Counter(),
    }
    for split in EVAL_SPLITS:
        rows = selected_rows_by_split.get(split, [])
        if rows:
            writers.write_table(
                split,
                pa.Table.from_pylist(rows, schema=compact_schema(metadata_columns)),
            )
            update_stats_from_rows(stats, split, rows, metadata_columns)

    columns = [field.name for field in compact_schema(metadata_columns)]
    dataset = ds.dataset(input_path, format="parquet")
    total_rows = total_rows or parquet_row_count(input_path)
    processed = 0
    progress = ProgressLogger("write split outputs", total_rows, progress_every_seconds)
    eval_array = np.fromiter(eval_molecules, dtype=np.uint32)
    selected_codes = np.fromiter(
        (
            left * int(pair_code_multiplier) + right
            for left, right in selected_pairs
        ),
        dtype=np.int64,
    )
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        left = batch.column("row_index_a").to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
        right = batch.column("row_index_b").to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
        touches_eval = np.isin(left, eval_array, assume_unique=False) | np.isin(
            right,
            eval_array,
            assume_unique=False,
        )
        codes = pair_code_arrays(left, right, pair_code_multiplier)
        selected_mask = np.isin(codes, selected_codes, assume_unique=False)
        train_mask = ~touches_eval
        thrown_mask = touches_eval & ~selected_mask

        if train_mask.any():
            train_batch = batch.filter(pa.array(train_mask))
            writers.write_table("train", pa.Table.from_batches([train_batch]))
            update_stats_from_masked_batch(
                stats,
                "train",
                batch,
                train_mask,
                metadata_columns,
                similarity_buckets,
            )
        thrown_count = int(thrown_mask.sum())
        if thrown_count:
            stats["thrown_out_pairs"]["eval_molecule_overlap"] += thrown_count
            codes_for_strata = stratum_code_arrays(
                batch,
                metadata_columns,
                similarity_buckets=similarity_buckets,
            )[thrown_mask]
            code_counts = np.bincount(codes_for_strata)
            stats["thrown_out_strata"].update(
                counter_from_code_counts(
                    Counter({int(code): int(code_counts[code]) for code in np.nonzero(code_counts)[0]}),
                    metadata_columns,
                    similarity_buckets,
                )
            )
        processed += batch.num_rows
        progress.update(
            processed,
            extra=(
                f"train={stats['rows_by_split']['train']:,} "
                f"validation={stats['rows_by_split']['validation']:,} "
                f"test={stats['rows_by_split']['test']:,} "
                f"thrown_out={stats['thrown_out_pairs']['eval_molecule_overlap']:,}"
            ),
        )
    progress.finish(
        processed,
        extra=(
            f"train={stats['rows_by_split']['train']:,} "
            f"validation={stats['rows_by_split']['validation']:,} "
            f"test={stats['rows_by_split']['test']:,} "
            f"thrown_out={stats['thrown_out_pairs']['eval_molecule_overlap']:,}"
        ),
    )
    return stats


def summarize_stats(stats: dict[str, Any], metadata_columns: list[str]) -> dict[str, Any]:
    return {
        "rows_by_split": dict(sorted(stats["rows_by_split"].items())),
        "transfer_label_counts": {
            split: dict(sorted(stats["transfer_label_counts"][split].items())) for split in SPLITS
        },
        "stratum_counts": {
            split: serializable_counter(stats["stratum_counts"][split], metadata_columns)
            for split in SPLITS
        },
        "unique_molecules": {split: len(stats["unique_molecules"][split]) for split in SPLITS},
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
    return {"molecule_overlap": overlaps, "n_errors": len(errors), "errors": errors[:100]}


def build(args: argparse.Namespace) -> dict[str, Any]:
    populated_outputs = [
        args.output_dir / "metadata.json",
        args.output_dir / "train",
        args.output_dir / "validation",
        args.output_dir / "test",
    ]
    if args.output_dir.exists() and not args.overwrite and any(path.exists() for path in populated_outputs):
        raise FileExistsError(f"{args.output_dir} already contains split output; pass --overwrite")
    if args.overwrite and args.output_dir.exists() and args.reuse_bucketed_input:
        for child_name in ("metadata.json", "train", "validation", "test"):
            child = args.output_dir / child_name
            if child.is_dir():
                shutil.rmtree(child)
            elif child.exists():
                child.unlink()
    elif args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_bucketed_input:
        thresholds = read_similarity_quantile_metadata(args)
        if thresholds is None:
            thresholds = []
            args._quantile_sample_strategy = "reused_bucketed_input_thresholds_unavailable"
            args._quantile_values_seen = 0
            args._quantile_values_sampled = 0
            args._quantile_files_read = 0
            args._quantile_row_groups_read = 0
            write_similarity_quantile_metadata(args, thresholds)
            print(
                f"[{utc_now()}] skip similarity quantile estimation: "
                "reusing precomputed similarity_bucket labels; thresholds unavailable",
                file=sys.stderr,
                flush=True,
            )
    else:
        thresholds = compute_similarity_thresholds(args)
        write_similarity_quantile_metadata(args, thresholds)
    bucketed_input = ensure_bucketed_input(args, thresholds)
    source_strata = read_counter_checkpoint(args, "source_strata.json")
    if source_strata is None:
        source_strata = collect_stratum_counts(
            bucketed_input,
            args.metadata_columns,
            args.batch_size,
            progress_label="collect source stratum counts",
            progress_every_seconds=args.progress_every_seconds,
            total_rows=getattr(args, "_bucketed_rows", None),
            similarity_buckets=args.similarity_buckets,
        )
        write_counter_checkpoint(
            args,
            "source_strata.json",
            source_strata,
            extra={"source": "bucketed_input"},
        )

    selected_pairs: dict[tuple[int, int], str] = {}
    selected_rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in EVAL_SPLITS}
    selection_metadata: dict[str, Any] = {}
    excluded_molecules: set[int] = set()
    for split in EVAL_SPLITS:
        checkpointed_selection = read_selection_checkpoint(args, split)
        if checkpointed_selection is not None:
            pairs, molecules, meta, rows = checkpointed_selection
            selected_pairs.update(pairs)
            selected_rows_by_split[split] = rows
            excluded_molecules.update(molecules)
            selection_metadata[split] = meta
            continue

        if excluded_molecules:
            available_checkpoint = f"{split}_available_strata.json"
            available_strata = read_counter_checkpoint(args, available_checkpoint)
            if available_strata is None:
                available_strata = collect_stratum_counts(
                    bucketed_input,
                    args.metadata_columns,
                    args.batch_size,
                    excluded_molecules=excluded_molecules,
                    progress_label=f"collect {split} available strata",
                    progress_every_seconds=args.progress_every_seconds,
                    total_rows=getattr(args, "_bucketed_rows", None),
                    similarity_buckets=args.similarity_buckets,
                )
                write_counter_checkpoint(
                    args,
                    available_checkpoint,
                    available_strata,
                    extra={"source": "bucketed_input", "excluded_molecules": len(excluded_molecules)},
                )
        else:
            available_strata = source_strata
        pairs, molecules, meta, rows = select_eval_split(
            input_path=bucketed_input,
            metadata_columns=args.metadata_columns,
            batch_size=args.batch_size,
            target_pairs=args.eval_pairs_per_split,
            available_strata=available_strata,
            excluded_molecules=excluded_molecules,
            seed=args.seed,
            split=split,
            args=args,
        )
        selected_pairs.update(pairs)
        selected_rows_by_split[split] = rows
        excluded_molecules.update(molecules)
        selection_metadata[split] = meta
        write_selection_checkpoint(args, split, pairs, molecules, meta, rows)

    source_meta_path = args.input_dir / "metadata.json"
    source_meta = json.loads(source_meta_path.read_text()) if source_meta_path.exists() else {}
    pair_code_multiplier = int(
        source_meta.get("load_stats", {}).get("records_loaded")
        or (max(excluded_molecules) + 1 if excluded_molecules else 1_000_000)
    )
    pair_code_multiplier = max(pair_code_multiplier + 1, 1_000_000)
    stats = write_split_outputs(
        input_path=bucketed_input,
        output_dir=args.output_dir,
        selected_pairs=selected_pairs,
        selected_rows_by_split=selected_rows_by_split,
        eval_molecules=excluded_molecules,
        metadata_columns=args.metadata_columns,
        batch_size=args.batch_size,
        compression=args.parquet_compression,
        progress_every_seconds=args.progress_every_seconds,
        similarity_buckets=args.similarity_buckets,
        pair_code_multiplier=pair_code_multiplier,
        total_rows=getattr(args, "_bucketed_rows", None),
    )
    validation = validate(stats)
    write_json(
        checkpoint_path(args, "write_stats.json"),
        {
            "created_at_utc": utc_now(),
            "write_stats": summarize_stats(stats, args.metadata_columns),
            "validation": validation,
        },
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_pair_dir": str(args.input_dir),
        "source_pair_metadata": {
            "schema_version": source_meta.get("schema_version"),
            "pairs_written": source_meta.get("pairs_written"),
            "candidate_pairs_seen": source_meta.get("candidate_pairs_seen"),
        },
        "output_dir": str(args.output_dir),
        "split_version": SPLIT_VERSION,
        "seed": args.seed,
        "target_pairs": {split: args.eval_pairs_per_split for split in EVAL_SPLITS},
        "selection_policy": {
            "order": "validation_then_test_then_train",
            "eval_allocation": "largest_remainder_proportional_by_stratum",
            "eval_pair_selection": "within needed strata, prefer pairs with two reused molecules, then one, then zero",
            "train_policy": "all pairs not touching validation/test molecules",
            "cross_split_pair_policy": "throw_out_any_pair_touching_validation_or_test_molecules",
            "molecule_overlap_allowed": False,
            "stratification_fields": ["transfer_label", "similarity_bucket", *args.metadata_columns],
            "metadata_stratification_mode": "presence_codes",
            "candidate_pool": "deterministic bounded reservoir per allocated stratum before greedy molecule reuse",
        },
        "metadata_columns": args.metadata_columns,
        "similarity_thresholds": [compact_float(value) for value in thresholds],
        "similarity_bucket_policy": (
            "provided_fixed_thresholds"
            if args.similarity_thresholds
            else "estimated_quantile_thresholds"
        ),
        "similarity_quantile_thresholds": [compact_float(value) for value in thresholds],
        "similarity_quantile_thresholds_available": bool(thresholds),
        "similarity_quantile_estimation": {
            "mode": "provided"
            if args.similarity_thresholds
            else getattr(args, "_quantile_sample_strategy", "proportional_parquet_sample"),
            "values_seen": int(getattr(args, "_quantile_values_seen", 0)),
            "values_sampled": int(getattr(args, "_quantile_values_sampled", 0)),
            "max_quantile_values": args.max_quantile_values,
            "files_read": int(getattr(args, "_quantile_files_read", 0)),
            "row_groups_read": int(getattr(args, "_quantile_row_groups_read", 0)),
        },
        "source_strata": serializable_counter(source_strata, args.metadata_columns),
        "selection": {
            split: {
                **{
                    key: value
                    for key, value in selection_metadata[split].items()
                    if key not in {"selected_strata", "allocation"}
                },
                "selected_strata": serializable_counter(
                    selection_metadata[split].get("selected_strata", Counter()),
                    args.metadata_columns,
                ),
                "allocation": serializable_counter(
                    Counter(selection_metadata[split].get("allocation", {})),
                    args.metadata_columns,
                ),
            }
            for split in EVAL_SPLITS
        },
        "write_stats": summarize_stats(stats, args.metadata_columns),
        "validation": validation,
        "files": {
            "train": "train/",
            "validation": "validation/",
            "test": "test/",
            "metadata": "metadata.json",
        },
    }
    write_json(args.output_dir / "metadata.json", metadata)
    if not args.keep_bucketed_input and not args.reuse_bucketed_input and bucketed_input.exists():
        shutil.rmtree(bucketed_input)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-columns", nargs="+", default=DEFAULT_METADATA_COLUMNS)
    parser.add_argument("--eval-pairs-per-split", type=int, default=30_000)
    parser.add_argument("--similarity-buckets", type=int, default=5)
    parser.add_argument("--similarity-thresholds", type=float, nargs="+", default=None)
    parser.add_argument(
        "--max-quantile-values",
        type=int,
        default=5_000_000,
        help="Maximum deterministic sample size used to estimate similarity quantiles.",
    )
    parser.add_argument(
        "--candidate-pool-multiplier",
        type=int,
        default=50,
        help="Per-stratum candidate pool target as quota times this multiplier.",
    )
    parser.add_argument(
        "--min-candidates-per-stratum",
        type=int,
        default=100,
        help="Minimum candidate pool size for any allocated stratum.",
    )
    parser.add_argument(
        "--max-candidates-per-stratum",
        type=int,
        default=5_000,
        help="Maximum oversampled candidate pool size per stratum; quota is always retained.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--row-group-size", type=int, default=250_000)
    parser.add_argument(
        "--bucket-file-row-limit",
        type=int,
        default=10_000_000,
        help="Approximate rows per bucketed Parquet output file.",
    )
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument(
        "--progress-every-seconds",
        type=float,
        default=60.0,
        help="Write phase progress with rows/sec and ETA to stderr at this interval; set 0 to disable periodic updates.",
    )
    parser.add_argument("--keep-bucketed-input", action="store_true")
    parser.add_argument(
        "--reuse-bucketed-input",
        action="store_true",
        help="Reuse output_dir/_bucketed_input and skip rebuilding similarity_bucket labels.",
    )
    parser.add_argument(
        "--resume-checkpoints",
        action="store_true",
        help="Reuse completed checkpoint files under output_dir/checkpoints.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.eval_pairs_per_split < 1:
        parser.error("--eval-pairs-per-split must be positive")
    if args.similarity_buckets < 2:
        parser.error("--similarity-buckets must be at least 2")
    if args.similarity_thresholds is not None and len(args.similarity_thresholds) != args.similarity_buckets - 1:
        parser.error("--similarity-thresholds must provide similarity_buckets - 1 values")
    if args.batch_size < 1 or args.row_group_size < 1:
        parser.error("batch and row-group sizes must be positive")
    if args.bucket_file_row_limit < args.batch_size:
        parser.error("--bucket-file-row-limit must be at least --batch-size")
    if args.max_quantile_values < 1:
        parser.error("--max-quantile-values must be positive")
    if args.candidate_pool_multiplier < 1:
        parser.error("--candidate-pool-multiplier must be positive")
    if args.min_candidates_per_stratum < 1 or args.max_candidates_per_stratum < 1:
        parser.error("candidate pool sizes must be positive")
    if args.progress_every_seconds < 0:
        parser.error("--progress-every-seconds cannot be negative")
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
