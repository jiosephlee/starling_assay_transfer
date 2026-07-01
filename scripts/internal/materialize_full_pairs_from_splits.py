#!/usr/bin/env python3
"""Materialize full split pair rows from lightweight split rows and a base dataset."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_transfer import (  # noqa: E402
    SPLITS,
    compact_json,
    iter_parquet_rows,
    parquet_files_from_input,
    prepare_output_dir,
    read_jsonl_gz,
    utc_now,
    write_json,
)


DEFAULT_BASE_INPUT = "datasets/base/Oral_bioavailability_cleaned"
DEFAULT_SPLIT_DIR = Path("datasets/pairs_split/generic_transfer_pair_splits")
DEFAULT_OUTPUT_DIR = Path("datasets/pairs_split_full/generic_transfer_pair_splits_full")
DEFAULT_METADATA_COLUMNS = [
    "support_text",
    "molecule_name",
    "bioavailability_report_type",
    "species_or_population",
    "dose",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
    "extra_details",
]
SCHEMA_VERSION = "generic_transfer_pair_splits_full_v1"
COMPACT_SPLIT_SCHEMA_VERSION = "generic_transfer_pair_splits_compact_v1"
LABEL_TEXT = {0: "not_transfer", 1: "transfer"}


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
            rows = f"{current:,}/{self.total:,} ({pct:.2f}%)"
        else:
            eta = None
            rows = f"{current:,}"
        suffix = f" {extra}" if extra else ""
        print(
            f"[{utc_now()}] {self.phase}: rows={rows} "
            f"rate={rate:,.0f}/s elapsed={format_duration(elapsed)} eta={format_duration(eta)}{suffix}",
            file=sys.stderr,
            flush=True,
        )
        self.last = now

    def finish(self, current: int, *, extra: str = "") -> None:
        self.update(current, force=True, extra=extra)


def load_base_rows(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    columns = sorted(set([args.smiles_column, *args.metadata_columns]))
    rows: dict[int, dict[str, Any]] = {}
    for row_index, row in enumerate(
        iter_parquet_rows(
            args.base_input,
            columns=columns,
            batch_size=args.read_batch_size,
            max_rows=args.max_base_rows,
        )
    ):
        rows[row_index] = row
    if not rows:
        raise RuntimeError("no base rows loaded")
    return rows


def materialize_molecule(
    molecule: dict[str, Any],
    *,
    row_index: int,
    base_rows: dict[int, dict[str, Any]],
    metadata_columns: list[str],
    smiles_column: str,
    strict_smiles: bool,
) -> dict[str, Any]:
    base = base_rows.get(row_index)
    if base is None:
        raise KeyError(f"base row index not found: {row_index}")
    base_smiles = base.get(smiles_column)
    if strict_smiles and str(base_smiles) != str(molecule.get("canonical_smiles")):
        raise AssertionError(
            f"SMILES mismatch for row {row_index}: pair={molecule.get('canonical_smiles')} base={base_smiles}"
        )
    out = dict(molecule)
    out["metadata"] = {column: base.get(column) for column in metadata_columns}
    return out


def materialize_row(
    row: dict[str, Any],
    *,
    base_rows: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    out = dict(row)
    out["schema_version"] = SCHEMA_VERSION
    out["source_schema_version"] = row.get("schema_version")
    left_index = int(row.get("row_index_a", row["molecule_a"]["row_index"]))
    right_index = int(row.get("row_index_b", row["molecule_b"]["row_index"]))
    out["molecule_a"] = materialize_molecule(
        row["molecule_a"],
        row_index=left_index,
        base_rows=base_rows,
        metadata_columns=args.metadata_columns,
        smiles_column=args.smiles_column,
        strict_smiles=args.strict_smiles,
    )
    out["molecule_b"] = materialize_molecule(
        row["molecule_b"],
        row_index=right_index,
        base_rows=base_rows,
        metadata_columns=args.metadata_columns,
        smiles_column=args.smiles_column,
        strict_smiles=args.strict_smiles,
    )
    return out


def load_base_table(args: argparse.Namespace) -> pa.Table:
    columns = sorted(set([args.smiles_column, *args.metadata_columns]))
    tables = [
        pq.read_table(path, columns=columns)
        for path in parquet_files_from_input(args.base_input)
    ]
    if not tables:
        raise RuntimeError("no base Parquet files loaded")
    table = pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]
    if args.max_base_rows is not None:
        table = table.slice(0, args.max_base_rows)
    if table.num_rows <= 0:
        raise RuntimeError("no base rows loaded")
    return table


def metadata_struct_type(metadata_columns: list[str]) -> pa.StructType:
    return pa.struct([(column, pa.large_string()) for column in metadata_columns])


def molecule_struct_type(metadata_columns: list[str]) -> pa.StructType:
    return pa.struct(
        [
            ("record_id", pa.string()),
            ("row_index", pa.uint32()),
            ("canonical_smiles", pa.large_string()),
            ("metadata", metadata_struct_type(metadata_columns)),
        ]
    )


def parquet_output_schema(metadata_columns: list[str]) -> pa.Schema:
    return pa.schema(
        [
            ("schema_version", pa.string()),
            ("source_schema_version", pa.string()),
            ("pair_id", pa.string()),
            ("record_id_a", pa.string()),
            ("record_id_b", pa.string()),
            ("row_index_a", pa.uint32()),
            ("row_index_b", pa.uint32()),
            ("group_id", pa.string()),
            ("molecule_a", molecule_struct_type(metadata_columns)),
            ("molecule_b", molecule_struct_type(metadata_columns)),
            ("transfer_label", pa.string()),
            ("value_difference", pa.float32()),
            ("T_transfer", pa.float32()),
            ("T_not_transfer", pa.float32()),
            ("weighted_tanimoto", pa.float32()),
            ("similarity_bucket", pa.int8()),
            ("split", pa.string()),
            ("split_version", pa.string()),
            ("eval_subset", pa.string()),
        ]
    )


def constant_array(value: str | None, length: int, type_: pa.DataType = pa.string()) -> pa.Array:
    if value is None:
        return pa.nulls(length, type=type_)
    return pa.array([value] * length, type=type_)


def take_column(base_table: pa.Table, column: str, indices: pa.Array) -> pa.Array:
    return pc.take(base_table[column].combine_chunks(), indices)


def metadata_struct(base_table: pa.Table, indices: pa.Array, metadata_columns: list[str]) -> pa.StructArray:
    return pa.StructArray.from_arrays(
        [take_column(base_table, column, indices) for column in metadata_columns],
        names=metadata_columns,
    )


def molecule_struct(
    *,
    base_table: pa.Table,
    indices: pa.Array,
    metadata_columns: list[str],
    smiles_column: str,
) -> pa.StructArray:
    record_ids = pc.cast(indices, pa.string())
    return pa.StructArray.from_arrays(
        [
            record_ids,
            indices,
            take_column(base_table, smiles_column, indices),
            metadata_struct(base_table, indices, metadata_columns),
        ],
        names=["record_id", "row_index", "canonical_smiles", "metadata"],
    )


def label_array(labels: pa.Array) -> pa.Array:
    is_transfer = pc.equal(labels, pa.scalar(1, type=pa.int8()))
    return pc.if_else(
        is_transfer,
        pa.scalar("transfer", type=pa.string()),
        pa.scalar("not_transfer", type=pa.string()),
    )


def compact_batch_to_full_table(
    batch: pa.RecordBatch,
    *,
    base_table: pa.Table,
    split: str,
    split_version: str | None,
    metadata_columns: list[str],
    smiles_column: str,
    source_schema_version: str,
    transfer_threshold: float,
    not_transfer_threshold: float,
) -> pa.Table:
    length = batch.num_rows
    left = batch.column("row_index_a").cast(pa.uint32())
    right = batch.column("row_index_b").cast(pa.uint32())
    left_text = pc.cast(left, pa.string())
    right_text = pc.cast(right, pa.string())
    pair_id = pc.binary_join_element_wise(left_text, right_text, ":")
    schema = parquet_output_schema(metadata_columns)
    table = pa.Table.from_arrays(
        [
            constant_array(SCHEMA_VERSION, length),
            constant_array(source_schema_version, length),
            pair_id,
            left_text,
            right_text,
            left,
            right,
            constant_array(None, length),
            molecule_struct(
                base_table=base_table,
                indices=left,
                metadata_columns=metadata_columns,
                smiles_column=smiles_column,
            ),
            molecule_struct(
                base_table=base_table,
                indices=right,
                metadata_columns=metadata_columns,
                smiles_column=smiles_column,
            ),
            label_array(batch.column("transfer_label")),
            batch.column("value_difference").cast(pa.float32()),
            pa.array([transfer_threshold] * length, type=pa.float32()),
            pa.array([not_transfer_threshold] * length, type=pa.float32()),
            batch.column("weighted_tanimoto").cast(pa.float32()),
            batch.column("similarity_bucket").cast(pa.int8()),
            constant_array(split, length),
            constant_array(split_version, length),
            constant_array(split if split in {"validation", "test"} else None, length),
        ],
        schema=schema,
    )
    return table


class RollingParquetWriter:
    def __init__(
        self,
        split_dir: Path,
        schema: pa.Schema,
        *,
        file_row_limit: int,
        compression: str,
    ) -> None:
        self.split_dir = split_dir
        self.schema = schema
        self.file_row_limit = file_row_limit
        self.compression = compression
        self.part_id = 0
        self.rows_in_part = 0
        self.writer: pq.ParquetWriter | None = None
        self.split_dir.mkdir(parents=True, exist_ok=True)

    def write_table(self, table: pa.Table) -> None:
        offset = 0
        while offset < table.num_rows:
            if self.writer is None:
                self.writer = pq.ParquetWriter(
                    self.split_dir / f"part-{self.part_id:05d}.parquet",
                    self.schema,
                    compression=self.compression,
                    use_dictionary=False,
                    write_statistics=True,
                )
                self.rows_in_part = 0
            remaining = table.num_rows - offset
            space = max(self.file_row_limit - self.rows_in_part, 1)
            take = min(remaining, space)
            self.writer.write_table(table.slice(offset, take))
            offset += take
            self.rows_in_part += take
            if self.rows_in_part >= self.file_row_limit:
                self.close_part()

    def close_part(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None
        self.part_id += 1
        self.rows_in_part = 0

    def close(self) -> None:
        self.close_part()


def split_parquet_path(split_dir: Path, split: str) -> Path:
    path = split_dir / split
    if path.exists():
        return path
    parquet_file = split_dir / f"{split}.parquet"
    if parquet_file.exists():
        return parquet_file
    raise FileNotFoundError(f"no compact Parquet split found for {split} under {split_dir}")


def parquet_row_count(path: Path) -> int:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
    else:
        files = [path]
    return sum(pq.ParquetFile(file_path).metadata.num_rows for file_path in files)


def prepare_parquet_output(output_dir: Path, splits: list[str], overwrite: bool) -> None:
    expected = [output_dir / split for split in splits] + [output_dir / "metadata.json"]
    present = [path for path in expected if path.exists()]
    if present and not overwrite:
        formatted = "\n".join(str(path) for path in present)
        raise FileExistsError(f"output files exist; pass --overwrite:\n{formatted}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in present:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()


def build_jsonl(args: argparse.Namespace) -> dict[str, Any]:
    for split in args.splits:
        path = args.split_dir / f"{split}.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(path)
    prepare_output_dir(
        args.output_dir,
        [f"{split}.jsonl.gz" for split in args.splits] + ["metadata.json"],
        args.overwrite,
    )
    base_rows = load_base_rows(args)
    stats: dict[str, Any] = {
        "rows_by_split": Counter(),
        "eval_subset_counts": {split: Counter() for split in args.splits},
    }
    for split in args.splits:
        with gzip.open(
            args.output_dir / f"{split}.jsonl.gz",
            "wt",
            compresslevel=args.gzip_compresslevel,
        ) as handle:
            for row in read_jsonl_gz(args.split_dir / f"{split}.jsonl.gz", max_rows=args.max_rows_per_split):
                out = materialize_row(row, base_rows=base_rows, args=args)
                handle.write(compact_json(out) + "\n")
                stats["rows_by_split"][split] += 1
                stats["eval_subset_counts"][split][out.get("eval_subset") or "none"] += 1

    source_metadata_path = args.split_dir / "metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text()) if source_metadata_path.exists() else {}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "base_input": args.base_input,
        "source_split_dir": str(args.split_dir),
        "source_split_metadata": {
            "schema_version": source_metadata.get("schema_version"),
            "split_version": source_metadata.get("split_version"),
            "selection_policy": source_metadata.get("selection_policy"),
            "write_stats": source_metadata.get("write_stats"),
        },
        "output_dir": str(args.output_dir),
        "splits": list(args.splits),
        "metadata_columns": args.metadata_columns,
        "smiles_column": args.smiles_column,
        "strict_smiles": args.strict_smiles,
        "rows_by_split": dict(sorted(stats["rows_by_split"].items())),
        "eval_subset_counts": {
            split: dict(sorted(stats["eval_subset_counts"][split].items())) for split in args.splits
        },
        "note": "Full metadata is materialized after split selection and molecule-overlap discard.",
    }
    write_json(args.output_dir / "metadata.json", metadata)
    return metadata


def build_compact_parquet(args: argparse.Namespace) -> dict[str, Any]:
    prepare_parquet_output(args.output_dir, list(args.splits), args.overwrite)
    base_table = load_base_table(args)
    source_metadata_path = args.split_dir / "metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text()) if source_metadata_path.exists() else {}
    source_pair_metadata = source_metadata.get("source_pair_metadata") or {}
    source_schema_version = str(source_metadata.get("schema_version") or COMPACT_SPLIT_SCHEMA_VERSION)
    split_version = source_metadata.get("split_version")
    thresholds = source_pair_metadata.get("thresholds") or {}
    transfer_threshold = float(thresholds.get("transfer", args.transfer_threshold))
    not_transfer_threshold = float(thresholds.get("not_transfer", args.not_transfer_threshold))
    schema = parquet_output_schema(args.metadata_columns)
    stats: dict[str, Any] = {
        "rows_by_split": Counter(),
        "eval_subset_counts": {split: Counter() for split in args.splits},
        "files_by_split": Counter(),
    }
    for split in args.splits:
        input_path = split_parquet_path(args.split_dir, split)
        total_rows = parquet_row_count(input_path)
        max_rows = min(total_rows, args.max_rows_per_split) if args.max_rows_per_split else total_rows
        progress = ProgressLogger(f"materialize {split}", max_rows, args.progress_every_seconds)
        writer = RollingParquetWriter(
            args.output_dir / split,
            schema,
            file_row_limit=args.parquet_file_row_limit,
            compression=args.parquet_compression,
        )
        dataset = ds.dataset(input_path, format="parquet")
        rows = 0
        try:
            for batch in dataset.to_batches(batch_size=args.batch_size):
                if args.max_rows_per_split is not None and rows >= args.max_rows_per_split:
                    break
                if args.max_rows_per_split is not None and rows + batch.num_rows > args.max_rows_per_split:
                    batch = batch.slice(0, args.max_rows_per_split - rows)
                table = compact_batch_to_full_table(
                    batch,
                    base_table=base_table,
                    split=split,
                    split_version=split_version,
                    metadata_columns=args.metadata_columns,
                    smiles_column=args.smiles_column,
                    source_schema_version=source_schema_version,
                    transfer_threshold=transfer_threshold,
                    not_transfer_threshold=not_transfer_threshold,
                )
                writer.write_table(table)
                rows += table.num_rows
                stats["rows_by_split"][split] += table.num_rows
                stats["eval_subset_counts"][split][split if split in {"validation", "test"} else "none"] += table.num_rows
                progress.update(rows, extra=f"parts_written={writer.part_id + int(writer.writer is not None):,}")
        finally:
            writer.close()
        stats["files_by_split"][split] = len(sorted((args.output_dir / split).glob("*.parquet")))
        progress.finish(rows, extra=f"parts_written={stats['files_by_split'][split]:,}")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "base_input": args.base_input,
        "source_split_dir": str(args.split_dir),
        "source_split_metadata": {
            "schema_version": source_metadata.get("schema_version"),
            "split_version": source_metadata.get("split_version"),
            "selection_policy": source_metadata.get("selection_policy"),
            "write_stats": source_metadata.get("write_stats"),
        },
        "output_dir": str(args.output_dir),
        "input_format": "compact_parquet",
        "output_format": "parquet",
        "splits": list(args.splits),
        "metadata_columns": args.metadata_columns,
        "smiles_column": args.smiles_column,
        "strict_smiles": False,
        "rows_by_split": dict(sorted(stats["rows_by_split"].items())),
        "eval_subset_counts": {
            split: dict(sorted(stats["eval_subset_counts"][split].items())) for split in args.splits
        },
        "files_by_split": dict(sorted(stats["files_by_split"].items())),
        "parquet": {
            "compression": args.parquet_compression,
            "file_row_limit": args.parquet_file_row_limit,
            "batch_size": args.batch_size,
        },
        "row_schema": "nested molecule_a/molecule_b structs with base metadata joined by row_index",
        "note": "Full metadata is materialized after split selection and molecule-overlap discard.",
    }
    write_json(args.output_dir / "metadata.json", metadata)
    return metadata


def infer_input_format(split_dir: Path, splits: list[str]) -> str:
    if all((split_dir / f"{split}.jsonl.gz").exists() for split in splits):
        return "jsonl"
    if all((split_dir / split).exists() or (split_dir / f"{split}.parquet").exists() for split in splits):
        return "compact_parquet"
    raise FileNotFoundError(f"could not infer split input format under {split_dir}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_format = infer_input_format(args.split_dir, list(args.splits)) if args.input_format == "auto" else args.input_format
    if input_format == "jsonl":
        if args.output_format != "jsonl":
            raise ValueError("legacy jsonl input currently supports only --output-format jsonl")
        return build_jsonl(args)
    if input_format == "compact_parquet":
        if args.output_format != "parquet":
            raise ValueError("compact parquet input currently supports only --output-format parquet")
        return build_compact_parquet(args)
    raise ValueError(f"unsupported input format: {input_format}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", default=DEFAULT_BASE_INPUT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-format", choices=("auto", "jsonl", "compact_parquet"), default="auto")
    parser.add_argument("--output-format", choices=("jsonl", "parquet"), default="jsonl")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--metadata-columns", nargs="+", default=DEFAULT_METADATA_COLUMNS)
    parser.add_argument("--read-batch-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--max-base-rows", type=int, default=None)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--gzip-compresslevel", type=int, default=1)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--parquet-file-row-limit", type=int, default=10_000_000)
    parser.add_argument("--progress-every-seconds", type=float, default=60.0)
    parser.add_argument("--transfer-threshold", type=float, default=10.0)
    parser.add_argument("--not-transfer-threshold", type=float, default=30.0)
    parser.add_argument("--strict-smiles", action="store_true", default=True)
    parser.add_argument("--no-strict-smiles", dest="strict_smiles", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.read_batch_size < 1:
        parser.error("--read-batch-size must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_base_rows is not None and args.max_base_rows < 1:
        parser.error("--max-base-rows must be positive")
    if args.max_rows_per_split is not None and args.max_rows_per_split < 1:
        parser.error("--max-rows-per-split must be positive")
    if not 0 <= args.gzip_compresslevel <= 9:
        parser.error("--gzip-compresslevel must be between 0 and 9")
    if args.parquet_file_row_limit < 1:
        parser.error("--parquet-file-row-limit must be positive")
    if args.progress_every_seconds < 0:
        parser.error("--progress-every-seconds cannot be negative")
    if args.output_format == "jsonl" and args.input_format == "compact_parquet":
        parser.error("compact parquet input requires --output-format parquet")
    if args.output_format == "parquet" and args.input_format == "jsonl":
        parser.error("legacy jsonl input requires --output-format jsonl")
    return args


def main() -> None:
    metadata = build(parse_args())
    print(
        json.dumps(
            {
                "output_dir": metadata["output_dir"],
                "rows_by_split": metadata["rows_by_split"],
                "eval_subset_counts": metadata["eval_subset_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
