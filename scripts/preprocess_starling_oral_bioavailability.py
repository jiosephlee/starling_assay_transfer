#!/usr/bin/env python3
"""Build a numeric Starling oral-bioavailability dataset and optionally upload it.

The output keeps exactly the original Starling data columns, but replaces
oral_bioavailability_value with the cleaned numeric percent value from
Kiria-Nozan/Starling-bioavailability-clean. Rows without a cleaned numeric match
are dropped.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_transfer import compact_float, replace_dir_from_tmp, upload_folder_to_hf, utc_now, write_json  # noqa: E402


DEFAULT_SOURCE_REPO = "starling-labs/Oral_Bioavailability"
DEFAULT_SOURCE_FILE = "data/train-00000-of-00001.parquet"
DEFAULT_CLEAN_REPO = "Kiria-Nozan/Starling-bioavailability-clean"
DEFAULT_CLEAN_FILE = "data/molecule_records.jsonl.gz"
DEFAULT_OUTPUT_DIR = Path("datasets/base/starling_oral_bioavailability_numeric")
VALUE_COLUMN = "oral_bioavailability_value"


def load_clean_values(path: Path, max_clean_rows: int | None) -> tuple[dict[int, dict[str, Any]], Counter[str]]:
    clean: dict[int, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_index = int(row["source_index"])
            value = row.get("oral_bioavailability_value_percent")
            if value is None:
                stats["missing_numeric_value"] += 1
                continue
            if source_index in clean:
                stats["duplicate_source_index"] += 1
                continue
            clean[source_index] = row
            stats["clean_rows"] += 1
            if max_clean_rows is not None and stats["clean_rows"] >= max_clean_rows:
                break
    return clean, stats


def output_schema(source_schema: pa.Schema) -> pa.Schema:
    fields: list[pa.Field] = []
    for field in source_schema:
        if field.name == VALUE_COLUMN:
            fields.append(pa.field(field.name, pa.float64()))
        else:
            fields.append(field)
    return pa.schema(fields)


def compare_metadata(source_row: dict[str, Any], clean_row: dict[str, Any]) -> list[str]:
    metadata = clean_row.get("metadata") or {}
    mismatches: list[str] = []
    for key, value in source_row.items():
        if key == VALUE_COLUMN:
            continue
        if metadata.get(key) != value:
            mismatches.append(key)
    return mismatches


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(
        hf_hub_download(args.source_repo, args.source_file, repo_type="dataset", revision=args.source_revision)
    )
    clean_path = Path(
        hf_hub_download(args.clean_repo, args.clean_file, repo_type="dataset", revision=args.clean_revision)
    )
    clean_values, clean_stats = load_clean_values(clean_path, args.max_clean_rows)
    if not clean_values:
        raise RuntimeError("no cleaned numeric rows loaded")

    source = pq.ParquetFile(source_path)
    schema = output_schema(source.schema_arrow)
    rows_buffer: list[dict[str, Any]] = []
    written = 0
    source_rows = 0
    audit_mismatches: Counter[str] = Counter()
    unmatched_source_rows = 0

    tmp_parent = args.output_dir.parent
    tmp_parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.TemporaryDirectory(prefix=f".{args.output_dir.name}.", dir=tmp_parent) as tmp_name:
        tmp = Path(tmp_name)
        output_path = tmp / "train.parquet"
        writer = pq.ParquetWriter(output_path, schema=schema, compression=args.compression)
        try:
            for batch in source.iter_batches(batch_size=args.batch_size):
                for source_row in batch.to_pylist():
                    if args.max_source_rows is not None and source_rows >= args.max_source_rows:
                        break
                    clean_row = clean_values.get(source_rows)
                    source_rows += 1
                    if clean_row is None:
                        unmatched_source_rows += 1
                        continue
                    mismatches = compare_metadata(source_row, clean_row)
                    if mismatches:
                        for key in mismatches:
                            audit_mismatches[key] += 1
                        if args.strict_audit:
                            raise AssertionError(
                                f"source row {source_rows - 1} differs from cleaned metadata: {mismatches}"
                            )
                    out = dict(source_row)
                    out[VALUE_COLUMN] = float(clean_row["oral_bioavailability_value_percent"])
                    rows_buffer.append(out)
                    written += 1
                    if len(rows_buffer) >= args.row_group_size:
                        writer.write_table(pa.Table.from_pylist(rows_buffer, schema=schema))
                        rows_buffer.clear()
                if args.max_source_rows is not None and source_rows >= args.max_source_rows:
                    break
            if rows_buffer:
                writer.write_table(pa.Table.from_pylist(rows_buffer, schema=schema))
                rows_buffer.clear()
        finally:
            writer.close()

        metadata = {
            "schema_version": "starling_oral_bioavailability_numeric_v1",
            "created_at_utc": utc_now(),
            "source_repo": args.source_repo,
            "source_revision": args.source_revision,
            "source_file": args.source_file,
            "clean_repo": args.clean_repo,
            "clean_revision": args.clean_revision,
            "clean_file": args.clean_file,
            "output_dir": str(args.output_dir),
            "output_file": "train.parquet",
            "source_rows_seen": source_rows,
            "clean_rows_loaded": int(clean_stats["clean_rows"]),
            "rows_written": written,
            "unmatched_source_rows_seen": unmatched_source_rows,
            "value_column": VALUE_COLUMN,
            "value_policy": "replace original string value with cleaned numeric percent; drop unmatched rows",
            "columns": [field.name for field in schema],
            "audit_mismatches": dict(sorted(audit_mismatches.items())),
            "strict_audit": args.strict_audit,
            "max_source_rows": args.max_source_rows,
            "max_clean_rows": args.max_clean_rows,
            "parquet": {
                "compression": args.compression,
                "row_group_size": args.row_group_size,
            },
        }
        write_json(tmp / "dataset_info.json", metadata)
        (tmp / "README.md").write_text(
            "---\n"
            "dataset_info:\n"
            "  features:\n"
            + "\n".join(f"  - name: {field.name}\n    dtype: {field.type}" for field in schema)
            + "\n  splits:\n  - name: train\n"
            f"    num_examples: {written}\n"
            "---\n\n"
            "# Starling Oral Bioavailability Numeric\n\n"
            "This dataset keeps the original `starling-labs/Oral_Bioavailability` columns and "
            "replaces `oral_bioavailability_value` with the cleaned numeric percent value from "
            "`Kiria-Nozan/Starling-bioavailability-clean`.\n",
        )
        replace_dir_from_tmp(tmp, args.output_dir)

    if args.repo_id:
        upload_folder_to_hf(
            folder_path=args.output_dir,
            repo_id=args.repo_id,
            private=args.private,
            path_in_repo=args.path_in_repo,
            commit_message=args.commit_message,
        )
        metadata["uploaded_repo_id"] = args.repo_id
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--source-file", default=DEFAULT_SOURCE_FILE)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--clean-repo", default=DEFAULT_CLEAN_REPO)
    parser.add_argument("--clean-file", default=DEFAULT_CLEAN_FILE)
    parser.add_argument("--clean-revision", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--row-group-size", type=int, default=50_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--max-clean-rows", type=int, default=None)
    parser.add_argument("--strict-audit", action="store_true")
    parser.add_argument("--repo-id", default=None, help="Upload destination dataset repo ID.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--path-in-repo", default=None)
    parser.add_argument("--commit-message", default="Upload numeric Starling oral bioavailability")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in ("batch_size", "row_group_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_source_rows", "max_clean_rows"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    metadata = build(parse_args())
    print(
        json.dumps(
            {
                "output_dir": metadata["output_dir"],
                "rows_written": metadata["rows_written"],
                "source_rows_seen": metadata["source_rows_seen"],
                "audit_mismatches": metadata["audit_mismatches"],
                "uploaded_repo_id": metadata.get("uploaded_repo_id"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
