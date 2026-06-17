#!/usr/bin/env python3
"""Materialize full split pair rows from lightweight split rows and a base dataset."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_transfer import SPLITS, compact_json, iter_parquet_rows, prepare_output_dir, read_jsonl_gz, utc_now, write_json  # noqa: E402


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


def build(args: argparse.Namespace) -> dict[str, Any]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", default=DEFAULT_BASE_INPUT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--metadata-columns", nargs="+", default=DEFAULT_METADATA_COLUMNS)
    parser.add_argument("--read-batch-size", type=int, default=8192)
    parser.add_argument("--max-base-rows", type=int, default=None)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--gzip-compresslevel", type=int, default=1)
    parser.add_argument("--strict-smiles", action="store_true", default=True)
    parser.add_argument("--no-strict-smiles", dest="strict_smiles", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.read_batch_size < 1:
        parser.error("--read-batch-size must be positive")
    if args.max_base_rows is not None and args.max_base_rows < 1:
        parser.error("--max-base-rows must be positive")
    if args.max_rows_per_split is not None and args.max_rows_per_split < 1:
        parser.error("--max-rows-per-split must be positive")
    if not 0 <= args.gzip_compresslevel <= 9:
        parser.error("--gzip-compresslevel must be between 0 and 9")
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
