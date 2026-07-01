#!/usr/bin/env python3
"""Add a reproducible Tianang-style condition key to Oral Bioavailability v2."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_transfer import parquet_files_from_input, utc_now, write_json  # noqa: E402


DEFAULT_INPUT = "datasets/base/Oral_bioavailability_cleaned_v2"
DEFAULT_OUTPUT_DIR = Path("datasets/base/Oral_bioavailability_cleaned_v2_condition_key")
KEY_COLUMN = "condition_key_repro"
KEY_FIELDS = (
    "species_or_population",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
)
NORMALIZED_SPECIES_COLUMN = "species_or_population_normalized"
UNSPECIFIED_TOKEN = "not specified"
NULL_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "not applicable",
    "not reported",
    "not specified",
    "not stated",
    "null",
    "unknown",
    "unspecified",
}


def normalize_key_part(value: Any, *, allow_unspecified: bool) -> str | None:
    if value is None:
        return UNSPECIFIED_TOKEN if allow_unspecified else None
    text = str(value).strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    if text in NULL_VALUES:
        return UNSPECIFIED_TOKEN if allow_unspecified else None
    return text


def condition_key(row: dict[str, Any]) -> str | None:
    species = normalize_key_part(row.get("species_or_population"), allow_unspecified=False)
    if species is None:
        return None
    return " | ".join(
        [
            species,
            normalize_key_part(row.get("oral_exposure_mode"), allow_unspecified=True) or UNSPECIFIED_TOKEN,
            normalize_key_part(row.get("qualifying_conditions"), allow_unspecified=True) or UNSPECIFIED_TOKEN,
            normalize_key_part(row.get("comparator"), allow_unspecified=True) or UNSPECIFIED_TOKEN,
        ]
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tables = [pq.read_table(path) for path in parquet_files_from_input(args.input)]
    table = pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]
    missing = [field for field in KEY_FIELDS if field not in table.column_names]
    if missing:
        raise KeyError(f"missing condition-key field(s): {', '.join(missing)}")
    if NORMALIZED_SPECIES_COLUMN not in table.column_names:
        raise KeyError(f"missing compatibility field: {NORMALIZED_SPECIES_COLUMN}")

    rows = table.select(list(KEY_FIELDS) + [NORMALIZED_SPECIES_COLUMN]).to_pylist()
    keys = [condition_key(row) for row in rows]
    key_counts = Counter(key for key in keys if key is not None)
    condition_key_null_count = sum(1 for key in keys if key is None)
    normalized_species_null_count = sum(1 for row in rows if row.get(NORMALIZED_SPECIES_COLUMN) is None)

    if KEY_COLUMN in table.column_names:
        table = table.drop([KEY_COLUMN])
    table = table.append_column(KEY_COLUMN, pa.array(keys, type=pa.string()))
    pq.write_table(table, args.output_dir / "train.parquet", compression=args.compression)

    metadata = {
        "schema_version": "starling_oral_bioavailability_numeric_v2_condition_key_v1",
        "created_at_utc": utc_now(),
        "input": args.input,
        "output_dir": str(args.output_dir),
        "output_file": "train.parquet",
        "rows_written": table.num_rows,
        "added_columns": [KEY_COLUMN],
        "condition_key_column": KEY_COLUMN,
        "condition_key_fields": list(KEY_FIELDS),
        "condition_key_policy": (
            "lowercase, strip, collapse whitespace, and normalize dash variants; "
            "species_or_population is required and null/null-like values produce a null condition key; "
            f"missing/null-like non-species condition fields are encoded as {UNSPECIFIED_TOKEN!r}; "
            "source columns are otherwise unchanged"
        ),
        "compatibility_gate_column": NORMALIZED_SPECIES_COLUMN,
        "condition_key_null_count": condition_key_null_count,
        "normalized_species_null_count": normalized_species_null_count,
        "condition_key_counts": {
            "unique": len(key_counts),
            "top_25": dict(key_counts.most_common(25)),
        },
        "columns": table.column_names,
        "parquet": {"compression": args.compression},
    }
    write_json(args.output_dir / "dataset_info.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    metadata = build(parse_args())
    print(
        json.dumps(
            {
                "output_dir": metadata["output_dir"],
                "rows_written": metadata["rows_written"],
                "condition_key_unique": metadata["condition_key_counts"]["unique"],
                "condition_key_null_count": metadata["condition_key_null_count"],
                "normalized_species_null_count": metadata["normalized_species_null_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
