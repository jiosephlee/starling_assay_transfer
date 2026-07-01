#!/usr/bin/env python3
"""Build Oral Bioavailability v3 base data.

v3 starts from the cleaned Starling oral-bioavailability base, adds the v2
normalized species/population column and reproducible condition key, then removes
only molecules from TDC Bioavailability_Ma train/valid/test by raw or RDKit
canonical SMILES match.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
INTERNAL_DIR = SCRIPT_DIR / "internal"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(INTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(INTERNAL_DIR))

from common_transfer import parquet_files_from_input, utc_now, write_json  # noqa: E402
from species_normalization import (  # noqa: E402
    NORMALIZED_COLUMN,
    normalized_species_or_population,
)
from create_oral_bioavailability_condition_key_base import (  # noqa: E402
    KEY_COLUMN,
    condition_key,
)


DEFAULT_INPUT = "datasets/base/Oral_bioavailability_cleaned"
DEFAULT_OUTPUT_DIR = Path("datasets/base/Oral_bioavailability_cleaned_v3")
DEFAULT_TDC_DIR = Path("tdc/official_tianang")
DEFAULT_REFERENCE_DIR = Path("datasets/exclusions/tdc_bioavailability_ma_v3")
DEFAULT_TDC_SPLITS = ("train", "valid", "test")

CHEM: Any = None
CANONICAL_CACHE: dict[str, str | None] = {}


def init_rdkit() -> None:
    global CHEM
    if CHEM is not None:
        return
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.warning")
    CHEM = Chem


def canonical_smiles(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in CANONICAL_CACHE:
        return CANONICAL_CACHE[text]
    init_rdkit()
    mol = CHEM.MolFromSmiles(text)
    if mol is None:
        CANONICAL_CACHE[text] = None
        return None
    CANONICAL_CACHE[text] = CHEM.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return CANONICAL_CACHE[text]


def normalized_set(values: list[Any]) -> set[str]:
    return {str(value).strip() for value in values if value is not None and str(value).strip()}


def iter_tdc_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in args.tdc_splits:
        path = args.tdc_dir / split / f"{args.tdc_task}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw_row = json.loads(line)
                raw = str(raw_row.get("drug") or raw_row.get("Drug") or "").strip()
                can = canonical_smiles(raw)
                rows.append(
                    {
                        "task": args.tdc_task,
                        "split": split,
                        "source_file": str(path),
                        "source_row_number": row_number,
                        "raw_smiles": raw or None,
                        "canonical_smiles": can,
                        "Y": raw_row.get("Y"),
                        "valid_smiles": can is not None,
                    }
                )
                if args.max_tdc_rows is not None and len(rows) >= args.max_tdc_rows:
                    return rows
    if not rows:
        raise RuntimeError(f"no TDC rows found under {args.tdc_dir}")
    return rows


def load_base_table(args: argparse.Namespace) -> pa.Table:
    tables = [
        pq.read_table(path)
        for path in parquet_files_from_input(args.input)
    ]
    if not tables:
        raise RuntimeError("no base Parquet files loaded")
    table = pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]
    if args.max_rows is not None:
        table = table.slice(0, args.max_rows)
    if args.smiles_column not in table.column_names:
        raise KeyError(f"base input missing SMILES column: {args.smiles_column}")
    return table


def add_repro_columns(table: pa.Table) -> pa.Table:
    rows = table.to_pylist()
    species = [normalized_species_or_population(row.get("species_or_population")) for row in rows]
    keys = [
        condition_key({**row, NORMALIZED_COLUMN: species_value})
        for row, species_value in zip(rows, species, strict=True)
    ]
    for column in (NORMALIZED_COLUMN, KEY_COLUMN):
        if column in table.column_names:
            table = table.drop([column])
    table = table.append_column(NORMALIZED_COLUMN, pa.array(species, type=pa.string()))
    table = table.append_column(KEY_COLUMN, pa.array(keys, type=pa.string()))
    return table


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    if overwrite and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_reference_artifacts(reference_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    schema = pa.schema(
        [
            ("task", pa.string()),
            ("split", pa.string()),
            ("source_file", pa.string()),
            ("source_row_number", pa.int32()),
            ("raw_smiles", pa.large_string()),
            ("canonical_smiles", pa.large_string()),
            ("Y", pa.int64()),
            ("valid_smiles", pa.bool_()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), reference_dir / "tdc_bioavailability_ma_smiles.parquet")
    raw_unique = sorted(normalized_set([row["raw_smiles"] for row in rows]))
    canonical_unique = sorted(normalized_set([row["canonical_smiles"] for row in rows]))
    (reference_dir / "raw_smiles.txt").write_text("\n".join(raw_unique) + "\n")
    (reference_dir / "canonical_smiles.txt").write_text("\n".join(canonical_unique) + "\n")
    by_split = Counter(row["split"] for row in rows)
    metadata = {
        "schema_version": "tdc_bioavailability_ma_smiles_reference_v1",
        "created_at_utc": utc_now(),
        "rows": len(rows),
        "valid_smiles_rows": sum(1 for row in rows if row["valid_smiles"]),
        "invalid_smiles_rows": sum(1 for row in rows if not row["valid_smiles"]),
        "unique_raw_smiles": len(raw_unique),
        "unique_canonical_smiles": len(canonical_unique),
        "counts_by_split": dict(sorted(by_split.items())),
    }
    write_json(reference_dir / "metadata.json", metadata)
    return metadata


def build(args: argparse.Namespace) -> dict[str, Any]:
    prepare_dir(args.output_dir, args.overwrite)
    prepare_dir(args.reference_dir, args.overwrite)

    tdc_rows = iter_tdc_rows(args)
    reference_metadata = write_reference_artifacts(args.reference_dir, tdc_rows)
    table = add_repro_columns(load_base_table(args))

    tdc_raw = normalized_set([row["raw_smiles"] for row in tdc_rows])
    tdc_canonical = normalized_set([row["canonical_smiles"] for row in tdc_rows])
    base_raw = [None if value is None else str(value).strip() for value in table[args.smiles_column].to_pylist()]
    base_canonical = [canonical_smiles(value) for value in base_raw]
    raw_matches = [value in tdc_raw or value in tdc_canonical if value else False for value in base_raw]
    canonical_matches = [
        value in tdc_raw or value in tdc_canonical if value else False
        for value in base_canonical
    ]
    remove_mask = [raw or can for raw, can in zip(raw_matches, canonical_matches, strict=True)]
    filtered = table.filter(pc.invert(pa.array(remove_mask, type=pa.bool_())))

    pq.write_table(filtered, args.output_dir / "train.parquet", compression=args.compression)
    matched_raw_values = sorted({raw for raw, remove in zip(base_raw, remove_mask, strict=True) if remove and raw})
    matched_canonical_values = sorted(
        {can for can, remove in zip(base_canonical, remove_mask, strict=True) if remove and can}
    )
    (args.reference_dir / "matched_base_raw_smiles.txt").write_text("\n".join(matched_raw_values) + "\n")
    (args.reference_dir / "matched_base_canonical_smiles.txt").write_text(
        "\n".join(matched_canonical_values) + "\n"
    )

    metadata = {
        "schema_version": "starling_oral_bioavailability_numeric_v3",
        "created_at_utc": utc_now(),
        "input": args.input,
        "output_dir": str(args.output_dir),
        "output_file": "train.parquet",
        "tdc_task": args.tdc_task,
        "tdc_splits": list(args.tdc_splits),
        "tdc_reference_dir": str(args.reference_dir),
        "tdc_reference_metadata": reference_metadata,
        "smiles_column": args.smiles_column,
        "rows_input": table.num_rows,
        "rows_removed": int(sum(remove_mask)),
        "rows_written": filtered.num_rows,
        "added_or_replaced_columns": [NORMALIZED_COLUMN, KEY_COLUMN],
        "removal_policy": (
            "remove base rows when stripped raw base SMILES or RDKit canonical base SMILES "
            "matches stripped raw or RDKit canonical TDC Bioavailability_Ma train/valid/test SMILES"
        ),
        "base_smiles_stats": {
            "invalid_or_missing_canonical_smiles": sum(1 for value in base_canonical if value is None),
            "raw_match_rows": int(sum(raw_matches)),
            "canonical_match_rows": int(sum(canonical_matches)),
            "matched_unique_raw_smiles": len(matched_raw_values),
            "matched_unique_canonical_smiles": len(matched_canonical_values),
        },
        "columns": filtered.column_names,
        "parquet": {"compression": args.compression},
    }
    write_json(args.output_dir / "dataset_info.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tdc-dir", type=Path, default=DEFAULT_TDC_DIR)
    parser.add_argument("--tdc-task", default="Bioavailability_Ma")
    parser.add_argument("--tdc-splits", nargs="+", default=list(DEFAULT_TDC_SPLITS))
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-tdc-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    metadata = build(parse_args())
    print(
        json.dumps(
            {
                "output_dir": metadata["output_dir"],
                "rows_input": metadata["rows_input"],
                "rows_removed": metadata["rows_removed"],
                "rows_written": metadata["rows_written"],
                "tdc_unique_raw_smiles": metadata["tdc_reference_metadata"]["unique_raw_smiles"],
                "tdc_unique_canonical_smiles": metadata["tdc_reference_metadata"]["unique_canonical_smiles"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
