#!/usr/bin/env python3
"""Prepare a raw Starling assay CSV into a base-ready Parquet for one task.

This is the new first phase of the assay-transfer pipeline (before the base builder).
Driven entirely by a :class:`TaskSpec`, it:

1. loads a canonical extraction input from ``datasets/_source/<domain>/``,
2. resolves the internal ``global_identifier`` ("SMILES:<int>") to real SMILES via
   ``final_smiles_mapping_v2.parquet``,
3. filters to the task's comparable-endpoint family,
4. parses the messy value string into a numeric ``property_value`` (native units in
   ``property_value_raw``, comparison-scale in ``property_value``),
5. maps the raw columns onto the task's canonical metadata columns,

and writes ``<output_dir>/base.parquet`` plus a ``prepare_report.json`` with resolution
and parse yields. The base builder then consumes this (with ``--skip-tdc-exclusion``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.taskspecs import available, get_task  # noqa: E402

DEFAULT_ID_SMILES_MAP = "datasets/_source/final_smiles_mapping_v2.parquet"


def load_id_smiles_map(map_path: Path, needed_ids: set[str]) -> dict[str, str]:
    """Load only the ``global_identifier -> smiles`` rows we need."""
    table = pq.read_table(map_path, columns=["global_identifier", "smiles"])
    mask = pc.is_in(table["global_identifier"], value_set=pa.array(sorted(needed_ids)))
    table = table.filter(mask)
    ids = table["global_identifier"].to_pylist()
    smis = table["smiles"].to_pylist()
    mapping: dict[str, str] = {}
    for gid, smi in zip(ids, smis):
        if gid is not None and smi is not None and str(smi).strip():
            mapping[gid] = str(smi).strip()
    return mapping


def _prepare_record(
    row: dict[str, Any], spec: Any, id_map: dict[str, str]
) -> tuple[dict[str, Any] | None, str | None]:
    gid = row.get(spec.raw_id_column)
    if not gid:
        return None, "no_id"
    smiles = id_map.get(gid)
    if smiles is None:
        return None, "unresolved_id"
    if spec.property_filter_column is not None:
        if row.get(spec.property_filter_column) not in spec.property_filter_values:
            return None, "filtered_out_property"
    units = row.get(spec.raw_units_column) if spec.raw_units_column else None
    native = spec.parse_value(row.get(spec.raw_value_column), units)
    if native is None:
        return None, "unparseable_value"
    if not spec.in_range(native):
        return None, "out_of_range_value"
    transformed = spec.transform_value(native)
    if transformed is None:
        return None, "untransformable_value"
    record: dict[str, Any] = {
        "smiles": smiles,
        "property_value": float(transformed),
        "property_value_raw": float(native),
        "pmid": row.get("pmid"),
        "extraction_id": row.get("extraction_id"),
    }
    for canonical, raw_col in spec.raw_metadata_map.items():
        value = row.get(raw_col)
        record[canonical] = value if value is not None and str(value).strip() else None
    record.update(spec.derive(record))
    return record, None


def _prepare_records(
    rows: list[dict[str, Any]], spec: Any, id_map: dict[str, str]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for row in rows:
        record, rejection = _prepare_record(row, spec, id_map)
        if rejection is not None:
            stats[rejection] += 1
            continue
        if record is None:
            raise AssertionError("accepted row did not produce a record")
        records.append(record)
        stats["records_kept"] += 1
    return records, stats


def _records_table(records: list[dict[str, Any]], spec: Any) -> pa.Table:
    base = ["smiles", "property_value", "property_value_raw", "pmid", "extraction_id"]
    columns = list(
        dict.fromkeys(base + list(spec.raw_metadata_map) + list(spec.derived_columns))
    )
    arrays: dict[str, pa.Array] = {}
    for column in columns:
        values = [record.get(column) for record in records]
        if column in ("property_value", "property_value_raw"):
            arrays[column] = pa.array(values, type=pa.float64())
        else:
            strings = [None if value is None else str(value) for value in values]
            arrays[column] = pa.array(strings, type=pa.string())
    return pa.table(arrays)


def _build_report(
    args: argparse.Namespace,
    spec: Any,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    id_map: dict[str, str],
    stats: Counter[str],
    table: pa.Table,
) -> dict[str, Any]:
    needed_ids = {row.get(spec.raw_id_column) for row in rows if row.get(spec.raw_id_column)}
    return {
        "task": spec.name,
        "input": str(args.input),
        "id_smiles_map": str(args.id_smiles_map),
        "output_dir": str(args.output_dir),
        "rows_total": len(rows),
        "unique_ids_seen": len(needed_ids),
        "ids_resolved": len(id_map),
        "records_kept": stats.get("records_kept", 0),
        "unique_molecules": len({record["smiles"] for record in records}),
        "drop_stats": dict(sorted(stats.items())),
        "columns": table.column_names,
        "value_column": "property_value",
        "value_raw_column": "property_value_raw",
        "metadata_columns": list(spec.raw_metadata_map.keys()),
        "derived_columns": list(spec.derived_columns.keys()),
        "property_filter": {
            "column": spec.property_filter_column,
            "values": list(spec.property_filter_values),
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    spec = get_task(args.task)
    if not spec.raw_value_column:
        raise ValueError(f"task {spec.name!r} does not define a raw_value_column (not preparable)")
    rows = list(csv.DictReader(args.input.open()))
    needed = {row.get(spec.raw_id_column) for row in rows if row.get(spec.raw_id_column)}
    id_map = load_id_smiles_map(args.id_smiles_map, needed)
    records, stats = _prepare_records(rows, spec, id_map)
    if not records:
        raise RuntimeError("no records survived preparation; check filters / value parsing")
    table = _records_table(records, spec)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output_dir / "base.parquet", compression=args.compression)
    report = _build_report(args, spec, rows, records, id_map, stats, table)
    (args.output_dir / "prepare_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=available())
    parser.add_argument("--input", type=Path, required=True, help="Raw assay CSV.")
    parser.add_argument("--id-smiles-map", type=Path, default=Path(DEFAULT_ID_SMILES_MAP))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
