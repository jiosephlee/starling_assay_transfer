#!/usr/bin/env python3
"""Prepare stage: raw source parquet -> canonical base records.

Source-generic replacement for the oral-specific base builders. Driven by the policy
engine (:mod:`pipeline.endpoints`, :mod:`pipeline.policy`) rather than hard-coded columns,
it turns one source's ``extractions.parquet`` into the normalized record schema of
``docs/assay_transfer_design.md`` section 3 / 14:

For every raw row it

1. assigns a canonical endpoint and canonicalizes the value onto the comparison scale
   (quarantining rows whose endpoint is disabled or whose value/unit is unresolved);
2. derives ``species_exact`` with the explicit-only resolver (join / label field only —
   the raw species/context columns are retained unchanged for model input); and
3. keeps the source's condition/context columns as retrieval metadata (support text and
   ids excluded).

Output: ``<output_dir>/base.parquet`` + ``prepare_report.json`` (assignment coverage,
quarantine reasons by endpoint, species coverage). Condition-key profiles beyond
``same_endpoint`` / ``same_species_same_endpoint`` (i.e. ``most_specific``) require
normalized condition fields that are not built yet and are added in a later increment.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.endpoints import load_endpoint_resolver  # noqa: E402
from pipeline.normalize.assay_species_normalization import (  # noqa: E402
    SPECIES_EXACT_COLUMN,
    resolve_species_record,
)
from pipeline.normalize.conditions import CONDITION_COLUMNS, normalize_conditions  # noqa: E402

DEFAULT_SOURCE_DIR = _REPO_ROOT / "datasets" / "starling_assays" / "datasets"

# Columns never carried into base as retrieval metadata (ids, provenance, leakage text,
# and the raw value/unit/endpoint columns, which are re-expressed canonically).
_GLOBAL_RESERVED = {
    "pmid",
    "extraction_id",
    "global_identifier",
    "confidence",
    "paragraph_idx",
    "support_text",
    "smiles",
}

# Fixed canonical columns emitted first, in this order. Normalized condition columns
# (for most_specific keying) follow species_exact.
_CANONICAL_COLUMNS = [
    "smiles",
    "source_id",
    "canonical_endpoint_id",
    "canonical_endpoint_key",
    "metric_type",
    "property_value",
    "property_value_native",
    SPECIES_EXACT_COLUMN,
    *CONDITION_COLUMNS,
    "pmid",
    "extraction_id",
]


def _metadata_columns(schema_names: list[str], reserved: set[str]) -> list[str]:
    return sorted(name for name in schema_names if name not in reserved)


def _prepare_records(
    source: str, rows: list[dict[str, Any]], resolver: Any, metadata_cols: list[str]
) -> tuple[list[dict[str, Any]], collections.Counter, collections.Counter]:
    records: list[dict[str, Any]] = []
    assigned: collections.Counter = collections.Counter()
    quarantine: collections.Counter = collections.Counter()
    for row in rows:
        smiles = row.get("smiles")
        if not smiles or not str(smiles).strip():
            quarantine["missing_smiles"] += 1
            continue
        assignment, reason = resolver.assign(source, row)
        if assignment is None:
            quarantine[reason] += 1
            continue
        species = resolve_species_record(row, source)
        record: dict[str, Any] = {
            "smiles": str(smiles).strip(),
            "source_id": source,
            "canonical_endpoint_id": assignment.canonical_endpoint_id,
            # same_endpoint firewall key; species / most-specific profiles extend this
            # at pair-construction time.
            "canonical_endpoint_key": assignment.canonical_endpoint_id,
            "metric_type": assignment.metric_type,
            "property_value": float(assignment.transformed_value),
            "property_value_native": float(assignment.native_value),
            SPECIES_EXACT_COLUMN: species,
            "pmid": row.get("pmid"),
            "extraction_id": row.get("extraction_id"),
        }
        record.update(normalize_conditions(row))
        for col in metadata_cols:
            value = row.get(col)
            record[col] = value if value is not None and str(value).strip() else None
        records.append(record)
        assigned[assignment.canonical_endpoint_id] += 1
    return records, assigned, quarantine


def _records_table(records: list[dict[str, Any]], metadata_cols: list[str]) -> pa.Table:
    columns = list(dict.fromkeys(_CANONICAL_COLUMNS + metadata_cols))
    arrays: dict[str, pa.Array] = {}
    for column in columns:
        values = [record.get(column) for record in records]
        if column in ("property_value", "property_value_native"):
            arrays[column] = pa.array(values, type=pa.float64())
        else:
            arrays[column] = pa.array(
                [None if v is None else str(v) for v in values], type=pa.string()
            )
    return pa.table(arrays)


def build(args: argparse.Namespace) -> dict[str, Any]:
    parquet = args.input or (args.source_dir / args.source / "extractions.parquet")
    if not parquet.exists():
        raise FileNotFoundError(f"source parquet not found: {parquet}")
    resolver = load_endpoint_resolver()
    table = pq.read_table(parquet)
    rows = table.to_pylist()
    reserved = set(_GLOBAL_RESERVED)
    cols = resolver.source_columns(args.source)
    for key in ("raw_endpoint_column", "raw_value_column", "raw_unit_column"):
        col = cols.get(key)
        if col and col != "embedded_in_measured_value":
            reserved.add(col)
    metadata_cols = _metadata_columns(table.schema.names, reserved)

    records, assigned, quarantine = _prepare_records(args.source, rows, resolver, metadata_cols)
    if not records:
        raise RuntimeError(f"no records survived preparation for source {args.source!r}")
    out_table = _records_table(records, metadata_cols)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, args.output_dir / "base.parquet", compression=args.compression)

    species_known = sum(1 for r in records if r[SPECIES_EXACT_COLUMN])
    condition_coverage = {
        col: round(sum(1 for r in records if r.get(col)) / len(records), 4)
        for col in CONDITION_COLUMNS
    }
    report = {
        "source": args.source,
        "input": str(parquet),
        "output_dir": str(args.output_dir),
        "endpoint_resolver_version": resolver.version,
        "enabled_endpoints": resolver.enabled_endpoints(),
        "rows_total": len(rows),
        "records_kept": len(records),
        "unique_molecules": len({r["smiles"] for r in records}),
        "assigned_by_endpoint": dict(assigned.most_common()),
        "quarantine_by_reason": dict(quarantine.most_common()),
        "species_exact_coverage": round(species_known / len(records), 4),
        "condition_field_coverage": condition_coverage,
        "metadata_columns": metadata_cols,
        "columns": out_table.column_names,
    }
    (args.output_dir / "prepare_report.json").write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Source collection id (e.g. q2, q3, q4).")
    parser.add_argument("--input", type=Path, default=None, help="Override source parquet path.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
