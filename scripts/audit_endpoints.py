#!/usr/bin/env python3
"""Inventory raw endpoint aliases, units, and value-parse yields per source.

This implements design step 1 of ``docs/assay_transfer_design.md`` section 16: before
any endpoint assignment or metric policy can be applied, we must know the actual raw
vocabulary each source uses. For every source collection declared in
``configs/assay_transfer/v1/endpoints.yaml`` it reads the canonical parquet and reports,
per raw endpoint category:

- row count,
- numeric-parse yield of the raw value column, and
- the distinct raw unit strings (top-N) when the source carries a units column.

Output is a JSON report (and a human summary) used to design the raw-category alias map
and the unit-canonicalization tables. It never mutates any dataset.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_ENDPOINTS = _REPO_ROOT / "configs" / "assay_transfer" / "v1" / "endpoints.yaml"
DEFAULT_SOURCE_DIR = _REPO_ROOT / "datasets" / "starling_assays" / "datasets"

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _has_number(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return _NUMBER_RE.search(text.replace("−", "-")) is not None


def _column(table: Any, name: str | None) -> list[Any]:
    if name and name in table.schema.names:
        return table.column(name).to_pylist()
    return [None] * table.num_rows


def audit_source(source: str, spec: dict[str, Any], parquet: Path, top_units: int) -> dict[str, Any]:
    ep_col = spec.get("raw_endpoint_column")
    val_col = spec.get("raw_value_column")
    unit_col = spec.get("raw_unit_column")
    has_unit_col = bool(unit_col) and unit_col != "embedded_in_measured_value"

    table = pq.read_table(parquet)
    endpoints = _column(table, ep_col)
    values = _column(table, val_col)
    units = _column(table, unit_col) if has_unit_col else [None] * table.num_rows

    by_cat: dict[str, dict[str, Any]] = {}
    for ep, val, unit in zip(endpoints, values, units):
        cat = (ep or "").strip() or "∅"
        entry = by_cat.setdefault(
            cat, {"rows": 0, "numeric": 0, "nonempty_value": 0, "units": collections.Counter()}
        )
        entry["rows"] += 1
        if val is not None and str(val).strip():
            entry["nonempty_value"] += 1
            if _has_number(val):
                entry["numeric"] += 1
        if has_unit_col and unit is not None and str(unit).strip():
            entry["units"][str(unit).strip()] += 1

    categories = []
    for cat, entry in sorted(by_cat.items(), key=lambda kv: -kv[1]["rows"]):
        categories.append(
            {
                "raw_category": cat,
                "rows": entry["rows"],
                "numeric": entry["numeric"],
                "nonempty_value": entry["nonempty_value"],
                "distinct_units": len(entry["units"]),
                "top_units": entry["units"].most_common(top_units),
            }
        )
    return {
        "source": source,
        "parquet": str(parquet),
        "rows": table.num_rows,
        "endpoint_column": ep_col,
        "value_column": val_col,
        "unit_column": unit_col,
        "has_unit_column": has_unit_col,
        "distinct_categories": len(by_cat),
        "categories": categories,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    endpoints_cfg = yaml.safe_load(args.endpoints.read_text())
    sources = endpoints_cfg.get("source_collections", {})
    report: dict[str, Any] = {"endpoints_config": str(args.endpoints), "sources": {}}
    for source, spec in sources.items():
        parquet = args.source_dir / source / "extractions.parquet"
        if not parquet.exists():
            report["sources"][source] = {"error": f"missing parquet {parquet}"}
            continue
        report["sources"][source] = audit_source(source, spec, parquet, args.top_units)
    return report


def _print_summary(report: dict[str, Any]) -> None:
    for source, s in report["sources"].items():
        if "error" in s:
            print(f"## {source}: {s['error']}")
            continue
        print(f"\n## {source}  rows={s['rows']}  distinct_categories={s['distinct_categories']}")
        for c in s["categories"]:
            units = ", ".join(f"{u}({n})" for u, n in c["top_units"]) or ("n/a" if not s["has_unit_column"] else "∅")
            print(
                f"  {c['rows']:>6}  {c['raw_category']:<42} numeric {c['numeric']}/{c['nonempty_value']}"
                f"  units[{c['distinct_units']}]: {units}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--top-units", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None, help="Write JSON report here.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build(args)
    _print_summary(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=False, default=list))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
