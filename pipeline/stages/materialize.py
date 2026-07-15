#!/usr/bin/env python3
"""Materialize stage: pairs + base -> final build parquet (model input contract).

Inlines the retrieved record's metadata ``Z_A`` (assay context, species, normalized
conditions, source) onto each pair by joining ``retrieved_row_index`` back into the base
table, producing the self-contained pair artifact of
``docs/assay_transfer_design.md`` sections 10, 14. Each materialized row carries exactly
what the model sees plus the label and its provenance:

- query structure ``x_B`` (``query_smiles``);
- retrieved structure ``x_A`` (``retrieved_smiles``), value ``y_A`` (``retrieved_value``),
  and metadata ``Z_A`` (``retrieved_*`` columns);
- the shared setting ``K`` (``k_profile`` + ``setting_key``); and
- ``transfer_label`` plus the query-aggregate diagnostics (``query_value_mean``/``n``/
  ``std``) which are label provenance only, never model inputs (section 10).

The base list must be passed in the same order as the pairs stage so ``retrieved_row_index``
aligns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Base columns re-expressed on the pair already (not duplicated as retrieved_* metadata).
_ALREADY_ON_PAIR = {"smiles", "property_value", "property_value_native", "canonical_endpoint_key"}


def _load_base_rows(base_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in base_paths:
        p = path / "base.parquet" if path.is_dir() else path
        rows.extend(pq.read_table(p).to_pylist())
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_paths = [Path(p) for p in args.base]
    base_rows = _load_base_rows(base_paths)
    pairs_table = pq.read_table(args.pairs)
    pairs = pairs_table.to_pylist()
    if not pairs:
        raise RuntimeError("no pairs to materialize")

    # Retrieved metadata columns Z_A: base columns not already carried on the pair.
    base_cols = [c for c in base_rows[0].keys() if c not in _ALREADY_ON_PAIR] if base_rows else []

    out_rows: list[dict[str, Any]] = []
    for p in pairs:
        idx = int(p["retrieved_row_index"])
        base = base_rows[idx]
        row = dict(p)
        for c in base_cols:
            row[f"retrieved_{c}"] = base.get(c)
        out_rows.append(row)

    columns = list(out_rows[0].keys())
    arrays = {}
    for c in columns:
        vals = [r.get(c) for r in out_rows]
        first = next((v for v in vals if v is not None), None)
        if isinstance(first, bool):
            arrays[c] = pa.array(vals, type=pa.bool_())
        elif isinstance(first, int) and not isinstance(first, bool):
            arrays[c] = pa.array(vals, type=pa.int64())
        elif isinstance(first, float):
            arrays[c] = pa.array(vals, type=pa.float64())
        else:
            arrays[c] = pa.array([None if v is None else str(v) for v in vals], type=pa.string())
    table = pa.table(arrays)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output_dir / "dataset.parquet", compression="zstd")

    manifest = {
        "stage": "materialize",
        "pairs_input": str(args.pairs),
        "base_inputs": [str(p) for p in base_paths],
        "rows": len(out_rows),
        "retrieved_metadata_columns": [f"retrieved_{c}" for c in base_cols],
        "columns": table.column_names,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="pairs.parquet from the pairs stage.")
    parser.add_argument("--base", nargs="+", required=True, help="Base dirs/parquets (same order as pairs).")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
