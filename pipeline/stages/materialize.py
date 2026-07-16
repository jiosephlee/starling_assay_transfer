#!/usr/bin/env python3
"""Materialize stage: selected candidates + base -> final v2 transfer-example artifact.

Inlines the retrieved record's metadata ``Z_A`` (assay context, species, normalized
conditions, source) onto each selected candidate by joining ``retrieved_row_index`` back
into the base table, producing the self-contained transfer-example artifact of
``docs/assay_transfer_design_v2.md`` section 16.2. Each materialized row carries exactly
what the model sees plus the target and its provenance:

- query structure ``x_B`` (``query_smiles``) — structure only (section 3);
- retrieved structure ``x_A`` (``retrieved_smiles``), value ``y_A`` (``retrieved_value``),
  and metadata ``Z_A`` (``retrieved_*`` columns);
- the shared setting ``K`` (``k_profile`` + ``setting_key``);
- the **continuous primary target** (``continuous_target``) and the **nullable** hard
  binary label (``binary_label``); and
- evidence counts/fractions, assay concept, Tanimoto bucket, and ``candidate_id`` (label
  provenance / sampling strata, never model inputs).

Accepts one or more selected split parquets (train/validation/test); the ``split`` column
is preserved for render_hf. The base list must be passed in the same order as the pairs
stage so ``retrieved_row_index`` aligns.
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


# v2 transfer-example fields that must survive to the materialized artifact (section 16.2).
_REQUIRED_V2_FIELDS = (
    "candidate_id", "query_smiles", "retrieved_smiles", "retrieved_value",
    "continuous_target", "binary_label", "n_records", "assay_concept", "tanimoto_bucket",
)


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_paths = [Path(p) for p in args.base]
    base_rows = _load_base_rows(base_paths)
    pairs: list[dict[str, Any]] = []
    for p in args.pairs:
        p = Path(p)
        f = p / "selected.parquet" if p.is_dir() else p
        pairs.extend(pq.read_table(f).to_pylist())
    if not pairs:
        raise RuntimeError("no selected candidates to materialize")
    missing = [c for c in _REQUIRED_V2_FIELDS if c not in pairs[0]]
    if missing:
        raise ValueError(f"selected candidates missing v2 fields: {missing}")

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

    from collections import Counter

    hard = sum(1 for r in out_rows if r.get("binary_label") is not None)
    manifest = {
        "stage": "materialize",
        "pairs_inputs": [str(p) for p in args.pairs],
        "base_inputs": [str(p) for p in base_paths],
        "rows": len(out_rows),
        "rows_by_split": dict(Counter(r.get("split") for r in out_rows)),
        "hard_binary_rows": hard,
        "hard_binary_coverage": round(hard / len(out_rows), 4),
        "retrieved_metadata_columns": [f"retrieved_{c}" for c in base_cols],
        "columns": table.column_names,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", required=True,
                        help="Selected split parquets (selected.parquet dirs/files).")
    parser.add_argument("--base", nargs="+", required=True, help="Base dirs/parquets (same order as pairs).")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
