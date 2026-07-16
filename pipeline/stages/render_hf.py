#!/usr/bin/env python3
"""Render-HF stage: materialized build -> templated Hugging Face train/val/test parquet.

Applies a per-build jinja template (``templates/*.jinja``) to the fixed materialized pair
schema, producing prompt/completion HF splits (``dataset_system_design.md`` section 9). The
data schema is fixed; only the prompt renderer is swappable. Since the no-source-value
variant is removed (``assay_transfer_design.md`` sections 2, 10), the retrieval value
``y_A`` is always rendered and there is no no-source-value template.

Splits are taken from each row's ``split`` column. Output: ``<output_dir>/<split>/data.parquet``
for train/validation/test, plus a ``dataset_info.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Template

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SPLITS = ("train", "validation", "test")
DEFAULT_TEMPLATE = _REPO_ROOT / "templates" / "assay_transfer_classification.jinja"

# transfer_label -> completion token.
_COMPLETION = {"transfer": " transfer", "not_transfer": " not_transfer"}


def build(args: argparse.Namespace) -> dict[str, Any]:
    template = Template(args.template.read_text(), keep_trailing_newline=True)
    rows = pq.read_table(args.dataset).to_pylist()
    if not rows:
        raise RuntimeError("no materialized rows to render")

    _LABEL_NAME = {1: "transfer", 0: "not_transfer"}
    by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    label_counts: dict[str, Counter] = {s: Counter() for s in SPLITS}
    for row in rows:
        split = row.get("split")
        if split not in by_split:
            continue
        # binary_label is nullable (1 / 0 / None); continuous_target is the primary signal.
        blabel = row.get("binary_label")
        name = _LABEL_NAME.get(blabel)
        by_split[split].append(
            {
                "prompt": template.render(row=row),
                "completion": _COMPLETION[name] if name else None,  # only hard-binary rows
                "continuous_target": row.get("continuous_target"),
                "binary_label": name,  # "transfer" / "not_transfer" / null
                "query_smiles": row.get("query_smiles"),
                "retrieved_smiles": row.get("retrieved_smiles"),
                "retrieved_value": row.get("retrieved_value"),
                "canonical_endpoint_id": row.get("canonical_endpoint_id"),
                "assay_concept": row.get("assay_concept"),
                "tanimoto_bucket": row.get("tanimoto_bucket"),
                "k_profile": row.get("k_profile"),
                "candidate_id": row.get("candidate_id"),
            }
        )
        label_counts[split][name or "null"] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _FLOAT = {"continuous_target", "retrieved_value"}
    written = {}
    for split, recs in by_split.items():
        if not recs:
            continue
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        cols = list(recs[0].keys())
        arrays = {}
        for c in cols:
            vals = [r[c] for r in recs]
            if c in _FLOAT:
                arrays[c] = pa.array(vals, type=pa.float64())
            else:
                arrays[c] = pa.array([None if v is None else str(v) for v in vals], type=pa.string())
        pq.write_table(pa.table(arrays), split_dir / "data.parquet", compression="zstd")
        written[split] = len(recs)

    info = {
        "stage": "render_hf",
        "dataset_input": str(args.dataset),
        "template": str(args.template),
        "primary_target": "continuous_target",
        "rows_per_split": written,
        "binary_label_counts": {s: dict(label_counts[s]) for s in SPLITS if written.get(s)},
        "completion_map": _COMPLETION,
    }
    (args.output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="dataset.parquet from materialize.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
