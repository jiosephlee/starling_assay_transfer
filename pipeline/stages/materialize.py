#!/usr/bin/env python3
"""Materialize selected v3 candidates with retrieval context and provenance."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

REQUIRED = {"candidate_id", "binary_label", "retrieval_record_id", "query_smiles",
            "retrieved_smiles", "assay_concept", "tanimoto_bucket"}


def _file(path: Path, default: str) -> Path:
    return path / default if path.is_dir() else path


def _load_selected(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(pq.read_table(_file(path, "selected.parquet")).to_pylist())
    return rows


def _base_indexes(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[tuple, str]]:
    index, query_choices = {}, {}
    for path in paths:
        for row in pq.read_table(_file(path, "records.parquet")).to_pylist():
            child_id = row["child_id"]
            if child_id in index:
                raise ValueError(f"duplicate retrieval child_id: {child_id}")
            index[child_id] = row
            key = (row["canonical_endpoint_key"], row["canonical_smiles"])
            choice = (child_id, row.get("source_smiles") or row["canonical_smiles"])
            if key not in query_choices or choice[0] < query_choices[key][0]:
                query_choices[key] = choice
    return index, {key: value[1] for key, value in query_choices.items()}


def _validate(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("no selected candidates to materialize")
    missing = REQUIRED - set(rows[0])
    if missing:
        raise ValueError(f"selected candidates missing fields: {sorted(missing)}")
    invalid = [row["candidate_id"] for row in rows if row.get("binary_label") not in (0, 1)]
    if invalid:
        raise ValueError(f"selected candidates contain null/invalid binary labels: {invalid[:3]}")


def _materialize(rows: list[dict[str, Any]], index: dict[str, dict[str, Any]],
                 query_original: dict[tuple, str]) -> list[dict[str, Any]]:
    output = []
    for candidate in rows:
        retrieval = index.get(candidate["retrieval_record_id"])
        if retrieval is None:
            raise KeyError(f"missing retrieval child {candidate['retrieval_record_id']}")
        row = dict(candidate)
        row["retrieved_original_smiles"] = retrieval.get("source_smiles") or row["retrieved_smiles"]
        row["retrieved_measurement_label"] = retrieval.get("measurement_label")
        query_key = (row["canonical_endpoint_key"], row["query_smiles"])
        row["query_original_smiles"] = query_original.get(query_key, row["query_smiles"])
        for key, value in retrieval.items():
            if key.startswith("context_") or key in {
                "parent_provenance_id", "record_id", "input_sha256", "scalar_is_approximate"
            }:
                row[f"retrieved_{key}"] = value
        output.append(row)
    return output


def build(args: argparse.Namespace) -> dict[str, Any]:
    selected = _load_selected([Path(path) for path in args.pairs])
    _validate(selected)
    index, query_original = _base_indexes([Path(path) for path in args.base])
    rows = _materialize(selected, index, query_original)
    table = pa.Table.from_pylist(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output_dir / "dataset.parquet", compression="zstd")
    manifest = {"stage": "materialize_v3", "rows": len(rows),
                "rows_by_split": dict(Counter(row["split"] for row in rows)),
                "binary_label_counts": dict(Counter(str(row["binary_label"]) for row in rows)),
                "retrieval_join_key": "child_id", "columns": table.column_names}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--base", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
