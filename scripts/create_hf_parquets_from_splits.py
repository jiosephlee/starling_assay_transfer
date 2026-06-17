#!/usr/bin/env python3
"""Render generic transfer split JSONL files into HF-style Parquet datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Environment, Template

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_transfer import SPLITS, read_jsonl_gz, utc_now, write_json  # noqa: E402


DEFAULT_SPLIT_DIR = Path("datasets/pairs_split/generic_transfer_pair_splits")
DEFAULT_OUTPUT_DIR = Path("datasets/pairs_split_hf/generic_transfer_hf_parquet")
DEFAULT_TEMPLATE = Path("templates/generic_transfer_classification.jinja")
TRANSFER_COMPLETIONS = {"transfer": "A", "not_transfer": "B"}
HF_SCHEMA_VERSION = "generic_transfer_hf_parquet_v1"


def compile_template(path: Path) -> Template:
    env = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
    return env.from_string(path.read_text())


def render_template(template: Template, values: dict[str, Any]) -> str:
    return template.render(**values).strip() + "\n"


def row_label(row: dict[str, Any]) -> str:
    label = row.get("transfer_label")
    if label not in TRANSFER_COMPLETIONS:
        raise ValueError(f"invalid transfer label for {row.get('pair_id')}: {label!r}")
    return str(label)


def build_values(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "molecule_a": {
            "canonical_smiles": row["molecule_a"]["canonical_smiles"],
            "metadata": row["molecule_a"].get("metadata") or {},
        },
        "molecule_b": {
            "canonical_smiles": row["molecule_b"]["canonical_smiles"],
            "metadata": row["molecule_b"].get("metadata") or {},
        },
        "group_id": row.get("group_id"),
    }


def metadata_for_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(row["pair_id"]),
        "pair_id": str(row["pair_id"]),
        "split": row.get("split"),
        "split_version": row.get("split_version"),
        "eval_subset": row.get("eval_subset"),
        "group_id": row.get("group_id"),
        "record_id_a": str(row.get("record_id_a")),
        "record_id_b": str(row.get("record_id_b")),
        "canonical_smiles_a": row["molecule_a"]["canonical_smiles"],
        "canonical_smiles_b": row["molecule_b"]["canonical_smiles"],
        "transfer_label": row.get("transfer_label"),
        "weighted_tanimoto": (
            None if row.get("weighted_tanimoto") is None else float(row["weighted_tanimoto"])
        ),
        "value_difference": (
            None if row.get("value_difference") is None else float(row["value_difference"])
        ),
        "metadata_a_json": json.dumps(row["molecule_a"].get("metadata") or {}, sort_keys=True),
        "metadata_b_json": json.dumps(row["molecule_b"].get("metadata") or {}, sort_keys=True),
        "tool_version": HF_SCHEMA_VERSION,
    }


def hf_row(row: dict[str, Any], template: Template) -> dict[str, Any]:
    label = row_label(row)
    return {
        "prompt": render_template(template, build_values(row)),
        "completion": TRANSFER_COMPLETIONS[label],
        "metadata": metadata_for_row(row),
    }


def validate_rendered_row(row: dict[str, Any]) -> None:
    prompt = row["prompt"]
    required = ["## Molecule A", "## Molecule B", "(A) transfer", "(B) not transfer", "Answer:"]
    missing = [text for text in required if text not in prompt]
    if missing:
        raise AssertionError(f"rendered prompt missing required text: {missing}")
    forbidden = [
        "transfer_label",
        "value_difference",
        "oral_bioavailability_value",
        "T_transfer",
        "T_not_transfer",
        "weighted_tanimoto",
    ]
    exposed = [text for text in forbidden if text in prompt]
    if exposed:
        raise AssertionError(f"rendered prompt exposes target/source leakage fields: {exposed}")
    if row["completion"] not in {"A", "B"}:
        raise AssertionError(f"invalid completion: {row['completion']}")


def schema() -> pa.Schema:
    metadata_type = pa.struct(
        [
            ("sample_id", pa.large_string()),
            ("pair_id", pa.large_string()),
            ("split", pa.string()),
            ("split_version", pa.string()),
            ("eval_subset", pa.string()),
            ("group_id", pa.large_string()),
            ("record_id_a", pa.large_string()),
            ("record_id_b", pa.large_string()),
            ("canonical_smiles_a", pa.large_string()),
            ("canonical_smiles_b", pa.large_string()),
            ("transfer_label", pa.string()),
            ("weighted_tanimoto", pa.float64()),
            ("value_difference", pa.float64()),
            ("metadata_a_json", pa.large_string()),
            ("metadata_b_json", pa.large_string()),
            ("tool_version", pa.string()),
        ]
    )
    return pa.schema(
        [
            ("prompt", pa.large_string()),
            ("completion", pa.string()),
            ("metadata", metadata_type),
        ]
    )


class ParquetRowWriter:
    def __init__(self, path: Path, *, row_group_size: int, compression: str) -> None:
        self.schema = schema()
        self.writer = pq.ParquetWriter(path, self.schema, compression=compression)
        self.row_group_size = row_group_size
        self.buffer: list[dict[str, Any]] = []

    def write(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.row_group_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()

    def __enter__(self) -> "ParquetRowWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def prepare_output(output_dir: Path, splits: Iterable[str], overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [output_dir / f"{split}.parquet" for split in splits]
    expected.append(output_dir / "dataset_info.json")
    present = [path for path in expected if path.exists()]
    if present and not overwrite:
        formatted = "\n".join(str(path) for path in present)
        raise FileExistsError(f"output files exist; pass --overwrite:\n{formatted}")
    for path in present:
        path.unlink()


def source_metadata(split_dir: Path) -> dict[str, Any]:
    path = split_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build(args: argparse.Namespace) -> dict[str, Any]:
    if not args.template.exists():
        raise FileNotFoundError(args.template)
    for split in args.splits:
        path = args.split_dir / f"{split}.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(path)
    prepare_output(args.output_dir, args.splits, args.overwrite)
    template = compile_template(args.template)
    rendered_counts: dict[str, int] = {}
    completion_counts: dict[str, dict[str, int]] = {}

    for split in args.splits:
        count = 0
        completions: Counter[str] = Counter()
        with ParquetRowWriter(
            args.output_dir / f"{split}.parquet",
            row_group_size=args.parquet_row_group_size,
            compression=args.parquet_compression,
        ) as writer:
            for source_row in read_jsonl_gz(
                args.split_dir / f"{split}.jsonl.gz",
                max_rows=args.max_rows_per_split,
            ):
                row = hf_row(source_row, template)
                if count == 0:
                    validate_rendered_row(row)
                writer.write(row)
                count += 1
                completions[row["completion"]] += 1
        rendered_counts[split] = count
        completion_counts[split] = dict(sorted(completions.items()))

    metadata = {
        "schema_version": HF_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_split_dir": str(args.split_dir),
        "source_split_metadata": source_metadata(args.split_dir),
        "output_dir": str(args.output_dir),
        "included_splits": list(args.splits),
        "template": str(args.template),
        "completion_mapping": dict(TRANSFER_COMPLETIONS),
        "row_schema": ["prompt", "completion", "metadata"],
        "rendered_counts": rendered_counts,
        "completion_counts": completion_counts,
        "max_rows_per_split": args.max_rows_per_split,
        "parquet": {
            "row_group_size": args.parquet_row_group_size,
            "compression": args.parquet_compression,
        },
        "prompt_leakage_policy": "validation rejects target label/value/similarity field names in rendered prompt",
    }
    write_json(args.output_dir / "dataset_info.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--parquet-row-group-size", type=int, default=50_000)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_rows_per_split is not None and args.max_rows_per_split < 1:
        parser.error("--max-rows-per-split must be positive")
    if args.parquet_row_group_size < 1:
        parser.error("--parquet-row-group-size must be positive")
    return args


def main() -> None:
    metadata = build(parse_args())
    print(
        json.dumps(
            {
                "output_dir": metadata["output_dir"],
                "rendered_counts": metadata["rendered_counts"],
                "completion_counts": metadata["completion_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
