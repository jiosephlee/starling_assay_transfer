#!/usr/bin/env python3
"""Tokenize rendered HF transfer Parquets for TRL causal-LM SFT."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_transfer import SPLITS, json_default, utc_now, write_json  # noqa: E402


DEFAULT_INPUT_DIR = Path("datasets/pairs_split_hf/generic_transfer_hf_parquet")
DEFAULT_OUTPUT_DIR = Path("datasets/pairs_split_hf/tokenized/generic_transfer/qwen3_8b")
DEFAULT_TOKENIZER = "Qwen/Qwen3-8B"
SCHEMA_VERSION = "generic_transfer_hf_causal_lm_tokenized_v1"


class LengthStats:
    def __init__(self) -> None:
        self.n = 0
        self.total = 0
        self.minimum: int | None = None
        self.maximum: int | None = None

    def update_many(self, values: list[int]) -> None:
        if not values:
            return
        self.n += len(values)
        self.total += sum(values)
        batch_min = min(values)
        batch_max = max(values)
        self.minimum = batch_min if self.minimum is None else min(self.minimum, batch_min)
        self.maximum = batch_max if self.maximum is None else max(self.maximum, batch_max)

    def to_json(self) -> dict[str, Any]:
        return {
            "count": self.n,
            "min": self.minimum,
            "max": self.maximum,
            "mean": None if self.n == 0 else self.total / self.n,
        }


def tokenized_schema(metadata_type: pa.DataType) -> pa.Schema:
    return pa.schema(
        [
            ("input_ids", pa.list_(pa.int32())),
            ("attention_mask", pa.list_(pa.int8())),
            ("labels", pa.list_(pa.int32())),
            ("completion", pa.string()),
            ("metadata", metadata_type),
            ("prompt_length", pa.int32()),
            ("completion_length", pa.int32()),
            ("sequence_length", pa.int32()),
        ]
    )


def output_path(output_dir: Path, split: str) -> Path:
    return output_dir / f"{split}.parquet"


def tokenize_split(
    *,
    input_file: Path,
    output_file: Path,
    tokenizer: Any,
    split: str,
    batch_size: int,
    row_group_size: int,
    compression: str,
    max_rows: int | None,
) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(input_file)
    metadata_type = parquet_file.schema_arrow.field("metadata").type
    writer = pq.ParquetWriter(
        output_file,
        tokenized_schema(metadata_type),
        compression=compression,
    )
    rows = 0
    completion_counts: Counter[str] = Counter()
    prompt_lengths = LengthStats()
    completion_lengths = LengthStats()
    sequence_lengths = LengthStats()
    buffer: list[dict[str, Any]] = []
    eos_id = int(tokenizer.eos_token_id)
    try:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["prompt", "completion", "metadata"],
        ):
            if max_rows is not None and rows >= max_rows:
                break
            py_rows = batch.to_pylist()
            if max_rows is not None:
                py_rows = py_rows[: max_rows - rows]
            prompts = [str(row["prompt"]) for row in py_rows]
            completions = [str(row["completion"]) for row in py_rows]
            prompt_encoded = tokenizer(prompts, add_special_tokens=False)["input_ids"]
            completion_encoded = tokenizer(completions, add_special_tokens=False)["input_ids"]
            batch_prompt_lengths: list[int] = []
            batch_completion_lengths: list[int] = []
            batch_sequence_lengths: list[int] = []
            for row, prompt_ids_raw, completion_ids_raw in zip(
                py_rows,
                prompt_encoded,
                completion_encoded,
                strict=True,
            ):
                prompt_ids = [int(token_id) for token_id in prompt_ids_raw]
                completion_ids = [int(token_id) for token_id in completion_ids_raw]
                input_ids = prompt_ids + completion_ids + [eos_id]
                labels = [-100] * len(prompt_ids) + completion_ids + [eos_id]
                attention_mask = [1] * len(input_ids)
                record = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "completion": str(row["completion"]),
                    "metadata": row["metadata"],
                    "prompt_length": len(prompt_ids),
                    "completion_length": len(completion_ids) + 1,
                    "sequence_length": len(input_ids),
                }
                buffer.append(record)
                completion_counts[str(row["completion"])] += 1
                batch_prompt_lengths.append(record["prompt_length"])
                batch_completion_lengths.append(record["completion_length"])
                batch_sequence_lengths.append(record["sequence_length"])
                rows += 1
                if len(buffer) >= row_group_size:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=writer.schema))
                    buffer.clear()
            prompt_lengths.update_many(batch_prompt_lengths)
            completion_lengths.update_many(batch_completion_lengths)
            sequence_lengths.update_many(batch_sequence_lengths)
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=writer.schema))
            buffer.clear()
    finally:
        writer.close()
    return {
        "rows": rows,
        "completion_counts": dict(sorted(completion_counts.items())),
        "prompt_lengths": prompt_lengths.to_json(),
        "completion_lengths": completion_lengths.to_json(),
        "sequence_lengths": sequence_lengths.to_json(),
    }


def validate_outputs(output_dir: Path, expected_counts: dict[str, int]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for split, expected in expected_counts.items():
        parquet_file = pq.ParquetFile(output_path(output_dir, split))
        rows = parquet_file.metadata.num_rows
        if rows != expected:
            raise AssertionError(f"{split}: expected {expected} rows, found {rows}")
        if rows == 0:
            stats[split] = {"rows": rows}
            continue
        first = next(
            parquet_file.iter_batches(
                batch_size=1,
                columns=[
                    "input_ids",
                    "attention_mask",
                    "labels",
                    "prompt_length",
                    "completion_length",
                    "sequence_length",
                ],
            )
        ).to_pylist()[0]
        if len(first["input_ids"]) != first["sequence_length"]:
            raise AssertionError(f"{split}: bad input_ids length")
        if len(first["attention_mask"]) != first["sequence_length"]:
            raise AssertionError(f"{split}: bad attention_mask length")
        if len(first["labels"]) != first["sequence_length"]:
            raise AssertionError(f"{split}: bad labels length")
        if any(label != -100 for label in first["labels"][: first["prompt_length"]]):
            raise AssertionError(f"{split}: prompt labels are not masked")
        if all(label == -100 for label in first["labels"][first["prompt_length"] :]):
            raise AssertionError(f"{split}: completion labels are all masked")
        stats[split] = {
            "rows": rows,
            "num_row_groups": parquet_file.metadata.num_row_groups,
            "first_sequence_length": first["sequence_length"],
        }
    return stats


def prepare_output(output_dir: Path, splits: list[str], overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [output_path(output_dir, split) for split in splits]
    expected.append(output_dir / "tokenized_dataset_info.json")
    present = [path for path in expected if path.exists()]
    if present and not overwrite:
        formatted = "\n".join(str(path) for path in present)
        raise FileExistsError(f"output files exist; pass --overwrite:\n{formatted}")
    for path in present:
        path.unlink()


def build(args: argparse.Namespace) -> dict[str, Any]:
    for split in args.splits:
        path = args.input_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
    prepare_output(args.output_dir, args.splits, args.overwrite)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError(f"{args.tokenizer} has no eos_token_id")

    split_stats: dict[str, Any] = {}
    for split in args.splits:
        split_stats[split] = tokenize_split(
            input_file=args.input_dir / f"{split}.parquet",
            output_file=output_path(args.output_dir, split),
            tokenizer=tokenizer,
            split=split,
            batch_size=args.batch_size,
            row_group_size=args.row_group_size,
            compression=args.compression,
            max_rows=args.max_rows_per_split,
        )
    expected_counts = {split: int(split_stats[split]["rows"]) for split in args.splits}
    validation = validate_outputs(args.output_dir, expected_counts)
    source_info_path = args.input_dir / "dataset_info.json"
    source_info = json.loads(source_info_path.read_text()) if source_info_path.exists() else {}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_dataset_dir": str(args.input_dir),
        "source_dataset_info": {
            "schema_version": source_info.get("schema_version"),
            "source_split_dir": source_info.get("source_split_dir"),
            "completion_mapping": source_info.get("completion_mapping"),
            "row_schema": source_info.get("row_schema"),
        },
        "output_dir": str(args.output_dir),
        "tokenizer": {
            "requested": args.tokenizer,
            "name_or_path": tokenizer.name_or_path,
            "class": tokenizer.__class__.__name__,
            "vocab_size": tokenizer.vocab_size,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "tokenization": {
            "format": "causal_lm_sft",
            "text_construction": "prompt + completion + eos_token",
            "label_policy": "prompt token labels are -100; completion and eos token labels are supervised",
            "add_special_tokens": False,
        },
        "splits": list(args.splits),
        "output_format": "parquet",
        "compression": args.compression,
        "row_group_size": args.row_group_size,
        "batch_size": args.batch_size,
        "max_rows_per_split": args.max_rows_per_split,
        "columns": [
            "input_ids",
            "attention_mask",
            "labels",
            "completion",
            "metadata",
            "prompt_length",
            "completion_length",
            "sequence_length",
        ],
        "split_stats": split_stats,
        "validation": validation,
    }
    write_json(args.output_dir / "tokenized_dataset_info.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--row-group-size", type=int, default=50_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.row_group_size < 1:
        parser.error("batch and row-group sizes must be positive")
    if args.max_rows_per_split is not None and args.max_rows_per_split < 1:
        parser.error("--max-rows-per-split must be positive")
    return args


def main() -> None:
    metadata = build(parse_args())
    print(
        json.dumps(
            {
                "output_dir": metadata["output_dir"],
                "tokenizer": metadata["tokenizer"],
                "split_stats": {
                    split: {"rows": stats["rows"], "sequence_lengths": stats["sequence_lengths"]}
                    for split, stats in metadata["split_stats"].items()
                },
                "validation": metadata["validation"],
            },
            indent=2,
            sort_keys=True,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
