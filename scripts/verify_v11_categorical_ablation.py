#!/usr/bin/env python3
"""Verify one paired V11 continuous/categorical ablation release."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from pipeline.v11_contract import DEFAULT_REGISTRY
from pipeline.v3_policy import file_sha256
from scripts.build_v11_categorical_ablation import EVAL_SPLITS
from scripts.verify_v11_intern_raw_pair import _verify_manifest, verify


def _manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _pair_ids(path: Path, start: int = 0) -> Iterator[str]:
    position = 0
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=["pair_id"]):
        for value in batch.column(0).to_pylist():
            if position >= start:
                yield str(value)
            position += 1


def _digest(values: Iterator[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _verify_variants(continuous: dict, categorical: dict) -> None:
    if continuous.get("dataset_variant") != "continuous_only":
        raise AssertionError("continuous root has incorrect variant")
    if categorical.get("dataset_variant") != "with_categorical":
        raise AssertionError("categorical root has incorrect variant")
    if continuous["task_id"] != categorical["task_id"]:
        raise AssertionError("paired roots belong to different tasks")
    if set(continuous["train_components"]) != {"continuous"}:
        raise AssertionError("continuous-only component inventory is invalid")
    if set(categorical["train_components"]) != {"continuous", "categorical"}:
        raise AssertionError("with-categorical component inventory is invalid")
    provenance = ("eligible_source_sha256", "prompt_projection", "heldout_reservation")
    if any(continuous[field] != categorical[field] for field in provenance):
        raise AssertionError("paired variants do not share frozen provenance")


def _verify_shared_eval(continuous_root: Path, categorical_root: Path) -> None:
    for split in EVAL_SPLITS:
        first = continuous_root / split / "data.parquet"
        second = categorical_root / split / "data.parquet"
        if file_sha256(first) != file_sha256(second):
            raise AssertionError(f"{split}: variants are not byte-identical")


def _verify_additive_train(
    continuous_root: Path, categorical_root: Path,
    continuous: dict, categorical: dict,
) -> None:
    first_path = continuous_root / "train" / "data.parquet"
    combined_path = categorical_root / "train" / "data.parquet"
    continuous_rows = continuous["rows"]["train"]
    category = categorical["train_components"]["categorical"]
    expected_total = continuous_rows + category["achieved_pairs"]
    if categorical["rows"]["train"] != expected_total:
        raise AssertionError("with-categorical train is not additive")
    first_table = pq.read_table(first_path)
    combined_prefix = pq.read_table(combined_path).slice(0, continuous_rows)
    if not first_table.equals(combined_prefix):
        raise AssertionError("continuous train rows changed between variants")
    first_digest = _digest(_pair_ids(first_path))
    prefix_digest = _digest(iter(_take(_pair_ids(combined_path), continuous_rows)))
    suffix_digest = _digest(_pair_ids(combined_path, continuous_rows))
    if first_digest != prefix_digest:
        raise AssertionError("continuous train component changed between variants")
    if first_digest != categorical["train_components"]["continuous"]["pair_id_sha256"]:
        raise AssertionError("continuous component digest mismatch")
    if suffix_digest != category["pair_id_sha256"]:
        raise AssertionError("categorical component digest mismatch")


def _take(values: Iterator[str], count: int) -> Iterator[str]:
    for index, value in enumerate(values):
        if index >= count:
            break
        yield value


def _degree_maxima(path: Path, start: int, count: int) -> tuple[int, int]:
    query: Counter = Counter()
    retrieval: Counter = Counter()
    position, end = 0, start + count
    columns = ["query_record_id", "retrieval_record_id"]
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=columns):
        batch_start, position = position, position + batch.num_rows
        if position <= start:
            continue
        if batch_start >= end:
            break
        offset = max(0, start - batch_start)
        length = min(position, end) - (batch_start + offset)
        selected = batch.slice(offset, length)
        query.update(str(value) for value in selected.column(0).to_pylist())
        retrieval.update(str(value) for value in selected.column(1).to_pylist())
    return max(query.values(), default=0), max(retrieval.values(), default=0)


def _component_ranges(manifest: dict) -> list[tuple[str, int, int]]:
    order = manifest["measurement_kind_policy"]["train_components"]
    if set(order) != set(manifest["train_components"]):
        raise AssertionError("train component order does not match component inventory")
    ranges, start = [], 0
    for name in order:
        count = int(manifest["train_components"][name]["achieved_pairs"])
        ranges.append((name, start, count))
        start += count
    if start != manifest["rows"]["train"]:
        raise AssertionError("train component ranges do not cover the train split")
    return ranges


def _verify_caps(root: Path, manifest: dict) -> None:
    path = root / "train" / "data.parquet"
    for name, start, count in _component_ranges(manifest):
        component = manifest["train_components"][name]
        if component["achieved_pairs"] > component["max_pairs"]:
            raise AssertionError(f"{name}: component exceeded its global cap")
        if component["query_degree_cap"] != 6 or component["retrieval_degree_cap"] != 6:
            raise AssertionError(f"{name}: component degree caps drifted")
        query_max, retrieval_max = _degree_maxima(path, start, count)
        if query_max != component.get("observed_max_query_degree"):
            raise AssertionError(f"{name}: observed query degree does not match manifest")
        if retrieval_max != component.get("observed_max_retrieval_degree"):
            raise AssertionError(f"{name}: observed retrieval degree does not match manifest")
        if query_max > component["query_degree_cap"]:
            raise AssertionError(f"{name}: rows exceed query degree cap")
        if retrieval_max > component["retrieval_degree_cap"]:
            raise AssertionError(f"{name}: rows exceed retrieval degree cap")


def verify_pair(
    continuous_root: Path, categorical_root: Path,
    source: Path, registry: Path = DEFAULT_REGISTRY,
) -> None:
    continuous = _manifest(continuous_root)
    categorical = _manifest(categorical_root)
    _verify_variants(continuous, categorical)
    verify(categorical_root, source, registry)
    _verify_manifest(continuous_root, continuous)
    _verify_shared_eval(continuous_root, categorical_root)
    _verify_additive_train(continuous_root, categorical_root, continuous, categorical)
    _verify_caps(continuous_root, continuous)
    _verify_caps(categorical_root, categorical)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-root", type=Path, required=True)
    parser.add_argument("--categorical-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    verify_pair(args.continuous_root, args.categorical_root, args.source, args.registry)
    print("V11 categorical-ablation pair verification passed")


if __name__ == "__main__":
    main()
