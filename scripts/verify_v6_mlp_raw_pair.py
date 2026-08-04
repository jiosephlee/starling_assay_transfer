#!/usr/bin/env python3
"""Verify the full compact V6 MLP artifact and shared benchmark membership."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pipeline.pair_core import (TARGET_SMOOTHING, TARGET_TEMPERATURE, molecule_splits,
                                oriented_pair, pair_id, record_key)
from pipeline.v6_mlp import assay_paragraph, membership_hash


def _sources(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = pq.read_table(path).to_pylist()
    return rows, {record_key(row): row for row in rows}


def _verify_records(dataset: Path, sources: list[dict], source_index: dict[str, dict],
                    expected_records: int, expected_split: dict) -> list[dict]:
    records = pq.read_table(dataset / "records.parquet").to_pylist()
    if len(records) != expected_records or any("value" in column for column in records[0]):
        raise AssertionError("model record table count or value-hiding contract failed")
    assignments = molecule_splits(sources)
    split_counts = {name: 0 for name in ("train", "validation", "test")}
    for index, row in enumerate(records):
        source = source_index[row["record_id"]]
        if row["record_index"] != index or row["assay_paragraph"] != assay_paragraph(source):
            raise AssertionError("record index or source assay paragraph mismatch")
        expected = assignments[source["canonical_smiles"]]
        if row["split"] != expected or row["canonical_smiles"] != source["canonical_smiles"]:
            raise AssertionError("record split or molecule provenance mismatch")
        split_counts[expected] += 1
    if split_counts != expected_split:
        raise AssertionError(f"record split mismatch: {split_counts} (expected {expected_split})")
    return records


def _numeric_sources(records: list[dict], source_index: dict[str, dict]) -> dict[str, np.ndarray]:
    source_rows = [source_index[row["record_id"]] for row in records]
    return {"value": np.asarray([row["comparison_value"] for row in source_rows]),
            "a": np.asarray([row["transfer_max"] for row in source_rows]),
            "b": np.asarray([row["not_transfer_min"] for row in source_rows]),
            "smiles": np.asarray([row["canonical_smiles"] for row in source_rows], dtype=object),
            "condition": np.asarray([row["canonical_endpoint_key"] for row in source_rows], dtype=object),
            "records": np.asarray(source_rows, dtype=object)}


def _verify_targets(batch, numeric: dict[str, np.ndarray]) -> None:
    query = np.asarray(batch.column("query_record_index")).astype(np.int64)
    retrieval = np.asarray(batch.column("retrieval_record_index")).astype(np.int64)
    if np.any(numeric["smiles"][query] == numeric["smiles"][retrieval]):
        raise AssertionError("same-molecule MLP pair")
    if np.any(numeric["condition"][query] != numeric["condition"][retrieval]):
        raise AssertionError("cross-condition MLP pair")
    distance = np.abs(numeric["value"][retrieval] - numeric["value"][query])
    a, b = numeric["a"][query], numeric["b"][query]
    z = math.log(9.0) * (a + b - 2.0 * distance) / (b - a)
    sigmoid = 1.0 / (1.0 + np.exp(np.clip(-z / TARGET_TEMPERATURE, -700, 700)))
    target = TARGET_SMOOTHING + (1.0 - 2.0 * TARGET_SMOOTHING) * sigmoid
    for name, expected in (("distance", distance), ("target_z", z), ("target_a", target)):
        actual = np.asarray(batch.column(name))
        if not np.allclose(actual, expected, rtol=2e-5, atol=2e-5):
            raise AssertionError(f"{name} reconstruction mismatch")


def _verify_train(dataset: Path, numeric: dict[str, np.ndarray], train_rows: int, group_size: int) -> None:
    path, rows_seen = dataset / "train.parquet", 0
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != train_rows:
        raise AssertionError(f"training artifact is not exactly {train_rows:,} rows")
    for batch in parquet.iter_batches(batch_size=131072):
        size = len(batch)
        group = np.asarray(batch.column("group_index"))
        member = np.asarray(batch.column("member_index"))
        expected = np.arange(rows_seen, rows_seen + size, dtype=np.int64)
        if np.any(group != expected // group_size) or np.any(member != expected % group_size):
            raise AssertionError(f"incomplete or interleaved {group_size}-candidate group")
        _verify_targets(batch, numeric)
        rows_seen += size
    # Distinctness via prefix-bucketed group-by. A flat count_distinct materializes the distinct
    # keys into one contiguous Arrow array, which overflows the 2 GiB single-array cap past ~67M
    # rows (32 B keys). Grouping by the first pair_id byte (uniform for sha256) keeps each bucket's
    # distinct set ~train_rows/256, so no single array is large; the buckets are disjoint by prefix
    # so summing their distinct counts is exact.
    pair_ids = pq.read_table(path, columns=["pair_id"])["pair_id"]
    if len(pair_ids) != train_rows:
        raise AssertionError("directed training pair reused")
    buckets = pa.table({"prefix": pc.binary_slice(pair_ids, 0, 1), "pid": pair_ids})
    grouped = buckets.group_by("prefix").aggregate([("pid", "count_distinct")])
    if pc.sum(grouped["pid_count_distinct"]).as_py() != train_rows:
        raise AssertionError("directed training pair reused")


def _verify_benchmarks(dataset: Path, intern: Path) -> None:
    expected_counts = {"validation": 2000, "test": 2000,
                       "validation_ranking": 20000, "test_ranking": 20000}
    for split, expected in expected_counts.items():
        original = pq.read_table(intern / split / "data.parquet", columns=["pair_id"]).to_pylist()
        compact = pq.read_table(dataset / f"{split}.parquet", columns=["pair_id"]).to_pylist()
        original_ids = [row["pair_id"] for row in original]
        compact_ids = [row["pair_id"].hex() for row in compact]
        if len(compact) != expected or compact_ids != original_ids:
            raise AssertionError(f"{split} is not byte-identical ordered benchmark membership")
        if membership_hash(compact_ids) != membership_hash(original_ids):
            raise AssertionError(f"{split} membership hash mismatch")


def _verify_pair_id_samples(dataset: Path, numeric: dict[str, np.ndarray]) -> None:
    table = pq.read_table(dataset / "train.parquet",
                          columns=["query_record_index", "retrieval_record_index", "pair_id"])
    for index in range(0, len(table), 199):
        query = numeric["records"][table["query_record_index"][index].as_py()]
        retrieval = numeric["records"][table["retrieval_record_index"][index].as_py()]
        if not oriented_pair(query, retrieval) or table["pair_id"][index].as_py().hex() != pair_id(query, retrieval):
            raise AssertionError("sampled direction or pair ID mismatch")


def verify(dataset: Path, source: Path, intern: Path, expected_records: int = 138_806,
           expected_split: dict | None = None, train_rows: int = 20_000_000,
           group_size: int = 40) -> None:
    sources, source_index = _sources(source)
    expected_split = expected_split or {"train": 116060, "validation": 11276, "test": 11470}
    records = _verify_records(dataset, sources, source_index, expected_records, expected_split)
    numeric = _numeric_sources(records, source_index)
    _verify_train(dataset, numeric, train_rows, group_size)
    _verify_pair_id_samples(dataset, numeric)
    _verify_benchmarks(dataset, intern)


def _parse_split(text: str) -> dict:
    out = {}
    for item in text.split(","):
        name, _, value = item.partition("=")
        out[name.strip()] = int(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/parquet/assay_transfer_raw_pair_v6_mlp"))
    parser.add_argument("--source", type=Path, default=Path("datasets/eligible/assay_transfer_soft_evidence_v4/records.parquet"))
    parser.add_argument("--intern", type=Path, default=Path("datasets/hf_parquet/assay_transfer_raw_pair_v6_intern"))
    parser.add_argument("--expected-records", type=int, default=138_806)
    parser.add_argument("--expected-split", type=_parse_split, default=None,
                        help="e.g. train=116041,validation=11276,test=11470")
    parser.add_argument("--train-rows", type=int, default=20_000_000)
    parser.add_argument("--group-size", type=int, default=40)
    args = parser.parse_args()
    verify(args.dataset, args.source, args.intern, args.expected_records, args.expected_split,
           args.train_rows, args.group_size)
    print("V6 MLP verification passed")


if __name__ == "__main__":
    main()
