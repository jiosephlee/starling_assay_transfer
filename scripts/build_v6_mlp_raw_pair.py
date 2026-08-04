#!/usr/bin/env python3
"""Build the compact 20M-row V6 MLP dataset and shared held-out benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.pair_core import molecule_splits, record_key
from pipeline.v3_policy import file_sha256
from pipeline.v6_mlp import (compact_pair, condition_stats, iter_mlp_groups,
                             membership_hash, model_record)


def _load_records(path: Path, expected_records: int = 138_806) -> list[dict]:
    rows = pq.read_table(path).to_pylist()
    if expected_records is not None and len(rows) != expected_records:
        raise ValueError(f"expected {expected_records:,} eligible records, found {len(rows):,}")
    return rows


def _indices(rows: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    record_indices = {record_key(row): index for index, row in enumerate(sorted(rows, key=record_key))}
    molecules = sorted({str(row["canonical_smiles"]) for row in rows})
    return record_indices, {smiles: index for index, smiles in enumerate(molecules)}


def _write_records(output: Path, rows: list[dict], splits: dict[str, str],
                   indices: dict[str, int], molecule_indices: dict[str, int]) -> Path:
    ordered = sorted(rows, key=lambda row: indices[record_key(row)])
    values = [model_record(row, indices[record_key(row)], molecule_indices[row["canonical_smiles"]],
                           splits[row["canonical_smiles"]]) for row in ordered]
    path = output / "records.parquet"
    pq.write_table(pa.Table.from_pylist(values), path, compression="zstd")
    return path


def _pair_schema() -> pa.Schema:
    return pa.schema([pa.field("group_index", pa.uint32()), pa.field("member_index", pa.uint8()),
                      pa.field("query_record_index", pa.uint32()),
                      pa.field("retrieval_record_index", pa.uint32()),
                      pa.field("pair_id", pa.binary(32)), pa.field("retrieval_value", pa.float32()),
                      pa.field("retrieval_is_approximate", pa.bool_()), pa.field("distance", pa.float32()),
                      pa.field("target_z", pa.float32()), pa.field("target_a", pa.float32()),
                      pa.field("target_b", pa.float32())])


def _flush(writer: pq.ParquetWriter, rows: list[dict]) -> None:
    writer.write_table(pa.Table.from_pylist(rows, schema=_pair_schema()))
    rows.clear()


def _write_train(output: Path, rows: list[dict], indices: dict[str, int],
                 stats: dict[str, tuple[float, float]], groups: int, size: int) -> Path:
    path, buffer = output / "train.parquet", []
    writer = pq.ParquetWriter(path, _pair_schema(), compression="zstd")
    for group_index, query, candidates in iter_mlp_groups(rows, groups, size):
        buffer.extend(compact_pair(query, candidate, group_index, member, indices, stats)
                      for member, candidate in enumerate(candidates))
        if len(buffer) >= 100_000:
            _flush(writer, buffer)
        if (group_index + 1) % 10_000 == 0:
            print(f"built {group_index + 1:,}/{groups:,} MLP groups", flush=True)
    if buffer:
        _flush(writer, buffer)
    writer.close()
    return path


def _benchmark_rows(path: Path, source: dict[str, dict], indices: dict[str, int],
                    stats: dict[str, tuple[float, float]], ranking: bool) -> list[dict]:
    rows, output = pq.read_table(path).to_pylist(), []
    group_ids = {}
    for index, row in enumerate(rows):
        query, retrieval = source[row["query_record_id"]], source[row["retrieval_record_id"]]
        group_name = row.get("ranking_query_id", f"ordinary-{index:06d}")
        group_index = group_ids.setdefault(group_name, len(group_ids))
        member = row.get("ranking_member_index", 0)
        item = compact_pair(query, retrieval, group_index, member, indices, stats)
        item["pair_id"] = bytes.fromhex(row["pair_id"])
        if ranking:
            item["ranking_query_id"] = group_name
        else:
            item["binary_label"] = int(row["binary_label"])
        output.append(item)
    return output


def _write_benchmarks(output: Path, intern: Path, source: dict[str, dict],
                      indices: dict[str, int], stats: dict[str, tuple[float, float]]) -> dict[str, Path]:
    paths = {}
    for split in ("validation", "test", "validation_ranking", "test_ranking"):
        ranking = split.endswith("ranking")
        rows = _benchmark_rows(intern / split / "data.parquet", source, indices, stats, ranking)
        path = output / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        paths[split] = path
    return paths


def _benchmark_hashes(intern: Path) -> dict[str, str]:
    output = {}
    for split in ("validation", "test", "validation_ranking", "test_ranking"):
        rows = pq.read_table(intern / split / "data.parquet", columns=["pair_id"]).to_pylist()
        output[split] = membership_hash([row["pair_id"] for row in rows])
    return output


def build(source_path: Path, intern: Path, output: Path, groups: int, size: int,
          expected_records: int = 138_806,
          dataset_version: str = "assay_transfer_raw_pair_v6_mlp") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows = _load_records(source_path, expected_records)
    splits, (indices, molecule_indices) = molecule_splits(rows), _indices(rows)
    source = {record_key(row): row for row in rows}
    train = [row for row in rows if splits[row["canonical_smiles"]] == "train"]
    stats = condition_stats(train)
    paths = {"records": _write_records(output, rows, splits, indices, molecule_indices)}
    paths["train"] = _write_train(output, train, indices, stats, groups, size)
    paths.update(_write_benchmarks(output, intern, source, indices, stats))
    counts = {name: pq.ParquetFile(path).metadata.num_rows for name, path in paths.items()}
    manifest = {"version": dataset_version, "source": str(source_path),
                "intern_benchmark": str(intern), "train_groups": groups, "group_size": size,
                "rows": counts, "benchmark_membership_sha256": _benchmark_hashes(intern),
                "parquet_sha256": {name: file_sha256(path) for name, path in paths.items()},
                "normalization": {key: {"mean": mean, "std": std}
                                  for key, (mean, std) in stats.items()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/eligible/assay_transfer_soft_evidence_v4/records.parquet"))
    parser.add_argument("--intern", type=Path, default=Path("datasets/hf_parquet/assay_transfer_raw_pair_v6_intern"))
    parser.add_argument("--output", type=Path, default=Path("datasets/parquet/assay_transfer_raw_pair_v6_mlp"))
    parser.add_argument("--groups", type=int, default=500_000)
    parser.add_argument("--group-size", type=int, default=40)
    parser.add_argument("--expected-records", type=int, default=138_806)
    parser.add_argument("--dataset-version", default="assay_transfer_raw_pair_v6_mlp")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.intern, args.output, args.groups, args.group_size,
                           args.expected_records, args.dataset_version), indent=2))


if __name__ == "__main__":
    main()
