#!/usr/bin/env python3
"""Validate V6 raw-pair provenance, target math, and benchmark cardinalities."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.v6_intern import oriented_pair, render_prompt, target_for, unordered_pair_id


def _rows(root: Path, split: str) -> list[dict]:
    return pq.read_table(root / split / "data.parquet").to_pylist()


def _verify_row(row: dict, source: dict[str, dict]) -> None:
    query, retrieval = source[row["query_record_id"]], source[row["retrieval_record_id"]]
    if row["query_smiles"] == row["retrieval_smiles"]:
        raise AssertionError("self-molecule pair")
    if query["canonical_endpoint_key"] != retrieval["canonical_endpoint_key"]:
        raise AssertionError("cross-condition pair")
    if row["prompt"] != render_prompt(query, retrieval):
        raise AssertionError("prompt does not preserve independent source contexts")
    if not oriented_pair(query, retrieval):
        raise AssertionError("pair uses the rejected deterministic direction")
    if "value hidden" not in row["prompt"]:
        raise AssertionError("query block is not marked value-hidden")
    a, b = row["target_a"], row["target_b"]
    if not (.1 <= a <= .9 and math.isclose(a + b, 1.0, abs_tol=1e-6)):
        raise AssertionError("invalid stored A/B target")
    expected = target_for(query, retrieval)
    if any(not math.isclose(row[key], expected[key], abs_tol=1e-6) for key in expected):
        raise AssertionError("target reconstruction mismatch")


def _source_rows(path: Path) -> dict[str, dict]:
    rows = pq.read_table(path).to_pylist()
    return {str(row["child_id"]): row for row in rows}


def _record_packing(row: dict, chunks: dict, groups: dict) -> None:
    chunk_id = row["packed_chunk_index"]
    batch_id = row["optimizer_batch_index"]
    position = row["optimizer_batch_position"]
    state = chunks.setdefault(chunk_id, {"tokens": 0, "batch": batch_id, "position": position})
    if state["batch"] != batch_id or state["position"] != position:
        raise AssertionError("inconsistent packed-chunk assignment")
    state["tokens"] += row["templated_token_count"]
    group_id = row["listnet_group_index"]
    previous = groups.setdefault(group_id, chunk_id)
    if previous != chunk_id:
        raise AssertionError("ListNet group split across packed chunks")


def _verify_packing(chunks: dict, groups: dict, rows_seen: int) -> None:
    if rows_seen != 1_998_840 or len(groups) != rows_seen // 4:
        raise AssertionError("unexpected retained training or ListNet-group count")
    if len(chunks) != 249_856 or set(chunks) != set(range(249_856)):
        raise AssertionError("packed chunks are not a complete sequential assignment")
    batches: dict[int, set[int]] = {}
    for chunk_id, state in chunks.items():
        if state["tokens"] > 4096:
            raise AssertionError(f"chunk {chunk_id} exceeds 4,096 tokens")
        if state["position"] != chunk_id % 256 or state["batch"] != chunk_id // 256:
            raise AssertionError("optimizer assignment does not match chunk index")
        batches.setdefault(state["batch"], set()).add(chunk_id)
    if len(batches) != 976 or any(len(chunk_ids) != 256 for chunk_ids in batches.values()):
        raise AssertionError("optimizer batch is not exactly 256 packed chunks")


def _verify_train_flat(root: Path, source: dict[str, dict]) -> None:
    """v6_5: flat per-pair train split, packing left to SFT. Only per-row + uniqueness checks."""
    path = root / "train" / "data.parquet"
    parquet, seen = pq.ParquetFile(path), set()
    for batch in parquet.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            _verify_row(row, source)
            if row["pair_id"] in seen:
                raise AssertionError("training pair reused")
            seen.add(row["pair_id"])


def _verify_train(root: Path, source: dict[str, dict]) -> None:
    path = root / "train" / "data.parquet"
    parquet, seen, pending, chunks, groups = pq.ParquetFile(path), set(), [], {}, {}
    rows_seen = 0
    for batch in parquet.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            _verify_row(row, source)
            if row["pair_id"] in seen:
                raise AssertionError("training pair reused")
            seen.add(row["pair_id"])
            _record_packing(row, chunks, groups)
            pending.append(row)
            rows_seen += 1
            if len(pending) == 4:
                group_ids = {item["listnet_group_id"] for item in pending}
                members = {item["listnet_member_index"] for item in pending}
                molecule_counts = {}
                for item in pending:
                    smiles = item["retrieval_smiles"]
                    molecule_counts[smiles] = molecule_counts.get(smiles, 0) + 1
                if len(group_ids) != 1 or members != {0, 1, 2, 3}:
                    raise AssertionError("incomplete or interleaved training list")
                if max(molecule_counts.values()) > 2:
                    raise AssertionError("more than two list records share a retrieval molecule")
                pending = []
    if pending:
        raise AssertionError("incomplete final ListNet group")
    _verify_packing(chunks, groups, rows_seen)


def verify(root: Path, source_path: Path, listnet: bool = False) -> None:
    source, all_pairs = _source_rows(source_path), set()
    (_verify_train if listnet else _verify_train_flat)(root, source)
    for split, expected in (("validation", 2000), ("test", 2000),
                            ("validation_ranking", 20000), ("test_ranking", 20000)):
        rows = _rows(root, split)
        if len(rows) != expected:
            raise AssertionError(f"{split}: expected {expected}, got {len(rows)}")
        if len({row["pair_id"] for row in rows}) != len(rows):
            raise AssertionError(f"duplicate oriented pair in {split}")
        for row in rows:
            _verify_row(row, source)
            pair = tuple(sorted((row["query_record_id"], row["retrieval_record_id"])))
            if pair in all_pairs:
                raise AssertionError("unordered pair reused across benchmark subsets")
            all_pairs.add(pair)
    for split in ("validation", "test"):
        ordinary = {row["pair_id"] for row in _rows(root, split)}
        ranking = {row["pair_id"] for row in _rows(root, split + "_ranking")}
        if ordinary & ranking:
            raise AssertionError(f"ordinary/ranking overlap in {split}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        default=Path("datasets/eligible/assay_transfer_soft_evidence_v4/records.parquet"))
    parser.add_argument("--listnet", action="store_true",
                        help="enforce v6 ListNet 4-member group + offline packing invariants")
    args = parser.parse_args()
    verify(args.root, args.source, listnet=args.listnet)
    print("V6 raw-pair verification passed")


if __name__ == "__main__":
    main()
