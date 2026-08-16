#!/usr/bin/env python3
"""Verify V12/V12.1 lineage, leakage, degree, and ranking contracts."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from pipeline.v12_ranking import MINIMUM_SAME_CATEGORY
from pipeline.v3_policy import file_sha256
from scripts import build_v12_release as v12_release


def _source(source: Path) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    rows, parents = {}, defaultdict(set)
    for batch in pq.ParquetFile(source).iter_batches(8192):
        for row in batch.to_pylist():
            rows[str(row["child_id"])] = row
            parents[str(row["dataset_split"])].add(str(row["normalized_parent_identity_key"]))
    if any(parents[left] & parents[right] for left, right in (
        ("train", "validation"), ("train", "test"), ("validation", "test")
    )):
        raise ValueError("eligible source has cross-split normalized-parent overlap")
    return rows, parents


def _calibration(source: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    path = source.parent / "train_value_calibration.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        document = json.load(handle)
    counts = Counter(
        str(row["pair_bucket_key"]) for row in rows.values()
        if row["dataset_split"] == "train"
    )
    for bucket, entry in document["buckets"].items():
        if int(entry["record_count"]) != counts[bucket] or counts[bucket] < 25:
            raise ValueError(f"invalid train-only calibration count for {bucket}")
    eligible_buckets = {str(row["pair_bucket_key"]) for row in rows.values()}
    if eligible_buckets != set(document["buckets"]):
        raise ValueError("eligible records and train-calibrated bucket inventory differ")


def _gold_keys(path: Path) -> set[str]:
    return {
        str(json.loads(line)["molecule_identity_key"])
        for line in path.read_text(encoding="utf-8").splitlines() if line
    }


def _gold_lineage(source: Path, parents: Mapping[str, set[str]]) -> None:
    split = json.loads((source.parent / "split_manifest.json").read_text(encoding="utf-8"))
    paths = {name: Path(value) for name, value in split["gold_paths"].items()}
    for name, path in paths.items():
        if split["gold_sha256"][name] != file_sha256(path):
            raise ValueError(f"live gold {name} file changed")
    gold = {name: _gold_keys(path) for name, path in paths.items()}
    if parents["test"] - (gold["valid"] | gold["test"]):
        raise ValueError("test parents exceed the gold valid+test universe")
    if parents["validation"] - gold["train"]:
        raise ValueError("validation parents exceed the gold train universe")
    if len(parents["validation"]) != len(parents["test"]) // 2:
        raise ValueError("validation parent count is not floor(test parent count / 2)")


def _train(
    path: Path, source: Mapping[str, Mapping[str, Any]], components: Mapping[str, Any],
) -> None:
    query_degree, retrieval_degree = Counter(), Counter()
    seen = set()
    for batch in pq.ParquetFile(path).iter_batches(8192):
        for row in batch.to_pylist():
            query = source[str(row["query_record_id"])]
            retrieval = source[str(row["retrieval_record_id"])]
            if query["dataset_split"] != "train" or retrieval["dataset_split"] != "train":
                raise ValueError("train pair uses a held-out record")
            if row["query_record_id"] == row["retrieval_record_id"]:
                raise ValueError("train contains a self-pair")
            if row["pair_id"] in seen:
                raise ValueError("train contains a duplicate directed pair")
            seen.add(row["pair_id"])
            query_degree[str(row["query_record_id"])] += 1
            retrieval_degree[str(row["retrieval_record_id"])] += 1
    if max(query_degree.values(), default=0) > 6 or max(retrieval_degree.values(), default=0) > 6:
        raise ValueError("train exceeds the 6/6 query/retrieval degree caps")
    expected = sum(int(value["achieved_pairs"]) for value in components.values())
    if len(seen) != expected:
        raise ValueError("train component counts do not match unique pair count")


def _ranking(
    path: Path, split: str, source: Mapping[str, Mapping[str, Any]], expected_rows: int,
) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in pq.ParquetFile(path).iter_batches(8192):
        for row in batch.to_pylist():
            query = source[str(row["query_record_id"])]
            retrieval = source[str(row["retrieval_record_id"])]
            if query["dataset_split"] != split or retrieval["dataset_split"] != "train":
                raise ValueError(f"{split} ranking crosses its query/train pool contract")
            groups[str(row["ranking_query_id"])].append(row)
    if sum(map(len, groups.values())) != expected_rows:
        raise ValueError(f"{split} ranking row count mismatch")
    for query_id, members in groups.items():
        indices = sorted(int(row["ranking_member_index"]) for row in members)
        if len(members) != 24 or indices != list(range(24)):
            raise ValueError(f"{query_id} is not one complete 24-candidate list")
        retrievals = [source[str(row["retrieval_record_id"])] for row in members]
        parents = {str(row["normalized_parent_identity_key"]) for row in retrievals}
        if len(parents) < 2:
            raise ValueError(f"{query_id} has fewer than two retrieval parents")
        if members[0]["ranking_family"] == "categorical":
            category = str(members[0]["ranking_query_category_id"])
            values = [str(row["canonical_category_id"]) for row in retrievals]
            if values.count(category) < MINIMUM_SAME_CATEGORY or not any(
                value != category for value in values
            ):
                raise ValueError(f"{query_id} violates categorical ranking constraints")
        elif any(row.get("calibration_sample_standard_deviation") in {None, 0} for row in members):
            raise ValueError(f"{query_id} lacks global SD measurement metadata")


def _reference_membership(root: Path, reference: Path) -> None:
    for split in v12_release.SPLITS:
        current = root / split / "data.parquet"
        frozen = reference / split / "data.parquet"
        if v12_release._pair_id_sha256(current) != v12_release._pair_id_sha256(frozen):
            raise ValueError(f"V12.1 changed {split} ordered pair membership")
        if split.endswith("ranking") and (
            v12_release._ranking_identity_sha256(current)
            != v12_release._ranking_identity_sha256(frozen)
        ):
            raise ValueError(f"V12.1 changed {split} ranking identities")


def verify(root: Path, source: Path, reference: Path | None = None) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["eligible_source_sha256"] != file_sha256(source):
        raise ValueError("release eligible-source hash mismatch")
    rows, parents = _source(source)
    _gold_lineage(source, parents)
    _calibration(source, rows)
    _train(root / "train/data.parquet", rows, manifest["train_components"])
    for split in ("validation", "test"):
        _ranking(
            root / f"{split}_ranking/data.parquet", split, rows,
            int(manifest["rows"][f"{split}_ranking"]),
        )
    for split in v12_release.SPLITS:
        path = root / split / "data.parquet"
        if manifest["parquet_sha256"][split] != file_sha256(path):
            raise ValueError(f"{split} Parquet hash mismatch")
    if reference is not None:
        _reference_membership(root, reference)
    return {
        "status": "ok", "version": manifest["version"],
        "source_parent_counts": {key: len(value) for key, value in sorted(parents.items())},
        "rows": manifest["rows"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-v12", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.source, args.reference_v12), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
