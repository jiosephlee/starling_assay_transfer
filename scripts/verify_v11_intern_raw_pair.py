#!/usr/bin/env python3
"""Verify one task-separated v11 Intern raw-pair release."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.pair_core import oriented_pair
from pipeline.source_normalization.starling_txagent_eligible_v7 import heldout_union_molecules
from pipeline.v11_contract import DEFAULT_REGISTRY, load_registry
from pipeline.v11_prompt_rendering import render_prompt
from pipeline.v11_targets import target_for
from pipeline.v3_policy import file_sha256


SPLITS = ("train", "validation", "validation_ranking", "test", "test_ranking")
VERSIONS = {
    "v11-txagent-v7-task-separated",
    "v11-txagent-v7-categorical-ablation",
}


def _source_rows(path: Path) -> dict[str, dict]:
    return {row["child_id"]: row for row in pq.read_table(path).to_pylist()}


def _verify_manifest(root: Path, manifest: dict) -> None:
    if manifest.get("version") not in VERSIONS:
        raise AssertionError("unexpected v11 release version")
    for split in SPLITS:
        path = root / split / "data.parquet"
        if pq.ParquetFile(path).metadata.num_rows != manifest["rows"][split]:
            raise AssertionError(f"{split}: row count does not match manifest")
        if file_sha256(path) != manifest["parquet_sha256"][split]:
            raise AssertionError(f"{split}: hash does not match manifest")


def _verify_metadata(row: dict) -> None:
    metadata = row["metadata"]
    soft = metadata["soft_targets"]
    if set(soft) != {"(A)", "(B)"} or not math.isclose(sum(soft.values()), 1.0):
        raise AssertionError("invalid metadata.soft_targets")
    if not math.isclose(soft["(A)"], row["target_a"], abs_tol=1e-8):
        raise AssertionError("metadata A probability mismatch")
    if not math.isclose(soft["(B)"], row["target_b"], abs_tol=1e-8):
        raise AssertionError("metadata B probability mismatch")
    expected = {
        key: row[key]
        for key in ("assay_concept", "task_id", "source_id", "measurement_kind")
    }
    if metadata["groups"] != expected:
        raise AssertionError("metadata groups mismatch")


def _verify_row(row: dict, source: dict[str, dict], registry_path: Path) -> None:
    query = source[row["query_record_id"]]
    retrieval = source[row["retrieval_record_id"]]
    if row["query_smiles"] == row["retrieval_smiles"]:
        raise AssertionError("self-molecule pair")
    if query["pair_bucket_key"] != retrieval["pair_bucket_key"]:
        raise AssertionError("cross-bucket pair")
    kinds = {query["measurement_kind"], retrieval["measurement_kind"], row["measurement_kind"]}
    if len(kinds) != 1:
        raise AssertionError("pair measurement-kind mismatch")
    if not oriented_pair(query, retrieval):
        raise AssertionError("rejected deterministic direction")
    if row["prompt"] != render_prompt(query, retrieval, registry_path=registry_path):
        raise AssertionError("prompt reconstruction mismatch")
    plain_lines = any(line.startswith("SMILES:") for line in row["prompt"].splitlines())
    if row["prompt"].count("<SMILES>") != 2 or plain_lines:
        raise AssertionError("Intern SMILES tags are missing or malformed")
    expected = target_for(query, retrieval)
    if any(not math.isclose(row[key], value, abs_tol=1e-8) for key, value in expected.items()):
        raise AssertionError("target reconstruction mismatch")
    completion = "(A)" if row["target_a"] >= 0.5 else "(B)"
    if row["completion"] != completion:
        raise AssertionError("completion disagrees with target")
    _verify_metadata(row)


def _verify_split(
    root: Path, split: str, source: dict[str, dict], registry_path: Path,
) -> set[tuple[str, str]]:
    seen_ids: set[str] = set()
    unordered: set[tuple[str, str]] = set()
    for batch in pq.ParquetFile(root / split / "data.parquet").iter_batches(8192):
        for row in batch.to_pylist():
            if row["pair_id"] in seen_ids:
                raise AssertionError(f"{split}: duplicate oriented pair")
            seen_ids.add(row["pair_id"])
            unordered.add(tuple(sorted((row["query_record_id"], row["retrieval_record_id"]))))
            _verify_row(row, source, registry_path)
    return unordered


def _verify_ranking_width(root: Path, split: str, width: int) -> None:
    groups: dict[str, set[int]] = {}
    path = root / f"{split}_ranking" / "data.parquet"
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=["ranking_query_id", "ranking_member_index"]):
        for row in batch.to_pylist():
            groups.setdefault(row["ranking_query_id"], set()).add(row["ranking_member_index"])
    expected = set(range(width))
    if any(members != expected for members in groups.values()):
        raise AssertionError(f"{split}: incomplete ranking anchor")


def _verify_membership(root: Path, source: dict[str, dict], task: str, registry: dict) -> None:
    def molecules(split: str) -> set[str]:
        table = pq.read_table(root / split / "data.parquet", columns=["query_smiles", "retrieval_smiles"])
        return set(table["query_smiles"].to_pylist()) | set(table["retrieval_smiles"].to_pylist())

    train = molecules("train")
    evaluation = set().union(*(molecules(split) for split in SPLITS[1:]))
    reserved = heldout_union_molecules(task, list(source.values()), registry)
    if train & (evaluation | reserved):
        raise AssertionError("train overlaps evaluation or scaffold-reserved molecules")
    if not evaluation <= reserved:
        raise AssertionError("evaluation escaped scaffold-reserved pool")


def _measurement_kinds(path: Path) -> set[str]:
    kinds: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(8192, columns=["measurement_kind"]):
        kinds.update(str(value) for value in batch.column(0).to_pylist())
    return kinds


def _verify_measurement_policy(root: Path, manifest: dict) -> None:
    policy = manifest.get("measurement_kind_policy")
    if not policy:
        return
    if policy.get("reservation") != "all_eligible_measurement_kinds":
        raise AssertionError("reservation policy is not all-kind")
    expected_eval = set(policy["evaluation"])
    for split in SPLITS[1:]:
        actual = _measurement_kinds(root / split / "data.parquet")
        if actual != expected_eval:
            raise AssertionError(f"{split}: measurement kinds violate evaluation policy")
    expected_train: set[str] = set()
    for component in manifest["train_components"].values():
        expected_train.update(component["achieved_by_measurement_kind"])
    actual_train = _measurement_kinds(root / "train" / "data.parquet")
    if actual_train != expected_train:
        raise AssertionError("train measurement kinds violate component policy")


def verify(root: Path, source_path: Path, registry_path: Path) -> None:
    registry = load_registry(registry_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    source = _source_rows(source_path)
    _verify_manifest(root, manifest)
    if file_sha256(source_path) != manifest["eligible_source_sha256"]:
        raise AssertionError("eligible source hash mismatch")
    if file_sha256(registry_path) != manifest["prompt_projection"]["sha256"]:
        raise AssertionError("prompt projection hash mismatch")
    pairs = {split: _verify_split(root, split, source, registry_path) for split in SPLITS}
    for split in ("validation", "test"):
        if pairs[split] & pairs[f"{split}_ranking"]:
            raise AssertionError(f"{split}: ordinary/ranking pair overlap")
        _verify_ranking_width(root, split, manifest["construction"]["ranking_anchor_width"])
    _verify_measurement_policy(root, manifest)
    _verify_membership(root, source, manifest["task_id"], registry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    verify(args.root, args.source, args.registry)
    print("V11 Intern raw-pair release verification passed")


if __name__ == "__main__":
    main()
