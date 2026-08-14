#!/usr/bin/env python3
"""Verify one V11.1 percentile-distance-CDF release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from pipeline.pair_core import oriented_pair
from pipeline.source_normalization.starling_txagent_eligible_v7 import heldout_union_molecules
from pipeline.v11_1_targets import (
    TARGET_CONTRACT, attach_distance_calibrations, load_distance_calibration,
    parsed_calibrations, target_for,
)
from pipeline.v11_contract import DEFAULT_REGISTRY, load_registry
from pipeline.v11_prompt_rendering import render_prompt
from pipeline.v3_policy import file_sha256
from scripts import verify_v11_intern_raw_pair as v11_verify
from scripts.build_v11_1_from_defined_split import (
    MEMBERSHIP_CONTRACT, REFERENCE_VERSION, _assert_invariants, _ranking_identity_sha256,
)
from scripts.build_v11_1_categorical_release import CALIBRATION_FILE, VERSION
from scripts.build_v11_categorical_ablation import _pair_id_sha256
from scripts.build_v11_categorical_ablation import SPLITS


def _source_rows(path: Path, calibration_path: Path) -> dict[str, dict]:
    rows = pq.read_table(path).to_pylist()
    document = load_distance_calibration(calibration_path)
    attached, _ = attach_distance_calibrations(rows, parsed_calibrations(document))
    return {row["child_id"]: row for row in attached}


def _verify_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("version") != VERSION or manifest.get("dataset_variant") != "with_categorical":
        raise AssertionError("unexpected V11.1 release identity")
    expected = {"README.md", "manifest.json", CALIBRATION_FILE}
    expected.update(f"{split}/data.parquet" for split in SPLITS)
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        raise AssertionError(f"unexpected V11.1 release topology: {sorted(actual)}")
    calibration = manifest["percentile_distance_calibration"]
    if calibration["path"] != CALIBRATION_FILE:
        raise AssertionError("V11.1 calibration path mismatch")
    if file_sha256(root / CALIBRATION_FILE) != calibration["sha256"]:
        raise AssertionError("V11.1 calibration hash mismatch")
    membership = manifest.get("reference_membership", {})
    if membership.get("schema_version") != MEMBERSHIP_CONTRACT:
        raise AssertionError("V11.1 lacks the frozen V11 membership contract")
    if membership.get("reference_version") != REFERENCE_VERSION:
        raise AssertionError("V11.1 membership does not reference V11")
    for split in SPLITS:
        path = root / split / "data.parquet"
        if pq.ParquetFile(path).metadata.num_rows != manifest["rows"][split]:
            raise AssertionError(f"{split}: row count does not match manifest")
        if file_sha256(path) != manifest["parquet_sha256"][split]:
            raise AssertionError(f"{split}: hash does not match manifest")


def _verify_reference_manifest(
    reference_root: Path, membership: Mapping[str, Any], task_id: str,
) -> dict[str, Any]:
    path = reference_root / "manifest.json"
    if file_sha256(path) != membership["reference_manifest_sha256"]:
        raise AssertionError("frozen reference manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != REFERENCE_VERSION or manifest.get("task_id") != task_id:
        raise AssertionError("frozen reference release identity mismatch")
    return manifest


def _verify_frozen_split(
    root: Path, reference_root: Path, split: str, membership: Mapping[str, Any],
) -> None:
    output_path = root / split / "data.parquet"
    reference_path = reference_root / split / "data.parquet"
    expected = membership["splits"][split]
    if file_sha256(reference_path) != expected["parquet_sha256"]:
        raise AssertionError(f"{split}: frozen reference Parquet hash mismatch")
    if _pair_id_sha256(output_path) != expected["ordered_pair_id_sha256"]:
        raise AssertionError(f"{split}: ordered V11 pair membership changed")
    if _pair_id_sha256(reference_path) != expected["ordered_pair_id_sha256"]:
        raise AssertionError(f"{split}: reference pair sequence hash changed")
    for actual, expected_row in zip(
        _parquet_rows(output_path), _parquet_rows(reference_path), strict=True
    ):
        _assert_invariants(expected_row, actual)
    if split == "validation_ranking":
        _verify_ranking_identity(output_path, reference_path, expected)


def _parquet_rows(path: Path):
    for batch in pq.ParquetFile(path).iter_batches(8192):
        yield from batch.to_pylist()


def _verify_ranking_identity(
    output_path: Path, reference_path: Path, expected: Mapping[str, Any],
) -> None:
    digest = expected["ranking_identity_sha256"]
    if _ranking_identity_sha256(reference_path) != digest:
        raise AssertionError("reference ranking identity hash changed")
    if _ranking_identity_sha256(output_path) != digest:
        raise AssertionError("V11.1 ranking groups or member order changed")


def _verify_metadata(row: Mapping[str, Any]) -> None:
    if "target_z" in row or "target_z" in row["metadata"]:
        raise AssertionError("V11.1 must not emit target_z")
    metadata, distribution = row["metadata"], row["target_distribution"]
    soft = metadata["soft_targets"]
    expected_soft = {"(A)": distribution["transfer"], "(B)": distribution["nontransfer"]}
    if soft != expected_soft or metadata.get("target_contract") != TARGET_CONTRACT:
        raise AssertionError("V11.1 soft-target metadata mismatch")
    expected_groups = {key: row[key] for key in (
        "assay_concept", "task_id", "source_id", "measurement_kind"
    )}
    if metadata["groups"] != expected_groups:
        raise AssertionError("V11.1 metadata groups mismatch")


def _verify_row(
    row: dict, source: Mapping[str, dict], registry: Mapping[str, Any], registry_path: Path,
) -> None:
    query, retrieval = source[row["query_record_id"]], source[row["retrieval_record_id"]]
    required_equal = (
        "source_id", "measurement_kind", "canonical_endpoint_key",
        "canonical_measurement_scale_id", "pair_bucket_key",
    )
    if any(query.get(field) != retrieval.get(field) for field in required_equal):
        raise AssertionError("V11.1 pair crosses its comparison contract")
    if query["canonical_smiles"] == retrieval["canonical_smiles"]:
        raise AssertionError("V11.1 contains a same-molecule pair")
    if row["split"] == "train" and not oriented_pair(query, retrieval):
        raise AssertionError("V11.1 train pair violates deterministic orientation")
    if row["prompt"] != render_prompt(query, retrieval, registry_path=registry_path):
        raise AssertionError("V11.1 prompt reconstruction mismatch")
    expected = target_for(query, retrieval)
    if any(not v11_verify._same(row[key], value) for key, value in expected.items()):
        raise AssertionError("V11.1 target reconstruction mismatch")
    v11_verify._verify_side("query", query, row)
    v11_verify._verify_side("retrieval", retrieval, row)
    _verify_metadata(row)


def _verify_split(
    root: Path, split: str, source: Mapping[str, dict],
    registry: Mapping[str, Any], registry_path: Path,
) -> None:
    seen = set()
    for batch in pq.ParquetFile(root / split / "data.parquet").iter_batches(8192):
        for row in batch.to_pylist():
            if row["pair_id"] in seen:
                raise AssertionError(f"{split}: duplicate oriented pair")
            seen.add(row["pair_id"])
            _verify_row(row, source, registry, registry_path)


def verify(
    root: Path, source_path: Path, registry_path: Path = DEFAULT_REGISTRY,
    reference_root: Path | None = None,
) -> None:
    registry = load_registry(registry_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _verify_manifest(root, manifest)
    membership = manifest["reference_membership"]
    reference_root = reference_root or Path(membership["reference_root"])
    _verify_reference_manifest(reference_root, membership, manifest["task_id"])
    source = _source_rows(source_path, root / CALIBRATION_FILE)
    if file_sha256(source_path) != manifest["eligible_source_sha256"]:
        raise AssertionError("eligible source hash mismatch")
    if file_sha256(registry_path) != manifest["prompt_projection"]["sha256"]:
        raise AssertionError("prompt projection hash mismatch")
    v11_verify._verify_schema(root, manifest)
    for split in SPLITS:
        _verify_split(root, split, source, registry, registry_path)
        _verify_frozen_split(root, reference_root, split, membership)
    v11_verify._verify_ranking(root, manifest)
    v11_verify._verify_membership(root, source, manifest["task_id"], registry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--reference-release", type=Path)
    args = parser.parse_args()
    verify(args.root, args.source, args.registry, args.reference_release)
    print("V11.1 percentile-distance-CDF verification passed")


if __name__ == "__main__":
    main()
