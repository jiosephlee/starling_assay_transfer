#!/usr/bin/env python3
"""Verify one V11 value-CDF with-categorical release."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from pipeline.pair_core import oriented_pair
from pipeline.source_normalization.starling_txagent_eligible_v7 import heldout_union_molecules
from pipeline.v11_contract import DEFAULT_REGISTRY, load_registry, project_prompt_record
from pipeline.v11_prompt_rendering import render_prompt
from pipeline.v11_ranking import CATEGORICAL_MATCHES
from pipeline.v11_targets import TARGET_CONTRACT, target_for
from pipeline.v3_policy import file_sha256


SPLITS = ("train", "validation_ranking")
VERSION = "v11-txagent-v7-value-cdf-with-categorical"
SIDE_FIELDS = (
    "geometry_value", "finite_scalar_value", "value_percentile",
    "canonical_category_id", "canonical_category_rank",
    "canonical_measurement_scale_id", "canonical_unit_text",
    "raw_measurement_text", "raw_unit_text",
)


def _source_rows(path: Path) -> dict[str, dict]:
    return {row["child_id"]: row for row in pq.read_table(path).to_pylist()}


def _same(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def _verify_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("version") != VERSION or manifest.get("dataset_variant") != "with_categorical":
        raise AssertionError("unexpected V11 release identity")
    expected_files = {"README.md", "manifest.json"}
    expected_files.update(f"{split}/data.parquet" for split in SPLITS)
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise AssertionError(f"unexpected local release topology: {sorted(actual_files)}")
    for split in SPLITS:
        path = root / split / "data.parquet"
        if pq.ParquetFile(path).metadata.num_rows != manifest["rows"][split]:
            raise AssertionError(f"{split}: row count does not match manifest")
        if file_sha256(path) != manifest["parquet_sha256"][split]:
            raise AssertionError(f"{split}: hash does not match manifest")


def _verify_metadata(row: Mapping[str, Any]) -> None:
    metadata = row["metadata"]
    soft = metadata["soft_targets"]
    if set(soft) != {"(A)", "(B)"} or not math.isclose(sum(soft.values()), 1.0):
        raise AssertionError("invalid metadata.soft_targets")
    if not _same(soft["(A)"], row["target_a"]) or not _same(soft["(B)"], row["target_b"]):
        raise AssertionError("metadata soft targets disagree with row")
    expected = {key: row[key] for key in (
        "assay_concept", "task_id", "source_id", "measurement_kind"
    )}
    if metadata["groups"] != expected or metadata.get("target_contract") != TARGET_CONTRACT:
        raise AssertionError("metadata grouping or target contract mismatch")


def _verify_side(prefix: str, source: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    for field in SIDE_FIELDS:
        if not _same(row[f"{prefix}_{field}"], source.get(field)):
            raise AssertionError(f"{prefix}_{field} does not match eligible source")


def _verify_prompt_projection(query: dict, registry: Mapping[str, Any]) -> None:
    projected = project_prompt_record(query, query["task_id"], "query", registry)
    if "unit_text" in projected or "measurement_text" in projected:
        raise AssertionError("query prompt projection exposes measurement or unit")


def _verify_row(
    row: dict, source: Mapping[str, dict], registry: Mapping[str, Any], registry_path: Path,
) -> None:
    query, retrieval = source[row["query_record_id"]], source[row["retrieval_record_id"]]
    if query["canonical_smiles"] == retrieval["canonical_smiles"]:
        raise AssertionError("self-molecule pair")
    required_equal = (
        "source_id", "measurement_kind", "canonical_endpoint_key",
        "canonical_measurement_scale_id", "pair_bucket_key",
    )
    if any(query.get(field) != retrieval.get(field) for field in required_equal):
        raise AssertionError("pair crosses source, kind, endpoint, scale, or bucket")
    if row["split"] == "train" and not oriented_pair(query, retrieval):
        raise AssertionError("rejected deterministic pair orientation")
    if row["prompt"] != render_prompt(query, retrieval, registry_path=registry_path):
        raise AssertionError("prompt reconstruction mismatch")
    _verify_prompt_projection(query, registry)
    expected = target_for(query, retrieval)
    if any(not _same(row[key], value) for key, value in expected.items()):
        raise AssertionError("target reconstruction mismatch")
    _verify_side("query", query, row)
    _verify_side("retrieval", retrieval, row)
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


def _ranking_groups(path: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for batch in pq.ParquetFile(path).iter_batches(8192):
        for row in batch.to_pylist():
            groups[str(row["ranking_query_id"])].append(row)
    return groups


def _verify_group(query_id: str, rows: list[dict], width: int) -> None:
    if len(rows) != width or {row["ranking_member_index"] for row in rows} != set(range(width)):
        raise AssertionError(f"{query_id}: incomplete ranking group")
    singleton_fields = (
        "query_record_id", "source_id", "measurement_kind", "canonical_endpoint_key",
        "canonical_measurement_scale_id", "pair_bucket_key", "ranking_family",
        "calibration_sample_standard_deviation",
    )
    if any(len({row[field] for row in rows}) != 1 for field in singleton_fields):
        raise AssertionError(f"{query_id}: ranking group crosses its comparison contract")
    family = rows[0]["ranking_family"]
    kind = rows[0]["measurement_kind"]
    if family == "continuous" and kind != "continuous":
        raise AssertionError(f"{query_id}: continuous family has non-continuous rows")
    if family != "categorical":
        return
    if kind not in {"binary", "ordinal"}:
        raise AssertionError(f"{query_id}: categorical family has invalid kind")
    query_category = rows[0]["query_canonical_category_id"]
    same = sum(row["retrieval_canonical_category_id"] == query_category for row in rows)
    if same != CATEGORICAL_MATCHES or same == width:
        raise AssertionError(f"{query_id}: categorical group is not 5/{width - CATEGORICAL_MATCHES}")


def _verify_ranking(root: Path, manifest: Mapping[str, Any]) -> None:
    diagnostics = manifest["ranking_validation"]
    width = int(diagnostics["ranking_anchor_width"])
    if diagnostics.get("directed_pair_policy") != "each_direction_at_most_once":
        raise AssertionError("ranking directed-pair policy is missing")
    if int(diagnostics.get("maximum_unordered_pair_uses", 0)) != 2:
        raise AssertionError("ranking unordered-pair reuse policy is invalid")
    groups = _ranking_groups(root / "validation_ranking/data.parquet")
    for query_id, rows in groups.items():
        _verify_group(query_id, rows, width)
    expected = sum(
        sum(family["achieved_by_source"].values())
        for family in diagnostics["families"].values()
    )
    if len(groups) != expected:
        raise AssertionError("ranking group count disagrees with diagnostics")
    _verify_ranking_reuse(groups, manifest)


def _verify_ranking_reuse(
    groups: Mapping[str, list[dict]], manifest: Mapping[str, Any],
) -> None:
    degree, unordered = defaultdict(int), defaultdict(int)
    for rows in groups.values():
        for row in rows:
            degree[str(row["query_record_id"])] += 1
            degree[str(row["retrieval_record_id"])] += 1
            edge = tuple(sorted((str(row["query_record_id"]), str(row["retrieval_record_id"]))))
            unordered[edge] += 1
    cap = int(manifest["construction"]["ranking_record_degree_cap"])
    if max(degree.values(), default=0) > cap:
        raise AssertionError("ranking record degree cap exceeded")
    if max(unordered.values(), default=0) > 2:
        raise AssertionError("ranking unordered pair appears more than once per direction")


def _molecules(path: Path) -> set[str]:
    table = pq.read_table(path, columns=["query_smiles", "retrieval_smiles"])
    return set(table["query_smiles"].to_pylist()) | set(table["retrieval_smiles"].to_pylist())


def _verify_membership(
    root: Path, source: Mapping[str, dict], task_id: str, registry: Mapping[str, Any],
) -> None:
    reserved = heldout_union_molecules(task_id, list(source.values()), registry)
    train = _molecules(root / "train/data.parquet")
    validation = _molecules(root / "validation_ranking/data.parquet")
    if train & reserved:
        raise AssertionError("train overlaps the current scaffold gold union")
    if not validation <= reserved:
        raise AssertionError("ranking validation escaped the scaffold gold union")


def _verify_schema(root: Path, manifest: Mapping[str, Any]) -> None:
    schemas = [pq.read_schema(root / split / "data.parquet") for split in SPLITS]
    if not schemas[0].equals(schemas[1], check_metadata=False):
        raise AssertionError("train and validation_ranking schemas differ")
    if schemas[0].names != manifest["release_schema"]:
        raise AssertionError("release schema names disagree with manifest")


def verify(root: Path, source_path: Path, registry_path: Path = DEFAULT_REGISTRY) -> None:
    registry = load_registry(registry_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    source = _source_rows(source_path)
    _verify_manifest(root, manifest)
    if file_sha256(source_path) != manifest["eligible_source_sha256"]:
        raise AssertionError("eligible source hash mismatch")
    if file_sha256(registry_path) != manifest["prompt_projection"]["sha256"]:
        raise AssertionError("prompt projection hash mismatch")
    _verify_schema(root, manifest)
    for split in SPLITS:
        _verify_split(root, split, source, registry, registry_path)
    _verify_ranking(root, manifest)
    _verify_membership(root, source, manifest["task_id"], registry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    verify(args.root, args.source, args.registry)
    print("V11 value-CDF with-categorical verification passed")


if __name__ == "__main__":
    main()
