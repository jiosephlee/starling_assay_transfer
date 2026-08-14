#!/usr/bin/env python3
"""Verify the sole V11 with-categorical release and its component contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator, Mapping

import pyarrow.parquet as pq

from pipeline.v11_contract import DEFAULT_REGISTRY, heldout_reservation_manifest, load_registry
from scripts.verify_v11_intern_raw_pair import verify


REQUIRED_RANKING_COLUMNS = {
    "measurement_kind", "query_geometry_value", "retrieval_geometry_value",
    "query_value_percentile", "retrieval_value_percentile",
    "query_canonical_category_id", "retrieval_canonical_category_id",
    "canonical_measurement_scale_id", "absolute_geometry_difference",
    "calibration_sample_standard_deviation", "ranking_family",
}


def _manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _component_ranges(manifest: Mapping) -> list[tuple[str, int, int]]:
    order = manifest["measurement_kind_policy"]["train_components"]
    if order != ["continuous", "categorical"]:
        raise AssertionError("train component order is not continuous then categorical")
    ranges, start = [], 0
    for name in order:
        count = int(manifest["train_components"][name]["achieved_pairs"])
        ranges.append((name, start, count))
        start += count
    if start != manifest["rows"]["train"]:
        raise AssertionError("component ranges do not cover train")
    return ranges


def _range_rows(path: Path, start: int, count: int) -> Iterator[dict]:
    position, end = 0, start + count
    columns = ["pair_id", "query_record_id", "retrieval_record_id", "measurement_kind"]
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=columns):
        batch_start, position = position, position + batch.num_rows
        if position <= start:
            continue
        if batch_start >= end:
            break
        offset = max(0, start - batch_start)
        length = min(position, end) - (batch_start + offset)
        yield from batch.slice(offset, length).to_pylist()


def _range_diagnostics(path: Path, start: int, count: int) -> tuple[str, int, int, Counter]:
    digest, query, retrieval, kinds = hashlib.sha256(), Counter(), Counter(), Counter()
    for row in _range_rows(path, start, count):
        digest.update(str(row["pair_id"]).encode("utf-8") + b"\n")
        query[str(row["query_record_id"])] += 1
        retrieval[str(row["retrieval_record_id"])] += 1
        kinds[str(row["measurement_kind"])] += 1
    return digest.hexdigest(), max(query.values(), default=0), max(retrieval.values(), default=0), kinds


def _verify_components(root: Path, manifest: Mapping) -> None:
    path = root / "train/data.parquet"
    for name, start, count in _component_ranges(manifest):
        component = manifest["train_components"][name]
        digest, query_max, retrieval_max, kinds = _range_diagnostics(path, start, count)
        if count > int(component["max_pairs"]) or digest != component["pair_id_sha256"]:
            raise AssertionError(f"{name}: cap or pair sequence mismatch")
        expected_kinds = dict(sorted(component["achieved_by_measurement_kind"].items()))
        if dict(sorted(kinds.items())) != expected_kinds:
            raise AssertionError(f"{name}: measurement-kind counts mismatch")
        if query_max != component["observed_max_query_degree"]:
            raise AssertionError(f"{name}: query degree does not match manifest")
        if retrieval_max != component["observed_max_retrieval_degree"]:
            raise AssertionError(f"{name}: retrieval degree does not match manifest")
        if query_max > component["query_degree_cap"] or retrieval_max > component["retrieval_degree_cap"]:
            raise AssertionError(f"{name}: component degree cap exceeded")


def _verify_ranking_diagnostics(manifest: Mapping) -> None:
    families = manifest["ranking_validation"]["families"]
    if set(families) != {"continuous", "categorical"}:
        raise AssertionError("ranking diagnostics omit a family")
    for family, report in families.items():
        for source, target in report["target_by_source"].items():
            achieved = int(report["achieved_by_source"].get(source, 0))
            if achieved > int(target):
                raise AssertionError(f"{family}/{source}: anchor quota exceeded")
            if achieved < int(target) and source not in report["shortfalls"]:
                raise AssertionError(f"{family}/{source}: unexplained anchor shortfall")
            shortfall = report["shortfalls"].get(source)
            if shortfall and shortfall["reason"] not in {"no_eligible_rows", "constraints_exhausted"}:
                raise AssertionError(f"{family}/{source}: invalid shortfall reason")


def _verify_gold(manifest: Mapping, registry_path: Path) -> None:
    registry = load_registry(registry_path)
    live = heldout_reservation_manifest(manifest["task_id"], registry)
    if live != manifest["heldout_reservation"]:
        raise AssertionError("release does not pin the current V7 gold scaffold reservation")


def _verify_downstream_columns(root: Path) -> None:
    columns = set(pq.read_schema(root / "validation_ranking/data.parquet").names)
    missing = REQUIRED_RANKING_COLUMNS - columns
    if missing:
        raise AssertionError(f"ranking data lacks downstream columns: {sorted(missing)}")


def verify_release(
    root: Path, source: Path, registry: Path = DEFAULT_REGISTRY,
) -> None:
    manifest = _manifest(root)
    verify(root, source, registry)
    _verify_components(root, manifest)
    _verify_ranking_diagnostics(manifest)
    _verify_gold(manifest, registry)
    _verify_downstream_columns(root)


def verify_pair(
    _continuous_root: Path, categorical_root: Path, source: Path,
    registry: Path = DEFAULT_REGISTRY,
) -> None:
    """Compatibility entry point; the continuous-only argument is intentionally ignored."""
    verify_release(categorical_root, source, registry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    verify_release(args.root, args.source, args.registry)
    print("V11 with-categorical release verification passed")


if __name__ == "__main__":
    main()
