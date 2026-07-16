#!/usr/bin/env python3
"""Rank molecule split seeds by held-out Fg-high hard-label capacity."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.normalize.common_transfer import FingerprintCache  # noqa: E402
from pipeline.stages.compose_v3 import _eligible_row  # noqa: E402
from pipeline.stages.pairs import _label  # noqa: E402
from pipeline.stages.split import assign_split  # noqa: E402
from pipeline.v3_policy import V3Policies  # noqa: E402


def _rows(base: Path, policies: V3Policies) -> list[dict[str, Any]]:
    output = []
    for row in pq.read_table(base / "records.parquet").to_pylist():
        eligible, _ = _eligible_row(row, policies)
        if eligible is not None and eligible["assay_concept"] == "Fg":
            output.append(eligible)
    return output


def _fingerprints(rows: list[dict[str, Any]], policies: V3Policies) -> FingerprintCache:
    config = policies.fingerprint
    cache = FingerprintCache(radius=int(config["radius"]), nbits=int(config["n_bits"]))
    for smiles in {row["canonical_smiles"] for row in rows}:
        cache.get(smiles)
    return cache


def _high_neighbors(rows: list[dict[str, Any]], policies: V3Policies, cache: FingerprintCache) -> dict[str, set[str]]:
    boundary = float(policies.tanimoto["boundary"])
    molecules, neighbors = sorted({row["canonical_smiles"] for row in rows}), defaultdict(set)
    for index, left in enumerate(molecules):
        for right in molecules[index + 1:]:
            if cache.similarity(cache.get(left), cache.get(right)) >= boundary:
                neighbors[left].add(right)
                neighbors[right].add(left)
    return neighbors


def _split_rows(rows: list[dict[str, Any]], seed: int, ratios: tuple[float, float, float]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[assign_split(row["canonical_smiles"], seed, ratios)].append(row)
    return output


def _group_capacity(rows: list[dict[str, Any]], policies: V3Policies, neighbors: dict[str, set[str]]) -> int:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[row["canonical_endpoint_key"]].append(row)
    return sum(_endpoint_capacity(group, policies, neighbors) for group in by_key.values())


def _endpoint_capacity(rows: list[dict[str, Any]], policies: V3Policies, neighbors: dict[str, set[str]]) -> int:
    evidence: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        evidence[row["canonical_smiles"]].append(float(row["comparison_value"]))
    if len(evidence) < 2:
        return 0
    metric, total = policies.metric_for(rows[0]), 0
    for retrieved in rows:
        for query, values in evidence.items():
            if query not in neighbors[retrieved["canonical_smiles"]]:
                continue
            if _label(metric, float(retrieved["comparison_value"]), values)["binary_label"] is not None:
                total += 1
    return total


def _result(seed: int, rows: list[dict[str, Any]], policies: V3Policies,
            neighbors: dict[str, set[str]], ratios: tuple[float, float, float]) -> dict[str, int]:
    split_rows = _split_rows(rows, seed, ratios)
    return {"seed": seed, "validation_fg_high": _group_capacity(split_rows["validation"], policies, neighbors),
            "test_fg_high": _group_capacity(split_rows["test"], policies, neighbors)}


def _write(rows: list[dict[str, int]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "validation_fg_high", "test_fg_high", "minimum"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "minimum": min(row["validation_fg_high"], row["test_fg_high"])})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--seed-count", type=int, default=10000)
    parser.add_argument("--train-frac", type=float, default=None)
    parser.add_argument("--validation-frac", type=float, default=None)
    parser.add_argument("--test-frac", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policies = V3Policies(args.release)
    rows = _rows(args.base, policies)
    cache = _fingerprints(rows, policies)
    neighbors = _high_neighbors(rows, policies, cache)
    split = policies.sampling["split"]
    ratios = (float(args.train_frac or split["train"]), float(args.validation_frac or split["validation"]),
              float(args.test_frac or split["test"]))
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to one: {ratios}")
    results = [_result(seed, rows, policies, neighbors, ratios) for seed in range(args.seed_count)]
    results.sort(key=lambda item: (min(item["validation_fg_high"], item["test_fg_high"]), item["validation_fg_high"], item["test_fg_high"]), reverse=True)
    _write(results, args.output)
    print(results[:10])


if __name__ == "__main__":
    main()
