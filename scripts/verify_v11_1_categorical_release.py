#!/usr/bin/env python3
"""Verify V11.1 release targets and shared component contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.v11_contract import DEFAULT_REGISTRY
from scripts import verify_v11_categorical_ablation as component_verify
from scripts.verify_v11_1_intern_raw_pair import verify


def _verify_v11_1_columns(root: Path) -> None:
    for split in ("train", "validation_ranking"):
        columns = set(pq.read_schema(root / split / "data.parquet").names)
        required = {"percentile_distance", "percentile_distance_cdf", "target_a", "target_b"}
        if required - columns:
            raise AssertionError(f"{split}: missing V11.1 target audit columns")
        if "target_z" in columns:
            raise AssertionError(f"{split}: V11.1 schema must not contain target_z")


def verify_release(root: Path, source: Path, registry: Path = DEFAULT_REGISTRY) -> None:
    manifest = component_verify._manifest(root)
    verify(root, source, registry)
    component_verify._verify_components(root, manifest)
    component_verify._verify_ranking_diagnostics(manifest)
    component_verify._verify_gold(manifest, registry)
    component_verify._verify_downstream_columns(root)
    _verify_v11_1_columns(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    verify_release(args.root, args.source, args.registry)
    print("V11.1 release verification passed")


if __name__ == "__main__":
    main()
