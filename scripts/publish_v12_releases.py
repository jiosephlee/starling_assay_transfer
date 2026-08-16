#!/usr/bin/env python3
"""Verify, publish, and remotely verify all V12 or V12.1 task datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from scripts.build_v12_release import SPLITS, VARIANT, VERSION as V12_VERSION
from scripts.build_v12_1_from_defined_split import VERSION as V12_1_VERSION
from scripts.publish_v11_categorical_ablation import _publish_one, _verify_remote
from scripts.verify_v12_release import verify


TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction")
RELEASE_FILES = {
    "README.md", "manifest.json", "train/data.parquet",
    "validation_ranking/data.parquet", "test_ranking/data.parquet",
    "calibration/split_manifest.json", "calibration/train_value_calibration.json.gz",
    "calibration/train_percentile_distance_calibration.json.gz",
}


def hub_dataset_ids(version: str) -> dict[str, str]:
    if version not in {"v12", "v12.1"}:
        raise ValueError("publish version must be v12 or v12.1")
    return {
        task: (
            f"jiosephlee/assay-transfer-raw-pair-{version}-"
            f"{task.replace('_', '-')}-with-categorical-intern"
        )
        for task in TASKS
    }


def _expected_version(version: str) -> str:
    return V12_VERSION if version == "v12" else V12_1_VERSION


def _validate_release(
    version: str, root: Path, source: Path, reference: Path | None,
) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != _expected_version(version):
        raise ValueError(f"{root}: unexpected release version")
    if set(manifest.get("paths", {})) != set(SPLITS):
        raise ValueError(f"{root}: split topology is incomplete")
    if {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()} != RELEASE_FILES:
        raise ValueError(f"{root}: local release file topology is not canonical")
    return verify(root, source, reference)


def publish_tasks(
    version: str, release_root: Path, source_root: Path,
    reference_v12_root: Path | None = None, tasks: tuple[str, ...] = TASKS,
) -> list[dict[str, Any]]:
    api, receipts = HfApi(), []
    ids = hub_dataset_ids(version)
    for task in tasks:
        root = release_root / task / VARIANT
        source = source_root / task / "records.parquet"
        reference = (
            reference_v12_root / task / VARIANT if reference_v12_root is not None else None
        )
        local_verification = _validate_release(version, root, source, reference)
        commit = _publish_one(
            api, root, ids[task], RELEASE_FILES, set(),
            f"Publish assay-transfer {version} leakage-safe task dataset",
        )
        remote = _verify_remote(api, root, ids[task], RELEASE_FILES)
        receipts.append({
            **remote, "upload_commit": commit,
            "local_verification": local_verification,
        })
    return receipts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, choices=("v12", "v12.1"))
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-v12-root", type=Path)
    parser.add_argument(
        "--task", action="append", choices=TASKS,
        help="publish only this task; repeat for more than one (default: all three)",
    )
    args = parser.parse_args()
    if args.version == "v12.1" and args.reference_v12_root is None:
        parser.error("--reference-v12-root is required for V12.1")
    if args.version == "v12" and args.reference_v12_root is not None:
        parser.error("--reference-v12-root applies only to V12.1")
    return args


def main() -> None:
    args = _parse_args()
    receipts = publish_tasks(
        args.version, args.release_root, args.source_root, args.reference_v12_root,
        tuple(args.task or TASKS),
    )
    print(json.dumps(receipts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
