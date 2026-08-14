#!/usr/bin/env python3
"""Publish and remotely verify the three V11.1 task datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from scripts.build_v11_1_categorical_release import (
    CALIBRATION_FILE, HUB_DATASET_IDS, VARIANT,
)
from scripts.build_v11_1_from_defined_split import MEMBERSHIP_CONTRACT
from scripts.build_v11_1_intern_raw_pair import DEFAULT_OUTPUT_ROOT
from scripts.publish_v11_categorical_ablation import _publish_one, _verify_remote


RELEASE_FILES = {
    "README.md", "manifest.json", "train/data.parquet",
    "validation_ranking/data.parquet", CALIBRATION_FILE,
}


def publish_tasks(output_root: Path, task_ids: list[str]) -> list[dict[str, Any]]:
    api, releases = HfApi(), []
    for task_id in task_ids:
        root, repo_id = output_root / task_id / VARIANT, HUB_DATASET_IDS[task_id]
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        membership = manifest.get("reference_membership", {})
        if membership.get("schema_version") != MEMBERSHIP_CONTRACT:
            raise ValueError(f"{task_id}: refusing to publish non-frozen V11.1 membership")
        commit = _publish_one(
            api, root, repo_id, RELEASE_FILES, set(),
            "Publish V11.1 percentile-distance-CDF dataset",
        )
        verified = _verify_remote(api, root, repo_id, RELEASE_FILES)
        releases.append({**verified, "upload_commit": commit})
    return releases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task", action="append", choices=sorted(HUB_DATASET_IDS))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = args.task or list(HUB_DATASET_IDS)
    print(json.dumps(publish_tasks(args.output_root, tasks), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
