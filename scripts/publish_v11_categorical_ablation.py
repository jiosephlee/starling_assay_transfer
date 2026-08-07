#!/usr/bin/env python3
"""Publish and remotely verify the six public V11 categorical-ablation datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from pipeline.v3_policy import file_sha256
from scripts.build_v11_categorical_ablation import (
    DEFAULT_OUTPUT_ROOT,
    HUB_DATASET_IDS,
    VARIANTS,
)


RELEASE_FILES = {
    "README.md", "manifest.json", "train/data.parquet",
    "validation/data.parquet", "validation_ranking/data.parquet",
    "test/data.parquet", "test_ranking/data.parquet",
}


def _local_files(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def _publish_one(api: HfApi, root: Path, repo_id: str) -> str:
    files = _local_files(root)
    if files != RELEASE_FILES:
        raise ValueError(f"{repo_id}: unexpected local release topology: {sorted(files)}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(root), repo_id=repo_id, repo_type="dataset",
        commit_message="Publish V11 categorical-ablation Intern dataset",
    )
    return str(commit.oid)


def _lfs_sha256(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if not lfs:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def _verify_remote(api: HfApi, root: Path, repo_id: str) -> dict[str, Any]:
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True)
    if info.private:
        raise AssertionError(f"{repo_id}: dataset is not public")
    siblings = {item.rfilename: item for item in info.siblings}
    if set(siblings) != RELEASE_FILES | {".gitattributes"}:
        raise AssertionError(f"{repo_id}: remote topology mismatch")
    for relative in sorted(name for name in RELEASE_FILES if name.endswith(".parquet")):
        if _lfs_sha256(siblings[relative]) != file_sha256(root / relative):
            raise AssertionError(f"{repo_id}/{relative}: remote LFS hash mismatch")
    for relative in ("README.md", "manifest.json"):
        downloaded = Path(hf_hub_download(repo_id, relative, repo_type="dataset"))
        if file_sha256(downloaded) != file_sha256(root / relative):
            raise AssertionError(f"{repo_id}/{relative}: remote content mismatch")
    return {"repo_id": repo_id, "commit": info.sha, "public": not info.private}


def publish_tasks(output_root: Path, task_ids: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    api = HfApi()
    releases = []
    for task_id in task_ids:
        if task_id not in HUB_DATASET_IDS:
            raise ValueError(f"unknown V11 release task: {task_id}")
        repos = HUB_DATASET_IDS[task_id]
        for variant in VARIANTS:
            root = output_root / task_id / variant
            commit = _publish_one(api, root, repos[variant])
            verified = _verify_remote(api, root, repos[variant])
            releases.append({**verified, "upload_commit": commit})
    return releases


def publish_all(output_root: Path = DEFAULT_OUTPUT_ROOT) -> list[dict[str, Any]]:
    return publish_tasks(output_root, tuple(HUB_DATASET_IDS))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--task", action="append", choices=sorted(HUB_DATASET_IDS),
        help="publish only this task; repeat for multiple tasks (default: all tasks)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = args.task or list(HUB_DATASET_IDS)
    releases = publish_tasks(args.output_root, tasks)
    print(json.dumps(releases, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
