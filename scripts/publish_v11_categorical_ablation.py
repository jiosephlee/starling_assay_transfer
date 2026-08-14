#!/usr/bin/env python3
"""Publish and remotely verify the three V11 with-categorical datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import (
    CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download,
)

from pipeline.v3_policy import file_sha256
from scripts.build_v11_categorical_ablation import DEFAULT_OUTPUT_ROOT, HUB_DATASET_IDS, VARIANT


RELEASE_FILES = {
    "README.md", "manifest.json", "train/data.parquet",
    "validation_ranking/data.parquet",
}
OBSOLETE_FILES = {
    "validation/data.parquet", "test/data.parquet", "test_ranking/data.parquet",
}


def _local_files(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def _remote_files(api: HfApi, repo_id: str) -> set[str]:
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True)
    return {item.rfilename for item in info.siblings}


def _publish_one(
    api: HfApi, root: Path, repo_id: str,
    release_files: set[str] = RELEASE_FILES, obsolete_files: set[str] = OBSOLETE_FILES,
    commit_message: str = "Publish V11 value-CDF with-categorical dataset",
) -> str:
    files = _local_files(root)
    if files != release_files:
        raise ValueError(f"{repo_id}: unexpected local release topology: {sorted(files)}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    remote = _remote_files(api, repo_id)
    allowed = release_files | obsolete_files | {".gitattributes"}
    unexpected = remote - allowed
    if unexpected:
        raise ValueError(f"{repo_id}: refusing to delete unknown remote files: {sorted(unexpected)}")
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(root / name))
        for name in sorted(release_files)
    ]
    operations.extend(
        CommitOperationDelete(path_in_repo=name) for name in sorted(remote & obsolete_files)
    )
    commit = api.create_commit(
        repo_id=repo_id, repo_type="dataset", operations=operations,
        commit_message=commit_message,
    )
    return str(commit.oid)


def _lfs_sha256(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if not lfs:
        return None
    return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)


def _verify_remote(
    api: HfApi, root: Path, repo_id: str, release_files: set[str] = RELEASE_FILES,
) -> dict[str, Any]:
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True)
    if info.private:
        raise AssertionError(f"{repo_id}: dataset is not public")
    siblings = {item.rfilename: item for item in info.siblings}
    if set(siblings) != release_files | {".gitattributes"}:
        raise AssertionError(f"{repo_id}: remote topology mismatch")
    for relative in sorted(name for name in release_files if name.endswith((".parquet", ".json.gz"))):
        remote_hash = _lfs_sha256(siblings[relative])
        if remote_hash is None:
            downloaded = Path(hf_hub_download(repo_id, relative, repo_type="dataset"))
            remote_hash = file_sha256(downloaded)
        if remote_hash != file_sha256(root / relative):
            raise AssertionError(f"{repo_id}/{relative}: remote LFS hash mismatch")
    for relative in ("README.md", "manifest.json"):
        downloaded = Path(hf_hub_download(repo_id, relative, repo_type="dataset"))
        if file_sha256(downloaded) != file_sha256(root / relative):
            raise AssertionError(f"{repo_id}/{relative}: remote content mismatch")
    return {"repo_id": repo_id, "commit": info.sha, "public": not info.private}


def publish_tasks(output_root: Path, task_ids: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    api, releases = HfApi(), []
    for task_id in task_ids:
        if task_id not in HUB_DATASET_IDS:
            raise ValueError(f"unknown V11 release task: {task_id}")
        root, repo_id = output_root / task_id / VARIANT, HUB_DATASET_IDS[task_id]
        upload_commit = _publish_one(api, root, repo_id)
        releases.append({**_verify_remote(api, root, repo_id), "upload_commit": upload_commit})
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
    print(json.dumps(
        publish_tasks(args.output_root, args.task or list(HUB_DATASET_IDS)),
        indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
