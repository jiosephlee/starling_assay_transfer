#!/usr/bin/env python3
"""Upload a local dataset folder to a Hugging Face dataset repository."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_transfer import upload_folder_to_hf  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--path-in-repo", default=None)
    parser.add_argument("--commit-message", default="Upload dataset artifacts")
    args = parser.parse_args()
    if not args.folder.exists() or not args.folder.is_dir():
        parser.error("--folder must be an existing directory")
    return args


def main() -> None:
    args = parse_args()
    commit = upload_folder_to_hf(
        folder_path=args.folder,
        repo_id=args.repo_id,
        private=args.private,
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
    )
    print(json.dumps({"repo_id": args.repo_id, "folder": str(args.folder), "commit": commit}, indent=2))


if __name__ == "__main__":
    main()
