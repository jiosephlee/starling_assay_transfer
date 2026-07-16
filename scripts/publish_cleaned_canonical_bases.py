#!/usr/bin/env python3
"""Stage and optionally publish fully normalized canonical bases to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.source_normalization.hf_export import upload_export, write_all_clean_exports  # noqa: E402
from pipeline.source_normalization.io import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/source_normalization_v1.yaml")
    parser.add_argument("--base-root", type=Path, default=REPO_ROOT / "datasets/base/canonical_endpoints_v1")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "datasets/base/hf_cleaned")
    parser.add_argument("--namespace", default="jiosephlee")
    parser.add_argument("--upload", action="store_true", help="Create/update public Hugging Face datasets.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def publish(exports: list[dict], namespace: str) -> list[dict]:
    uploaded = []
    for export in exports:
        repo_id = f"{namespace}/{export['source_name']}_cleaned"
        revision = upload_export(export["output"], repo_id)
        payload = {"repo_id": repo_id, "dataset_content_commit": revision}
        write_json(payload, export["output"] / "metadata" / "upload.json")
        uploaded.append(payload)
    return uploaded


def main() -> None:
    args = parse_args()
    exports = write_all_clean_exports(load_config(args.config), args.base_root, args.output_root)
    result = {"staged": [{"source_name": item["source_name"], "rows": item["rows"]} for item in exports]}
    if args.upload:
        result["uploaded"] = publish(exports, args.namespace)
    print(result)


if __name__ == "__main__":
    main()
