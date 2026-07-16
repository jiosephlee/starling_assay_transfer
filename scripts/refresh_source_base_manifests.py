#!/usr/bin/env python3
"""Refresh canonical-base manifests after relocating unchanged pinned inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.source_normalization import refresh_manifests  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/source_normalization_v1.yaml")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = refresh_manifests(args.config, args.data_root, args.output_root, args.report)
    for manifest in manifests:
        print(f"refreshed {manifest['source_name']}: {manifest['stage_yields']['accepted_base_children']} base children")


if __name__ == "__main__":
    main()
