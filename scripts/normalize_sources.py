#!/usr/bin/env python3
"""Normalize Q1--Q4 and Starling into round-one artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rdkit import RDLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.source_normalization import run_normalization  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/source_normalization_v1.yaml")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    manifests = run_normalization(args.config, args.data_root, args.output_root, args.report)
    for manifest in manifests:
        counts = manifest["counts"]
        print(f"{manifest['source_id']}: {counts['accepted']} accepted, {counts['rejected']} rejected")


if __name__ == "__main__":
    main()
