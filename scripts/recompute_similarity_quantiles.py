#!/usr/bin/env python3
"""Recompute and save similarity quantile metadata for compact pair shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from create_splits_from_compact_pairs import (  # noqa: E402
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    compute_similarity_thresholds,
    write_similarity_quantile_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--similarity-buckets", type=int, default=5)
    parser.add_argument("--max-quantile-values", type=int, default=100_000_000)
    parser.add_argument("--progress-every-seconds", type=float, default=60.0)
    parser.add_argument("--similarity-thresholds", type=float, nargs="+", default=None)
    args = parser.parse_args()
    if args.similarity_buckets < 2:
        parser.error("--similarity-buckets must be at least 2")
    if args.max_quantile_values < 1:
        parser.error("--max-quantile-values must be positive")
    if args.similarity_thresholds is not None and len(args.similarity_thresholds) != args.similarity_buckets - 1:
        parser.error("--similarity-thresholds must provide similarity_buckets - 1 values")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = compute_similarity_thresholds(args)
    write_similarity_quantile_metadata(args, thresholds)
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "similarity_quantiles.json"),
                "similarity_quantile_thresholds": thresholds,
                "similarity_quantile_estimation": {
                    "mode": getattr(args, "_quantile_sample_strategy", "proportional_parquet_sample"),
                    "values_seen": int(getattr(args, "_quantile_values_seen", 0)),
                    "values_sampled": int(getattr(args, "_quantile_values_sampled", 0)),
                    "files_read": int(getattr(args, "_quantile_files_read", 0)),
                    "row_groups_read": int(getattr(args, "_quantile_row_groups_read", 0)),
                    "max_quantile_values": args.max_quantile_values,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
