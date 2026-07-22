#!/usr/bin/env python3
"""Create the shared deterministic 5,000-update V6 100M group schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starling_ml.v6_mlp_100m_data import build_group_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-cache", type=Path, default=Path(
        "ml/artifacts/v6_5_molformer_pubmedbert_100m/pair_cache"))
    parser.add_argument("--output", type=Path, default=Path(
        "ml/artifacts/v6_5_molformer_pubmedbert_100m/group_schedule.npy"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-groups", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=4878)
    parser.add_argument("--sampling-mode", choices=("with_replacement", "one_pass"),
                        default="with_replacement")
    args = parser.parse_args()
    result = build_group_schedule(args.pair_cache, args.output, args.steps,
                                  args.batch_groups, args.group_size, args.seed,
                                  args.sampling_mode)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
