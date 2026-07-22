#!/usr/bin/env python3
"""Create the immutable V6.5 pair-only cache used by the 100M MLP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starling_ml.v6_mlp_100m_data import build_pair_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(
        "datasets/parquet/assay_transfer_raw_pair_v6_5_mlp"))
    parser.add_argument("--output", type=Path, default=Path(
        "ml/artifacts/v6_5_molformer_pubmedbert_100m/pair_cache"))
    args = parser.parse_args()
    print(json.dumps(build_pair_cache(args.dataset, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
