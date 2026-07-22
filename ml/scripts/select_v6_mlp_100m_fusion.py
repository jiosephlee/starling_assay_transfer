#!/usr/bin/env python3
"""Select the V6.5 fusion winner using fixed validation-ranking queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starling_ml.v6_mlp_selection import select_fusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(
        "ml/artifacts/v6_5_molformer_pubmedbert_100m/runs"))
    parser.add_argument("--output", type=Path, default=Path(
        "ml/artifacts/v6_5_molformer_pubmedbert_100m/fusion_selection.json"))
    parser.add_argument("--step", type=int, default=1000)
    args = parser.parse_args()
    result = select_fusion(args.runs, args.output, args.step)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
