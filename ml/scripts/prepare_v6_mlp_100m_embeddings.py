#!/usr/bin/env python3
"""Precompute the immutable V6 MoLFormer and PubMedBERT embedding cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starling_ml.v6_embedding_cache import build_embedding_cache, reindex_embedding_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path(
        "datasets/parquet/assay_transfer_raw_pair_v6_5_mlp/records.parquet"))
    parser.add_argument("--output", type=Path, default=Path(
        "ml/artifacts/v6_5_molformer_pubmedbert_100m/embedding_cache"))
    parser.add_argument("--reuse-cache", type=Path)
    parser.add_argument("--reuse-records", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if bool(args.reuse_cache) != bool(args.reuse_records):
        parser.error("--reuse-cache and --reuse-records must be provided together")
    result = reindex_embedding_cache(args.reuse_cache, args.reuse_records,
                                     args.records, args.output) if args.reuse_cache else \
        build_embedding_cache(args.records, args.output, args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
