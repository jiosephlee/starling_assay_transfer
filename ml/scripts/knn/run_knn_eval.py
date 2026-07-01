#!/usr/bin/env python3
"""Thin CLI for record-native KNN evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starling_ml.knn_data import CANONICAL_QUERY_SPLITS, normalize_record_splits, read_yaml, resolve_path  # noqa: E402
from starling_ml.knn_pipeline import scorer_from_method, timestamped_run_dir, tune_and_evaluate  # noqa: E402
from starling_ml.knn_retrieval import RetrievalConfig  # noqa: E402
from starling_ml.record_knn_eval import ensure_record_knn_cache  # noqa: E402
from starling_ml.config import Config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ml/configs/knn_condition_key_v3.yaml")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--dataset-config")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--run-name", default="smoke")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--candidate-fractions", nargs="+", type=float, default=None)
    parser.add_argument("--k-values", nargs="+", type=int, default=None)
    parser.add_argument("--final-splits", nargs="+", default=None)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-queries-per-split", type=int, default=0)
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def cfg_get(cfg: dict[str, Any], args: argparse.Namespace, key: str, default: Any = None) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    return cfg.get("knn", {}).get(key, default)


def run() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = read_yaml(config_path)
    root = config_path.parents[2]
    dataset_dir = resolve_path(cfg_get(cfg, args, "dataset_dir"), root)
    cache_dir = resolve_path(cfg_get(cfg, args, "cache_dir", "ml/artifacts/record_knn_eval_cache"), root)
    if args.cache_only:
        print(json.dumps(build_caches(args, cfg, dataset_dir, cache_dir), indent=2, sort_keys=True))
        return
    output_root = resolve_path(cfg_get(cfg, args, "output_root"), root)
    run_dir = timestamped_run_dir(output_root, args.run_name)
    result = run_pipeline(args, cfg, config_path, dataset_dir, cache_dir, run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


def build_caches(args: argparse.Namespace, cfg: dict[str, Any], dataset_dir: Path, cache_dir: Path) -> dict[str, Any]:
    train_cfg = Config.from_yaml(args.config)
    fractions = [float(v) for v in cfg_get(cfg, args, "candidate_fractions", [0.10])]
    splits = normalize_record_splits(args.final_splits or list(CANONICAL_QUERY_SPLITS))
    paths: list[str] = []
    for split in splits:
        for fraction in fractions:
            path = ensure_record_knn_cache(
                cfg=train_cfg,
                dataset_dir=dataset_dir,
                cache_dir=cache_dir,
                split=split,
                dataset_config=cfg_get(cfg, args, "dataset_config", "full_metadata"),
                top_fraction=fraction,
                max_queries=int(args.max_queries_per_split or 0),
            )
            paths.append(str(path))
    return {"caches": paths}


def run_pipeline(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    config_path: Path,
    dataset_dir: Path,
    cache_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    methods = cfg_get(cfg, args, "methods", ["tanimoto_knn", "starling_rerank_knn"])
    scorers = [scorer_from_method(method, args, cfg, config_path) for method in methods]
    fractions = [float(v) for v in cfg_get(cfg, args, "candidate_fractions", [0.01, 0.05, 0.10])]
    k_values = [int(v) for v in cfg_get(cfg, args, "k_values", [1, 3, 5, 10])]
    final_splits = normalize_record_splits(args.final_splits or list(CANONICAL_QUERY_SPLITS))
    return tune_and_evaluate(
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        output_dir=run_dir,
        scorers=scorers,
        k_values=k_values,
        candidate_fractions=fractions,
        final_splits=final_splits,
        max_queries=int(args.max_queries_per_split or 0),
        require_cache=False,
    )


if __name__ == "__main__":
    run()
