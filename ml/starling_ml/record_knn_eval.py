"""Live-training adapter for record-native Starling KNN evaluation."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .knn_data import FULL_METADATA_CONFIG, load_record_dataset, normalize_record_split
from .knn_pipeline import compute_metrics, predict_split
from .knn_retrieval import RetrievalConfig, build_or_load_candidate_cache, candidate_cache_paths, cache_metadata
from .knn_scorers import MLPTransferScorer


@dataclass(frozen=True)
class RecordKnnResult:
    macro_f1: float
    accuracy: float
    transfer_precision: float
    transfer_recall: float
    n_queries: int
    elapsed_seconds: float
    candidate_fraction: float
    n_candidates: int
    k: int


def record_knn_result_dict(result: RecordKnnResult) -> dict[str, float | int]:
    return {
        "macro_f1": result.macro_f1,
        "accuracy": result.accuracy,
        "transfer_precision": result.transfer_precision,
        "transfer_recall": result.transfer_recall,
        "n_queries": result.n_queries,
        "elapsed_seconds": result.elapsed_seconds,
        "candidate_fraction": result.candidate_fraction,
        "n_candidates": result.n_candidates,
        "k": result.k,
    }


def evaluate_record_knn(
    *,
    cfg: Config,
    model: Any,
    dataset_dir: str | Path,
    cache_dir: str | Path,
    split: str = "validation_1",
    dataset_config: str = FULL_METADATA_CONFIG,
    top_fraction: float = 0.10,
    top_n: int | None = None,
    k: int = 10,
    batch_size: int = 4096,
    max_queries: int = 0,
    require_cache: bool = False,
) -> RecordKnnResult:
    start = time.perf_counter()
    canonical = normalize_record_split(split)
    retrieval = RetrievalConfig(top_fraction=top_fraction, top_n=top_n, max_queries=max_queries)
    dataset = load_record_dataset(dataset_dir, canonical, config_name=dataset_config, max_queries=max_queries)
    candidates = build_or_load_candidate_cache(dataset, cache_dir, retrieval, require_existing=require_cache)
    preds = record_knn_predictions(cfg=cfg, model=model, dataset=dataset, candidates=candidates, k=k, batch_size=batch_size)
    labels = dataset.queries["label"].to_numpy(dtype=np.int8)
    metrics = compute_metrics(labels, np.asarray(preds, dtype=np.int8))
    return _result_from_metrics(metrics, candidates, top_fraction, k, start)


def record_knn_predictions(
    *,
    cfg: Config,
    model: Any,
    dataset: Any,
    candidates: Any,
    k: int,
    batch_size: int,
) -> list[int]:
    scorer = mlp_scorer_from_config(cfg, model, int(batch_size))
    scorer.validate(dataset, {})
    scorer.prepare(dataset.split)
    from .knn_pipeline import VoteConfig

    result = predict_split(dataset, candidates, scorer, VoteConfig(k=int(k)))
    return result.predictions.astype(np.int8).tolist()


def ensure_record_knn_cache(
    *,
    cfg: Config,
    dataset_dir: str | Path,
    cache_dir: str | Path,
    split: str,
    top_fraction: float,
    dataset_config: str = FULL_METADATA_CONFIG,
    top_n: int | None = None,
    max_queries: int = 0,
    require_existing: bool = False,
) -> Path:
    canonical = normalize_record_split(split)
    retrieval = RetrievalConfig(top_fraction=top_fraction, top_n=top_n, max_queries=max_queries)
    dataset = load_record_dataset(dataset_dir, canonical, config_name=dataset_config, max_queries=max_queries)
    candidates = build_or_load_candidate_cache(dataset, cache_dir, retrieval, require_existing=require_existing)
    return candidates.data_path or candidate_cache_paths(cache_dir, cache_metadata(dataset, retrieval))[0]


def _result_from_metrics(metrics: dict[str, Any], candidates: Any, fraction: float, k: int, start: float) -> RecordKnnResult:
    return RecordKnnResult(
        macro_f1=float(metrics["macro_f1"]),
        accuracy=float(metrics["accuracy"]),
        transfer_precision=float(metrics["transfer_precision"]),
        transfer_recall=float(metrics["transfer_recall"]),
        n_queries=int(metrics["n_queries"]),
        elapsed_seconds=float(time.perf_counter() - start),
        candidate_fraction=float(fraction),
        n_candidates=int(candidates.positions.shape[1]),
        k=int(k),
    )


def mlp_scorer_from_config(cfg: Config, model: Any, batch_size: int) -> MLPTransferScorer:
    emb_manifest = Path(cfg.paths.embeddings_dir) / "manifest.json"
    return MLPTransferScorer(
        model=model,
        source_value_scale=float(cfg.model.source_value_scale),
        batch_size=int(batch_size),
        embedding_manifest_path=emb_manifest,
        base_parquet=cfg.paths.base_parquet,
    )


def _load_checkpoint_state(checkpoint: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    state_path = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    return load_file(str(state_path)) if state_path.suffix == ".safetensors" else torch.load(state_path, map_location="cpu")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ml/configs/shared_eval_same_species_v2.yaml")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--top-fraction", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = Config.from_yaml(args.config)
    cache_path = _ensure_from_args(args, cfg)
    if args.cache_only:
        print(json.dumps({"cache": str(cache_path)}, indent=2, sort_keys=True))
        return
    result = _evaluate_checkpoint(args, cfg)
    print(json.dumps(result, indent=2, sort_keys=True))


def _ensure_from_args(args: argparse.Namespace, cfg: Config) -> Path:
    return ensure_record_knn_cache(
        cfg=cfg,
        dataset_dir=args.dataset_dir or cfg.train.record_knn_eval_dataset_dir,
        cache_dir=args.cache_dir or cfg.train.record_knn_eval_cache_dir,
        split=args.split or cfg.train.record_knn_eval_splits[0],
        dataset_config=args.dataset_config or cfg.train.record_knn_eval_dataset_config,
        top_fraction=_arg_float(args.top_fraction, cfg.train.record_knn_eval_top_fraction),
        top_n=args.top_n,
        max_queries=args.max_queries,
    )


def _evaluate_checkpoint(args: argparse.Namespace, cfg: Config) -> dict[str, Any]:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --cache-only is set")
    import torch
    from .model import build_model

    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(_load_checkpoint_state(Path(args.checkpoint)), strict=False)
    model.to(args.device if torch.cuda.is_available() else "cpu")
    result = evaluate_record_knn(
        cfg=cfg,
        model=model,
        dataset_dir=args.dataset_dir or cfg.train.record_knn_eval_dataset_dir,
        cache_dir=args.cache_dir or cfg.train.record_knn_eval_cache_dir,
        split=args.split or cfg.train.record_knn_eval_splits[0],
        dataset_config=args.dataset_config or cfg.train.record_knn_eval_dataset_config,
        top_fraction=_arg_float(args.top_fraction, cfg.train.record_knn_eval_top_fraction),
        top_n=args.top_n,
        k=int(args.k if args.k is not None else cfg.train.record_knn_eval_k),
        batch_size=int(args.batch_size if args.batch_size is not None else cfg.train.record_knn_eval_batch_size),
        max_queries=args.max_queries,
        require_cache=True,
    )
    payload = record_knn_result_dict(result)
    payload.update({"checkpoint": args.checkpoint, "missing": len(missing), "unexpected": len(unexpected)})
    return payload


def _arg_float(value: float | None, default: float) -> float:
    return float(value if value is not None else default)


if __name__ == "__main__":
    main()
