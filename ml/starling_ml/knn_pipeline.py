"""Shared record KNN evaluation pipeline."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .knn_data import (
    CANONICAL_QUERY_SPLITS,
    MISSING_FIELDS,
    RecordKnnDataset,
    load_record_dataset,
    normalize_record_splits,
    read_yaml,
    resolve_path,
)
from .knn_retrieval import CandidateSet, RetrievalConfig, build_or_load_candidate_cache
from .knn_scorers import CandidateBatch, KnnScorer, MLPTransferScorer, TanimotoScorer


@dataclass(frozen=True)
class VoteConfig:
    k: int
    weighting: str = "score"
    tie_policy: str = "positive"


@dataclass(frozen=True)
class PredictionResult:
    predictions: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True)
class SplitEvaluation:
    split: str
    scorer: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    predictions: PredictionResult
    elapsed_seconds: float


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float | int]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    labels = np.asarray(labels, dtype=np.int8)
    preds = np.asarray(preds, dtype=np.int8)
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "transfer_precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "transfer_recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "n_queries": int(len(labels)),
    }


def weighted_vote(labels: np.ndarray, weights: np.ndarray, tie_policy: str = "positive") -> tuple[int, float]:
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not len(labels):
        return 0, 0.0
    if (weights <= 0).all() or not np.isfinite(weights).all():
        prob = float(labels.mean())
    else:
        clean = np.clip(weights, 0.0, None)
        prob = float(np.dot(labels, clean) / max(float(clean.sum()), 1e-12))
    pred = int(prob >= 0.5) if tie_policy == "positive" else int(prob > 0.5)
    return pred, prob


def evaluate_split(
    dataset: RecordKnnDataset,
    candidates: CandidateSet,
    scorer: KnnScorer,
    vote: VoteConfig,
) -> SplitEvaluation:
    start = time.perf_counter()
    scorer.validate(dataset, {})
    scorer.prepare(dataset.split)
    result = predict_split(dataset, candidates, scorer, vote)
    labels = dataset.queries["label"].to_numpy(dtype=np.int8)
    metrics = compute_metrics(labels, result.predictions)
    params = output_params(scorer, candidates, vote)
    return SplitEvaluation(dataset.split, scorer.name, params, metrics, result, time.perf_counter() - start)


def predict_split(
    dataset: RecordKnnDataset,
    candidates: CandidateSet,
    scorer: KnnScorer,
    vote: VoteConfig,
) -> PredictionResult:
    preds: list[int] = []
    probs: list[float] = []
    for query_pos in range(len(dataset.queries)):
        pred, prob = predict_query(dataset, candidates, scorer, vote, query_pos)
        preds.append(pred)
        probs.append(prob)
    return PredictionResult(np.asarray(preds, dtype=np.int8), np.asarray(probs, dtype=np.float32))


def predict_query(
    dataset: RecordKnnDataset,
    candidates: CandidateSet,
    scorer: KnnScorer,
    vote: VoteConfig,
    query_pos: int,
) -> tuple[int, float]:
    positions = candidates.positions[query_pos].astype(np.int64)
    batch = CandidateBatch(dataset, query_pos, positions, candidates.similarities[query_pos])
    scores = scorer.score_candidates(batch)
    order = np.argsort(scores)[::-1][: int(vote.k)]
    top_pos = positions[order]
    top_scores = scores[order] if vote.weighting == "score" else np.ones(len(order))
    labels = dataset.sources.iloc[top_pos]["label"].to_numpy(dtype=np.int8)
    return weighted_vote(labels, top_scores, vote.tie_policy)


def output_params(scorer: KnnScorer, candidates: CandidateSet, vote: VoteConfig) -> dict[str, Any]:
    top_policy = candidates.metadata["top_policy"]
    return {
        "k": int(vote.k),
        "candidate_fraction": top_policy.get("top_fraction"),
        "top_n": top_policy.get("top_n"),
        "vote_weighting": vote.weighting,
        "tie_policy": vote.tie_policy,
        "n_candidates": int(candidates.metadata["n_candidates"]),
        "retrieval_cache": Path(candidates.data_path).name if candidates.data_path else "",
        "scorer": scorer.name,
    }


def ensure_retrieval_cache(
    dataset_dir: str | Path,
    cache_dir: str | Path,
    split: str,
    config: RetrievalConfig,
    *,
    require_existing: bool = False,
) -> CandidateSet:
    dataset = load_record_dataset(dataset_dir, split, max_queries=config.max_queries)
    return build_or_load_candidate_cache(dataset, cache_dir, config, require_existing=require_existing)


def run_multi_split_eval(
    *,
    dataset_dir: str | Path,
    cache_dir: str | Path,
    splits: list[str],
    retrieval: RetrievalConfig,
    scorer: KnnScorer,
    vote: VoteConfig,
    require_cache: bool = False,
) -> list[SplitEvaluation]:
    results: list[SplitEvaluation] = []
    for split in normalize_record_splits(splits):
        dataset = load_record_dataset(dataset_dir, split, max_queries=retrieval.max_queries)
        candidates = build_or_load_candidate_cache(dataset, cache_dir, retrieval, require_existing=require_cache)
        results.append(evaluate_split(dataset, candidates, scorer, vote))
    return results


def tune_and_evaluate(
    *,
    dataset_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    scorers: list[KnnScorer],
    k_values: list[int],
    candidate_fractions: list[float],
    final_splits: list[str] | None = None,
    max_queries: int = 0,
    require_cache: bool = False,
) -> dict[str, Any]:
    final_splits = final_splits or list(CANONICAL_QUERY_SPLITS)
    rows, slice_rows, pred_rows, best = [], [], [], {}
    for scorer in scorers:
        selected = tune_scorer(dataset_dir, cache_dir, scorer, k_values, candidate_fractions, max_queries, require_cache)
        best[scorer.name] = selected
        append_grid_rows(rows, selected.pop("_grid_rows"))
        for split in normalize_record_splits(final_splits):
            result = run_selected(dataset_dir, cache_dir, split, scorer, selected, max_queries, require_cache)
            phase = "tune_selected" if split == "validation_1" else "frozen_eval"
            rows.append(metric_row(result, phase))
            slice_rows.extend(slice_metric_rows(result, load_record_dataset(dataset_dir, split, max_queries=max_queries)))
            pred_rows.extend(prediction_rows(result, load_record_dataset(dataset_dir, split, max_queries=max_queries)))
    write_pipeline_outputs(output_dir, rows, slice_rows, pred_rows, best)
    return {"output_dir": str(output_dir), "best_params": best}


def tune_scorer(
    dataset_dir: str | Path,
    cache_dir: str | Path,
    scorer: KnnScorer,
    k_values: list[int],
    fractions: list[float],
    max_queries: int,
    require_cache: bool,
) -> dict[str, Any]:
    grid_rows: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    for frac in fractions_for_scorer(scorer, fractions):
        retrieval = RetrievalConfig(top_fraction=float(frac), max_queries=max_queries)
        dataset = load_record_dataset(dataset_dir, "validation_1", max_queries=max_queries)
        candidates = build_or_load_candidate_cache(dataset, cache_dir, retrieval, require_existing=require_cache)
        for k in k_values:
            result = evaluate_split(dataset, candidates, scorer, VoteConfig(k=int(k)))
            row = metric_row(result, "tune_grid")
            grid_rows.append(row)
            best_result = choose_better(best_result, row)
    if best_result is None:
        raise RuntimeError(f"no tuning result for scorer {scorer.name}")
    return selected_params(best_result, grid_rows)


def fractions_for_scorer(scorer: KnnScorer, fractions: list[float]) -> list[float]:
    if isinstance(scorer, TanimotoScorer):
        return [1.0]
    return [float(value) for value in fractions]


def choose_better(current: dict[str, Any] | None, row: dict[str, Any]) -> dict[str, Any]:
    if current is None:
        return row
    left = (float(row["macro_f1"]), float(row["accuracy"]))
    right = (float(current["macro_f1"]), float(current["accuracy"]))
    return row if left > right else current


def selected_params(best_row: dict[str, Any], grid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("k", "candidate_fraction", "top_n", "vote_weighting", "tie_policy")
    selected = {key: best_row[key] for key in keys if key in best_row and best_row[key] not in ("", None)}
    selected["_grid_rows"] = grid_rows
    return selected


def run_selected(
    dataset_dir: str | Path,
    cache_dir: str | Path,
    split: str,
    scorer: KnnScorer,
    params: dict[str, Any],
    max_queries: int,
    require_cache: bool,
) -> SplitEvaluation:
    retrieval = RetrievalConfig(top_fraction=float(params.get("candidate_fraction", 1.0)), max_queries=max_queries)
    dataset = load_record_dataset(dataset_dir, split, max_queries=max_queries)
    candidates = build_or_load_candidate_cache(dataset, cache_dir, retrieval, require_existing=require_cache)
    return evaluate_split(dataset, candidates, scorer, VoteConfig(k=int(params["k"])))


def metric_row(result: SplitEvaluation, phase: str) -> dict[str, Any]:
    return {"split": result.split, "method": result.scorer, "phase": phase, **result.params, **result.metrics}


def prediction_rows(result: SplitEvaluation, dataset: RecordKnnDataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query, pred, prob in zip(
        dataset.queries.itertuples(index=False),
        result.predictions.predictions,
        result.predictions.probabilities,
        strict=True,
    ):
        rows.append(prediction_row(result, query, int(pred), float(prob)))
    return rows


def prediction_row(result: SplitEvaluation, query: Any, pred: int, prob: float) -> dict[str, Any]:
    return {
        "split": result.split,
        "method": result.scorer,
        "k": result.params["k"],
        "candidate_fraction": result.params.get("candidate_fraction"),
        "query_record_id": query.record_key,
        "query_row_index": int(query.row_index),
        "label": int(query.label),
        "prediction": pred,
        "prob_ge20": prob,
        "ob_bin": query.ob_bin,
        "missing_count_bucket": query.missing_count_bucket,
    }


def slice_metric_rows(result: SplitEvaluation, dataset: RecordKnnDataset) -> list[dict[str, Any]]:
    labels = dataset.queries["label"].to_numpy(dtype=np.int8)
    preds = result.predictions.predictions
    rows = [slice_row(result, "overall", "all", labels, preds)]
    for column in ("ob_bin", "missing_count_bucket"):
        rows.extend(slice_column(result, dataset, column, preds))
    if "metadata_missing_mask" in dataset.queries:
        rows.extend(missing_field_slices(result, dataset, preds))
    return rows


def slice_column(result: SplitEvaluation, dataset: RecordKnnDataset, column: str, preds: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = dataset.queries[column].astype(str)
    labels = dataset.queries["label"].to_numpy(dtype=np.int8)
    for value in sorted(set(values)):
        mask = np.asarray(values == value)
        rows.append(slice_row(result, column, str(value), labels[mask], preds[mask]))
    return rows


def missing_field_slices(result: SplitEvaluation, dataset: RecordKnnDataset, preds: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = dataset.queries["label"].to_numpy(dtype=np.int8)
    for pos, field in enumerate(MISSING_FIELDS):
        values = dataset.queries["metadata_missing_mask"].str[pos].map({"1": "true", "0": "false"})
        for value in sorted(set(values)):
            mask = np.asarray(values == value)
            rows.append(slice_row(result, f"{field}_missing", str(value), labels[mask], preds[mask]))
    return rows


def slice_row(result: SplitEvaluation, name: str, value: str, labels: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    return {
        "split": result.split,
        "method": result.scorer,
        "slice": name,
        "slice_value": value,
        **result.params,
        **compute_metrics(labels, preds),
    }


def append_grid_rows(rows: list[dict[str, Any]], grid_rows: list[dict[str, Any]]) -> None:
    rows.extend(grid_rows)


def write_pipeline_outputs(
    output_dir: str | Path,
    metrics: list[dict[str, Any]],
    slices: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    best: dict[str, Any],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_params.json").write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
    write_csv(out / "metrics.csv", metrics)
    write_csv(out / "metrics_by_slice.csv", slices)
    write_csv(out / "query_predictions.csv", predictions)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = fieldnames_for(rows)
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fieldnames_for(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def timestamped_run_dir(root: str | Path, run_name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_name)
    out = Path(root) / f"{time.strftime('%Y-%m-%d')}_{safe}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def load_starling_model(config_path: str | Path, checkpoint: str | Path, device: str) -> tuple[Any, Any]:
    import torch
    from safetensors.torch import load_file

    from .config import Config
    from .model import build_model

    cfg = Config.from_yaml(str(config_path))
    model = build_model(cfg)
    state_path = Path(checkpoint) / "model.safetensors" if Path(checkpoint).is_dir() else Path(checkpoint)
    state = load_file(str(state_path)) if state_path.suffix == ".safetensors" else torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return cfg, model.to(device).eval()


def scorer_from_method(method: str, args: Any, cfg: dict[str, Any], config_path: Path) -> KnnScorer:
    if method == "tanimoto_knn":
        return TanimotoScorer()
    if method != "starling_rerank_knn":
        raise ValueError(f"unknown KNN method {method!r}")
    device = default_device(getattr(args, "device", None))
    checkpoint = getattr(args, "checkpoint", None) or cfg.get("knn", {}).get("checkpoint")
    if not checkpoint:
        raise ValueError("starling_rerank_knn requires --checkpoint or knn.checkpoint")
    model_cfg, model = load_starling_model(config_path, checkpoint, device)
    batch_size = int(getattr(args, "batch_size", None) or cfg.get("knn", {}).get("batch_size", 4096))
    emb_manifest = Path(model_cfg.paths.embeddings_dir) / "manifest.json"
    return MLPTransferScorer(
        model,
        float(model_cfg.model.source_value_scale),
        batch_size,
        device,
        config_path,
        checkpoint,
        emb_manifest,
        model_cfg.paths.base_parquet,
    )


def default_device(value: str | None) -> str:
    if value:
        return value
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
