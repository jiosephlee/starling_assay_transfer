"""Frozen ordinary and ranking metric contract for V6 direct MLPs."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .v6_embedding_cache import CONCEPTS


def _f1(predicted: np.ndarray, labels: np.ndarray, positive: bool) -> float:
    tp = np.sum((predicted == positive) & (labels == positive))
    fp = np.sum((predicted == positive) & (labels != positive))
    fn = np.sum((predicted != positive) & (labels == positive))
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return float(2 * precision * recall / max(1e-12, precision + recall))


def _ordinary_slice(probability: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    labels, predicted = target >= 0.5, probability >= 0.5
    positive_predictions = int(np.sum(predicted))
    true_positive = int(np.sum(predicted & labels))
    return {"row_count": len(target), "accuracy": float(np.mean(predicted == labels)),
            "macro_f1": 0.5 * (_f1(predicted, labels, False) + _f1(predicted, labels, True)),
            "transfer_precision": true_positive / max(1, positive_predictions),
            "soft_mae": float(np.mean(np.abs(probability - target))), "parse_rate": 1.0}


def ordinary_metrics(probability: np.ndarray, target: np.ndarray,
                     concepts: np.ndarray) -> dict:
    probability, target, concepts = map(np.asarray, (probability, target, concepts))
    if not (len(probability) == len(target) == len(concepts)):
        raise ValueError("ordinary metric arrays must align")
    slices = {}
    for code, name in enumerate(CONCEPTS):
        mask = concepts == code
        slices[name] = _ordinary_slice(probability[mask], target[mask]) if mask.any() else _empty(False)
    return {"overall": _ordinary_slice(probability, target), "assay_concept": slices}


def _rankdata(values: np.ndarray) -> np.ndarray:
    order, ranks, start = np.argsort(values, kind="stable"), np.empty(len(values)), 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(target_z: np.ndarray, scores: np.ndarray) -> float:
    left, right = _rankdata(target_z), _rankdata(scores)
    if np.std(left) == 0 or np.std(right) == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _ndcg(target_a: np.ndarray, scores: np.ndarray, k: int) -> float:
    predicted = np.argsort(-scores, kind="stable")
    ideal = np.argsort(-target_a, kind="stable")
    top, ideal_top = predicted[:k], ideal[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    dcg = float(np.sum(target_a[top] * discounts))
    ideal_dcg = float(np.sum(target_a[ideal_top] * discounts))
    return dcg / ideal_dcg if ideal_dcg > 0 else math.nan


def _ranking_list(target_z: np.ndarray, target_a: np.ndarray,
                  scores: np.ndarray) -> dict[str, float]:
    predicted = np.argsort(-scores, kind="stable")
    maxima = target_z == np.max(target_z)
    return {"spearman": _spearman(target_z, scores),
            "ndcg_at_5": _ndcg(target_a, scores, 5),
            "ndcg_at_10": _ndcg(target_a, scores, 10),
            "top1_hit": float(maxima[predicted[0]]),
            "best_in_top10_hit": float(np.any(maxima[predicted[:10]]))}


def _group_rows(group_ids: np.ndarray) -> dict[int, list[int]]:
    output: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        output[int(group)].append(index)
    return output


def _macro(rows: list[dict[str, float]]) -> dict[str, float | int]:
    if not rows:
        return _empty(True)
    metrics = {}
    for key in rows[0]:
        finite = [row[key] for row in rows if math.isfinite(row[key])]
        metrics[key] = float(np.mean(finite)) if finite else math.nan
    return {"query_count": len(rows), **metrics}


def _empty(ranking: bool) -> dict[str, float | int | None]:
    names = ("spearman", "ndcg_at_5", "ndcg_at_10", "top1_hit", "best_in_top10_hit") \
        if ranking else ("accuracy", "macro_f1", "transfer_precision", "soft_mae", "parse_rate")
    return {"query_count" if ranking else "row_count": 0, **{name: None for name in names}}


def ranking_metrics(target_z: np.ndarray, target_a: np.ndarray, scores: np.ndarray,
                    group_ids: np.ndarray, concepts: np.ndarray) -> dict:
    arrays = tuple(map(np.asarray, (target_z, target_a, scores, group_ids, concepts)))
    if len({len(values) for values in arrays}) != 1:
        raise ValueError("ranking metric arrays must align")
    grouped, by_concept, all_rows = _group_rows(arrays[3]), defaultdict(list), []
    for indices in grouped.values():
        index = np.asarray(indices)
        if len(index) != 20:
            raise ValueError("ranking queries must contain exactly 20 candidates")
        codes = np.unique(arrays[4][index])
        if len(codes) != 1:
            raise ValueError("a ranking query crosses assay concepts")
        row = _ranking_list(arrays[0][index], arrays[1][index], arrays[2][index])
        all_rows.append(row)
        by_concept[int(codes[0])].append(row)
    slices = {name: _macro(by_concept[code]) for code, name in enumerate(CONCEPTS)}
    return {"overall": _macro(all_rows), "assay_concept": slices}


def _ordinary_wandb(split: str, metrics: dict) -> dict[str, float | int]:
    names = {"accuracy": "binary_accuracy", "macro_f1": "binary_macro_f1",
             "parse_rate": "binary_parse_rate", "transfer_precision": "transfer_precision",
             "soft_mae": "soft_mae"}
    output = {}
    for name, value in metrics["overall"].items():
        if name in names and value is not None:
            output[f"eval/{split}/overall/{names[name]}"] = value
    for concept, values in metrics["assay_concept"].items():
        for name, value in values.items():
            if name in names and value is not None:
                output[f"eval/{split}/assay_concept/{concept}/{names[name]}"] = value
    for metric in ("accuracy", "macro_f1"):
        finite = [values[metric] for values in metrics["assay_concept"].values()
                  if values[metric] is not None and math.isfinite(values[metric])]
        if finite:
            key = f"eval/{split}/assay_concept_avg_binary_{metric}"
            output[key] = float(np.mean(finite))
    return output


def _ranking_wandb(split: str, metrics: dict) -> dict[str, float | int]:
    output = {}
    sections = [("overall", metrics["overall"])] + [
        (f"assay_concept/{concept}", values)
        for concept, values in metrics["assay_concept"].items()]
    for section, values in sections:
        for name, value in values.items():
            if value is not None and (not isinstance(value, float) or math.isfinite(value)):
                output[f"eval/{split}/{section}/{name}"] = value
    return output


def wandb_metrics(split: str, metrics: dict) -> dict[str, float | int]:
    return _ranking_wandb(split, metrics) if "query_count" in metrics["overall"] \
        else _ordinary_wandb(split, metrics)
