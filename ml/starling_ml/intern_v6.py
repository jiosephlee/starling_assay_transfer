"""V6 Intern ListNet loss, collator, sequential sampler, and group-aware packing."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

import numpy as np


def listnet_loss(choice_scores: torch.Tensor, target_z: torch.Tensor, group_ids: torch.Tensor,
                 target_temperature: float = 1.0, model_temperature: float = 1.0) -> torch.Tensor:
    """ListNet CE over complete flattened groups, ignoring groups with fewer than two rows."""
    if choice_scores.ndim != 2 or choice_scores.shape[1] != 2:
        raise ValueError("choice_scores must be [documents, 2]")
    if target_temperature <= 0 or model_temperature <= 0:
        raise ValueError("ListNet temperatures must be positive")
    margins = choice_scores[:, 0] - choice_scores[:, 1]
    losses = []
    for group in group_ids.unique(sorted=True):
        mask = group_ids == group
        if int(mask.sum()) > 1:
            targets = F.softmax(target_z[mask] / target_temperature, dim=0)
            losses.append(-(targets * F.log_softmax(margins[mask] / model_temperature, dim=0)).sum())
    return torch.stack(losses).mean() if losses else margins.sum() * 0.0


def collate_intern_v6(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Flatten documents while retaining numeric ListNet group identifiers and targets."""
    fields = ("input_ids", "labels", "position_ids", "decision_offsets", "listnet_group_ids", "target_z")
    result: dict[str, torch.Tensor] = {}
    for field in fields:
        values = [item[field] for item in features if field in item]
        if values:
            result[field] = torch.cat(values) if values[0].ndim else torch.stack(values)
    return result


class PreassignedSequentialSampler(Sampler[int]):
    """Forbid runtime reshuffling of offline packed-chunk assignments."""

    def __init__(self, indices: Iterable[int]) -> None:
        self.indices = tuple(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class PackedGroup:
    group_id: str
    rows: tuple[dict[str, Any], ...]
    tokens: int


def group_bfd(groups: list[PackedGroup], max_tokens: int = 4096) -> list[list[PackedGroup]]:
    """Best-fit decreasing pack, never splitting an indivisible ListNet group."""
    chunks: list[list[PackedGroup]] = []
    remaining: list[int] = []
    for group in sorted(groups, key=lambda item: (-item.tokens, item.group_id)):
        if group.tokens > max_tokens:
            raise ValueError(f"oversized ListNet group: {group.group_id}")
        candidates = [index for index, space in enumerate(remaining) if space >= group.tokens]
        index = min(candidates, key=lambda item: remaining[item] - group.tokens) if candidates else None
        if index is None:
            chunks.append([group])
            remaining.append(max_tokens - group.tokens)
        else:
            chunks[index].append(group)
            remaining[index] -= group.tokens
    return chunks


def assign_optimizer_batches(chunks: list[list[PackedGroup]], chunks_per_update: int = 256) -> list[list[PackedGroup]]:
    """Drop only the trailing incomplete optimizer batch, preserving sequential chunks."""
    usable = len(chunks) // chunks_per_update * chunks_per_update
    return chunks[:usable]


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = _rankdata(left), _rankdata(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _ranking_list_metrics(relevance: np.ndarray, scores: np.ndarray, k: int) -> dict[str, float]:
    predicted = np.argsort(-scores, kind="stable")
    ideal = np.argsort(-relevance, kind="stable")
    discounts = 1.0 / np.log2(np.arange(2, min(k, len(relevance)) + 2))
    gains = np.exp2(relevance) - 1.0
    dcg = float(np.sum(gains[predicted[:k]] * discounts))
    ideal_dcg = float(np.sum(gains[ideal[:k]] * discounts))
    best = float(relevance[ideal[0]])
    top = predicted[:k]
    return {"ndcg_at_10": 0.0 if ideal_dcg == 0 else dcg / ideal_dcg,
            "top1_regret": best - float(relevance[predicted[0]]),
            "best_in_top10_regret": best - float(np.max(relevance[top])),
            "mean_top10_relevance": float(np.mean(relevance[top])),
            "spearman": _spearman(relevance, scores)}


def ranking_metrics(relevance: Iterable[float], scores: Iterable[float],
                    group_ids: Iterable[Any], k: int = 10) -> dict[str, float]:
    """Return query-macro ranking metrics without ever combining condition lists."""
    rel, predicted, groups = np.asarray(list(relevance)), np.asarray(list(scores)), list(group_ids)
    if len(rel) != len(predicted) or len(rel) != len(groups):
        raise ValueError("ranking relevance, scores, and group IDs must align")
    indices: dict[Any, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        indices[group].append(index)
    values = [_ranking_list_metrics(rel[rows], predicted[rows], k) for rows in indices.values()]
    if not values:
        raise ValueError("ranking metrics require at least one list")
    return {name: float(np.mean([row[name] for row in values])) for name in values[0]}
