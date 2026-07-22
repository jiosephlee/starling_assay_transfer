#!/usr/bin/env python3
"""Evaluate a completed V6 MLP checkpoint on frozen ordinary and ranking splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starling_ml.train_v6_mlp import Cache, _forward_scores, _model, _ranking_eval


@torch.no_grad()
def _direct_probabilities(model, cache: Cache, split_name: str, batch_size: int) -> np.ndarray:
    split, probabilities = cache.split(split_name), []
    model.eval()
    for start in range(0, len(split["target_a"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(split["target_a"])))
        logits, _ = _forward_scores(model, "direct", cache.features(split, indices), len(indices), 1)
        probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probabilities)


def _class_f1(predicted: np.ndarray, labels: np.ndarray, positive: int) -> float:
    tp = np.sum((predicted == positive) & (labels == positive))
    fp = np.sum((predicted == positive) & (labels != positive))
    fn = np.sum((predicted != positive) & (labels == positive))
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return float(2 * precision * recall / max(1e-12, precision + recall))


def _ordinary_metrics(probabilities: np.ndarray, target_a: np.ndarray) -> dict[str, float]:
    targets = np.stack([target_a, 1.0 - target_a], axis=-1)
    labels, predicted = (target_a >= 0.5).astype(np.int8), probabilities.argmax(axis=-1) == 0
    macro_f1 = 0.5 * (_class_f1(predicted, labels, 0) + _class_f1(predicted, labels, 1))
    return {"accuracy": float(np.mean(predicted == labels)),
            "macro_f1": macro_f1,
            "soft_nll": float(-np.mean(np.sum(targets * np.log(np.clip(probabilities, 1e-8, 1)), axis=-1))),
            "brier": float(np.mean(np.sum((probabilities - targets) ** 2, axis=-1)))}


def evaluate(checkpoint: Path, cache_path: Path, output: Path, device_name: str) -> dict:
    device, state = torch.device(device_name), torch.load(checkpoint, map_location="cpu", weights_only=False)
    variant = state["args"]["variant"]
    model = _model(variant).to(device)
    model.load_state_dict(state["model"])
    cache = Cache(cache_path, device)
    result = {"variant": variant, "checkpoint": str(checkpoint), "step": state["step"],
              "validation_ranking": _ranking_eval(model, variant, cache, "validation_ranking", 20, 128),
              "test_ranking": _ranking_eval(model, variant, cache, "test_ranking", 20, 128)}
    if variant == "direct":
        for split_name in ("validation", "test"):
            targets = np.asarray(cache.split(split_name)["target_a"])
            result[split_name] = _ordinary_metrics(
                _direct_probabilities(model, cache, split_name, 4096), targets)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("ml/artifacts/v6_mlp/cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.cache, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
