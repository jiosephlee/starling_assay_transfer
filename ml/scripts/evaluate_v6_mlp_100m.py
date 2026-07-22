#!/usr/bin/env python3
"""Evaluate the selected V6 100M checkpoint in the frozen split order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starling_ml.v6_mlp_100m import V6FusionMLP100M
from starling_ml.v6_mlp_100m_data import V6CachedPairs
from starling_ml.v6_mlp_metrics import ordinary_metrics, ranking_metrics


@torch.no_grad()
def _scores(model, cache: V6CachedPairs, split_name: str, batch_size: int):
    split, probabilities, margins = cache.split(split_name), [], []
    model.eval()
    for start in range(0, len(split["target_a"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(split["target_a"])))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cache.device.type == "cuda"):
            logits = model(*cache.features(split, indices))
        logits = logits.float()
        probabilities.append(torch.softmax(logits, -1)[:, 0].cpu().numpy())
        margins.append((logits[:, 0] - logits[:, 1]).cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(margins)


def _ordinary(model, cache, split_name: str, batch_size: int) -> dict:
    split, (probability, _) = cache.split(split_name), _scores(model, cache, split_name, batch_size)
    return ordinary_metrics(probability, np.asarray(split["target_a"]), cache.concepts(split))


def _ranking(model, cache, split_name: str, batch_size: int) -> dict:
    split, (_, scores) = cache.split(split_name), _scores(model, cache, split_name, batch_size)
    return ranking_metrics(np.asarray(split["target_z"]), np.asarray(split["target_a"]), scores,
                           np.asarray(split["group_index"]), cache.concepts(split))


def _load(checkpoint: Path, selection: Path, device: torch.device):
    chosen = json.loads(selection.read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mode = state["args"]["fusion_mode"]
    if chosen["winner"] != mode:
        raise ValueError("checkpoint fusion mode is not the frozen selection winner")
    model = V6FusionMLP100M(mode).to(device)
    model.load_state_dict(state["model"])
    return model, state, chosen


def evaluate(args) -> dict:
    device = torch.device(args.device)
    model, state, selection = _load(Path(args.checkpoint), Path(args.selection), device)
    cache = V6CachedPairs(Path(args.embedding_cache), Path(args.pair_cache), device)
    sections = [("validation", _ordinary), ("validation_ranking", _ranking),
                ("test", _ordinary), ("test_ranking", _ranking)]
    result = {"fusion_mode": selection["winner"], "checkpoint": args.checkpoint,
              "checkpoint_step": state["training"]["step"], "selection": selection}
    for split_name, function in sections:
        result[split_name] = function(model, cache, split_name, args.batch_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection", default=(
        "ml/artifacts/v6_5_molformer_pubmedbert_100m/fusion_selection.json"))
    parser.add_argument("--embedding-cache", default=(
        "ml/artifacts/v6_5_molformer_pubmedbert_100m/embedding_cache"))
    parser.add_argument("--pair-cache", default=(
        "ml/artifacts/v6_5_molformer_pubmedbert_100m/pair_cache"))
    parser.add_argument("--output", default=(
        "ml/artifacts/v6_5_molformer_pubmedbert_100m/final_evaluation.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))
