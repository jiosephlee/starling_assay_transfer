"""Train and evaluate V6 direct-prediction or graded-contrastive MLP models."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .intern_v6 import ranking_metrics
from .v6_mlp import (V6ContrastiveRetriever, V6DirectPredictor, direct_listnet_loss,
                     graded_list_loss, soft_ab_loss)


PAIR_NAMES = ("query_record_index", "retrieval_record_index", "retrieval_value",
              "retrieval_is_approximate", "target_z", "target_a", "group_index")


class Cache:
    def __init__(self, root: Path, device: torch.device):
        self.root, self.device = root, device
        self.molecule = torch.from_numpy(np.load(root / "molecule_features.npy")).to(device=device)
        self.assay = torch.from_numpy(np.load(root / "assay_features.npy")).to(device=device)
        self.record_molecule = torch.from_numpy(np.load(root / "record_molecule_index.npy")).long().to(device=device)
        self.splits = {}

    def split(self, name: str) -> dict[str, np.ndarray]:
        if name not in self.splits:
            self.splits[name] = {key: np.load(self.root / name / f"{key}.npy", mmap_mode="r")
                                 for key in PAIR_NAMES}
        return self.splits[name]

    def features(self, split: dict[str, np.ndarray], indices: np.ndarray):
        query = torch.as_tensor(np.asarray(split["query_record_index"][indices]), device=self.device).long()
        retrieval = torch.as_tensor(np.asarray(split["retrieval_record_index"][indices]), device=self.device).long()
        value = torch.as_tensor(np.asarray(split["retrieval_value"][indices]), device=self.device).float()
        approx = torch.as_tensor(np.asarray(split["retrieval_is_approximate"][indices]), device=self.device).float()
        qm = self.molecule[self.record_molecule[query].long()].float()
        rm = self.molecule[self.record_molecule[retrieval].long()].float()
        return qm, rm, self.assay[query].float(), self.assay[retrieval].float(), value, approx


def _group_indices(groups: np.ndarray, size: int) -> np.ndarray:
    return (groups[:, None] * size + np.arange(size, dtype=np.int64)[None, :]).reshape(-1)


def _targets(split: dict[str, np.ndarray], indices: np.ndarray, device: torch.device):
    z = torch.as_tensor(np.asarray(split["target_z"][indices]), device=device).float()
    target_a = torch.as_tensor(np.asarray(split["target_a"][indices]), device=device).float()
    return z, target_a


def _forward_scores(model, variant: str, features, group_count: int, group_size: int):
    output = model(*features)
    if variant == "direct":
        scores = output[:, 0] - output[:, 1]
    else:
        scores = output
    return output, scores.reshape(group_count, group_size)


def _train_loss(model, variant: str, features, z, target_a, group_count: int,
                group_size: int, listnet_weight: float) -> torch.Tensor:
    output, grouped_scores = _forward_scores(model, variant, features, group_count, group_size)
    grouped_z = z.reshape(group_count, group_size)
    if variant == "contrastive":
        return graded_list_loss(grouped_scores, grouped_z)
    loss = soft_ab_loss(output, target_a)
    if listnet_weight:
        loss = loss + listnet_weight * direct_listnet_loss(output, z, group_size)
    return loss


@torch.no_grad()
def _ranking_eval(model, variant: str, cache: Cache, split_name: str,
                  group_size: int, batch_groups: int) -> dict[str, float]:
    split, scores = cache.split(split_name), []
    groups = len(split["target_z"]) // group_size
    model.eval()
    for start in range(0, groups, batch_groups):
        current = min(batch_groups, groups - start)
        indices = _group_indices(np.arange(start, start + current), group_size)
        features = cache.features(split, indices)
        _, values = _forward_scores(model, variant, features, current, group_size)
        scores.append(values.float().cpu().numpy().reshape(-1))
    relevance = np.asarray(split["target_z"])
    result = ranking_metrics(relevance, np.concatenate(scores), np.asarray(split["group_index"]))
    model.train()
    return result


def _binary_macro_f1(predicted: np.ndarray, labels: np.ndarray) -> float:
    values = []
    for positive in (0, 1):
        tp = np.sum((predicted == positive) & (labels == positive))
        fp = np.sum((predicted == positive) & (labels != positive))
        fn = np.sum((predicted != positive) & (labels == positive))
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        values.append(2 * precision * recall / max(1e-12, precision + recall))
    return float(np.mean(values))


@torch.no_grad()
def _direct_classification_eval(model, cache: Cache, split_name: str) -> dict[str, float]:
    split, probabilities = cache.split(split_name), []
    model.eval()
    for start in range(0, len(split["target_a"]), 4096):
        indices = np.arange(start, min(start + 4096, len(split["target_a"])))
        logits = model(*cache.features(split, indices))
        probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    predicted = np.concatenate(probabilities).argmax(axis=-1)
    labels = (np.asarray(split["target_a"]) < 0.5).astype(np.int8)
    model.train()
    return {"accuracy": float(np.mean(predicted == labels)),
            "macro_f1": _binary_macro_f1(predicted, labels)}


def _model(variant: str) -> torch.nn.Module:
    if variant == "direct":
        return V6DirectPredictor()
    return V6ContrastiveRetriever()


def _save(path: Path, model, optimizer, step: int, args, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "args": vars(args), "metrics": metrics}, path)


def _log(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device, output = torch.device(args.device), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache, model = Cache(Path(args.cache), device), _model(args.variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_split, rng = cache.split("train"), np.random.default_rng(args.seed)
    group_count = len(train_split["target_z"]) // args.group_size
    best, started = float("-inf"), time.time()
    for step in range(1, args.max_steps + 1):
        groups = rng.integers(0, group_count, size=args.batch_groups)
        indices = _group_indices(groups, args.group_size)
        features = cache.features(train_split, indices)
        z, target_a = _targets(train_split, indices, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = _train_loss(model, args.variant, features, z, target_a, args.batch_groups,
                               args.group_size, args.listnet_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_steps == 0:
            _log(output / "metrics.jsonl", {"step": step, "train_loss": float(loss.detach()),
                                             "elapsed_seconds": time.time() - started})
        if step % args.eval_steps == 0 or step == args.max_steps:
            metrics = _ranking_eval(model, args.variant, cache, "validation_ranking",
                                    args.group_size_eval, args.eval_batch_groups)
            row = {"step": step, **{"validation_" + key: value for key, value in metrics.items()}}
            if args.variant == "direct":
                classification = _direct_classification_eval(model, cache, "validation")
                row.update({"validation_" + key: value for key, value in classification.items()})
            _log(output / "metrics.jsonl", row)
            _save(output / "last.pt", model, optimizer, step, args, metrics)
            if metrics["ndcg_at_10"] > best:
                best = metrics["ndcg_at_10"]
                _save(output / "best.pt", model, optimizer, step, args, metrics)
    (output / "completed.json").write_text(json.dumps({"steps": args.max_steps,
        "best_validation_ndcg_at_10": best, "elapsed_seconds": time.time() - started}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("direct", "contrastive"), required=True)
    parser.add_argument("--cache", default="ml/artifacts/v6_mlp/cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-groups", type=int, default=128)
    parser.add_argument("--eval-batch-groups", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=40)
    parser.add_argument("--group-size-eval", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--listnet-weight", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4878)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
