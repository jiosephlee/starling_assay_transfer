"""Train one cached-embedding V6 100M fusion ablation."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .v6_mlp_100m import FUSION_SPECS, V6FusionMLP100M, soft_ab_loss
from .v6_mlp_100m_data import V6CachedPairs, group_indices, validate_group_schedule
from .v6_mlp_metrics import ordinary_metrics, ranking_metrics, wandb_metrics


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return max(1, step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _optimizer(model: torch.nn.Module, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_factor(step, args.warmup_steps, args.schedule_steps))
    return optimizer, scheduler


def _rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint(path: Path, model, optimizer, scheduler, state: dict, args) -> None:
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "training": state,
               "rng": _rng_state(), "args": vars(args)}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _resume(path: Path, model, optimizer, scheduler) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng(payload["rng"])
    return payload["training"]


def _append(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _wandb_run(args, output: Path):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    id_path = output / "wandb_id.txt"
    run_id = id_path.read_text().strip() if id_path.exists() else wandb.util.generate_id()
    if not id_path.exists():
        id_path.write_text(run_id + "\n")
    name = args.wandb_run_name or (
        f"mlp-100m-{args.fusion_mode}_assay-transfer-raw-pair-v6-5-soft-a100")
    run = wandb.init(project=args.wandb_project, group=args.wandb_group,
                      name=name, id=run_id,
                      resume="allow", mode=args.wandb_mode, config=vars(args))
    run.define_metric("step")
    run.define_metric("train/*", step_metric="step")
    run.define_metric("eval/*", step_metric="step")
    return run


@torch.no_grad()
def _scores(model, cache: V6CachedPairs, split_name: str, batch_size: int,
            selected: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    split, probabilities, margins = cache.split(split_name), [], []
    selected = np.arange(len(split["target_a"])) if selected is None else selected
    was_training = model.training
    model.eval()
    for start in range(0, len(selected), batch_size):
        indices = selected[start:start + batch_size]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cache.device.type == "cuda"):
            logits = model(*cache.features(split, indices))
        logits = logits.float()
        probabilities.append(torch.softmax(logits, -1)[:, 0].cpu().numpy())
        margins.append((logits[:, 0] - logits[:, 1]).cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(probabilities), np.concatenate(margins)


def _validation(model, cache: V6CachedPairs, batch_size: int) -> dict:
    split = cache.split("validation")
    probabilities, _ = _scores(model, cache, "validation", batch_size)
    return ordinary_metrics(probabilities, np.asarray(split["target_a"]), cache.concepts(split))


def _ranking_validation(model, cache: V6CachedPairs, args) -> dict:
    split = cache.split("validation_ranking")
    indices = cache.ranking_subset("validation_ranking", args.ranking_eval_max_queries,
                                   args.ranking_eval_seed)
    _, scores = _scores(model, cache, "validation_ranking", args.eval_batch_size, indices)
    return ranking_metrics(np.asarray(split["target_z"])[indices],
                           np.asarray(split["target_a"])[indices], scores,
                           np.asarray(split["group_index"])[indices],
                           cache.concepts(split)[indices])


def _microbatches(groups: np.ndarray, size: int):
    for start in range(0, len(groups), size):
        yield groups[start:start + size]


def _train_update(model, cache, split, groups, optimizer, args) -> float:
    optimizer.zero_grad(set_to_none=True)
    chunks, total = list(_microbatches(groups, args.microbatch_groups)), 0.0
    for selected in chunks:
        indices = group_indices(selected, args.group_size)
        features, targets = cache.features(split, indices), cache.targets(split, indices)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cache.device.type == "cuda"):
            loss = soft_ab_loss(model(*features), targets)
        (loss / len(chunks)).backward()
        total += float(loss.detach()) / len(chunks)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
    optimizer.step()
    return total


def _initial_state(args) -> dict:
    return {"step": 0, "best_ndcg_at_5": float("-inf"),
            "best_ndcg_at_10": float("-inf"), "best_step": None,
            "last_eval_step": None, "fusion_mode": args.fusion_mode,
            "schedule_sha256": None}


def _ranking_improved(state: dict, ranking: dict) -> bool:
    ndcg5, ndcg10 = ranking["overall"]["ndcg_at_5"], ranking["overall"]["ndcg_at_10"]
    if not math.isfinite(ndcg5):
        return False
    current = (ndcg5, ndcg10 if math.isfinite(ndcg10) else -float("inf"))
    best = (state["best_ndcg_at_5"], state["best_ndcg_at_10"])
    if current <= best:
        return False
    state.update(best_ndcg_at_5=current[0], best_ndcg_at_10=current[1],
                 best_step=state["step"])
    return True


def _evaluate_and_save(model, cache, optimizer, scheduler, state, args, output, run) -> None:
    ordinary = _validation(model, cache, args.eval_batch_size)
    ranking = _ranking_validation(model, cache, args)
    phase = "start" if state["step"] == 0 else "step"
    row = {"step": state["step"], "phase": phase,
           "validation": ordinary, "ranking_validation": ranking}
    _append(output / "metrics.jsonl", row)
    flat = {"step": state["step"],
            "eval/validation/trigger_step": state["step"],
            "eval/ranking_validation/trigger_step": state["step"],
            **wandb_metrics("validation", ordinary),
            **wandb_metrics("ranking_validation", ranking)}
    if run:
        run.log(flat, step=state["step"])
    improved = _ranking_improved(state, ranking)
    state["last_eval_step"] = state["step"]
    save_last = state["step"] == args.stop_after or (
        state["step"] > 0 and state["step"] % args.checkpoint_steps == 0)
    if save_last:
        _checkpoint(output / "last.pt", model, optimizer, scheduler, state, args)
    if improved:
        _checkpoint(output / "best.pt", model, optimizer, scheduler, state, args)


def train(args) -> dict:
    _seed(args.seed)
    device, output = torch.device(args.device), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pair_path = Path(args.pair_cache)
    schedule, schedule_manifest = validate_group_schedule(
        Path(args.schedule), args.schedule_steps, pair_path)
    if schedule_manifest["batch_groups"] != args.batch_groups:
        raise ValueError("schedule and --batch-groups disagree")
    if schedule_manifest["group_size"] != args.group_size:
        raise ValueError("schedule and --group-size disagree")
    cache = V6CachedPairs(Path(args.embedding_cache), pair_path, device)
    model = V6FusionMLP100M(args.fusion_mode).to(device)
    optimizer, scheduler = _optimizer(model, args)
    state = _resume(Path(args.resume), model, optimizer, scheduler) if args.resume else _initial_state(args)
    state["schedule_sha256"] = schedule_manifest["schedule_sha256"]
    state["records_sha256"] = cache.embedding_manifest["records_sha256"]
    state["pair_dataset_manifest_sha256"] = cache.pair_manifest["dataset_manifest_sha256"]
    run, started, split = _wandb_run(args, output), time.time(), cache.split("train")
    if state.get("last_eval_step") != state["step"]:
        _evaluate_and_save(model, cache, optimizer, scheduler, state, args, output, run)
    while state["step"] < args.stop_after:
        groups = np.asarray(schedule[state["step"]])
        loss = _train_update(model, cache, split, groups, optimizer, args)
        state["step"] += 1
        scheduler.step()
        if state["step"] % args.log_steps == 0:
            row = {"step": state["step"], "train/loss": loss,
                   "train/learning_rate": scheduler.get_last_lr()[0]}
            _append(output / "metrics.jsonl", row)
            if run:
                run.log(row, step=state["step"])
        if state["step"] % args.eval_steps == 0 or state["step"] == args.stop_after:
            _evaluate_and_save(model, cache, optimizer, scheduler, state, args, output, run)
    result = {**state, "elapsed_seconds": time.time() - started}
    (output / "completed.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    if run:
        run.finish()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-mode", choices=tuple(FUSION_SPECS), required=True)
    parser.add_argument("--embedding-cache", default=(
        "ml/artifacts/v6_5_molformer_pubmedbert_100m/embedding_cache"))
    parser.add_argument("--pair-cache", default=(
        "ml/artifacts/v6_5_molformer_pubmedbert_100m/pair_cache"))
    parser.add_argument("--schedule", default=(
        "ml/artifacts/v6_5_molformer_pubmedbert_100m/group_schedule.npy"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--schedule-steps", type=int, default=5000)
    parser.add_argument("--stop-after", type=int, default=1000)
    parser.add_argument("--batch-groups", type=int, default=128)
    parser.add_argument("--microbatch-groups", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=40)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--checkpoint-steps", type=int, default=250)
    parser.add_argument("--ranking-eval-max-queries", type=int, default=100)
    parser.add_argument("--ranking-eval-seed", type=int, default=42)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4878)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-project", default="assay-transfer-soft")
    parser.add_argument("--wandb-group", default="assay-transfer-raw-pair-v6-5-soft")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args()
    if args.batch_groups % args.microbatch_groups:
        parser.error("--microbatch-groups must divide --batch-groups")
    return args


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, sort_keys=True))
