"""Train the transfer-classification model with the HuggingFace Trainer.

The DataLoader ships only integer indices; the model gathers frozen embeddings from GPU
buffers, so per-step cost is a few small matmuls and batch sizes can be very large.

Usage:
    python -m starling_ml.train --config ml/configs/default.yaml
    # multi-GPU:
    torchrun --nproc_per_node=8 -m starling_ml.train --config ml/configs/default.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.data import Subset
from transformers import Trainer, TrainerCallback, TrainingArguments

from . import benchmark_spec
from .config import Config
from .data import PairDataset, build_split_memmap, collate_pairs
from .metrics import binary_metrics, simple_transfer_metrics
from .model import build_model


def _metric_fn(metric_set: str):
    if metric_set == "simple_transfer":
        return simple_transfer_metrics
    if metric_set == "binary":
        return binary_metrics
    raise ValueError(f"unknown train.metric_set={metric_set!r}")


def _compute_metrics_for(
    metric_set: str,
    *,
    eval_subset_codes: np.ndarray | None = None,
    eval_subset_names: list[str] | None = None,
    similarity_bucket_codes: np.ndarray | None = None,
    similarity_bucket_names: list[str] | None = None,
):
    metric_fn = _metric_fn(metric_set)
    subset_codes = np.asarray(eval_subset_codes) if eval_subset_codes is not None else None
    subset_names = list(eval_subset_names or [])
    sim_codes = np.asarray(similarity_bucket_codes) if similarity_bucket_codes is not None else None
    sim_names = list(similarity_bucket_names or [])

    def _compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = np.asarray(logits).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        out = metric_fn(logits, labels)
        if subset_codes is not None and len(labels) == len(subset_codes):
            for code, name in enumerate(subset_names):
                mask = subset_codes == code
                if not mask.any():
                    continue
                for metric_name, value in metric_fn(logits[mask], labels[mask]).items():
                    out[f"{name}_{metric_name}"] = value
        if sim_codes is not None and len(labels) == len(sim_codes):
            for code, name in enumerate(sim_names):
                mask = sim_codes == code
                if not mask.any():
                    continue
                for metric_name, value in metric_fn(logits[mask], labels[mask]).items():
                    out[f"similarity_{name}_{metric_name}"] = value
        return out

    return _compute_metrics


class WandbValMetricsCallback(TrainerCallback):
    """Mirror full-validation metrics to the requested W&B `val/...` namespace."""

    REQUESTED_KEYS = (
        "accuracy",
        "macro_f1",
        "parse_rate",
        "label/A/precision",
        "label/A/recall",
        "label/A/f1",
        "label/A/predicted",
        "label/B/precision",
        "label/B/recall",
        "label/B/f1",
        "label/B/predicted",
    )

    @staticmethod
    def _lookup(logs: dict[str, float], key: str):
        candidates = (
            f"eval_val_{key}",
            f"eval_val_{key.replace('/', '_')}",
            f"eval_val/{key}",
        )
        for candidate in candidates:
            if candidate in logs:
                return logs[candidate]
        return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        payload = {"val/trigger_step": int(state.global_step)}
        for key in self.REQUESTED_KEYS:
            value = self._lookup(logs, key)
            if value is not None:
                payload[f"val/{key}"] = value
        if len(payload) == 1:
            return
        if "wandb" not in args.report_to:
            return
        try:
            import wandb
        except Exception:
            return
        if wandb.run is not None:
            wandb.log(payload)


class SimpleValidationWandbCallback(TrainerCallback):
    """Log benchmark validation metrics and train-step metrics to W&B."""

    METRICS = ("macro_f1", "accuracy", "transfer_precision", "transfer_recall", "entropy")
    TRAIN_METRICS = ("loss", "grad_norm", "learning_rate")
    PREFIX_MAP = (
        ("eval_val_", ("val/",)),
        ("eval_val_no_overlap_", ("val/no_overlap/", "val/double_unseen/")),
        ("eval_val_a_seen_only_", ("val/a_seen_only/", "val/query_unseen/")),
        ("eval_val_both_seen_", ("val/both_seen/",)),
    )

    def __init__(self, project: str, run_name: str, group: str | None = None) -> None:
        self.project = project
        self.run_name = run_name or None
        self.group = group or None

    @classmethod
    def _payload(cls, logs: dict[str, float]) -> dict[str, float]:
        payload: dict[str, float] = {}
        for metric in cls.TRAIN_METRICS:
            value = logs.get(metric)
            if isinstance(value, (int, float)):
                payload[f"train/{metric}"] = float(value)
        for prefix, out_prefixes in cls.PREFIX_MAP:
            for metric in cls.METRICS:
                key = f"{prefix}{metric}"
                value = logs.get(key)
                if isinstance(value, (int, float)):
                    for out_prefix in out_prefixes:
                        payload[f"{out_prefix}{metric}"] = float(value)
        return payload

    def _ensure_run(self):
        import wandb

        if wandb.run is None:
            wandb.init(project=self.project, name=self.run_name, group=self.group)
        return wandb

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._ensure_run()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        payload = self._payload(logs)
        if not payload:
            return
        wandb = self._ensure_run()
        wandb.log(payload, step=int(state.global_step))
        if wandb.run is not None:
            for key, value in payload.items():
                wandb.run.summary[key] = value

    def on_train_end(self, args, state, control, **kwargs):
        # Finish the run explicitly in main after the final evaluation callback chain.
        return


class MetricsCsvCallback(TrainerCallback):
    """Persist per-log metrics to ``ml/results/runs/<run>/metrics.csv`` (tidy: epoch, split,
    metric, value) so runs store results first-class — same schema as ``report.parse_run_log``."""

    def __init__(self, out_csv: str):
        self.out_csv = out_csv

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        epoch = logs.get("epoch", state.epoch)
        rows = []
        eval_prefixes = (
            ("eval_val_no_overlap_", "val/no_overlap"),
            ("eval_val_a_seen_only_", "val/a_seen_only"),
            ("eval_val_both_seen_", "val/both_seen"),
            ("eval_val_similarity_", "val/similarity"),
            ("eval_val_", "val"),
            ("eval_train_sample_", "train_sample"),
        )
        for key, value in logs.items():
            if not isinstance(value, (int, float)) or key == "epoch":
                continue
            split = metric = None
            for prefix, split_name in eval_prefixes:
                if key.startswith(prefix):
                    split, metric = split_name, key[len(prefix):]
                    break
            if split is None and key in ("loss", "grad_norm", "learning_rate"):
                split, metric = "train", key
            if split is None or metric is None:
                continue
            rows.append((epoch, split, metric, value))
            if split == "val/no_overlap":
                rows.append((epoch, "val/double_unseen", metric, value))
            elif split == "val/a_seen_only":
                rows.append((epoch, "val/query_unseen", metric, value))
        if not rows:
            return
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)
        write_header = not os.path.exists(self.out_csv)
        with open(self.out_csv, "a", newline="") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(["epoch", "split", "metric", "value"])
            writer.writerows(rows)


class BestModelCallback(TrainerCallback):
    """Save the model each time the configured metric improves, replacing the previous best.

    Writes a model-only checkpoint (frozen buffers are non-persistent) to ``<output_dir>/best`` plus
    ``best_metric.json``. Rank-0 only; needs a ``trainer`` ref (set after Trainer construction).
    """

    def __init__(self, best_dir: str, metric: str = "eval_val_macro_f1"):
        self.best_dir = best_dir
        self.metric = metric
        self.best: float | None = None
        self.trainer = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.trainer is None or not state.is_world_process_zero or not metrics:
            return
        value = metrics.get(self.metric)
        if value is None:  # e.g. the train_sample eval pass — wrong prefix
            return
        if self.best is None or value > self.best:
            self.best = float(value)
            shutil.rmtree(self.best_dir, ignore_errors=True)
            os.makedirs(self.best_dir, exist_ok=True)
            self.trainer.save_model(self.best_dir)
            with open(os.path.join(self.best_dir, "best_metric.json"), "w") as fh:
                json.dump(
                    {
                        "metric": self.metric,
                        "value": self.best,
                        "step": int(state.global_step),
                        "epoch": float(state.epoch) if state.epoch is not None else None,
                    },
                    fh,
                )
            print(f"[best] {self.metric}={self.best:.4f} @ step {state.global_step} -> {self.best_dir}")


class TdcKnnEvalCallback(TrainerCallback):
    """Run downstream TDC Bioavailability_Ma KNN eval at a lower cadence."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.trainer = None
        self.last_eval_step: int | None = None
        self.best_cb = BestModelCallback(
            os.path.join(cfg.paths.output_dir, benchmark_spec.BEST_TDC_CHECKPOINT_DIR),
            metric="tdc_train_macro_f1",
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.trainer is None or not state.is_world_process_zero:
            return
        step = int(state.global_step)
        if step <= 0 or step % max(1, int(self.cfg.train.tdc_eval_steps)) != 0:
            return
        if self.last_eval_step == step:
            return
        from .tdc_knn_eval import evaluate_tdc_knn

        result = evaluate_tdc_knn(
            cfg=self.cfg,
            model=self.trainer.model,
            tdc_path=self.cfg.train.tdc_eval_path,
            cache_dir=self.cfg.train.tdc_eval_cache_dir,
            top_fraction=self.cfg.train.tdc_eval_top_fraction,
            k=self.cfg.train.tdc_eval_k,
            batch_size=self.cfg.train.tdc_eval_batch_size,
        )
        payload = {
            "tdc_train_macro_f1": result.macro_f1,
            "tdc_train_accuracy": result.accuracy,
            "tdc_train_transfer_precision": result.transfer_precision,
            "tdc_train_transfer_recall": result.transfer_recall,
            "tdc_train_n_queries": result.n_queries,
            "tdc_train_elapsed_seconds": result.elapsed_seconds,
        }
        if metrics is not None:
            metrics.update(payload)
        if "wandb" in args.report_to:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.define_metric("tdc/train/*", step_metric="tdc/train/training_step")
                    wandb_payload = {
                        f"tdc/train/{key.removeprefix('tdc_train_')}": value for key, value in payload.items()
                    }
                    wandb_payload["tdc/train/training_step"] = step
                    wandb.log(wandb_payload)
                    wandb.run.summary["tdc/train/macro_f1"] = result.macro_f1
            except Exception:
                pass
        self.best_cb.trainer = self.trainer
        self.best_cb.on_evaluate(args, state, control, metrics=payload, **kwargs)
        self.last_eval_step = step
        print(
            f"[tdc] train Bioavailability_Ma macro_f1={result.macro_f1:.4f} "
            f"@ step {step} ({result.elapsed_seconds:.1f}s)"
        )


def _dist_barrier_if_needed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


class RecordKnnEvalCallback(TrainerCallback):
    """Run record-level KNN eval with cached Tanimoto candidates and the live model."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.trainer = None
        self.last_eval_step: int | None = None
        self.best_cb = BestModelCallback(
            os.path.join(cfg.paths.output_dir, benchmark_spec.BEST_RECORD_KNN_CHECKPOINT_DIR),
            metric="record_knn_validation_1_macro_f1",
        )

    def _should_run(self, step: int) -> bool:
        if step <= 0:
            return False
        if step % max(1, int(self.cfg.train.record_knn_eval_steps)) != 0:
            return False
        return self.last_eval_step != step

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            from .record_knn_eval import ensure_record_knn_cache

            for split in self.cfg.train.record_knn_eval_splits:
                ensure_record_knn_cache(
                    cfg=self.cfg,
                    dataset_dir=self.cfg.train.record_knn_eval_dataset_dir,
                    cache_dir=self.cfg.train.record_knn_eval_cache_dir,
                    split=split,
                    dataset_config=self.cfg.train.record_knn_eval_dataset_config,
                    top_fraction=self.cfg.train.record_knn_eval_top_fraction,
                    max_queries=self.cfg.train.record_knn_eval_max_queries,
                    require_existing=True,
                )
        _dist_barrier_if_needed()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.trainer is None:
            return
        step = int(state.global_step)
        if not self._should_run(step):
            return
        payload = None
        if state.is_world_process_zero:
            payload = self._evaluate_splits()
            if metrics is not None:
                metrics.update(payload)
            self._log_wandb(args, step, payload)
            self.best_cb.trainer = self.trainer
            self.best_cb.on_evaluate(args, state, control, metrics=payload, **kwargs)
            print(
                f"[record-knn] validation_1 macro_f1={payload['record_knn_validation_1_macro_f1']:.4f} "
                f"@ step {step}"
            )
        self.last_eval_step = step
        _dist_barrier_if_needed()

    def _evaluate_splits(self) -> dict[str, float | int]:
        from .record_knn_eval import evaluate_record_knn, record_knn_result_dict

        payload: dict[str, float | int] = {}
        for split in self.cfg.train.record_knn_eval_splits:
            result = evaluate_record_knn(
                cfg=self.cfg,
                model=self.trainer.model,
                dataset_dir=self.cfg.train.record_knn_eval_dataset_dir,
                cache_dir=self.cfg.train.record_knn_eval_cache_dir,
                split=split,
                dataset_config=self.cfg.train.record_knn_eval_dataset_config,
                top_fraction=self.cfg.train.record_knn_eval_top_fraction,
                k=self.cfg.train.record_knn_eval_k,
                batch_size=self.cfg.train.record_knn_eval_batch_size,
                max_queries=self.cfg.train.record_knn_eval_max_queries,
                require_cache=True,
            )
            for key, value in record_knn_result_dict(result).items():
                payload[f"record_knn_{split}_{key}"] = value
        return payload

    def _log_wandb(self, args, step: int, payload: dict[str, float | int]) -> None:
        if "wandb" not in args.report_to:
            return
        try:
            import wandb

            if wandb.run is None:
                return
            wandb_payload = self._wandb_payload(step, payload)
            for split in self.cfg.train.record_knn_eval_splits:
                wandb.define_metric(f"record_knn/{split}/*", step_metric=f"record_knn/{split}/training_step")
            wandb.log(wandb_payload)
            wandb.run.summary["record_knn/validation_1/macro_f1"] = payload["record_knn_validation_1_macro_f1"]
        except Exception:
            pass

    def _wandb_payload(self, step: int, payload: dict[str, float | int]) -> dict[str, float | int]:
        out: dict[str, float | int] = {}
        for key, value in payload.items():
            prefix = "record_knn_"
            if not key.startswith(prefix):
                continue
            split, metric = self._split_metric_name(key.removeprefix(prefix))
            out[f"record_knn/{split}/{metric}"] = value
            out[f"record_knn/{split}/training_step"] = step
        return out

    @staticmethod
    def _split_metric_name(value: str) -> tuple[str, str]:
        for split in ("validation_1", "validation_2", "test"):
            prefix = f"{split}_"
            if value.startswith(prefix):
                return split, value.removeprefix(prefix)
        return "unknown", value


def _load_checkpoint_state(checkpoint: str) -> dict:
    safet = os.path.join(checkpoint, "model.safetensors")
    binf = os.path.join(checkpoint, "pytorch_model.bin")
    if os.path.exists(safet):
        from safetensors.torch import load_file

        return load_file(safet)
    if os.path.exists(binf):
        return torch.load(binf, map_location="cpu")
    raise FileNotFoundError(f"no model weights (model.safetensors / pytorch_model.bin) in {checkpoint}")


def _run_tdc_valid_best_val_eval(cfg: Config, trainer: Trainer) -> None:
    if not cfg.train.tdc_eval_enabled or not cfg.train.tdc_eval_valid_on_best_val:
        return
    if not trainer.is_world_process_zero():
        return

    best_dir = os.path.join(cfg.paths.output_dir, benchmark_spec.BEST_VAL_CHECKPOINT_DIR)
    if not os.path.exists(best_dir):
        print(f"[tdc] skip valid Bioavailability_Ma eval; missing best checkpoint {best_dir}")
        return

    from .tdc_knn_eval import tdc_eval_result_dict

    model = _load_tdc_best_model(cfg, trainer, best_dir)
    result = _evaluate_tdc_valid(cfg, model)
    result_payload = tdc_eval_result_dict(result)
    out_path = os.path.join(cfg.paths.output_dir, "tdc_valid_best_val_knn.json")
    _write_tdc_valid_json(best_dir, cfg.train.tdc_eval_valid_path, out_path, result_payload)
    _log_tdc_valid_wandb(trainer, result_payload)
    print(
        f"[tdc] valid Bioavailability_Ma macro_f1={result.macro_f1:.4f} "
        f"from {best_dir} ({result.elapsed_seconds:.1f}s) -> {out_path}"
    )


def _load_tdc_best_model(cfg: Config, trainer: Trainer, best_dir: str):
    device = next(trainer.model.parameters()).device
    model = build_model(cfg)
    state = _load_checkpoint_state(best_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[tdc] loaded best-val checkpoint for valid eval missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


def _evaluate_tdc_valid(cfg: Config, model):
    from .tdc_knn_eval import evaluate_tdc_knn

    return evaluate_tdc_knn(
        cfg=cfg,
        model=model,
        tdc_path=cfg.train.tdc_eval_valid_path,
        cache_dir=cfg.train.tdc_eval_cache_dir,
        top_fraction=cfg.train.tdc_eval_top_fraction,
        k=cfg.train.tdc_eval_k,
        batch_size=cfg.train.tdc_eval_batch_size,
    )


def _write_tdc_valid_json(best_dir: str, tdc_path: str, out_path: str, metrics: dict) -> None:
    with open(out_path, "w") as fh:
        json.dump({"checkpoint": best_dir, "tdc_path": tdc_path, "metrics": metrics}, fh, indent=2, sort_keys=True)


def _log_tdc_valid_wandb(trainer: Trainer, metrics: dict) -> None:
    if "wandb" not in trainer.args.report_to:
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.define_metric("tdc/valid/*", step_metric="tdc/valid/training_step")
            payload = {"tdc/valid/training_step": int(trainer.state.global_step)}
            payload.update({f"tdc/valid/{key}": value for key, value in metrics.items()})
            wandb.log(payload)
            for key, value in metrics.items():
                wandb.run.summary[f"tdc/valid/{key}"] = value
    except Exception:
        pass


def _run_record_knn_best_val_eval(cfg: Config, trainer: Trainer) -> None:
    if not cfg.train.record_knn_eval_enabled:
        return
    if not trainer.is_world_process_zero():
        _dist_barrier_if_needed()
        return

    reports = _evaluate_record_knn_final_checkpoints(cfg, trainer)
    _write_record_knn_final_reports(cfg, trainer, reports)
    _dist_barrier_if_needed()


def _evaluate_record_knn_final_checkpoints(cfg: Config, trainer: Trainer) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for label, dirname in _record_knn_final_checkpoint_specs():
        checkpoint_dir = os.path.join(cfg.paths.output_dir, dirname)
        if not os.path.exists(checkpoint_dir):
            print(f"[record-knn] skip {label} final eval; missing checkpoint {checkpoint_dir}")
            continue
        model = _load_record_knn_model(cfg, trainer, checkpoint_dir, label)
        reports[label] = {
            "checkpoint": checkpoint_dir,
            "metrics": _evaluate_record_knn_final_splits(cfg, model),
        }
    return reports


def _record_knn_final_checkpoint_specs() -> tuple[tuple[str, str], ...]:
    return (
        ("best_val", benchmark_spec.BEST_VAL_CHECKPOINT_DIR),
        ("record_knn_best", benchmark_spec.BEST_RECORD_KNN_CHECKPOINT_DIR),
    )


def _load_record_knn_model(cfg: Config, trainer: Trainer, checkpoint_dir: str, label: str):
    device = next(trainer.model.parameters()).device
    model = build_model(cfg)
    state = _load_checkpoint_state(checkpoint_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[record-knn] loaded {label} checkpoint missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


def _evaluate_record_knn_final_splits(cfg: Config, model) -> dict[str, dict[str, float | int]]:
    from .record_knn_eval import evaluate_record_knn, record_knn_result_dict

    payload = {}
    for split in cfg.train.record_knn_final_splits:
        result = evaluate_record_knn(
            cfg=cfg,
            model=model,
            dataset_dir=cfg.train.record_knn_eval_dataset_dir,
            cache_dir=cfg.train.record_knn_eval_cache_dir,
            split=split,
            dataset_config=cfg.train.record_knn_eval_dataset_config,
            top_fraction=cfg.train.record_knn_eval_top_fraction,
            k=cfg.train.record_knn_eval_k,
            batch_size=cfg.train.record_knn_eval_batch_size,
            max_queries=cfg.train.record_knn_eval_max_queries,
            require_cache=True,
        )
        payload[split] = record_knn_result_dict(result)
    return payload


def _write_record_knn_best_val_json(cfg: Config, best_dir: str, out_path: str, metrics: dict[str, Any]) -> None:
    with open(out_path, "w") as fh:
        json.dump(
            {
                "checkpoint": best_dir,
                "dataset_dir": cfg.train.record_knn_eval_dataset_dir,
                "dataset_config": cfg.train.record_knn_eval_dataset_config,
                "splits": cfg.train.record_knn_final_splits,
                "metrics": metrics,
            },
            fh,
            indent=2,
            sort_keys=True,
        )


def _write_record_knn_final_reports(cfg: Config, trainer: Trainer, reports: dict[str, dict[str, Any]]) -> None:
    best_val_report = reports.get("best_val")
    if best_val_report is not None:
        _write_record_knn_best_val_report(cfg, trainer, best_val_report)

    comparison_path = os.path.join(cfg.paths.output_dir, "record_knn_checkpoint_comparison.json")
    comparison = _record_knn_checkpoint_comparison(reports)
    _write_record_knn_comparison_json(cfg, comparison_path, reports, comparison)
    _log_record_knn_comparison_wandb(trainer, reports, comparison)
    print(f"[record-knn] checkpoint comparison -> {comparison_path}")


def _write_record_knn_best_val_report(cfg: Config, trainer: Trainer, report: dict[str, Any]) -> None:
    result_payload = report["metrics"]
    out_path = os.path.join(cfg.paths.output_dir, "record_knn_best_val.json")
    _write_record_knn_best_val_json(cfg, report["checkpoint"], out_path, result_payload)
    _log_record_knn_best_val_wandb(trainer, result_payload)
    print(
        f"[record-knn] best-val validation_1 macro_f1={result_payload['validation_1']['macro_f1']:.4f} "
        f"from {report['checkpoint']} -> {out_path}"
    )


def _record_knn_checkpoint_comparison(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    best_val = reports.get("best_val", {}).get("metrics", {})
    record_knn = reports.get("record_knn_best", {}).get("metrics", {})
    out: dict[str, Any] = {}
    for split in sorted(set(best_val) | set(record_knn)):
        split_out = _compare_record_knn_split(best_val.get(split, {}), record_knn.get(split, {}))
        if split_out:
            out[split] = split_out
    return out


def _compare_record_knn_split(best_val: dict[str, Any], record_knn: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in ("macro_f1", "accuracy", "transfer_precision", "transfer_recall"):
        if metric not in best_val or metric not in record_knn:
            continue
        best_value = float(best_val[metric])
        record_value = float(record_knn[metric])
        delta = record_value - best_value
        out[metric] = {
            "best_val": best_value,
            "record_knn_best": record_value,
            "record_knn_best_minus_best_val": delta,
            "winner": _comparison_winner(delta),
        }
    return out


def _comparison_winner(delta: float) -> str:
    if delta > 0:
        return "record_knn_best"
    if delta < 0:
        return "best_val"
    return "tie"


def _write_record_knn_comparison_json(
    cfg: Config,
    out_path: str,
    reports: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
) -> None:
    with open(out_path, "w") as fh:
        json.dump(
            {
                "dataset_dir": cfg.train.record_knn_eval_dataset_dir,
                "dataset_config": cfg.train.record_knn_eval_dataset_config,
                "splits": cfg.train.record_knn_final_splits,
                "checkpoints": reports,
                "comparison": comparison,
            },
            fh,
            indent=2,
            sort_keys=True,
        )


def _log_record_knn_best_val_wandb(trainer: Trainer, metrics: dict[str, dict[str, float | int]]) -> None:
    if "wandb" not in trainer.args.report_to:
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.define_metric("record_knn/best_val/*", step_metric="record_knn/best_val/training_step")
            wandb_payload = {"record_knn/best_val/training_step": int(trainer.state.global_step)}
            wandb_payload.update(_flat_record_knn_best_val(metrics))
            wandb.log(wandb_payload)
            for key, value in _flat_record_knn_best_val(metrics).items():
                wandb.run.summary[key] = value
    except Exception:
        pass


def _flat_record_knn_best_val(metrics: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for split, split_metrics in metrics.items():
        for key, value in split_metrics.items():
            out[f"record_knn/best_val/{split}/{key}"] = value
    return out


def _log_record_knn_comparison_wandb(
    trainer: Trainer,
    reports: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
) -> None:
    if "wandb" not in trainer.args.report_to:
        return
    try:
        import wandb

        if wandb.run is not None:
            step = int(trainer.state.global_step)
            payload = {"record_knn/checkpoint_comparison/training_step": step}
            payload.update(_flat_record_knn_report_metrics(reports))
            payload.update(_flat_record_knn_comparison(comparison))
            wandb.define_metric(
                "record_knn/checkpoint_comparison/*",
                step_metric="record_knn/checkpoint_comparison/training_step",
            )
            wandb.log(payload)
            for key, value in payload.items():
                if key.endswith("/training_step"):
                    continue
                wandb.run.summary[key] = value
    except Exception:
        pass


def _flat_record_knn_report_metrics(reports: dict[str, dict[str, Any]]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    report = reports.get("record_knn_best")
    if report is None:
        return out
    for split, split_metrics in report["metrics"].items():
        for key, value in split_metrics.items():
            out[f"record_knn/record_knn_best/{split}/{key}"] = value
    return out


def _flat_record_knn_comparison(comparison: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for split, split_metrics in comparison.items():
        for key, values in split_metrics.items():
            out[f"record_knn/checkpoint_comparison/{split}/{key}_delta"] = values[
                "record_knn_best_minus_best_val"
            ]
    return out


def _rank_info() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def _memmap_ready_marker(cfg: Config, train_split: str, eval_split: str) -> str:
    run_id = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT") or "single"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_id)
    value_tag = "source_value" if cfg.model.use_source_value else "plain"
    subset_tag = "subset" if cfg.train.eval_subset_metrics else "nosubset"
    return os.path.join(
        cfg.paths.memmap_dir, f".ready_{safe}_{value_tag}_{subset_tag}_{train_split}_{eval_split}.json"
    )


def _build_split_memmap_for_config(cfg: Config, split: str, rebuild: bool = False) -> dict:
    return build_split_memmap(
        cfg.paths.splits_dir,
        cfg.paths.memmap_dir,
        split,
        rebuild,
        base_parquet=cfg.paths.base_parquet,
        use_source_value=cfg.model.use_source_value,
        source_value_scale=cfg.model.source_value_scale,
        store_eval_subset=bool(cfg.train.eval_subset_metrics and split != "train"),
        eval_subset_names=tuple(cfg.train.eval_subset_names),
        store_similarity_bucket=bool(cfg.train.eval_similarity_bucket_metrics and split != "train"),
        similarity_bucket_names=tuple(cfg.train.eval_similarity_bucket_names),
    )


def _prepare_memmaps_distributed(cfg: Config, train_split: str, eval_split: str, rebuild: bool) -> None:
    """Build memmaps once under torchrun; all nonzero ranks wait for rank 0."""

    rank, world_size = _rank_info()
    if world_size <= 1:
        _build_split_memmap_for_config(cfg, train_split, rebuild)
        _build_split_memmap_for_config(cfg, eval_split, rebuild)
        return

    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)

    marker = _memmap_ready_marker(cfg, train_split, eval_split)
    if rank == 0:
        if os.path.exists(marker):
            os.unlink(marker)
        train_meta = _build_split_memmap_for_config(cfg, train_split, rebuild)
        eval_meta = _build_split_memmap_for_config(cfg, eval_split, rebuild)
        os.makedirs(cfg.paths.memmap_dir, exist_ok=True)
        with open(marker, "w") as fh:
            json.dump(
                {
                    "train_split": train_split,
                    "eval_split": eval_split,
                    "train_count": train_meta["count"],
                    "eval_count": eval_meta["count"],
                    "created_at": time.time(),
                },
                fh,
            )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    elif rank != 0:
        while not os.path.exists(marker):
            time.sleep(1.0)

    if rank != 0:
        # Re-read metadata without rebuilding; this verifies local visibility and signatures.
        _build_split_memmap_for_config(cfg, train_split, rebuild=False)
        _build_split_memmap_for_config(cfg, eval_split, rebuild=False)


def main() -> None:
    args = parse_train_args()
    cfg = setup_training_config(args)
    _prepare_memmaps_distributed(cfg, args.train_split, args.eval_split, args.rebuild_memmap)
    train_ds, eval_ds, eval_datasets = build_training_datasets(cfg, args)
    model = build_model(cfg)
    trainer = build_trainer(cfg, model, train_ds, eval_ds, eval_datasets)
    add_training_callbacks(cfg, trainer)
    trainer.train()
    trainer.evaluate(metric_key_prefix="eval")
    _run_tdc_valid_best_val_eval(cfg, trainer)
    _run_record_knn_best_val_eval(cfg, trainer)


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ml/configs/default.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--rebuild-memmap", action="store_true")
    return parser.parse_args()


def setup_training_config(args: argparse.Namespace) -> Config:
    cfg = Config.from_yaml(args.config).apply_overrides(args.overrides)
    tc = cfg.train
    if tc.metric_set not in {"binary", "simple_transfer"}:
        raise ValueError(f"unknown train.metric_set={tc.metric_set!r}")

    if tc.report_to == "wandb":
        # HF Trainer reads the wandb project from this env var; don't clobber a user override.
        os.environ.setdefault("WANDB_PROJECT", tc.wandb_project)

    if tc.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return cfg


def build_training_datasets(cfg: Config, args: argparse.Namespace):
    tc = cfg.train
    train_ds = PairDataset(cfg.paths.memmap_dir, args.train_split)
    eval_ds = PairDataset(cfg.paths.memmap_dir, args.eval_split)
    eval_datasets = {"val": eval_ds}
    if tc.train_eval_samples > 0:
        eval_datasets["train_sample"] = train_sample_subset(train_ds, tc)
    return train_ds, eval_ds, eval_datasets


def train_sample_subset(train_ds: PairDataset, tc) -> Subset:
    rng = np.random.default_rng(tc.seed)
    n_sample = min(tc.train_eval_samples, len(train_ds))
    sample_idx = rng.choice(len(train_ds), size=n_sample, replace=False).tolist()
    return Subset(train_ds, sample_idx)


def build_trainer(cfg: Config, model, train_ds: PairDataset, eval_ds: PairDataset, eval_datasets: dict) -> Trainer:
    tc = cfg.train
    eval_subset_codes = (
        np.asarray(eval_ds.eval_subset)
        if tc.eval_subset_metrics and eval_ds.eval_subset is not None
        else None
    )
    similarity_bucket_codes = (
        np.asarray(eval_ds.similarity_bucket)
        if tc.eval_similarity_bucket_metrics and eval_ds.similarity_bucket is not None
        else None
    )
    training_args = build_training_args(cfg)
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_datasets,
        data_collator=collate_pairs,
        compute_metrics=_compute_metrics_for(
            tc.metric_set,
            eval_subset_codes=eval_subset_codes,
            eval_subset_names=list(tc.eval_subset_names),
            similarity_bucket_codes=similarity_bucket_codes,
            similarity_bucket_names=list(tc.eval_similarity_bucket_names),
        ),
        callbacks=base_callbacks(cfg),
    )


def build_training_args(cfg: Config) -> TrainingArguments:
    tc = cfg.train
    warmup_kwargs = {"warmup_steps": tc.warmup_steps} if tc.warmup_steps and tc.warmup_steps > 0 else {"warmup_ratio": tc.warmup_ratio}
    hf_report_to = [] if tc.wandb_simple_validation_only or tc.report_to == "none" else [tc.report_to]
    return TrainingArguments(
        output_dir=cfg.paths.output_dir,
        per_device_train_batch_size=tc.per_device_batch_size,
        per_device_eval_batch_size=tc.per_device_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        **warmup_kwargs,
        max_grad_norm=tc.max_grad_norm,
        lr_scheduler_type=tc.lr_scheduler_type,
        # num_train_epochs > 0 takes precedence (epoch-based run); else max_steps.
        max_steps=(-1 if tc.num_train_epochs > 0 else tc.max_steps),
        num_train_epochs=(tc.num_train_epochs if tc.num_train_epochs > 0 else 3.0),
        eval_strategy="steps",
        eval_steps=tc.eval_steps,
        eval_on_start=True,
        save_strategy="no",  # no checkpoints / model saving
        logging_steps=tc.logging_steps,
        dataloader_num_workers=tc.dataloader_num_workers,
        dataloader_pin_memory=True,
        bf16=tc.bf16,
        tf32=tc.tf32,
        optim="adamw_torch_fused",
        torch_compile=tc.torch_compile,
        seed=tc.seed,
        report_to=hf_report_to,
        run_name=(tc.run_name or None),
        remove_unused_columns=False,  # our inputs are not model-signature columns
        label_names=["labels"],
        load_best_model_at_end=False,
    )


def base_callbacks(cfg: Config) -> list[TrainerCallback]:
    tc = cfg.train
    callbacks: list[TrainerCallback] = []
    if tc.wandb_simple_validation_only:
        callbacks.append(SimpleValidationWandbCallback(tc.wandb_project, run_display_name(cfg), os.environ.get("WANDB_RUN_GROUP")))
    elif tc.wandb_val_mirror:
        callbacks.append(WandbValMetricsCallback())
    callbacks.append(MetricsCsvCallback(metrics_csv_path(cfg)))
    return callbacks


def run_display_name(cfg: Config) -> str:
    return cfg.train.run_name or os.path.basename(cfg.paths.output_dir.rstrip("/"))


def metrics_csv_path(cfg: Config) -> str:
    return os.path.join("ml/results", cfg.paths.dataset, "runs", run_display_name(cfg), "metrics.csv")


def add_training_callbacks(cfg: Config, trainer: Trainer) -> None:
    tc = cfg.train
    best_cb = BestModelCallback(
        os.path.join(cfg.paths.output_dir, benchmark_spec.BEST_VAL_CHECKPOINT_DIR),
        metric=tc.best_metric,
    )
    trainer.add_callback(best_cb)
    best_cb.trainer = trainer
    if tc.tdc_eval_enabled:
        tdc_cb = TdcKnnEvalCallback(cfg)
        trainer.add_callback(tdc_cb)
        tdc_cb.trainer = trainer
    if tc.record_knn_eval_enabled:
        record_cb = RecordKnnEvalCallback(cfg)
        trainer.add_callback(record_cb)
        record_cb.trainer = trainer
    if tc.wandb_simple_validation_only and trainer.is_world_process_zero():
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass
    if torch.cuda.is_available() and trainer.is_world_process_zero():
        peak = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.max_memory_reserved() / 1e9
        print(f"[vram] peak allocated {peak:.1f} GB | reserved {reserved:.1f} GB (per device)")
    print(f"[done] training complete; best checkpoints saved under {cfg.paths.output_dir}")


if __name__ == "__main__":
    main()
