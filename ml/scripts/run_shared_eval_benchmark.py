#!/usr/bin/env python3
"""Run shared-eval MLP precompute, HP sweeps, final training, and HF upload.

The benchmark has two lanes:

    source_value:    bidirectional train, source value enabled, GPUs 4-7
    no_source_value: unidirectional train, source value disabled, GPUs 0-3

Sweep jobs are single-GPU with gradient accumulation 4. Final jobs use 4 GPUs
with gradient accumulation 1 so the effective global batch is comparable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starling_ml import benchmark_spec as spec  # noqa: E402

LRS = spec.LEARNING_RATES
EFFECTIVE_BATCH_SIZES = spec.EFFECTIVE_BATCH_SIZES
BASE_RESULTS_ROOT = Path("ml/results")
RECORD_KNN_DATASET_DIR = "datasets/starling_eval/condition_key_v3_record_splits_hf"
RECORD_KNN_DATASET_CONFIG = "full_metadata"
RECORD_KNN_CACHE_DIR = "ml/artifacts/record_knn_eval_cache/condition_key_v3_record_splits_hf"
RECORD_KNN_EVAL_SPLITS = ("validation_1",)
RECORD_KNN_FINAL_SPLITS = ("validation_1", "validation_2")
RECORD_KNN_TOP_FRACTION = "0.10"
RECORD_KNN_K = "10"
RECORD_KNN_MAX_QUERIES = "0"


@dataclass(frozen=True)
class Universe:
    key: str
    config: str
    final_port: int
    source_repo_id: str
    no_source_repo_id: str


@dataclass(frozen=True)
class Lane:
    key: str
    run_prefix: str
    use_source_value: bool
    sweep_gpus: tuple[str, str, str]
    final_gpus: str
    sweep_root: Path
    final_root: Path
    results_root: Path
    final_group: str
    split_version: str
    final_port_offset: int = 0


@dataclass(frozen=True)
class ModelPreset:
    key: str
    run_tag: str
    final_run_suffix: str
    overrides: tuple[str, ...]


UNIVERSES = (
    Universe(
        key="condition_key",
        config="ml/configs/shared_eval_condition_key.yaml",
        final_port=29640,
        source_repo_id="jiosephlee/starling-transfer-shared-eval-condition-key-source-value",
        no_source_repo_id="jiosephlee/starling-transfer-shared-eval-condition-key-no-source-value",
    ),
    Universe(
        key="same_species_v2",
        config="ml/configs/shared_eval_same_species_v2.yaml",
        final_port=29641,
        source_repo_id="jiosephlee/starling-transfer-shared-eval-same-species-v2-source-value",
        no_source_repo_id="jiosephlee/starling-transfer-shared-eval-same-species-v2-no-source-value",
    ),
    Universe(
        key="no_constraints",
        config="ml/configs/shared_eval_no_constraints.yaml",
        final_port=29642,
        source_repo_id="jiosephlee/starling-transfer-shared-eval-no-constraints-source-value",
        no_source_repo_id="jiosephlee/starling-transfer-shared-eval-no-constraints-no-source-value",
    ),
)

MODEL_PRESETS = {
    "default": ModelPreset(
        key="default",
        run_tag="step_logging_300_macro_f1_v1",
        final_run_suffix="step_logging_3000_macro_f1_v1",
        overrides=(),
    ),
    "small_lt10m_v1": ModelPreset(
        key="small_lt10m_v1",
        run_tag="step_logging_300_macro_f1_small_lt10m_v1",
        final_run_suffix="step_logging_3000_macro_f1_small_lt10m_v1",
        overrides=(
            "model.mol_hidden=384",
            "model.mol_out=256",
            "model.meta_field_proj=32",
            "model.d_model=384",
            "model.d_ff=1024",
            "model.n_blocks=6",
        ),
    ),
    "large_400m_v1": ModelPreset(
        key="large_400m_v1",
        run_tag="step_logging_300_macro_f1_large_400m_v1",
        final_run_suffix="step_logging_3000_macro_f1_large_400m_v1",
        overrides=(),
    ),
}


def _gpu_tuple(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or not value.strip():
        return default
    gpus = tuple(part.strip() for part in value.split(",") if part.strip())
    if not gpus:
        raise ValueError("GPU list must contain at least one device id")
    return gpus


def build_lanes(
    run_tag: str,
    *,
    sweep_gpus: str | None = None,
    final_gpus: str | None = None,
    split_version: str = "v3",
) -> tuple[Lane, ...]:
    source_sweep_gpus = _gpu_tuple(sweep_gpus, ("4", "5", "6"))
    nosv_sweep_gpus = _gpu_tuple(sweep_gpus, ("0", "1", "2"))
    source_final_gpus = ",".join(_gpu_tuple(final_gpus, ("4", "5", "6", "7")))
    nosv_final_gpus = ",".join(_gpu_tuple(final_gpus, ("0", "1", "2", "3")))
    return (
        Lane(
            key="source_value",
            run_prefix="srcval",
            use_source_value=True,
            sweep_gpus=source_sweep_gpus,
            final_gpus=source_final_gpus,
            sweep_root=Path(f"ml/artifacts/hp_sweeps/shared_eval_{run_tag}"),
            final_root=Path(f"ml/artifacts/runs/shared_eval_{run_tag}"),
            results_root=Path(f"ml/results/shared_eval_{run_tag}"),
            final_group=f"shared_eval_final_source_value_{run_tag}",
            split_version=split_version,
        ),
        Lane(
            key="no_source_value",
            run_prefix="nosv",
            use_source_value=False,
            sweep_gpus=nosv_sweep_gpus,
            final_gpus=nosv_final_gpus,
            sweep_root=Path(f"ml/artifacts/hp_sweeps/shared_eval_no_source_value_{run_tag}"),
            final_root=Path(f"ml/artifacts/runs/shared_eval_no_source_value_{run_tag}"),
            results_root=Path(f"ml/results/shared_eval_no_source_value_{run_tag}"),
            final_group=f"shared_eval_final_no_source_value_{run_tag}",
            split_version=split_version,
            final_port_offset=-100,
        ),
    )


def preset_with_suffix(preset: ModelPreset, suffix: str) -> ModelPreset:
    if not suffix:
        return preset
    safe = suffix.strip("_")
    return ModelPreset(
        key=preset.key,
        run_tag=f"{preset.run_tag}_{safe}",
        final_run_suffix=f"{preset.final_run_suffix}_{safe}",
        overrides=preset.overrides,
    )


def env_with_pythonpath(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "ml" if not current else f"ml:{current}"
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update(extra)
    return env


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=Path.cwd(), env=env, stdout=log, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def selected_lanes(name: str, lanes: tuple[Lane, ...]) -> tuple[Lane, ...]:
    if name == "both":
        return lanes
    for lane in lanes:
        if lane.key == name:
            return (lane,)
    raise ValueError(f"unknown lane: {name}")


def universe_by_key() -> dict[str, Universe]:
    return {universe.key: universe for universe in UNIVERSES}


def selected_universes(name: str) -> tuple[Universe, ...]:
    if name == "all":
        return UNIVERSES
    if name == "all_supported":
        return UNIVERSES
    by_key = universe_by_key()
    if name not in by_key:
        raise ValueError(f"unknown universe: {name}")
    return (by_key[name],)


def lane_dataset(lane: Lane, universe: Universe) -> str:
    suffix = "" if lane.split_version == "v3" else f"_{lane.split_version}"
    if lane.use_source_value:
        return f"shared_eval_{universe.key}{suffix}"
    return f"shared_eval_{universe.key}_no_source_value{suffix}"


def lane_memmap_dir(lane: Lane, universe: Universe) -> str:
    if lane.use_source_value:
        if lane.split_version == "v3_v2":
            return {
                "condition_key": "ml/artifacts/memmap_shared_eval_condition_key_source_value_v3_v2",
                "same_species_v2": "ml/artifacts/memmap_shared_eval_same_species_v2_source_value_v3_v2",
                "no_constraints": "ml/artifacts/memmap_shared_eval_no_constraints_source_value_v3_v2",
            }[universe.key]
        return {
            "condition_key": "ml/artifacts/memmap_shared_eval_condition_key_source_value",
            "same_species_v2": "ml/artifacts/memmap_shared_eval_same_species_source_value",
            "no_constraints": "ml/artifacts/memmap_shared_eval_no_constraints_source_value",
        }[universe.key]
    if lane.split_version == "v3_v2":
        return f"ml/artifacts/memmap_shared_eval_{universe.key}_no_source_value_v3_v2"
    return f"ml/artifacts/memmap_shared_eval_{universe.key}_no_source_value"


def lane_splits_dir(lane: Lane, universe: Universe) -> str:
    if lane.use_source_value:
        if lane.split_version == "v3_v2":
            return {
                "condition_key": "datasets/pairs_split_full/oral_bioavailability_condition_key_shared_eval_full_v3_v2",
                "same_species_v2": "datasets/pairs_split_full/oral_bioavailability_same_species_v2_shared_eval_full_v3_v2",
                "no_constraints": "datasets/pairs_split_full/oral_bioavailability_no_constraints_shared_eval_full_v3_v2",
            }[universe.key]
        return {
            "condition_key": "datasets/pairs_split_full/oral_bioavailability_condition_key_shared_eval_full_v3",
            "same_species_v2": "datasets/pairs_split_full/oral_bioavailability_same_species_v2_shared_eval_full_v3",
            "no_constraints": "datasets/pairs_split_full/oral_bioavailability_no_constraints_shared_eval_full_v3",
        }[universe.key]
    if lane.split_version == "v3_v2":
        return (
            "datasets/pairs_split_full/"
            f"oral_bioavailability_{universe.key}_shared_eval_unidirectional_full_v3_v2"
        )
    return (
        "datasets/pairs_split_full/"
        f"oral_bioavailability_{universe.key}_shared_eval_unidirectional_full_v3"
    )


def lane_overrides(lane: Lane, universe: Universe) -> list[str]:
    overrides = [
        f"paths.dataset={lane_dataset(lane, universe)}",
        f"paths.memmap_dir={lane_memmap_dir(lane, universe)}",
        f"paths.splits_dir={lane_splits_dir(lane, universe)}",
        f"model.use_source_value={'true' if lane.use_source_value else 'false'}",
    ]
    return overrides


def train_cmd(
    python: str,
    config: str,
    overrides: list[str],
    *,
    rebuild_memmap: bool,
) -> list[str]:
    cmd = [python, "-m", "starling_ml.train", "--config", config]
    if rebuild_memmap:
        cmd.append("--rebuild-memmap")
    cmd.extend(["--set", *overrides])
    return cmd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as fh:
        return json.load(fh)


def git_status_summary() -> str:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.stdout


def write_run_manifest(
    *,
    lane: Lane,
    universe: Universe,
    preset: ModelPreset,
    winner: dict[str, str],
    output_dir: Path,
    cmd: list[str],
    n_devices: int,
    run_name: str,
    returncode: int,
) -> None:
    splits_dir = Path(lane_splits_dir(lane, universe))
    split_meta_path = splits_dir / "metadata.json"
    memmap_dir = Path(lane_memmap_dir(lane, universe))
    manifest = {
        "run_name": run_name,
        "lane": lane.key,
        "universe": universe.key,
        "split_version": lane.split_version,
        "model_preset": preset.key,
        "run_tag": preset.run_tag,
        "final_run_suffix": preset.final_run_suffix,
        "splits_dir": str(splits_dir),
        "split_metadata_sha256": sha256_file(split_meta_path) if split_meta_path.exists() else "",
        "split_metadata": read_json_if_exists(split_meta_path),
        "memmap_dir": str(memmap_dir),
        "memmap_train_meta": read_json_if_exists(memmap_dir / "train" / "meta.json"),
        "memmap_validation_meta": read_json_if_exists(memmap_dir / "validation" / "meta.json"),
        "winner": winner,
        "command": cmd,
        "gpu_count": int(n_devices),
        "final_gpus": lane.final_gpus,
        "returncode": int(returncode),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_status_short": git_status_summary(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def read_best_metrics(dataset: str, run_name: str) -> dict[str, str]:
    path = BASE_RESULTS_ROOT / dataset / "runs" / run_name / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    best: dict[str, float] = {}
    wanted = ("transfer_precision", "transfer_recall", "macro_f1", "accuracy")
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("split") != "val/no_overlap":
                continue
            metric = row.get("metric")
            if metric not in wanted:
                continue
            try:
                value = float(row.get("value", "nan"))
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            best[metric] = value if metric not in best else max(best[metric], value)
    missing = [metric for metric in wanted if metric not in best]
    if missing:
        raise RuntimeError(f"missing validation no_overlap metrics {missing} in {path}")
    return {f"best_{metric}": f"{best[metric]:.6f}" for metric in wanted}


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def write_csv_and_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.with_suffix(".csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def _merge_rows_by_universe(
    existing: list[dict[str, str]],
    updates: list[dict[str, str]],
) -> list[dict[str, str]]:
    updated_keys = {(row["lane"], row["universe"]) for row in updates}
    merged = [row for row in existing if (row.get("lane", ""), row.get("universe", "")) not in updated_keys]
    merged.extend(updates)
    return merged


def candidate_name(lane: Lane, universe: Universe, lr: str, batch_size: str, run_tag: str) -> str:
    return f"hp_{lane.run_prefix}_{universe.key}_lr{lr}_bs{batch_size}_ga4_s300_{run_tag}"


def final_run_name(
    lane: Lane,
    universe: Universe,
    winner: dict[str, str],
    suffix: str = "",
) -> str:
    effective_batch = winner.get("effective_batch_size") or winner["per_device_batch_size"]
    name = (
        f"final_{lane.run_prefix}_{universe.key}_lr{winner['lr']}"
        f"_ebs{effective_batch}_ga1"
    )
    return f"{name}_{suffix}" if suffix else name


def _run_sweep_candidate(
    lane: Lane,
    universe: Universe,
    gpu: str,
    python: str,
    preset: ModelPreset,
    dataset: str,
    lr: str,
    effective_batch_size: str,
    *,
    rebuild_memmap: bool,
) -> dict[str, str]:
    log_dir = lane.sweep_root / universe.key
    batch = spec.batch_plan(effective_batch_size, n_devices=1, gradient_accumulation_steps=4)
    run_name = candidate_name(lane, universe, lr, str(effective_batch_size), preset.run_tag)
    output_dir = log_dir / run_name
    results_dir = BASE_RESULTS_ROOT / dataset / "runs" / run_name
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(results_dir, ignore_errors=True)
    env = sweep_env(lane, universe, gpu)
    overrides = sweep_overrides(lane, universe, preset, run_name, lr, batch, output_dir)
    cmd = train_cmd(python, universe.config, overrides, rebuild_memmap=rebuild_memmap)
    log_path = log_dir / f"{run_name}.log"
    returncode = run_logged(cmd, log_path, env)
    row = sweep_result_row(lane, universe, run_name, lr, batch, returncode, log_path)
    update_sweep_metrics(row, dataset, run_name, log_path, returncode)
    shutil.rmtree(output_dir, ignore_errors=True)
    return row


def sweep_env(lane: Lane, universe: Universe, gpu: str) -> dict[str, str]:
    return env_with_pythonpath(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "WANDB_PROJECT": spec.HP_SWEEP_PROJECT,
            "WANDB_RUN_GROUP": f"shared_eval_hp_{universe.key}_{lane.key}_v1",
        }
    )


def sweep_overrides(
    lane: Lane,
    universe: Universe,
    preset: ModelPreset,
    run_name: str,
    lr: str,
    batch: spec.BatchPlan,
    output_dir: Path,
) -> list[str]:
    return [
        *lane_overrides(lane, universe),
        *preset.overrides,
        *sweep_training_overrides(batch),
        f"train.run_name={run_name}",
        f"train.learning_rate={lr}",
        f"train.per_device_batch_size={batch.per_device_batch_size}",
        "loss.kind=bce",
        "loss.pos_weight=1.85",
        f"paths.output_dir={output_dir}",
    ]


def sweep_training_overrides(batch: spec.BatchPlan) -> list[str]:
    return [
        "train.num_train_epochs=0",
        f"train.max_steps={spec.SWEEP_MAX_STEPS}",
        f"train.eval_steps={spec.SWEEP_EVAL_STEPS}",
        "train.logging_steps=1",
        f"train.gradient_accumulation_steps={batch.gradient_accumulation_steps}",
        "train.torch_compile=false",
        "train.train_eval_samples=0",
        "train.eval_subset_metrics=true",
        "train.eval_similarity_bucket_metrics=false",
        "train.wandb_simple_validation_only=false",
        "train.wandb_val_mirror=false",
        "train.metric_set=simple_transfer",
        f"train.best_metric={spec.HP_SELECT_METRIC}",
        "train.report_to=wandb",
        f"train.wandb_project={spec.HP_SWEEP_PROJECT}",
    ]


def sweep_result_row(
    lane: Lane,
    universe: Universe,
    run_name: str,
    lr: str,
    batch: spec.BatchPlan,
    returncode: int,
    log_path: Path,
) -> dict[str, str]:
    return {
        "lane": lane.key,
        "universe": universe.key,
        "run_name": run_name,
        "lr": lr,
        "effective_batch_size": str(batch.effective_batch_size),
        "per_device_batch_size": str(batch.per_device_batch_size),
        "status": "ok" if returncode == 0 else "failed",
        "best_transfer_precision": "",
        "best_transfer_recall": "",
        "best_macro_f1": "",
        "best_accuracy": "",
        "returncode": str(returncode),
        "log": str(log_path),
    }


def update_sweep_metrics(row: dict[str, str], dataset: str, run_name: str, log_path: Path, returncode: int) -> None:
    if returncode == 0:
        try:
            row.update(read_best_metrics(dataset, run_name))
        except Exception as exc:  # keep sweeping even if parsing failed.
            row["status"] = "metric_parse_failed"
            row["returncode"] = f"0:{type(exc).__name__}:{exc}"
    else:
        text = log_path.read_text(errors="ignore").lower()
        if "out of memory" in text or "cuda oom" in text:
            row["status"] = "oom"


def sweep_universe(
    lane: Lane,
    universe: Universe,
    gpus: tuple[str, ...],
    python: str,
    rebuild_memmap: bool,
    preset: ModelPreset,
) -> list[dict[str, str]]:
    log_dir = lane.sweep_root / universe.key
    result_path = log_dir / "results.tsv"
    rows: list[dict[str, str]] = read_tsv_rows(result_path)
    pending = pending_sweep_candidates(lane, universe, rows, preset)
    if not pending:
        return rows
    dataset = lane_dataset(lane, universe)
    maybe_run_rebuild_sweep(lane, universe, gpus, python, preset, dataset, result_path, rows, pending, rebuild_memmap)
    if pending:
        run_pending_sweeps(lane, universe, gpus, python, preset, dataset, result_path, rows, pending)
    return rows


def pending_sweep_candidates(
    lane: Lane,
    universe: Universe,
    rows: list[dict[str, str]],
    preset: ModelPreset,
) -> list[tuple[str, str]]:
    completed = {row.get("run_name", "") for row in rows if row.get("status") in {"ok", "oom"}}
    return [
        (lr, effective_batch_size)
        for lr in LRS
        for effective_batch_size in EFFECTIVE_BATCH_SIZES
        if candidate_name(lane, universe, lr, str(effective_batch_size), preset.run_tag) not in completed
    ]


def maybe_run_rebuild_sweep(
    lane: Lane,
    universe: Universe,
    gpus: tuple[str, ...],
    python: str,
    preset: ModelPreset,
    dataset: str,
    result_path: Path,
    rows: list[dict[str, str]],
    pending: list[tuple[str, str]],
    rebuild_memmap: bool,
) -> None:
    if not rebuild_memmap or rows or not pending:
        return
    lr, effective_batch_size = pending.pop(0)
    row = _run_sweep_candidate(lane, universe, gpus[0], python, preset, dataset, lr, effective_batch_size, rebuild_memmap=True)
    rows.append(row)
    write_rows(result_path, rows)


def run_pending_sweeps(
    lane: Lane,
    universe: Universe,
    gpus: tuple[str, ...],
    python: str,
    preset: ModelPreset,
    dataset: str,
    result_path: Path,
    rows: list[dict[str, str]],
    pending: list[tuple[str, str]],
) -> None:
    with ThreadPoolExecutor(max_workers=min(len(gpus), len(pending))) as pool:
        futures = submit_sweep_futures(pool, lane, universe, gpus, python, preset, dataset, pending)
        for future in as_completed(futures):
            rows.append(future.result())
            write_rows(result_path, rows)


def submit_sweep_futures(
    pool: ThreadPoolExecutor,
    lane: Lane,
    universe: Universe,
    gpus: tuple[str, ...],
    python: str,
    preset: ModelPreset,
    dataset: str,
    pending: list[tuple[str, str]],
) -> dict[Any, tuple[str, str, str]]:
    futures = {}
    for idx, (lr, effective_batch_size) in enumerate(pending):
        gpu = gpus[idx % len(gpus)]
        futures[pool.submit(_run_sweep_candidate, lane, universe, gpu, python, preset, dataset, lr, effective_batch_size, rebuild_memmap=False)] = (lr, effective_batch_size, gpu)
    return futures


def winner_for(rows: list[dict[str, str]], universe: Universe) -> dict[str, str]:
    valid = [row for row in rows if row["universe"] == universe.key and row["status"] == "ok"]
    if not valid:
        raise RuntimeError(f"no successful HP candidates for {universe.key}")

    def key(row: dict[str, str]):
        return (
            float(row["best_macro_f1"]),
            float(row["best_accuracy"]),
            float(row["best_transfer_precision"]),
            float(row["best_transfer_recall"]),
            -int(row.get("effective_batch_size") or row["per_device_batch_size"]),
            -float(row["lr"]),
        )

    return max(valid, key=key)


def run_precompute(python: str, run_tag: str) -> None:
    env = env_with_pythonpath({"CUDA_VISIBLE_DEVICES": "7"})
    cmd = [
        python,
        "-m",
        "starling_ml.precompute_embeddings",
        "--config",
        "ml/configs/shared_eval_condition_key.yaml",
    ]
    code = run_logged(cmd, Path(f"ml/artifacts/hp_sweeps/shared_eval_{run_tag}/precompute.log"), env)
    if code != 0:
        raise SystemExit(f"precompute failed; see ml/artifacts/hp_sweeps/shared_eval_{run_tag}/precompute.log")


def run_sweeps(
    python: str,
    lanes: tuple[Lane, ...],
    universes: tuple[Universe, ...],
    rebuild_memmap: bool,
    preset: ModelPreset,
) -> dict[str, list[dict[str, str]]]:
    rows_by_lane: dict[str, list[dict[str, str]]] = {lane.key: [] for lane in lanes}
    with ThreadPoolExecutor(max_workers=len(lanes) * len(universes)) as pool:
        futures = {}
        for lane in lanes:
            if len(lane.sweep_gpus) < len(universes):
                raise ValueError(
                    f"lane {lane.key} has {len(lane.sweep_gpus)} sweep GPUs for {len(universes)} universes"
                )
            if len(universes) == 1:
                gpu_groups = {universes[0]: lane.sweep_gpus}
            else:
                gpu_groups = {universe: (gpu,) for universe, gpu in zip(universes, lane.sweep_gpus)}
            for universe, gpus in gpu_groups.items():
                futures[pool.submit(sweep_universe, lane, universe, gpus, python, rebuild_memmap, preset)] = (
                    lane,
                    universe,
                )
        for future in as_completed(futures):
            lane, universe = futures[future]
            try:
                rows_by_lane[lane.key].extend(future.result())
            except Exception as exc:
                raise RuntimeError(f"sweep failed for {lane.key}/{universe.key}") from exc

    for lane in lanes:
        rows = rows_by_lane[lane.key]
        all_results_path = lane.sweep_root / "all_results.tsv"
        write_rows(all_results_path, _merge_rows_by_universe(read_tsv_rows(all_results_path), rows))
        winners = [winner_for(rows, universe) for universe in universes]
        winners_path = lane.results_root / "tables" / "hp_sweep_winners"
        write_csv_and_md(winners_path, _merge_rows_by_universe(read_csv_rows(winners_path.with_suffix(".csv")), winners))
    return rows_by_lane


def read_winners(lane: Lane) -> list[dict[str, str]]:
    path = lane.results_root / "tables" / "hp_sweep_winners.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as fh:
        return list(csv.DictReader(fh))


def repo_id_for(lane: Lane, universe: Universe) -> str:
    return universe.source_repo_id if lane.use_source_value else universe.no_source_repo_id


def run_upload(
    lane: Lane,
    universe: Universe,
    run_dir: Path,
    python: str,
    *,
    public_hf: bool,
) -> None:
    repo_id = repo_id_for(lane, universe)
    log_path = lane.final_root / f"{run_dir.name}.upload.log"
    cmd = [
        python,
        "-m",
        "starling_ml.export_upload",
        "--run",
        str(run_dir),
        "--config",
        universe.config,
        "--out",
        str(run_dir / "export"),
        "--repo-id",
        repo_id,
    ]
    if public_hf:
        cmd.append("--public")
    code = run_logged(cmd, log_path, env_with_pythonpath())
    if code != 0:
        raise SystemExit(f"upload failed for {lane.key}/{universe.key}; see {log_path}")


def run_final(
    lane: Lane,
    universe: Universe,
    winner: dict[str, str],
    python: str,
    preset: ModelPreset,
    *,
    final_run_suffix: str = "",
    rebuild_memmap: bool,
    upload_hf: bool,
    public_hf: bool,
) -> None:
    run_name = final_run_name(lane, universe, winner, final_run_suffix)
    output_dir = lane.final_root / run_name
    log_path = lane.final_root / f"{run_name}.log"
    n_devices = len([gpu for gpu in lane.final_gpus.split(",") if gpu.strip()])
    env = final_env(lane)
    overrides = final_overrides(lane, universe, winner, preset, run_name, output_dir, n_devices)
    prebuild_record_knn_caches(python, universe.config, lane.final_root / f"{run_name}_record_knn_cache.log")
    train = train_cmd(python, universe.config, overrides, rebuild_memmap=rebuild_memmap)
    cmd = final_distributed_cmd(python, train, n_devices, universe.final_port + lane.final_port_offset)
    code = run_logged(cmd, log_path, env)
    if code != 0:
        raise SystemExit(f"final run failed for {lane.key}/{universe.key}; see {log_path}")
    write_run_manifest(
        lane=lane,
        universe=universe,
        preset=preset,
        winner=winner,
        output_dir=output_dir,
        cmd=cmd,
        n_devices=n_devices,
        run_name=run_name,
        returncode=code,
    )
    if upload_hf:
        run_upload(lane, universe, output_dir, python, public_hf=public_hf)


def final_env(lane: Lane) -> dict[str, str]:
    return env_with_pythonpath(
        {
            "CUDA_VISIBLE_DEVICES": lane.final_gpus,
            "WANDB_PROJECT": spec.FULL_RUN_PROJECT,
            "WANDB_RUN_GROUP": lane.final_group,
        }
    )


def final_overrides(
    lane: Lane,
    universe: Universe,
    winner: dict[str, str],
    preset: ModelPreset,
    run_name: str,
    output_dir: Path,
    n_devices: int,
) -> list[str]:
    effective = winner.get("effective_batch_size") or winner["per_device_batch_size"]
    batch = spec.batch_plan(effective, n_devices=n_devices, gradient_accumulation_steps=1)
    return [
        *lane_overrides(lane, universe),
        *preset.overrides,
        *final_training_overrides(),
        f"train.run_name={run_name}",
        f"train.learning_rate={winner['lr']}",
        f"train.per_device_batch_size={batch.per_device_batch_size}",
        f"paths.output_dir={output_dir}",
    ]


def final_training_overrides() -> list[str]:
    return [
        "train.gradient_accumulation_steps=1",
        "train.num_train_epochs=0",
        "train.max_steps=3000",
        "train.eval_steps=50",
        "train.logging_steps=1",
        "train.train_eval_samples=0",
        "train.eval_subset_metrics=true",
        "train.eval_similarity_bucket_metrics=true",
        "train.wandb_simple_validation_only=false",
        "train.wandb_val_mirror=false",
        "train.metric_set=simple_transfer",
        f"train.best_metric={spec.HP_SELECT_METRIC}",
        "train.report_to=wandb",
        f"train.wandb_project={spec.FULL_RUN_PROJECT}",
        "train.tdc_eval_enabled=false",
        *record_knn_training_overrides(),
    ]


def record_knn_training_overrides() -> list[str]:
    return [
        "train.record_knn_eval_enabled=true",
        f"train.record_knn_eval_dataset_dir={RECORD_KNN_DATASET_DIR}",
        f"train.record_knn_eval_dataset_config={RECORD_KNN_DATASET_CONFIG}",
        'train.record_knn_eval_splits=["validation_1"]',
        'train.record_knn_final_splits=["validation_1","validation_2"]',
        f"train.record_knn_eval_cache_dir={RECORD_KNN_CACHE_DIR}",
        "train.record_knn_eval_steps=500",
        f"train.record_knn_eval_top_fraction={RECORD_KNN_TOP_FRACTION}",
        f"train.record_knn_eval_k={RECORD_KNN_K}",
        "train.record_knn_eval_batch_size=4096",
        f"train.record_knn_eval_max_queries={RECORD_KNN_MAX_QUERIES}",
    ]


def final_distributed_cmd(python: str, train: list[str], n_devices: int, port: int) -> list[str]:
    return [
        python,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={n_devices}",
        f"--master_port={port}",
        *train[1:],
    ]


def prebuild_record_knn_caches(python: str, config: str, log_path: Path) -> None:
    for split in record_knn_prebuild_splits():
        split_log = log_path.with_name(f"{log_path.stem}_{split}{log_path.suffix}")
        code = run_logged(record_knn_cache_cmd(python, config, split), split_log, env_with_pythonpath())
        if code != 0:
            raise SystemExit(f"record KNN cache prebuild failed for {split}; see {split_log}")


def record_knn_prebuild_splits() -> tuple[str, ...]:
    return tuple(dict.fromkeys((*RECORD_KNN_EVAL_SPLITS, *RECORD_KNN_FINAL_SPLITS)))


def record_knn_cache_cmd(python: str, config: str, split: str) -> list[str]:
    return [
        python,
        "-m",
        "starling_ml.record_knn_eval",
        "--config",
        config,
        "--cache-only",
        "--split",
        split,
        "--dataset-dir",
        RECORD_KNN_DATASET_DIR,
        "--dataset-config",
        RECORD_KNN_DATASET_CONFIG,
        "--cache-dir",
        RECORD_KNN_CACHE_DIR,
        "--top-fraction",
        RECORD_KNN_TOP_FRACTION,
        "--max-queries",
        RECORD_KNN_MAX_QUERIES,
    ]


def run_finals_for_lane(
    lane: Lane,
    universes: tuple[Universe, ...],
    python: str,
    preset: ModelPreset,
    *,
    winners: list[dict[str, str]] | None,
    final_run_suffix: str,
    rebuild_memmap: bool,
    upload_hf: bool,
    public_hf: bool,
) -> None:
    winner_by_universe = {row["universe"]: row for row in (winners or read_winners(lane))}
    for universe in universes:
        run_final(
            lane,
            universe,
            winner_by_universe[universe.key],
            python,
            preset,
            final_run_suffix=final_run_suffix,
            rebuild_memmap=rebuild_memmap,
            upload_hf=upload_hf,
            public_hf=public_hf,
        )


def run_finals(
    python: str,
    lanes: tuple[Lane, ...],
    universes: tuple[Universe, ...],
    preset: ModelPreset,
    *,
    rows_by_lane: dict[str, list[dict[str, str]]] | None,
    final_run_suffix: str,
    rebuild_memmap: bool,
    upload_hf: bool,
    public_hf: bool,
) -> None:
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = {}
        for lane in lanes:
            winners = None
            if rows_by_lane is not None and lane.key in rows_by_lane:
                winners = [winner_for(rows_by_lane[lane.key], universe) for universe in universes]
            futures[
                pool.submit(
                    run_finals_for_lane,
                    lane,
                    universes,
                    python,
                    preset,
                    winners=winners,
                    final_run_suffix=final_run_suffix,
                    rebuild_memmap=rebuild_memmap,
                    upload_hf=upload_hf,
                    public_hf=public_hf,
                )
            ] = lane
        for future in as_completed(futures):
            lane = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"finals failed for lane {lane.key}") from exc


def run_uploads_only(
    python: str,
    lanes: tuple[Lane, ...],
    universes: tuple[Universe, ...],
    *,
    final_run_suffix: str,
    public_hf: bool,
) -> None:
    for lane in lanes:
        winner_by_universe = {row["universe"]: row for row in read_winners(lane)}
        for universe in universes:
            run_name = final_run_name(lane, universe, winner_by_universe[universe.key], final_run_suffix)
            run_upload(lane, universe, lane.final_root / run_name, python, public_hf=public_hf)


def main() -> None:
    args = benchmark_arg_parser().parse_args()
    preset, lanes, universes, final_run_suffix = benchmark_context(args)
    run_selected_phase(args, preset, lanes, universes, final_run_suffix)


def benchmark_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "precompute", "sweeps", "finals", "uploads"),
        default="all",
    )
    parser.add_argument(
        "--lane",
        choices=("both", "source_value", "no_source_value"),
        default="both",
    )
    parser.add_argument(
        "--universe",
        choices=("condition_key", "same_species_v2", "no_constraints", "all", "all_supported"),
        default="all",
        help="Dataset universe(s) to run. all_supported is kept as an alias for all.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--sweep-gpus",
        default=None,
        help="Comma-separated physical GPU ids for per-universe single-GPU sweeps.",
    )
    parser.add_argument(
        "--final-gpus",
        default=None,
        help="Comma-separated physical GPU ids for each distributed final run.",
    )
    parser.add_argument(
        "--model-preset",
        choices=tuple(MODEL_PRESETS),
        default="default",
        help="Architecture preset to apply on top of the dataset config.",
    )
    parser.add_argument(
        "--split-version",
        choices=("v3", "v3_v2"),
        default="v3_v2",
        help="Shared-eval split lineage to use for path/memmap overrides.",
    )
    parser.add_argument(
        "--run-tag-suffix",
        default="",
        help="Suffix appended to sweep/final run tags, e.g. ssv2_v3_v2.",
    )
    parser.add_argument("--rebuild-memmap", action="store_true")
    parser.add_argument("--upload-hf", action="store_true")
    parser.add_argument("--public-hf", action="store_true")
    parser.add_argument(
        "--final-run-suffix",
        default="",
        help="Optional suffix for final run names/output dirs, e.g. rebuilt_splits_precision_v1.",
    )
    return parser


def benchmark_context(args: argparse.Namespace) -> tuple[ModelPreset, tuple[Lane, ...], tuple[Universe, ...], str]:
    preset = preset_with_suffix(MODEL_PRESETS[args.model_preset], args.run_tag_suffix)
    lanes = selected_lanes(
        args.lane,
        build_lanes(
            preset.run_tag,
            sweep_gpus=args.sweep_gpus,
            final_gpus=args.final_gpus,
            split_version=args.split_version,
        ),
    )
    universes = selected_universes(args.universe)
    final_run_suffix = args.final_run_suffix or preset.final_run_suffix
    return preset, lanes, universes, final_run_suffix


def run_selected_phase(
    args: argparse.Namespace,
    preset: ModelPreset,
    lanes: tuple[Lane, ...],
    universes: tuple[Universe, ...],
    final_run_suffix: str,
) -> None:
    if args.phase in {"all", "precompute"}:
        run_precompute(args.python, preset.run_tag)
    rows_by_lane = None
    if args.phase in {"all", "sweeps"}:
        rows_by_lane = run_sweeps(args.python, lanes, universes, args.rebuild_memmap, preset)
    if args.phase in {"all", "finals"}:
        run_finals(
            args.python,
            lanes,
            universes,
            preset,
            rows_by_lane=rows_by_lane,
            final_run_suffix=final_run_suffix,
            rebuild_memmap=args.rebuild_memmap,
            upload_hf=args.upload_hf,
            public_hf=args.public_hf,
        )
    if args.phase == "uploads":
        run_uploads_only(
            args.python,
            lanes,
            universes,
            final_run_suffix=final_run_suffix,
            public_hf=args.public_hf,
        )


if __name__ == "__main__":
    main()
