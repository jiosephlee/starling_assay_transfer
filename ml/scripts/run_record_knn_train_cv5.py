from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from starling_ml.config import Config
from starling_ml.evaluate import _load_state_dict
from starling_ml.knn_data import FULL_METADATA_CONFIG, load_record_sources
from starling_ml.knn_pipeline import compute_metrics
from starling_ml.model import build_model


DATASET_DIR = Path("datasets/starling_eval/condition_key_v3_record_splits_hf")
CACHE_DIR = Path("ml/artifacts/record_knn_eval_cache/condition_key_v3_record_splits_hf_train_cv5")
RESULTS_DIR = Path("ml/results/tables")
ROW_DIR = RESULTS_DIR / "eval_tracking_rerun_v1_train_cv_rows"
LOG_DIR = Path("tmp/record_knn_train_cv_work/logs")
RUN_GLOB = "ml/artifacts/runs/*eval_tracking_rerun_v1/final_*/run_manifest.json"
UNIVERSES = ("condition_key", "same_species_v2", "no_constraints")
METRIC_FIELDS = ("train_cv5_macro_f1", "train_cv5_accuracy")
TOP_FRACTION = 0.10
N_FOLDS = 5
SEED = 1729
K = 10
QUERY_CHUNK = 64
SCORE_BATCH = 131072


@dataclass(frozen=True)
class FoldCache:
    fold: int
    query_indices: np.ndarray
    source_indices: np.ndarray
    positions: np.ndarray


@dataclass(frozen=True)
class Task:
    label: str
    ckpt_dir: str
    manifest: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--ckpt-dir")
    parser.add_argument("--label")
    parser.add_argument("--out")
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--row-dir", type=Path, default=ROW_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--run-glob", default=RUN_GLOB)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--validate-rows", action="store_true")
    return parser.parse_args()


def stratified_folds(labels: np.ndarray) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    folds = [[] for _ in range(N_FOLDS)]
    for label in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        for fold, part in enumerate(np.array_split(idx, N_FOLDS)):
            folds[fold].extend(part.tolist())
    return [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]


def fold_data_path(fold: int) -> Path:
    stem = f"fold{fold}_top_fraction_{TOP_FRACTION:.2f}_seed_{SEED}"
    return CACHE_DIR / f"{stem}.npz"


def load_fold_caches(rows) -> list[FoldCache]:
    labels = rows["label"].to_numpy(dtype=np.int8)
    folds = stratified_folds(labels)
    all_idx = np.arange(len(rows), dtype=np.int64)
    caches = []
    for fold, query_idx in enumerate(folds):
        source_idx = np.setdiff1d(all_idx, query_idx, assume_unique=True)
        data = np.load(fold_data_path(fold))
        positions = data["positions"].astype(np.uint32)
        caches.append(FoldCache(fold, query_idx, source_idx, positions))
    return caches


def manifest_command(path: Path) -> list[str]:
    return json.loads(path.read_text())["command"]


def config_from_manifest(path: Path) -> Config:
    cmd = manifest_command(path)
    cfg_path = cmd[cmd.index("--config") + 1]
    overrides = cmd[cmd.index("--set") + 1 :]
    return Config.from_yaml(cfg_path).apply_overrides(overrides)


def run_parts(path: Path) -> tuple[str, str, str]:
    text = str(path)
    model = "large_400m_v1" if "large_400m_v1" in text else "small_lt10m_v1"
    lane = "no_source_value" if "no_source_value" in text else "source_value"
    universe = next(v for v in UNIVERSES if f"_{v}_" in text)
    return universe, lane, model


def evaluate_checkpoint(rows, folds: list[FoldCache], manifest: Path, ckpt_dir: str) -> dict:
    cfg = config_from_manifest(manifest)
    model = build_model(cfg)
    model.load_state_dict(_load_state_dict(str(manifest.parent / ckpt_dir)), strict=False)
    model.to("cuda").eval()
    metrics = []
    for fold in folds:
        print(f"[score] {ckpt_dir} fold {fold.fold}", flush=True)
        metrics.append(evaluate_fold(rows, fold, model, cfg))
    del model
    torch.cuda.empty_cache()
    return average_metrics(metrics)


def evaluate_fold(rows, fold: FoldCache, model, cfg: Config) -> dict:
    arrays = fold_arrays(rows, fold)
    labels, preds = [], []
    for start in range(0, len(arrays["query_labels"]), QUERY_CHUNK):
        stop = min(start + QUERY_CHUNK, len(arrays["query_labels"]))
        pred = predict_query_chunk(arrays, fold.positions[start:stop], start, model, cfg)
        preds.append(pred)
        labels.append(arrays["query_labels"][start:stop])
    return compute_metrics(np.concatenate(labels), np.concatenate(preds))


def fold_arrays(rows, fold: FoldCache) -> dict[str, np.ndarray]:
    sources = rows.iloc[fold.source_indices]
    queries = rows.iloc[fold.query_indices]
    return {
        "source_row_index": sources["row_index"].to_numpy(dtype=np.int64),
        "source_value": sources["oral_bioavailability_value"].to_numpy(dtype=np.float32),
        "source_labels": sources["label"].to_numpy(dtype=np.int8),
        "query_row_index": queries["row_index"].to_numpy(dtype=np.int64),
        "query_labels": queries["label"].to_numpy(dtype=np.int8),
    }


def predict_query_chunk(arrays: dict, cand: np.ndarray, offset: int, model, cfg: Config) -> np.ndarray:
    cand = cand.astype(np.int64)
    scores = score_query_chunk(arrays, cand, offset, model, cfg)
    top = np.argsort(scores, axis=1)[:, ::-1][:, :K]
    labels = arrays["source_labels"][cand]
    top_labels = np.take_along_axis(labels, top, axis=1)
    top_scores = np.take_along_axis(scores, top, axis=1)
    weights = np.clip(top_scores, 0.0, None)
    probs = (top_labels * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-12)
    return (probs >= 0.5).astype(np.int8)


def score_query_chunk(arrays: dict, cand: np.ndarray, offset: int, model, cfg: Config) -> np.ndarray:
    flat = cand.reshape(-1)
    local = np.repeat(np.arange(cand.shape[0], dtype=np.int64), cand.shape[1]) + int(offset)
    source_idx = arrays["source_row_index"][flat]
    query_idx = arrays["query_row_index"][local]
    source_value = arrays["source_value"][flat]
    return score_pairs(model, source_idx, query_idx, source_value, cfg).reshape(cand.shape)


def score_pairs(model, source_idx, query_idx, source_value, cfg: Config) -> np.ndarray:
    out = np.empty(len(source_idx), dtype=np.float32)
    use_sv = bool(getattr(model, "use_source_value", False))
    with torch.inference_mode():
        for start in range(0, len(source_idx), SCORE_BATCH):
            stop = min(start + SCORE_BATCH, len(source_idx))
            kwargs = pair_kwargs(source_idx, query_idx, source_value, start, stop, cfg, use_sv)
            out[start:stop] = torch.sigmoid(model(**kwargs)["logits"]).float().cpu().numpy()
    return out


def pair_kwargs(source_idx, query_idx, source_value, start: int, stop: int, cfg: Config, use_sv: bool) -> dict:
    kwargs = {
        "a_idx": torch.from_numpy(source_idx[start:stop]).to("cuda"),
        "b_idx": torch.from_numpy(query_idx[start:stop]).to("cuda"),
    }
    if use_sv:
        scaled = source_value[start:stop] / np.float32(cfg.model.source_value_scale)
        kwargs["source_value"] = torch.from_numpy(scaled).to("cuda")
    return kwargs


def average_metrics(metrics: list[dict]) -> dict[str, float]:
    return {
        "train_cv5_macro_f1": float(np.mean([m["macro_f1"] for m in metrics])),
        "train_cv5_accuracy": float(np.mean([m["accuracy"] for m in metrics])),
    }


def worker_main(args: argparse.Namespace) -> None:
    rows = load_record_sources(DATASET_DIR, config_name=FULL_METADATA_CONFIG)
    folds = load_fold_caches(rows)
    manifest = Path(args.manifest)
    universe, lane, model = run_parts(manifest.parent)
    metrics = evaluate_checkpoint(rows, folds, manifest, args.ckpt_dir)
    payload = {"label": args.label, "universe": universe, "lane": lane, "model": model, **metrics}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(format_row(payload), indent=2, sort_keys=True) + "\n")
    print(json.dumps(format_row(payload), sort_keys=True), flush=True)


def format_row(row: dict) -> dict:
    return {key: (f"{value:.6f}" if isinstance(value, float) else value) for key, value in row.items()}


def final_manifests(run_glob: str) -> list[Path]:
    return sorted(Path().glob(run_glob))


def tasks(run_glob: str) -> list[Task]:
    specs = [
        ("best_val_macro_f1_checkpoint", "best_val_macro_f1"),
        ("best_record_knn_val1_checkpoint", "best_record_knn_validation_1_macro_f1"),
    ]
    return [Task(label, ckpt, manifest) for label, ckpt in specs for manifest in final_manifests(run_glob)]


def validate_task_count(all_tasks: list[Task]) -> None:
    if len(all_tasks) != 24:
        raise RuntimeError(f"expected 24 tasks, found {len(all_tasks)}")


def task_stem(task: Task) -> str:
    return f"{task.label}__{task.manifest.parent.name}"


def row_path(task: Task, row_dir: Path) -> Path:
    return row_dir / f"{task_stem(task)}.json"


def task_done(task: Task, row_dir: Path) -> bool:
    return row_path(task, row_dir).exists()


def log_path(task: Task, log_dir: Path) -> Path:
    return log_dir / f"{task_stem(task)}.log"


def launch(task: Task, gpu: str, args: argparse.Namespace) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = add_pythonpath("ml", env.get("PYTHONPATH", ""))
    cmd = [sys.executable, __file__, "--worker", "--manifest", str(task.manifest)]
    cmd += ["--ckpt-dir", task.ckpt_dir, "--label", task.label, "--out", str(row_path(task, args.row_dir))]
    args.log_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(cmd, stdout=log_path(task, args.log_dir).open("w"), stderr=subprocess.STDOUT, env=env)


def add_pythonpath(path: str, current: str) -> str:
    if not current:
        return path
    parts = current.split(os.pathsep)
    return current if path in parts else os.pathsep.join([path, current])


def run_queue(all_tasks: list[Task], args: argparse.Namespace) -> None:
    pending = [task for task in all_tasks if not task_done(task, args.row_dir)]
    active: dict[str, tuple[Task, subprocess.Popen]] = {}
    while pending or active:
        start_available(pending, active, args)
        reap_finished(active, args.log_dir)
        time.sleep(5)


def start_available(
    pending: list[Task],
    active: dict[str, tuple[Task, subprocess.Popen]],
    args: argparse.Namespace,
) -> None:
    for gpu in gpu_ids(args.gpu_ids):
        if gpu in active or not pending:
            continue
        task = pending.pop(0)
        print(f"[manager] launch gpu={gpu} {task_stem(task)}", flush=True)
        active[gpu] = (task, launch(task, gpu, args))


def gpu_ids(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def reap_finished(active: dict[str, tuple[Task, subprocess.Popen]], log_dir: Path) -> None:
    for gpu, (task, proc) in list(active.items()):
        code = proc.poll()
        if code is None:
            continue
        del active[gpu]
        if code != 0:
            raise RuntimeError(f"task failed gpu={gpu} code={code}: {log_path(task, log_dir)}")
        print(f"[manager] done gpu={gpu} {task_stem(task)}", flush=True)


def dry_run(all_tasks: list[Task], row_dir: Path) -> None:
    completed = sum(task_done(task, row_dir) for task in all_tasks)
    print(f"tasks\t{len(all_tasks)}")
    print(f"completed\t{completed}")
    print(f"missing\t{len(all_tasks) - completed}")
    for task in all_tasks:
        print_task_status(task, row_dir)


def validate_rows(all_tasks: list[Task], row_dir: Path) -> None:
    task_by_path = {row_path(task, row_dir): task for task in all_tasks}
    row_files = sorted(row_dir.glob("*.json"))
    orphan_files = [path for path in row_files if path not in task_by_path]
    for path in row_files:
        if path in task_by_path:
            validate_row_file(path, task_by_path[path])
    if orphan_files:
        names = ", ".join(path.name for path in orphan_files)
        raise RuntimeError(f"found row files without matching tasks: {names}")
    print(f"validated\t{len(row_files)}")
    print(f"missing\t{len(all_tasks) - len(row_files)}")


def validate_row_file(path: Path, task: Task) -> None:
    payload = json.loads(path.read_text())
    missing = [field for field in required_row_fields() if field not in payload]
    if missing:
        raise RuntimeError(f"{path} missing fields: {missing}")
    validate_row_identity(path, payload, task)
    for field in METRIC_FIELDS:
        validate_metric_value(path, payload[field], field)


def required_row_fields() -> list[str]:
    return ["label", *output_fields()]


def validate_row_identity(path: Path, payload: dict, task: Task) -> None:
    universe, lane, model = run_parts(task.manifest.parent)
    expected = {"label": task.label, "universe": universe, "lane": lane, "model": model}
    mismatches = {key: (payload[key], value) for key, value in expected.items() if payload[key] != value}
    if mismatches:
        raise RuntimeError(f"{path} identity mismatches: {mismatches}")


def validate_metric_value(path: Path, value: object, field: str) -> None:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{path} has nonnumeric {field}: {value!r}") from exc
    if not 0.0 <= parsed <= 1.0:
        raise RuntimeError(f"{path} has out-of-range {field}: {value!r}")


def print_task_status(task: Task, row_dir: Path) -> None:
    universe, lane, model = run_parts(task.manifest.parent)
    status = "done" if task_done(task, row_dir) else "todo"
    parts = [status, task.label, universe, lane, model, str(task.manifest)]
    print("\t".join(parts))


def collect_rows(label: str, row_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(row_dir.glob(f"{label}__*.json")):
        payload = json.loads(path.read_text())
        rows.append({key: payload[key] for key in output_fields()})
    rows.sort(key=row_sort_key)
    return rows


def output_fields() -> list[str]:
    return ["universe", "lane", "model", *METRIC_FIELDS]


def row_sort_key(row: dict) -> tuple[int, int, int]:
    universes = {"condition_key": 0, "same_species_v2": 1, "no_constraints": 2}
    lanes = {"no_source_value": 0, "source_value": 1}
    models = {"large_400m_v1": 0, "small_lt10m_v1": 1}
    return universes[row["universe"]], lanes[row["lane"]], models[row["model"]]


def write_cv_table(rows: list[dict], label: str) -> Path:
    path = RESULTS_DIR / f"eval_tracking_rerun_v1_{label}_train_cv5_record_knn.tsv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_fields(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def append_to_table(table: Path, cv_rows: list[dict]) -> None:
    cv_by_key = {(r["universe"], r["lane"], r["model"]): r for r in cv_rows}
    with table.open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
        fields = merged_fields(list(rows[0].keys()))
    for row in rows:
        cv_row = cv_by_key[(row["universe"], row["lane"], row["model"])]
        row.update({field: cv_row[field] for field in METRIC_FIELDS})
    write_main_table(table, fields, rows)


def merged_fields(fields: list[str]) -> list[str]:
    return fields + [field for field in METRIC_FIELDS if field not in fields]


def write_main_table(table: Path, fields: list[str], rows: list[dict]) -> None:
    with table.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def table_for_label(label: str) -> Path:
    if label == "best_val_macro_f1_checkpoint":
        return RESULTS_DIR / "eval_tracking_rerun_v1_best_val_macro_f1_checkpoint_metrics.tsv"
    return RESULTS_DIR / "eval_tracking_rerun_v1_best_record_knn_val1_checkpoint_metrics.tsv"


def finalize_tables(row_dir: Path) -> None:
    for label in ("best_val_macro_f1_checkpoint", "best_record_knn_val1_checkpoint"):
        rows = collect_rows(label, row_dir)
        if len(rows) != 12:
            raise RuntimeError(f"expected 12 rows for {label}, found {len(rows)}")
        print(f"[manager] wrote {write_cv_table(rows, label)}", flush=True)
        append_to_table(table_for_label(label), rows)


def manager_main(args: argparse.Namespace) -> None:
    all_tasks = tasks(args.run_glob)
    validate_task_count(all_tasks)
    if args.dry_run:
        dry_run(all_tasks, args.row_dir)
        return
    if args.validate_rows:
        validate_rows(all_tasks, args.row_dir)
        return
    if args.finalize_only:
        validate_rows(all_tasks, args.row_dir)
        finalize_tables(args.row_dir)
        return
    run_queue(all_tasks, args)
    finalize_tables(args.row_dir)


def main() -> None:
    args = parse_args()
    worker_main(args) if args.worker else manager_main(args)


if __name__ == "__main__":
    main()
