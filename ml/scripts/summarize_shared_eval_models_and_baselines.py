#!/usr/bin/env python3
"""Summarize shared-eval MLP best validation metrics and simple baselines."""
from __future__ import annotations

import ast
import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


RUN_TAG = "step_logging_300_macro_f1_v1"
FINAL_SUFFIX = "step_logging_3000_macro_f1_v1"
OUT_DIR = Path("ml/results")
UNSEEN_PARTIAL_SUBSETS = ("no_overlap", "a_seen_only")


@dataclass(frozen=True)
class Lane:
    key: str
    label: str
    run_prefix: str
    run_root: Path
    split_suffix: str


@dataclass(frozen=True)
class Universe:
    key: str
    label: str


def build_lanes(run_tag: str) -> tuple[Lane, ...]:
    return (
        Lane(
            key="source_value",
            label="bidirectional/source_value",
            run_prefix="srcval",
            run_root=Path(f"ml/artifacts/runs/shared_eval_{run_tag}"),
            split_suffix="shared_eval_full",
        ),
        Lane(
            key="no_source_value",
            label="unidirectional/no_source_value",
            run_prefix="nosv",
            run_root=Path(f"ml/artifacts/runs/shared_eval_no_source_value_{run_tag}"),
            split_suffix="shared_eval_unidirectional_full",
        ),
    )

UNIVERSES = (
    Universe("condition_key", "condition_key"),
    Universe("same_species_v2", "same_species_v2"),
    Universe("no_constraints", "no_constraints"),
)


def f1_score(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp / denom)


def metrics_from_predictions(labels: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    labels_bool = labels.astype(bool)
    pred_bool = pred.astype(bool)
    tp = int(np.count_nonzero(pred_bool & labels_bool))
    fp = int(np.count_nonzero(pred_bool & ~labels_bool))
    tn = int(np.count_nonzero(~pred_bool & ~labels_bool))
    fn = int(np.count_nonzero(~pred_bool & labels_bool))
    transfer_precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    transfer_recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    transfer_f1 = f1_score(tp, fp, fn)
    not_transfer_f1 = f1_score(tn, fn, fp)
    return {
        "macro_f1": (transfer_f1 + not_transfer_f1) / 2.0,
        "accuracy": (tp + tn) / max(1, len(labels)),
        "transfer_precision": transfer_precision,
        "transfer_recall": transfer_recall,
    }


def split_dir(lane: Lane, universe: Universe) -> Path:
    return Path("datasets/pairs_split_full") / f"oral_bioavailability_{universe.key}_{lane.split_suffix}"


def read_validation_labels_and_scores(
    path: Path,
    *,
    eval_subsets: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for file in sorted((path / "validation").glob("*.parquet")):
        columns = ["transfer_label", "weighted_tanimoto"]
        if eval_subsets is not None:
            columns.append("eval_subset")
        table = pq.read_table(file, columns=columns)
        if eval_subsets is None:
            mask = None
        else:
            subset_set = set(eval_subsets)
            mask = np.array([value in subset_set for value in table["eval_subset"].to_pylist()], dtype=bool)
        label_values = np.array([value == "transfer" for value in table["transfer_label"].to_pylist()], dtype=bool)
        score_values = table["weighted_tanimoto"].to_numpy(zero_copy_only=False).astype(np.float64)
        if mask is not None:
            label_values = label_values[mask]
            score_values = score_values[mask]
        labels.append(
            label_values
        )
        scores.append(score_values)
    if not scores:
        raise FileNotFoundError(path / "validation")
    return np.concatenate(labels), np.concatenate(scores)


def tuned_threshold_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    candidates = np.unique(scores)
    candidates = np.concatenate(
        [
            np.array([-math.inf], dtype=np.float64),
            candidates,
            np.array([math.inf], dtype=np.float64),
        ]
    )
    best_threshold = float(candidates[0])
    best_metrics = metrics_from_predictions(labels, scores >= candidates[0])
    for threshold in candidates[1:]:
        current = metrics_from_predictions(labels, scores >= threshold)
        current_key = (
            current["macro_f1"],
            current["accuracy"],
            current["transfer_precision"],
            -float(threshold) if math.isfinite(float(threshold)) else -1e100,
        )
        best_key = (
            best_metrics["macro_f1"],
            best_metrics["accuracy"],
            best_metrics["transfer_precision"],
            -best_threshold if math.isfinite(best_threshold) else -1e100,
        )
        if current_key > best_key:
            best_threshold = float(threshold)
            best_metrics = current
    return best_threshold, best_metrics


def baseline_rows(
    lane: Lane,
    universe: Universe,
    *,
    eval_subsets: tuple[str, ...] | None,
    note: str,
) -> list[dict[str, str]]:
    labels, scores = read_validation_labels_and_scores(split_dir(lane, universe), eval_subsets=eval_subsets)
    majority_pred = np.full(labels.shape, bool(np.count_nonzero(labels) >= len(labels) / 2), dtype=bool)
    fixed_pred = scores >= 0.5
    tuned_threshold, tuned = tuned_threshold_metrics(labels, scores)
    baselines = [
        ("weighted_tanimoto>=0.5", None, metrics_from_predictions(labels, fixed_pred)),
        ("weighted_tanimoto_tuned_threshold", tuned_threshold, tuned),
        ("majority", None, metrics_from_predictions(labels, majority_pred)),
    ]
    rows = []
    for model_or_baseline, threshold, metrics in baselines:
        rows.append(
            {
                "lane": lane.key,
                "lane_label": lane.label,
                "universe": universe.key,
                "method": model_or_baseline,
                "best_val_macro_f1": f"{metrics['macro_f1']:.6f}",
                "best_val_accuracy": f"{metrics['accuracy']:.6f}",
                "best_val_transfer_precision": f"{metrics['transfer_precision']:.6f}",
                "best_val_macro_f1_step": "",
                "best_val_accuracy_step": "",
                "best_val_transfer_precision_step": "",
                "threshold": "" if threshold is None else f"{threshold:.8g}",
                "source": str(split_dir(lane, universe) / "validation"),
                "notes": note,
            }
        )
    return rows


def parse_eval_dicts(log_path: Path) -> list[dict[str, float]]:
    text = log_path.read_text(errors="ignore")
    matches = re.findall(r"\{'eval_val_loss': .*?\}", text)
    out = []
    for match in matches:
        try:
            value = ast.literal_eval(match)
        except Exception:
            continue
        if isinstance(value, dict) and "eval_val_macro_f1" in value:
            out.append(value)
    return out


def log_completed(log_path: Path) -> bool:
    return "[done] training complete" in log_path.read_text(errors="ignore")


def find_log(lane: Lane, universe: Universe, final_suffix: str) -> Path:
    matches = sorted(
        lane.run_root.glob(f"final_{lane.run_prefix}_{universe.key}_*_ga1_{final_suffix}.log")
    )
    if not matches:
        raise FileNotFoundError(f"no log for {lane.key}/{universe.key}")
    return matches[-1]


def metrics_csv_path(lane: Lane, universe: Universe, log_path: Path) -> Path:
    dataset = f"shared_eval_{universe.key}"
    if lane.key == "no_source_value":
        dataset = f"{dataset}_no_source_value"
    return OUT_DIR / dataset / "runs" / log_path.with_suffix("").name / "metrics.csv"


def best_metric(evals: list[dict[str, float]], metric: str) -> tuple[float, str]:
    key = f"eval_val_{metric}"
    best = max(evals, key=lambda row: float(row.get(key, float("-inf"))))
    step = ""
    if "epoch" in best:
        step = f"epoch={float(best['epoch']):.6g}"
    return float(best[key]), step


def best_subset_average(metrics_csv: Path, metric: str, subsets: tuple[str, ...]) -> tuple[float, str]:
    by_epoch: dict[str, dict[str, float]] = {}
    with metrics_csv.open() as fh:
        for row in csv.DictReader(fh):
            if row["metric"] != metric:
                continue
            prefix = "val/"
            if not row["split"].startswith(prefix):
                continue
            subset = row["split"][len(prefix):]
            if subset not in subsets:
                continue
            by_epoch.setdefault(row["epoch"], {})[subset] = float(row["value"])
    complete = [
        (epoch, values)
        for epoch, values in by_epoch.items()
        if all(subset in values for subset in subsets)
    ]
    if not complete:
        raise RuntimeError(f"no complete subset rows for {metric} in {metrics_csv}")
    epoch, values = max(complete, key=lambda item: sum(item[1][subset] for subset in subsets) / len(subsets))
    return sum(values[subset] for subset in subsets) / len(subsets), f"epoch={float(epoch):.6g}"


def model_row(
    lane: Lane,
    universe: Universe,
    *,
    eval_subsets: tuple[str, ...] | None,
    final_suffix: str,
    note: str,
) -> dict[str, str]:
    log_path = find_log(lane, universe, final_suffix)
    if eval_subsets is None:
        evals = parse_eval_dicts(log_path)
        if not evals:
            raise RuntimeError(f"no eval dicts found in {log_path}")
        macro_f1, macro_step = best_metric(evals, "macro_f1")
        accuracy, accuracy_step = best_metric(evals, "accuracy")
        precision, precision_step = best_metric(evals, "transfer_precision")
    else:
        metrics_path = metrics_csv_path(lane, universe, log_path)
        macro_f1, macro_step = best_subset_average(metrics_path, "macro_f1", eval_subsets)
        accuracy, accuracy_step = best_subset_average(metrics_path, "accuracy", eval_subsets)
        precision, precision_step = best_subset_average(metrics_path, "transfer_precision", eval_subsets)
    notes = note
    if lane.key == "no_source_value" and universe.key == "condition_key" and not log_completed(log_path):
        notes = f"{notes}; killed/incomplete run; best-so-far from log".strip("; ")
    return {
        "lane": lane.key,
        "lane_label": lane.label,
        "universe": universe.key,
        "method": "MLP",
        "best_val_macro_f1": f"{macro_f1:.6f}",
        "best_val_accuracy": f"{accuracy:.6f}",
        "best_val_transfer_precision": f"{precision:.6f}",
        "best_val_macro_f1_step": macro_step,
        "best_val_accuracy_step": accuracy_step,
        "best_val_transfer_precision_step": precision_step,
        "threshold": "",
        "source": str(log_path),
        "notes": notes,
    }


def markdown_table(rows: list[dict[str, str]]) -> str:
    cols = [
        "lane_label",
        "universe",
        "method",
        "best_val_macro_f1",
        "best_val_accuracy",
        "best_val_transfer_precision",
        "threshold",
        "notes",
    ]
    labels = {
        "lane_label": "lane",
        "universe": "dataset",
        "method": "method",
        "best_val_macro_f1": "best val macro-F1",
        "best_val_accuracy": "best val accuracy",
        "best_val_transfer_precision": "best val transfer precision",
        "threshold": "threshold",
        "notes": "notes",
    }
    lines = [
        "| " + " | ".join(labels[c] for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(c, "") for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("full", "unseen_or_partial"),
        default="full",
        help="full uses all validation rows; unseen_or_partial excludes both_seen.",
    )
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--final-suffix", default=FINAL_SUFFIX)
    parser.add_argument(
        "--out-stem-suffix",
        default="",
        help="Optional suffix appended to the output stem, e.g. small_lt10m_v1.",
    )
    args = parser.parse_args()
    lanes = build_lanes(args.run_tag)

    if args.scope == "full":
        eval_subsets = None
        stem = "shared_eval_model_and_baseline_summary"
        model_note = ""
        baseline_note = ""
    else:
        eval_subsets = UNSEEN_PARTIAL_SUBSETS
        stem = "shared_eval_model_and_baseline_summary_unseen_or_partial"
        model_note = "excludes both_seen; MLP metrics are mean of no_overlap and a_seen_only subset metrics"
        baseline_note = "excludes both_seen; baseline metrics computed exactly on no_overlap+a_seen_only rows"
    if args.out_stem_suffix:
        stem = f"{stem}_{args.out_stem_suffix}"

    rows: list[dict[str, str]] = []
    for lane in lanes:
        for universe in UNIVERSES:
            rows.append(
                model_row(
                    lane,
                    universe,
                    eval_subsets=eval_subsets,
                    final_suffix=args.final_suffix,
                    note=model_note,
                )
            )
            rows.extend(baseline_rows(lane, universe, eval_subsets=eval_subsets, note=baseline_note))
    rows.sort(key=lambda row: (row["lane"], row["universe"], row["method"] != "MLP", row["method"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{stem}.csv"
    md_path = OUT_DIR / f"{stem}.md"
    json_path = OUT_DIR / f"{stem}.json"
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(markdown_table(rows))
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
