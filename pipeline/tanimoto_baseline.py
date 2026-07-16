"""Tune and evaluate the assay-transfer V3 weighted-Tanimoto baseline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

DEFAULT_SELECTED = Path("datasets/select/assay_transfer_binary_v3/same_endpoint/selected")
DEFAULT_OUTPUT = Path("tables")
CONCEPTS = ("oral_bioavailability", "oral_exposure", "Fa", "Fg", "Fh")
BUCKETS = ("low", "high")


def load_split(selected_dir: Path, split: str) -> dict[str, np.ndarray]:
    """Load and validate columns required by the baseline and its slices."""
    path = selected_dir / split / "selected.parquet"
    columns = ["tanimoto", "binary_label", "assay_concept", "tanimoto_bucket"]
    table = pq.read_table(path, columns=columns)
    data = {
        "score": table["tanimoto"].to_numpy(zero_copy_only=False).astype(np.float64),
        "label": table["binary_label"].to_numpy(zero_copy_only=False).astype(np.int8),
        "concept": np.asarray(table["assay_concept"].to_pylist(), dtype=object),
        "bucket": np.asarray(table["tanimoto_bucket"].to_pylist(), dtype=object),
    }
    validate_split(data, split)
    return data


def validate_split(data: dict[str, np.ndarray], split: str) -> None:
    """Reject malformed data rather than silently changing the benchmark population."""
    lengths = {len(values) for values in data.values()}
    if lengths == {0} or len(lengths) != 1:
        raise ValueError(f"{split}: columns are empty or have unequal lengths")
    if not np.isfinite(data["score"]).all():
        raise ValueError(f"{split}: Tanimoto scores must all be finite")
    if np.any((data["score"] < 0.0) | (data["score"] > 1.0)):
        raise ValueError(f"{split}: Tanimoto scores must be within [0, 1]")
    if not np.isin(data["label"], (0, 1)).all():
        raise ValueError(f"{split}: binary labels must be 0 or 1")


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    """Compute row-level binary metrics with transfer as the positive class."""
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(np.count_nonzero(labels & predictions))
    fp = int(np.count_nonzero(~labels & predictions))
    tn = int(np.count_nonzero(~labels & ~predictions))
    fn = int(np.count_nonzero(labels & ~predictions))
    transfer_f1 = _f1(tp, fp, fn)
    not_transfer_f1 = _f1(tn, fn, fp)
    return {
        "n": len(labels), "actual_transfer": tp + fn, "predicted_transfer": tp + fp,
        "macro_f1": (transfer_f1 + not_transfer_f1) / 2.0,
        "accuracy": (tp + tn) / len(labels),
        "transfer_precision": _ratio(tp, tp + fp),
        "transfer_recall": _ratio(tp, tp + fn),
    }


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    return _ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def threshold_grid(count: int = 100) -> np.ndarray:
    if count < 2:
        raise ValueError("threshold count must be at least two")
    return np.linspace(0.0, 1.0, count, dtype=np.float64)


def sweep_thresholds(data: dict[str, np.ndarray], count: int = 100) -> list[dict]:
    rows = []
    for threshold in threshold_grid(count):
        metrics = binary_metrics(data["label"], data["score"] >= threshold)
        rows.append({"threshold": float(threshold), **metrics})
    return rows


def choose_threshold(rows: Iterable[dict]) -> dict:
    """Select macro-F1, accuracy, precision, then the smaller cutoff."""
    rows = list(rows)
    if not rows:
        raise ValueError("threshold sweep cannot be empty")
    return max(
        rows,
        key=lambda row: (
            row["macro_f1"], row["accuracy"], row["transfer_precision"], -row["threshold"]
        ),
    )


def evaluation_rows(split: str, data: dict[str, np.ndarray], threshold: float) -> list[dict]:
    rows = [_evaluation_row(split, "overall", "all", data, threshold, None)]
    for concept in CONCEPTS:
        mask = data["concept"] == concept
        rows.append(_evaluation_row(split, "assay_concept", concept, data, threshold, mask))
    for bucket in BUCKETS:
        mask = data["bucket"] == bucket
        rows.append(_evaluation_row(split, "tanimoto_bucket", bucket, data, threshold, mask))
    return rows


def _evaluation_row(split: str, slice_type: str, slice_value: str,
                    data: dict[str, np.ndarray], threshold: float,
                    mask: np.ndarray | None) -> dict:
    use = np.ones(len(data["label"]), dtype=bool) if mask is None else mask
    metrics = binary_metrics(data["label"][use], data["score"][use] >= threshold)
    return {
        "split": split, "slice_type": slice_type, "slice_value": slice_value,
        "threshold": threshold, **metrics,
    }


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(_formatted_row(row))


def _formatted_row(row: dict) -> dict:
    return {key: f"{value:.10f}" if key == "threshold" else
            f"{value:.6f}" if isinstance(value, float) else value
            for key, value in row.items()}


def write_markdown(path: Path, best: dict, evaluation: list[dict]) -> None:
    overall = [row for row in evaluation if row["slice_type"] == "overall"]
    lines = [
        "# Assay Transfer V3 Weighted-Tanimoto Baseline", "",
        "The stored weighted Tanimoto score is thresholded as transfer when score `>= t`.",
        "Exactly 100 thresholds from 0 through 1 were evaluated on train; the cutoff was",
        "selected by macro-F1, then accuracy, transfer precision, and the smaller threshold.", "",
        f"Selected threshold: `{best['threshold']:.10f}`",
        f"(train macro-F1 `{best['macro_f1']:.6f}`, accuracy `{best['accuracy']:.6f}`).", "",
        "| split | n | macro-F1 | accuracy | transfer precision | transfer recall |", 
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(_markdown_metric_row(row))
    lines.extend(["", "Detailed overall and slice metrics are in "
                  "`assay_transfer_v3_tanimoto_baseline.tsv`."])
    path.write_text("\n".join(lines) + "\n")


def _markdown_metric_row(row: dict) -> str:
    return (f"| {row['split']} | {row['n']} | {row['macro_f1']:.6f} | "
            f"{row['accuracy']:.6f} | {row['transfer_precision']:.6f} | "
            f"{row['transfer_recall']:.6f} |")


def run(selected_dir: Path, output_dir: Path, count: int = 100) -> tuple[dict, list[dict]]:
    train = load_split(selected_dir, "train")
    sweep = sweep_thresholds(train, count)
    best = choose_threshold(sweep)
    evaluation = []
    for split in ("validation", "test"):
        evaluation.extend(evaluation_rows(split, load_split(selected_dir, split), best["threshold"]))
    write_tsv(output_dir / "assay_transfer_v3_tanimoto_threshold_sweep.tsv", sweep)
    write_tsv(output_dir / "assay_transfer_v3_tanimoto_baseline.tsv", evaluation)
    write_markdown(output_dir / "assay_transfer_v3_tanimoto_baseline.md", best, evaluation)
    return best, evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-dir", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold-count", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best, rows = run(args.selected_dir, args.output_dir, args.threshold_count)
    print(f"selected threshold={best['threshold']:.10f}")
    for row in rows:
        if row["slice_type"] == "overall":
            print(f"{row['split']}: macro_f1={row['macro_f1']:.6f} accuracy={row['accuracy']:.6f}")


if __name__ == "__main__":
    main()
