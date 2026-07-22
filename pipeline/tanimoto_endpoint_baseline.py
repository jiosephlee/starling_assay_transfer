"""Tune and evaluate endpoint-specific weighted-Tanimoto thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from pipeline import tanimoto_baseline as baseline
except ModuleNotFoundError:  # Support direct execution from the repository root.
    import tanimoto_baseline as baseline


DEFAULT_SELECTED = Path("datasets/select/assay_transfer_binary_v4_fg_v3/same_endpoint/selected")
DEFAULT_OUTPUT = Path("tables")
DEFAULT_OUTPUT_STEM = "assay_transfer_v4_fg_v3_endpoint_tanimoto"
DEFAULT_REPORT_TITLE = "Assay Transfer V4-on-V3 Endpoint-Threshold Weighted-Tanimoto Baseline"


def subset(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    """Return a row-aligned subset without changing the source arrays."""
    return {key: values[mask] for key, values in data.items()}


def endpoint_threshold_sweep(train: dict[str, np.ndarray], count: int = 100
                             ) -> tuple[list[dict], dict[str, float], list[dict]]:
    """Evaluate every cutoff for every train endpoint and select one per endpoint."""
    sweep, thresholds, selected = [], {}, []
    for endpoint in sorted(set(train["endpoint"])):
        rows = baseline.sweep_thresholds(subset(train, train["endpoint"] == endpoint), count)
        sweep.extend({"canonical_endpoint_key": endpoint, **row} for row in rows)
        best = baseline.choose_threshold(rows)
        thresholds[endpoint] = best["threshold"]
        selected.append({"canonical_endpoint_key": endpoint, **best})
    return sweep, thresholds, selected


def predictions(data: dict[str, np.ndarray], thresholds: dict[str, float],
                fallback_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply endpoint thresholds, using the global train cutoff for unseen endpoints."""
    used_fallback = np.asarray([endpoint not in thresholds for endpoint in data["endpoint"]])
    cutoffs = np.asarray([thresholds.get(endpoint, fallback_threshold)
                          for endpoint in data["endpoint"]], dtype=np.float64)
    return data["score"] >= cutoffs, used_fallback


def evaluation_rows(split: str, data: dict[str, np.ndarray],
                    endpoint_thresholds: dict[str, float], fallback_threshold: float) -> list[dict]:
    """Evaluate frozen endpoint thresholds overall and across the standard slices."""
    predicted, used_fallback = predictions(data, endpoint_thresholds, fallback_threshold)
    rows = [_evaluation_row(split, "overall", "all", data, predicted, used_fallback, None)]
    for concept in baseline.CONCEPTS:
        rows.append(_evaluation_row(split, "assay_concept", concept, data, predicted,
                                    used_fallback, data["concept"] == concept))
    for bucket in baseline.BUCKETS:
        rows.append(_evaluation_row(split, "tanimoto_bucket", bucket, data, predicted,
                                    used_fallback, data["bucket"] == bucket))
    return rows


def _evaluation_row(split: str, slice_type: str, slice_value: str,
                    data: dict[str, np.ndarray], predicted: np.ndarray,
                    used_fallback: np.ndarray, mask: np.ndarray | None) -> dict:
    use = np.ones(len(data["label"]), dtype=bool) if mask is None else mask
    metrics = baseline.binary_metrics(data["label"][use], predicted[use])
    return {
        "split": split, "slice_type": slice_type, "slice_value": slice_value,
        "threshold_strategy": "per_endpoint", "n_endpoints": len(set(data["endpoint"][use])),
        "n_global_fallback": int(np.count_nonzero(used_fallback[use])), **metrics,
    }


def output_paths(output_dir: Path, output_stem: str) -> dict[str, Path]:
    """Return all endpoint-baseline report paths derived from an output stem."""
    return {
        "sweep": output_dir / f"{output_stem}_threshold_sweep.tsv",
        "thresholds": output_dir / f"{output_stem}_thresholds.tsv",
        "baseline": output_dir / f"{output_stem}_baseline.tsv",
        "markdown": output_dir / f"{output_stem}_baseline.md",
    }


def write_markdown(path: Path, report_title: str, global_best: dict,
                   selected: list[dict], evaluation: list[dict], baseline_filename: str) -> None:
    """Write a concise description and overall held-out metrics."""
    overall = [row for row in evaluation if row["slice_type"] == "overall"]
    lines = [
        f"# {report_title}", "",
        "Each canonical endpoint selects its own weighted-Tanimoto cutoff from exactly 100",
        "thresholds spanning 0 through 1 on its train rows. Selection uses macro-F1, then",
        "accuracy, transfer precision, and the smaller threshold.", "",
        f"Endpoint-specific thresholds: `{len(selected)}`.",
        f"Global fallback for endpoints absent from train: `{global_best['threshold']:.10f}`.", "",
        "| split | n | endpoints | global fallbacks | macro-F1 | accuracy | transfer precision | transfer recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_markdown_metric_row(row) for row in overall)
    lines.extend(["", f"Detailed metrics are in `{baseline_filename}`."])
    path.write_text("\n".join(lines) + "\n")


def _markdown_metric_row(row: dict) -> str:
    return (f"| {row['split']} | {row['n']} | {row['n_endpoints']} | "
            f"{row['n_global_fallback']} | {row['macro_f1']:.6f} | {row['accuracy']:.6f} | "
            f"{row['transfer_precision']:.6f} | {row['transfer_recall']:.6f} |")


def run(selected_dir: Path, output_dir: Path, count: int = 100,
        output_stem: str = DEFAULT_OUTPUT_STEM,
        report_title: str = DEFAULT_REPORT_TITLE) -> tuple[dict, list[dict]]:
    """Tune endpoint cutoffs on train, then score validation and test once."""
    train = baseline.load_split(selected_dir, "train")
    global_best = baseline.choose_threshold(baseline.sweep_thresholds(train, count))
    sweep, thresholds, selected = endpoint_threshold_sweep(train, count)
    evaluation = []
    for split in ("validation", "test"):
        evaluation.extend(evaluation_rows(split, baseline.load_split(selected_dir, split),
                                          thresholds, global_best["threshold"]))
    paths = output_paths(output_dir, output_stem)
    baseline.write_tsv(paths["sweep"], sweep)
    baseline.write_tsv(paths["thresholds"], selected)
    baseline.write_tsv(paths["baseline"], evaluation)
    write_markdown(paths["markdown"], report_title, global_best, selected, evaluation,
                   paths["baseline"].name)
    return global_best, evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-dir", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold-count", type=int, default=100)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--report-title", default=DEFAULT_REPORT_TITLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_best, rows = run(args.selected_dir, args.output_dir, args.threshold_count,
                            args.output_stem, args.report_title)
    print(f"global fallback threshold={global_best['threshold']:.10f}")
    for row in rows:
        if row["slice_type"] == "overall":
            print(f"{row['split']}: macro_f1={row['macro_f1']:.6f} accuracy={row['accuracy']:.6f}")


if __name__ == "__main__":
    main()
