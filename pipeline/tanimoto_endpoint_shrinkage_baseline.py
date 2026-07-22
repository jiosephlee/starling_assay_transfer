"""Tune support-gated endpoint-specific weighted-Tanimoto thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from pipeline import tanimoto_baseline as baseline
    from pipeline import tanimoto_endpoint_baseline as endpoint_baseline
except ModuleNotFoundError:  # Support direct execution from the repository root.
    import tanimoto_baseline as baseline
    import tanimoto_endpoint_baseline as endpoint_baseline


DEFAULT_SELECTED = Path("datasets/select/assay_transfer_binary_v4_fg_v3/same_endpoint/selected")
DEFAULT_OUTPUT = Path("tables")
DEFAULT_OUTPUT_STEM = "assay_transfer_v4_fg_v3_endpoint_shrinkage_tanimoto"
DEFAULT_REPORT_TITLE = "Assay Transfer V4-on-V3 Support-Gated Endpoint Weighted-Tanimoto Baseline"
ENDPOINT_THRESHOLD_COUNT = 25
SUPPORT_CANDIDATES = (25, 50, 100, 200, 500, 1000)


def endpoint_train_counts(train: dict[str, np.ndarray]) -> dict[str, int]:
    """Count train rows available to fit each endpoint-specific threshold."""
    return {endpoint: int(np.count_nonzero(train["endpoint"] == endpoint))
            for endpoint in set(train["endpoint"])}


def eligible_thresholds(thresholds: dict[str, float], train_counts: dict[str, int],
                        minimum_rows: int) -> dict[str, float]:
    """Keep endpoint thresholds only when they have sufficient train support."""
    return {endpoint: threshold for endpoint, threshold in thresholds.items()
            if train_counts[endpoint] >= minimum_rows}


def support_sweep(validation: dict[str, np.ndarray], thresholds: dict[str, float],
                  train_counts: dict[str, int], fallback_threshold: float) -> list[dict]:
    """Evaluate each train-support gate on validation without refitting thresholds."""
    rows = []
    for minimum_rows in SUPPORT_CANDIDATES:
        eligible = eligible_thresholds(thresholds, train_counts, minimum_rows)
        overall = endpoint_baseline.evaluation_rows(
            "validation", validation, eligible, fallback_threshold)[0]
        rows.append({"min_train_rows": minimum_rows, "n_endpoint_thresholds": len(eligible),
                     **overall})
    return rows


def choose_support(rows: list[dict]) -> dict:
    """Prefer validation macro-F1, accuracy, precision, then the safer larger gate."""
    if not rows:
        raise ValueError("support sweep cannot be empty")
    return max(rows, key=lambda row: (row["macro_f1"], row["accuracy"],
                                      row["transfer_precision"], row["min_train_rows"]))


def selected_threshold_rows(selected: list[dict], train_counts: dict[str, int],
                            minimum_rows: int) -> list[dict]:
    """Add train support to the endpoint thresholds retained by the selected gate."""
    rows = []
    for row in selected:
        endpoint = row["canonical_endpoint_key"]
        if train_counts[endpoint] >= minimum_rows:
            rows.append({"train_rows": train_counts[endpoint], **row})
    return rows


def output_paths(output_dir: Path, output_stem: str) -> dict[str, Path]:
    """Return report paths derived from a configurable output stem."""
    return {
        "sweep": output_dir / f"{output_stem}_threshold_sweep.tsv",
        "support": output_dir / f"{output_stem}_support_sweep.tsv",
        "thresholds": output_dir / f"{output_stem}_thresholds.tsv",
        "baseline": output_dir / f"{output_stem}_baseline.tsv",
        "markdown": output_dir / f"{output_stem}_baseline.md",
    }


def write_markdown(path: Path, report_title: str, global_best: dict, support: dict,
                   evaluation: list[dict], baseline_filename: str) -> None:
    """Write the support-selection rule and overall held-out metrics."""
    overall = [row for row in evaluation if row["slice_type"] == "overall"]
    lines = [
        f"# {report_title}", "",
        "Each endpoint is tuned on 25 thresholds spanning 0 through 1 using train rows only.",
        "A validation-selected minimum train support determines which endpoint cutoffs are used;",
        "all remaining rows use the global cutoff tuned on 100 train thresholds.", "",
        f"Selected minimum train rows: `{support['min_train_rows']}` "
        f"({support['n_endpoint_thresholds']} endpoint-specific thresholds).",
        f"Global fallback threshold: `{global_best['threshold']:.10f}`.", "",
        "| split | n | endpoints | global fallbacks | macro-F1 | accuracy | transfer precision | transfer recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(endpoint_baseline._markdown_metric_row(row) for row in overall)
    lines.extend(["", f"Detailed metrics are in `{baseline_filename}`."])
    path.write_text("\n".join(lines) + "\n")


def run(selected_dir: Path, output_dir: Path, output_stem: str = DEFAULT_OUTPUT_STEM,
        report_title: str = DEFAULT_REPORT_TITLE) -> tuple[dict, dict, list[dict]]:
    """Tune endpoint cutoffs, choose their support gate on validation, and test once."""
    train = baseline.load_split(selected_dir, "train")
    validation = baseline.load_split(selected_dir, "validation")
    global_best = baseline.choose_threshold(baseline.sweep_thresholds(train, count=100))
    sweep, all_thresholds, selected = endpoint_baseline.endpoint_threshold_sweep(
        train, ENDPOINT_THRESHOLD_COUNT)
    train_counts = endpoint_train_counts(train)
    support_rows = support_sweep(validation, all_thresholds, train_counts, global_best["threshold"])
    support = choose_support(support_rows)
    thresholds = eligible_thresholds(all_thresholds, train_counts, support["min_train_rows"])
    evaluation = endpoint_baseline.evaluation_rows(
        "validation", validation, thresholds, global_best["threshold"])
    evaluation.extend(endpoint_baseline.evaluation_rows(
        "test", baseline.load_split(selected_dir, "test"), thresholds, global_best["threshold"]))
    evaluation = [{"min_train_rows": support["min_train_rows"], **row} for row in evaluation]
    paths = output_paths(output_dir, output_stem)
    baseline.write_tsv(paths["sweep"], sweep)
    baseline.write_tsv(paths["support"], support_rows)
    baseline.write_tsv(paths["thresholds"], selected_threshold_rows(
        selected, train_counts, support["min_train_rows"]))
    baseline.write_tsv(paths["baseline"], evaluation)
    write_markdown(paths["markdown"], report_title, global_best, support, evaluation,
                   paths["baseline"].name)
    return global_best, support, evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-dir", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--report-title", default=DEFAULT_REPORT_TITLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_best, support, rows = run(args.selected_dir, args.output_dir,
                                     args.output_stem, args.report_title)
    print(f"global fallback threshold={global_best['threshold']:.10f}")
    print(f"selected minimum train rows={support['min_train_rows']}")
    for row in rows:
        if row["slice_type"] == "overall":
            print(f"{row['split']}: macro_f1={row['macro_f1']:.6f} accuracy={row['accuracy']:.6f}")


if __name__ == "__main__":
    main()
