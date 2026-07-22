#!/usr/bin/env python3
"""Plot official V6.5 100M MLP validation macro-F1 learning curves."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = {
    "Concat": Path("ml/artifacts/v6_5_molformer_pubmedbert_100m/runs/concat/metrics.jsonl"),
    "Difference": Path("ml/artifacts/v6_5_molformer_pubmedbert_100m/runs/difference/metrics.jsonl"),
    "Difference + product": Path(
        "ml/artifacts/v6_5_molformer_pubmedbert_100m/runs/difference_product/metrics.jsonl"),
}


def _curve(path: Path) -> tuple[list[float], list[float]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    measured = [row for row in rows if "validation" in row]
    examples_millions = [row["step"] * 128 * 40 / 1_000_000 for row in measured]
    return examples_millions, [row["validation"]["overall"]["macro_f1"] for row in measured]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label", default="Difference + product")
    parser.add_argument("--title", default="V6.5 100M MLP validation macro-F1 throughout training")
    parser.add_argument("--x-label", default="Sampled training rows (millions; with replacement)")
    return parser.parse_args()


def _draw(axis: plt.Axes, label: str, path: Path, color: str) -> None:
    x, y = _curve(path)
    axis.plot(x, y, marker="o", markersize=3.5, linewidth=2.2, label=label, color=color)
    best = max(range(len(y)), key=y.__getitem__)
    axis.scatter([x[best]], [y[best]], s=55, color=color, edgecolor="white", zorder=4)
    axis.annotate(f"best {y[best]:.3f}", (x[best], y[best]), xytext=(5, 7),
                  textcoords="offset points", fontsize=9, color=color)


def main() -> None:
    args = _arguments()
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(9, 5.4), dpi=180)
    if args.metrics:
        _draw(axis, args.label, args.metrics, "#16A34A")
    else:
        colors = ("#2563EB", "#EA580C", "#16A34A")
        for (label, path), color in zip(RUNS.items(), colors):
            _draw(axis, label, path, color)
    axis.set(title=args.title, xlabel=args.x_label, ylabel="Macro-F1")
    axis.set_xlim(left=0)
    axis.legend(frameon=True)
    fig.tight_layout()
    output = args.output or Path("tmp/v6_5_mlp_100m_rerun/validation_macro_f1_curve.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
