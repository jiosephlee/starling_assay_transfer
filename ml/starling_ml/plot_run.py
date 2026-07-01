"""Plot a run's training curves from its extracted metrics.csv.

Three panels vs epoch: AUROC (val vs train_sample = the generalization gap), loss (train vs val),
and val accuracy / macro-F1.

Usage:
    python -m starling_ml.plot_run --name full_1epoch
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .report import resolve_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", default=None, help="dataset label (else from --config)")
    ap.add_argument("--config", default=None, help="config to read paths.dataset from")
    ap.add_argument("--results-dir", default="ml/results")
    args = ap.parse_args()

    base = os.path.join(args.results_dir, resolve_dataset(args.dataset, args.config))
    df = pd.read_csv(os.path.join(base, "runs", args.name, "metrics.csv"))

    def series(split: str, metric: str):
        d = df[(df["split"] == split) & (df["metric"] == metric)].sort_values("epoch")
        return d["epoch"].to_numpy(), d["value"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    for split, label in (("val", "val (unseen mols)"), ("train_sample", "train sample")):
        x, y = series(split, "auroc")
        if len(x):
            ax.plot(x, y, marker=".", label=label)
    ax.set(title="AUROC", xlabel="epoch", ylabel="AUROC")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    x, y = series("train", "loss")
    if len(x):
        ax.plot(x, y, label="train loss", alpha=0.5, lw=1)
    x, y = series("val", "loss")
    if len(x):
        ax.plot(x, y, marker=".", label="val loss", color="C3")
    ax.set(title="Loss", xlabel="epoch", ylabel="loss")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    for metric, color in (("accuracy", "C0"), ("macro_f1", "C2")):
        x, y = series("val", metric)
        if len(x):
            ax.plot(x, y, marker=".", label=f"val {metric}", color=color)
    ax.set(title="Val accuracy / macro-F1", xlabel="epoch", ylabel="score")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(args.name)
    fig.tight_layout()
    out = os.path.join(base, "plots", f"{args.name}_curves.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"[plot] -> {out}")


if __name__ == "__main__":
    main()
