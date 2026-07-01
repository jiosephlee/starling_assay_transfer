"""Rebuild the 4-row val baseline table: majority / tanimoto>0.5 / tuned-tanimoto / model.

The tuned threshold is chosen to maximize macro-F1 over ALL of train; the model row is read from
a run's extracted ``metrics.csv`` (best val-AUROC eval step) — not by reloading a checkpoint, so it
is robust to architecture/version drift.

Usage:
    python -m starling_ml.compute_baselines --model-run full_1epoch
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .config import Config
from .data import labels_to_int8
from .report import resolve_dataset, write_table


def _load_tan_lab(split_dir: str) -> tuple[np.ndarray, np.ndarray]:
    tan, lab = [], []
    for f in sorted(glob.glob(os.path.join(split_dir, "*.parquet"))):
        t = pq.read_table(f, columns=["weighted_tanimoto", "transfer_label"])
        tan.append(t.column("weighted_tanimoto").to_numpy(zero_copy_only=False).astype(np.float32))
        lab.append(labels_to_int8(t.column("transfer_label")))
    return np.concatenate(tan), np.concatenate(lab).astype(np.int8)


def _macro_f1(tan: np.ndarray, y: np.ndarray, t: float) -> float:
    p = tan > t
    tp = np.sum(p & (y == 1)); fp = np.sum(p & (y == 0))
    fn = np.sum(~p & (y == 1)); tn = np.sum(~p & (y == 0))
    return 0.5 * (2 * tp / (2 * tp + fp + fn + 1e-12) + 2 * tn / (2 * tn + fp + fn + 1e-12))


def _row(name: str, y: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "baseline": name,
        "pred_pos": int(preds.sum()),
        "accuracy": round(accuracy_score(y, preds), 4),
        "macro_f1": round(f1_score(y, preds, average="macro", zero_division=0), 4),
        "pos_f1": round(f1_score(y, preds, zero_division=0), 4),
        "precision": round(precision_score(y, preds, zero_division=0), 4),
        "recall": round(recall_score(y, preds, zero_division=0), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="ml/configs/default.yaml")
    ap.add_argument("--model-run", default=None, help="run name under results/<dataset>/runs for the model row")
    ap.add_argument("--dataset", default=None, help="dataset label (else from --config)")
    ap.add_argument("--results-dir", default="ml/results")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    base = os.path.join(args.results_dir, resolve_dataset(args.dataset, args.config))

    tan_tr, y_tr = _load_tan_lab(os.path.join(cfg.paths.splits_dir, "train"))
    grid = np.unique(np.concatenate([np.arange(0.01, 0.31, 0.0025), [0.35, 0.4, 0.45, 0.5]]))
    scores = [_macro_f1(tan_tr, y_tr, t) for t in grid]
    t_star = float(grid[int(np.argmax(scores))])
    print(f"tuned t*={t_star:.4f} (train macro-F1={max(scores):.4f}, train n={len(y_tr):,})")

    tan_v, y_v = _load_tan_lab(os.path.join(cfg.paths.splits_dir, "validation"))
    rows = [
        _row("Majority (not_transfer)", y_v, np.zeros(len(y_v), np.int64)),
        _row("Tanimoto > 0.5", y_v, (tan_v > 0.5).astype(np.int64)),
        _row(f"Tanimoto > {t_star:.3f} (tuned)", y_v, (tan_v > t_star).astype(np.int64)),
    ]

    if args.model_run:
        mp = os.path.join(base, "runs", args.model_run, "metrics.csv")
        md = pd.read_csv(mp)
        v = md[md["split"] == "val"].pivot_table(index="epoch", columns="metric", values="value")
        best = v.loc[v["auroc"].idxmax()]  # operating point = best val-AUROC eval
        rows.append({
            "baseline": f"Model ({args.model_run})",
            "pred_pos": "-",
            "accuracy": round(float(best["accuracy"]), 4),
            "macro_f1": round(float(best["macro_f1"]), 4),
            "pos_f1": round(float(best.get("f1", np.nan)), 4),
            "precision": round(float(best["precision"]), 4),
            "recall": round(float(best["recall"]), 4),
        })

    df = pd.DataFrame(rows)
    stem = os.path.join(base, "tables", "val_baselines")
    write_table(df, stem)
    print(df.to_string(index=False))
    print(f"\n[baselines] -> {stem}.{{csv,md}}")


if __name__ == "__main__":
    main()
