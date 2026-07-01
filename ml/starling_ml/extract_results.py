"""Extract run metrics and sweep tables into ``ml/results/``.

Usage:
    python -m starling_ml.extract_results --log ml/artifacts/full_1epoch.log --name full_1epoch
    python -m starling_ml.extract_results --sweeps
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from .report import parse_run_log, resolve_dataset, write_table

# (source tsv, output table stem under results/tables)
_SWEEPS = [
    ("ml/artifacts/sweep_logs/results.tsv", "lr_batch_sweep"),
    ("ml/artifacts/headscan_logs/results.tsv", "head_capacity_scan"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", help="Trainer stdout log to parse")
    ap.add_argument("--name", help="run name (subdir under results/<dataset>/runs)")
    ap.add_argument("--sweeps", action="store_true", help="also clean sweep .tsv -> tables")
    ap.add_argument("--dataset", default=None, help="dataset label (else from --config)")
    ap.add_argument("--config", default=None, help="config to read paths.dataset from")
    ap.add_argument("--results-dir", default="ml/results")
    args = ap.parse_args()

    dataset = resolve_dataset(args.dataset, args.config)
    base = os.path.join(args.results_dir, dataset)

    if args.log:
        if not args.name:
            ap.error("--name is required with --log")
        df = parse_run_log(args.log)
        out_dir = os.path.join(base, "runs", args.name)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "metrics.csv")
        df.to_csv(out, index=False)
        n_val = int(((df["split"] == "val") & (df["metric"] == "auroc")).sum())
        print(f"[extract] {args.log} -> {out}  ({len(df)} rows, {n_val} val-AUROC points)")

    if args.sweeps:
        for src, name in _SWEEPS:
            if not os.path.exists(src):
                print(f"[extract] skip (missing) {src}")
                continue
            stem = os.path.join(base, "tables", name)
            write_table(pd.read_csv(src, sep="\t"), stem)
            print(f"[extract] {src} -> {stem}.{{csv,md}}")


if __name__ == "__main__":
    main()
