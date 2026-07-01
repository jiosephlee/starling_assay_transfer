"""Parsing + table helpers for the ``ml/results`` reporting layer.

Trainer prints each train log / eval as a brace-delimited dict to stdout, e.g.
``{'eval_val_auroc': '0.79', ..., 'epoch': '0.5'}``. ``parse_run_log`` turns a saved run log
into a tidy ``(epoch, split, metric, value)`` DataFrame; ``write_table`` writes any DataFrame
as both CSV and GitHub-flavored Markdown (no extra deps).
"""
from __future__ import annotations

import os
import re

import pandas as pd

# Each metric dict is a single brace block with no nested braces in these logs.
_DICT = re.compile(r"\{[^{}]*\}")
# Values are always single-quoted in the Trainer stdout (e.g. 'eval_val_auroc': '0.79').
_PAIR = re.compile(r"'([A-Za-z0-9_/]+)':\s*'([^']*)'")


def _to_float(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_run_log(path: str) -> pd.DataFrame:
    """Parse a Trainer stdout log into tidy rows: epoch, split, metric, value.

    ``split`` is one of ``train`` (per-step loss), ``val``, ``train_sample`` (the capacity-signal
    eval set). Metric names have their ``eval_val_`` / ``eval_train_sample_`` prefix stripped.
    """
    with open(path) as fh:
        text = fh.read()

    rows: list[dict] = []
    for block in _DICT.findall(text):
        kv = {k: v for k, v in _PAIR.findall(block)}
        epoch = _to_float(kv.get("epoch"))
        if epoch is None:
            continue
        if any(k.startswith("eval_val_") for k in kv):
            split, prefix = "val", "eval_val_"
        elif any(k.startswith("eval_train_sample_") for k in kv):
            split, prefix = "train_sample", "eval_train_sample_"
        elif "loss" in kv and "learning_rate" in kv:
            split, prefix = "train", ""
        else:
            continue  # final train_runtime summary etc.
        for key, raw in kv.items():
            if key == "epoch":
                continue
            if prefix and not key.startswith(prefix):
                continue
            value = _to_float(raw)
            if value is None:
                continue
            rows.append(
                {"epoch": epoch, "split": split, "metric": key[len(prefix):], "value": value}
            )
    return pd.DataFrame(rows, columns=["epoch", "split", "metric", "value"])


def resolve_dataset(dataset: str | None, config: str | None, default: str = "same_species_v2") -> str:
    """Pick the dataset label for ml/results/<dataset>/: explicit --dataset wins, else read
    ``paths.dataset`` from --config, else the default."""
    if dataset:
        return dataset
    if config:
        from .config import Config

        return Config.from_yaml(config).paths.dataset
    return default


def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(row[c]) else str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, stem: str) -> None:
    """Write ``<stem>.csv`` and ``<stem>.md`` (creating parent dirs)."""
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    df.to_csv(stem + ".csv", index=False)
    with open(stem + ".md", "w") as fh:
        fh.write(to_markdown(df))
