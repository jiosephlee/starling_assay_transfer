"""Shared metric helpers."""
from __future__ import annotations

import numpy as np


def binary_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int64)
    out: dict[str, float] = {}
    # AUROC needs both classes present.
    if len(np.unique(labels)) > 1:
        out["auroc"] = float(roc_auc_score(labels, probs))
    else:
        out["auroc"] = float("nan")
    out["accuracy"] = float(accuracy_score(labels, preds))
    out["f1"] = float(f1_score(labels, preds, zero_division=0))  # positive-class F1
    out["macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    out["precision"] = float(precision_score(labels, preds, zero_division=0))
    out["recall"] = float(recall_score(labels, preds, zero_division=0))
    out["positive_rate"] = float(labels.mean())
    out["parse_rate"] = 1.0
    precision, recall, f1, _support = precision_recall_fscore_support(
        labels.astype(np.int64),
        preds,
        labels=[1, 0],
        zero_division=0,
    )
    for idx, label_name in enumerate(("A", "B")):
        label_id = 1 if label_name == "A" else 0
        out[f"label/{label_name}/precision"] = float(precision[idx])
        out[f"label/{label_name}/recall"] = float(recall[idx])
        out[f"label/{label_name}/f1"] = float(f1[idx])
        out[f"label/{label_name}/predicted"] = int((preds == label_id).sum())
    return out


def simple_transfer_metrics(
    logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Return the compact benchmark metric set.

    The positive class is transfer (label 1), which is also the old label/A in
    ``binary_metrics``.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int64)
    clipped_probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    entropy = -(
        clipped_probs * np.log(clipped_probs)
        + (1.0 - clipped_probs) * np.log(1.0 - clipped_probs)
    )
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "transfer_precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "transfer_recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "entropy": float(entropy.mean()),
    }
