"""Evaluate a trained checkpoint on val/test and report metrics + the tanimoto baseline.

Usage:
    python -m starling_ml.evaluate --config ml/configs/default.yaml \
        --checkpoint ml/artifacts/runs/default --split test
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import PairDataset, build_split_memmap, collate_pairs, labels_to_int8
from .metrics import binary_metrics
from .model import build_model


def _load_state_dict(checkpoint: str) -> dict:
    safet = os.path.join(checkpoint, "model.safetensors")
    binf = os.path.join(checkpoint, "pytorch_model.bin")
    if os.path.exists(safet):
        from safetensors.torch import load_file

        return load_file(safet)
    if os.path.exists(binf):
        return torch.load(binf, map_location="cpu")
    raise FileNotFoundError(f"no model weights (model.safetensors / pytorch_model.bin) in {checkpoint}")


def _tanimoto_baseline(splits_dir: str, split: str) -> dict[str, float]:
    """AUROC of the precomputed weighted_tanimoto column as a similarity-only classifier."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(splits_dir, split, "*.parquet")))
    tan, lab = [], []
    for f in files:
        t = pq.read_table(f, columns=["weighted_tanimoto", "transfer_label"])
        tan.append(t.column("weighted_tanimoto").to_numpy(zero_copy_only=False))
        lab.append(labels_to_int8(t.column("transfer_label")))
    tan = np.concatenate(tan).astype(np.float64)
    labels = np.concatenate(lab).astype(np.float64)
    from sklearn.metrics import roc_auc_score

    auroc = float(roc_auc_score(labels, tan)) if len(np.unique(labels)) > 1 else float("nan")
    return {"tanimoto_auroc": auroc}


@torch.no_grad()
def _predict(model, dataset, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pairs)
    logits_all, labels_all = [], []
    amp = device == "cuda" and torch.cuda.is_bf16_supported()
    for batch in loader:
        a = batch["a_idx"].to(device)
        b = batch["b_idx"].to(device)
        model_inputs = {"a_idx": a, "b_idx": b}
        if "source_value" in batch:
            model_inputs["source_value"] = batch["source_value"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(**model_inputs)["logits"].float()
        logits_all.append(logits.cpu().numpy())
        labels_all.append(batch["labels"].numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ml/configs/default.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config).apply_overrides(args.overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg)
    state = _load_state_dict(args.checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Frozen buffers (mol_emb/meta_emb/pos_weight) are non-persistent, so expected-missing.
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()

    build_split_memmap(
        cfg.paths.splits_dir,
        cfg.paths.memmap_dir,
        args.split,
        base_parquet=cfg.paths.base_parquet,
        use_source_value=cfg.model.use_source_value,
        source_value_scale=cfg.model.source_value_scale,
    )
    dataset = PairDataset(cfg.paths.memmap_dir, args.split)

    logits, labels = _predict(model, dataset, cfg.train.per_device_batch_size, device)
    metrics = binary_metrics(logits, labels)
    baseline = _tanimoto_baseline(cfg.paths.splits_dir, args.split)

    print(f"\n=== {args.split} ({len(labels)} pairs) ===")
    for k, v in metrics.items():
        print(f"  model.{k:14s} {v:.4f}")
    for k, v in baseline.items():
        print(f"  {k:20s} {v:.4f}")
    verdict = "PASS" if metrics["auroc"] > baseline["tanimoto_auroc"] else "BELOW"
    print(f"  -> model AUROC {verdict} vs tanimoto baseline")


if __name__ == "__main__":
    main()
