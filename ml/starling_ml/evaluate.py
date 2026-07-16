"""Evaluate a trained v2 checkpoint: continuous distance metrics + binary on the labeled subset.

The v2 primary objective is continuous distance regression (assay_transfer_design_v2.md
15); the binary benchmark is evaluated only on the hard-binary-labeled subset via a
validation-calibrated cutoff (a full calibration sweep is a follow-up — this reports the
0.5-logit-cutoff binary metrics alongside the regression metrics).

Usage:
    python -m starling_ml.evaluate --config ml/configs/<v2>.yaml \
        --checkpoint ml/artifacts/runs/<run>/best --split test
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import DEFAULT_METADATA_FIELDS, PairDataset, build_split_memmap, collate_pairs
from .metrics import binary_metrics, regression_metrics
from .model import build_model


def _load_state_dict(checkpoint: str) -> dict:
    safet = os.path.join(checkpoint, "model.safetensors")
    binf = os.path.join(checkpoint, "pytorch_model.bin")
    if os.path.exists(safet):
        from safetensors.torch import load_file

        return load_file(safet)
    if os.path.exists(binf):
        return torch.load(binf, map_location="cpu")
    raise FileNotFoundError(f"no model weights in {checkpoint}")


def _index_maps(cfg: Config) -> tuple[dict, dict]:
    with open(os.path.join(cfg.paths.embeddings_dir, "smiles_index.json")) as fh:
        smiles_to_row = json.load(fh)
    with open(os.path.join(cfg.paths.embeddings_dir, "meta_index.json")) as fh:
        meta_to_row = json.load(fh)
    return smiles_to_row, meta_to_row


@torch.no_grad()
def _predict(model, dataset, batch_size: int, device: str):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pairs)
    dist_p, logit_p, dist_t, lab_t = [], [], [], []
    amp = device == "cuda" and torch.cuda.is_bf16_supported()
    for batch in loader:
        inputs = {k: batch[k].to(device) for k in ("a_idx", "b_idx", "meta_a_idx", "source_value")}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(**inputs)
        dist_p.append(out["distance"].float().cpu().numpy())
        logit_p.append(out["logits"].float().cpu().numpy())
        dist_t.append(batch["distance"].numpy())
        lab_t.append(batch["labels"].numpy())
    return (np.concatenate(dist_p), np.concatenate(logit_p),
            np.concatenate(dist_t), np.concatenate(lab_t))


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
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()

    smiles_to_row, meta_to_row = _index_maps(cfg)
    build_split_memmap(cfg.paths.dataset_parquet, cfg.paths.memmap_dir, args.split,
                       smiles_to_row, meta_to_row, metadata_fields=DEFAULT_METADATA_FIELDS)
    dataset = PairDataset(cfg.paths.memmap_dir, args.split)

    dist_p, logits, dist_t, labels = _predict(model, dataset, cfg.train.per_device_batch_size, device)

    print(f"\n=== {args.split} ({len(dist_t)} examples) ===")
    print("continuous distance (primary):")
    for k, v in regression_metrics(dist_p, dist_t).items():
        print(f"  {k:12s} {v:.4f}")

    # Binary benchmark on the hard-binary-labeled subset (0.5-logit cutoff; calibrate on
    # validation for the frozen test cutoff).
    hard = np.isfinite(labels)
    if hard.sum() > 0:
        print(f"binary (hard-labeled subset, {int(hard.sum())} examples):")
        for k, v in binary_metrics(logits[hard], labels[hard]).items():
            if "/" not in k:
                print(f"  {k:12s} {v:.4f}")


if __name__ == "__main__":
    main()
