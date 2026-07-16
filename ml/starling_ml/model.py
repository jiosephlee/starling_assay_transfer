"""TransferPairModel: frozen cached embeddings -> siamese MLPs -> residual SwiGLU head.

Architecture (per pair, A/B are interchangeable so every per-molecule part is siamese):

    mol_emb[a] (768) --\\                          mol_emb[b] (768) --\\
                  mol_mlp (shared 2-layer)                    mol_mlp (shared)
                       -> h_a (256)                                -> h_b (256)
    meta_emb[a] (7,384) -> per-field proj -> m_a (224)   (shared, field-specific)
    meta_emb[b] (7,384) -> per-field proj -> m_b (224)

    head( concat[h_a, h_b, m_a, m_b] (960) ) -> logit -> BCE/focal

The MolFormer (768) and MiniLM (384) embeddings are frozen, registered as non-persistent
GPU buffers (reproducible from the .npy cache, so excluded from checkpoints).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config, LossConfig, ModelConfig


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_embeddings(embeddings_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the cached frozen embeddings + presence mask, erroring clearly if missing.

    The model never computes MolFormer/MiniLM on the fly — these .npy files (produced by
    ``precompute_embeddings.py``) are the only source. Returns ``(mol_emb, meta_emb, meta_present)``.
    """
    mol_path = os.path.join(embeddings_dir, "molformer_emb.npy")
    meta_path = os.path.join(embeddings_dir, "metadata_emb.npy")
    present_path = os.path.join(embeddings_dir, "metadata_present.npy")
    for p in (mol_path, meta_path, present_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"missing precomputed embedding {p!r}. Run:\n"
                f"  python -m starling_ml.precompute_embeddings --config <config>"
            )
    return np.load(mol_path), np.load(meta_path), np.load(present_path)


def verify_embeddings_fresh(
    cfg: Config, mol_emb: np.ndarray, meta_emb: np.ndarray, meta_present: np.ndarray
) -> None:
    """Guard against stale/mismatched embeddings before training.

    Checks (a) the manifest exists, (b) shapes match the manifest, and (c) the base parquet
    hasn't changed since precompute (via the SHA256 recorded in the manifest). Set the env var
    ``STARLING_SKIP_EMB_CHECK=1`` to bypass (e.g. if the base parquet isn't on this node).
    """
    if os.environ.get("STARLING_SKIP_EMB_CHECK") == "1":
        return
    manifest_path = os.path.join(cfg.paths.embeddings_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"missing {manifest_path!r}; re-run precompute_embeddings to (re)generate the cache."
        )
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    for name, arr, key in (
        ("molformer", mol_emb, "mol_emb_shape"),
        ("metadata", meta_emb, "metadata_emb_shape"),
        ("metadata_present", meta_present, "metadata_present_shape"),
    ):
        expected = manifest.get(key)
        if expected is not None and list(arr.shape) != list(expected):
            raise RuntimeError(
                f"{name} embedding shape {list(arr.shape)} != manifest {expected}; re-run precompute."
            )

    expected_fields = manifest.get("metadata_fields")
    if expected_fields is not None and list(expected_fields) != list(cfg.embedding.metadata_fields):
        raise RuntimeError(
            "metadata fields in embedding manifest do not match config; "
            f"manifest={expected_fields}, config={cfg.embedding.metadata_fields}. "
            "Re-run precompute_embeddings with this config."
        )

    expected_hash = manifest.get("base_parquet_sha256")
    base = cfg.paths.base_parquet
    if expected_hash and os.path.exists(base):
        actual = _file_sha256(base)
        if actual != expected_hash:
            raise RuntimeError(
                f"stale embeddings: base parquet {base!r} has changed since precompute "
                f"(sha256 {actual[:12]}… != {expected_hash[:12]}…). Re-run precompute_embeddings, "
                f"or set STARLING_SKIP_EMB_CHECK=1 to override."
            )


class MetaFieldProjection(nn.Module):
    """Field-specific linear projections (384 -> proj) applied to all fields at once, with a
    learned per-field "missing" embedding.

    Distinct weights per field (so `dose` can be weighted differently from `extra_details`),
    but the same projections are reused for molecule A and B (siamese). For a missing field
    (presence flag 0) the projected text vector is replaced by a learned per-field vector, so
    "absent" is represented distinctly from any real value.
    """

    def __init__(self, n_fields: int, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_fields, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(n_fields, out_dim))
        self.missing_emb = nn.Parameter(torch.randn(n_fields, out_dim) * 0.02)
        nn.init.xavier_uniform_(self.weight)
        self.out_features = n_fields * out_dim

    def forward(self, x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        # x: (B, n_fields, in_dim); present: (B, n_fields) in {0,1}
        proj = torch.einsum("bfd,fdo->bfo", x, self.weight) + self.bias  # (B,F,O)
        present = present.unsqueeze(-1).to(proj.dtype)  # (B,F,1)
        proj = present * proj + (1.0 - present) * self.missing_emb  # missing -> learned per-field vec
        return proj.flatten(1)  # (B, n_fields*out_dim)


class SwiGLUBlock(nn.Module):
    """Pre-norm residual SwiGLU FFN block with optional LayerScale.

    LayerScale (Touvron et al., CaiT) multiplies the residual branch by a learnable per-channel
    ``gamma`` initialized to a small value (``layerscale_init``), so each block starts near-identity
    and the network learns how much each (deep) block contributes — essential for stably training
    deep residual stacks. ``layerscale_init <= 0`` disables it.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float, layerscale_init: float = 1e-4):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w_in = nn.Linear(d_model, 2 * d_ff)
        self.w_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gamma = (
            nn.Parameter(torch.full((d_model,), float(layerscale_init)))
            if layerscale_init and layerscale_init > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        a, b = self.w_in(h).chunk(2, dim=-1)
        h = self.w_out(F.silu(a) * b)
        if self.gamma is not None:
            h = self.gamma * h
        return x + self.dropout(h)


class TransferPairModel(nn.Module):
    def __init__(
        self,
        model_cfg: ModelConfig,
        loss_cfg: LossConfig,
        mol_emb: np.ndarray,
        meta_emb: np.ndarray,
        meta_present: np.ndarray,
    ):
        super().__init__()
        self.loss_cfg = loss_cfg
        n, mol_dim = mol_emb.shape
        _, n_fields, text_dim = meta_emb.shape

        # Frozen feature tables (non-persistent: reproducible from .npy, kept out of checkpoints).
        self.register_buffer("mol_emb", torch.from_numpy(mol_emb).float(), persistent=False)
        self.register_buffer("meta_emb", torch.from_numpy(meta_emb).float(), persistent=False)
        self.register_buffer("meta_present", torch.from_numpy(meta_present).float(), persistent=False)

        # Per-molecule MolFormer branch (siamese 2-layer MLP).
        self.mol_mlp = nn.Sequential(
            nn.Linear(mol_dim, model_cfg.mol_hidden),
            nn.LayerNorm(model_cfg.mol_hidden),
            nn.GELU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(model_cfg.mol_hidden, model_cfg.mol_out),
        )
        # Per-field metadata branch (siamese, field-specific).
        self.meta_proj = MetaFieldProjection(n_fields, text_dim, model_cfg.meta_field_proj)

        # Asymmetric head (assay_transfer_design_v2.md 3): retrieval side contributes
        # structure + metadata + y_A; the query side contributes structure ONLY.
        self.source_value_scale = float(model_cfg.source_value_scale)
        head_in = 2 * model_cfg.mol_out + self.meta_proj.out_features + 1  # h_a, h_b, m_a, y_A
        self.input_ln = nn.LayerNorm(head_in)
        self.in_proj = nn.Linear(head_in, model_cfg.d_model)
        self.blocks = nn.ModuleList(
            SwiGLUBlock(model_cfg.d_model, model_cfg.d_ff, model_cfg.dropout, model_cfg.layerscale_init)
            for _ in range(model_cfg.n_blocks)
        )
        self.out_ln = nn.LayerNorm(model_cfg.d_model)
        # v2 dual head: dist_out is the primary continuous distance; out is the masked
        # binary auxiliary. Both read the shared trunk output.
        self.dist_out = nn.Linear(model_cfg.d_model, 1)
        self.out = nn.Linear(model_cfg.d_model, 1)

        self.register_buffer("pos_weight", torch.tensor(float(loss_cfg.pos_weight)), persistent=False)

    # ---- branches ----
    def _encode_retrieval(self, a_idx: torch.Tensor, meta_a_idx: torch.Tensor) -> torch.Tensor:
        h = self.mol_mlp(self.mol_emb[a_idx])
        m = self.meta_proj(self.meta_emb[meta_a_idx], self.meta_present[meta_a_idx])
        return torch.cat([h, m], dim=-1)

    def _encode_query(self, b_idx: torch.Tensor) -> torch.Tensor:
        return self.mol_mlp(self.mol_emb[b_idx])  # structure only

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(self.input_ln(x))
        for block in self.blocks:
            x = block(x)
        return self.out_ln(x)

    def forward(self, a_idx, b_idx, meta_a_idx, source_value, distance=None, labels=None):
        za = self._encode_retrieval(a_idx, meta_a_idx)  # [h_a | m_a]
        zb = self._encode_query(b_idx)                  # [h_b]
        y_a = (source_value.to(dtype=za.dtype, device=za.device) / self.source_value_scale).unsqueeze(-1)
        trunk = self._trunk(torch.cat([za, zb, y_a], dim=-1))
        distance_pred = F.softplus(self.dist_out(trunk)).squeeze(-1)  # D_expected >= 0
        logits = self.out(trunk).squeeze(-1)
        out = {"distance": distance_pred, "logits": logits}
        if distance is not None:
            out["loss"] = self._loss(distance_pred, logits, distance, labels)
        return out

    # ---- loss: primary continuous + optional masked binary auxiliary ----
    def _loss(self, distance_pred, logits, distance_target, labels) -> torch.Tensor:
        lc = self.loss_cfg
        target = distance_target.to(dtype=distance_pred.dtype, device=distance_pred.device)
        if lc.regression_kind == "mse":
            reg = F.mse_loss(distance_pred, target)
        else:
            reg = F.huber_loss(distance_pred, target, delta=lc.huber_delta)
        total = lc.continuous_weight * reg
        if lc.aux_binary_weight > 0 and labels is not None:
            labels = labels.to(dtype=logits.dtype, device=logits.device)
            mask = torch.isfinite(labels)  # NaN marks a missing hard binary label
            if mask.any():
                total = total + lc.aux_binary_weight * self._binary_loss(logits[mask], labels[mask])
        return total

    def _binary_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = self.loss_cfg.label_smoothing
        if eps > 0:
            targets = targets * (1.0 - eps) + 0.5 * eps
        if self.loss_cfg.kind == "focal":
            ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            p = torch.sigmoid(logits)
            p_t = p * targets + (1.0 - p) * (1.0 - targets)
            alpha = self.loss_cfg.focal_alpha
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            return (alpha_t * (1.0 - p_t).pow(self.loss_cfg.focal_gamma) * ce).mean()
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


def build_model(cfg: Config) -> TransferPairModel:
    mol_emb, meta_emb, meta_present = load_embeddings(cfg.paths.embeddings_dir)
    verify_embeddings_fresh(cfg, mol_emb, meta_emb, meta_present)
    return TransferPairModel(cfg.model, cfg.loss, mol_emb, meta_emb, meta_present)
