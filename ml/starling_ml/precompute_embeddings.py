"""Precompute frozen per-molecule embeddings for the 82K base molecules.

Outputs (into ``paths.embeddings_dir``):
  * ``molformer_emb.npy``  float16 (N, 768)   - MolFormer-XL, mean-pooled over tokens
  * ``metadata_emb.npy``   float16 (N, F, 384) - MiniLM, one vector per metadata field
  * ``manifest.json``      dims, model names, base-parquet checksum (staleness guard)

Because the ~2.5B training pairs reference only these ~82K unique molecules, we encode
each molecule exactly once here and the training loop just gathers by row index.

Usage:
    python -m starling_ml.precompute_embeddings --config ml/configs/default.yaml
    python -m starling_ml.precompute_embeddings --limit 256   # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from .config import Config


def _load_base_columns(path: str, metadata_fields: list[str], limit: int | None) -> dict[str, list]:
    import pyarrow.parquet as pq

    columns = ["smiles", *metadata_fields]
    table = pq.read_table(path, columns=columns)
    if limit is not None:
        table = table.slice(0, limit)
    return {c: table.column(c).to_pylist() for c in columns}


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_transformers_compat() -> None:
    """MolFormer-XL's remote modeling code targets transformers 4.x. Two small shims let
    it load under transformers 5.x (the embeddings are identical — these only patch removed
    import surfaces, not model math):

      1. ``transformers.onnx.OnnxConfig`` (removed) — only used to declare an ONNX-export
         subclass we never instantiate; provide a stub module.
      2. ``transformers.pytorch_utils.find_pruneable_heads_and_indices`` (removed) — only
         used by head-pruning, never during inference; restore the canonical 4.x impl.
    """
    import sys
    import types

    # 1. onnx stub
    try:
        import transformers.onnx  # noqa: F401
    except Exception:
        mod = types.ModuleType("transformers.onnx")
        mod.__file__ = "<shim>"

        def __getattr__(name: str):  # PEP 562 module-level getattr
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)  # don't fake dunders (e.g. __file__, __path__)
            return type(name, (object,), {})

        mod.__getattr__ = __getattr__  # type: ignore[attr-defined]
        sys.modules["transformers.onnx"] = mod

    # 2. find_pruneable_heads_and_indices
    import transformers.pytorch_utils as pu

    if not hasattr(pu, "find_pruneable_heads_and_indices"):
        import torch

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def _patch_model_compat(model) -> None:
    """Make MolFormer-XL's 4.x remote code runnable under transformers 5.x:

      * Restore ``get_head_mask`` (dropped from the mixin); with ``head_mask=None`` it just
        returns a list of Nones.
      * Rebuild the rotary embeddings' non-persistent ``inv_freq``/``cos_cached``/``sin_cached``
        buffers. transformers 5.x materializes the model on the meta device, so buffers
        computed in ``__init__`` (and excluded from the checkpoint) come back as NaN — we
        recompute them from each module's deterministic ``dim``/``base``.
    """
    import torch
    import types

    if not hasattr(model, "get_head_mask"):

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                raise NotImplementedError("head_mask not supported by compat shim")
            return [None] * num_hidden_layers

        model.get_head_mask = types.MethodType(get_head_mask, model)
    if not hasattr(model, "warn_if_padding_and_no_attention_mask"):
        model.warn_if_padding_and_no_attention_mask = types.MethodType(
            lambda self, *a, **k: None, model
        )

    # Rebuild rotary caches that the meta-device init left as NaN.
    n_fixed = 0
    for module in model.modules():
        if hasattr(module, "_set_cos_sin_cache") and hasattr(module, "inv_freq"):
            device = module.inv_freq.device
            dim, base = module.dim, module.base
            inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            with torch.no_grad():
                module.inv_freq.copy_(inv)
                module._set_cos_sin_cache(module.max_position_embeddings, device, torch.float32)
            n_fixed += 1
    if n_fixed:
        print(f"[molformer] rebuilt rotary caches for {n_fixed} modules")


def _mean_pool(last_hidden, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # (B, T, 1)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def compute_molformer(smiles: list[str], cfg: Config) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    _ensure_transformers_compat()
    name = cfg.embedding.molformer_model
    print(f"[molformer] loading {name}")
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        name, trust_remote_code=True, deterministic_eval=True
    )
    _patch_model_compat(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    bs = cfg.embedding.smiles_batch_size
    out = np.empty((len(smiles), cfg.embedding.mol_emb_dim), dtype=np.float16)
    amp = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(smiles), bs):
            batch = smiles[i : i + bs]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=cfg.embedding.max_smiles_tokens,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(device_type="cuda", dtype=amp, enabled=(device == "cuda")):
                hidden = model(**enc).last_hidden_state  # (B, T, 768)
            pooled = _mean_pool(hidden.float(), enc["attention_mask"])
            out[i : i + len(batch)] = pooled.cpu().numpy().astype(np.float16)
            if i % (bs * 20) == 0:
                done = i + len(batch)
                rate = done / max(time.time() - t0, 1e-6)
                print(f"[molformer] {done}/{len(smiles)} ({rate:.0f} mol/s)")
    return out


def compute_metadata(columns: dict[str, list], cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Embed each metadata field with MiniLM and emit a presence mask.

    Returns ``(emb, present)`` where ``emb`` is ``(N, F, 384)`` float16 with **zeros for missing
    fields** (the model overrides these via a learned per-field missing embedding), and ``present``
    is ``(N, F)`` uint8 (1 = field non-null and non-empty after trim). Only present values are sent
    through the encoder.
    """
    from sentence_transformers import SentenceTransformer

    name = cfg.embedding.text_encoder_model
    print(f"[metadata] loading {name}")
    device = "cuda" if _cuda() else "cpu"
    encoder = SentenceTransformer(name, device=device)

    n = len(columns["smiles"])
    n_fields = cfg.embedding.n_meta_fields
    out = np.zeros((n, n_fields, cfg.embedding.text_emb_dim), dtype=np.float16)  # missing -> zeros
    present = np.zeros((n, n_fields), dtype=np.uint8)
    for f_idx, field_name in enumerate(cfg.embedding.metadata_fields):
        raw = columns[field_name]
        mask = np.array(
            [(v is not None and str(v).strip() != "") for v in raw], dtype=bool
        )
        present[:, f_idx] = mask
        idx = np.nonzero(mask)[0]
        print(f"[metadata] field {f_idx + 1}/{n_fields}: {field_name} ({len(idx)}/{n} present)")
        if len(idx) == 0:
            continue
        texts = [str(raw[i]) for i in idx]
        emb = encoder.encode(
            texts,
            batch_size=cfg.embedding.text_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        out[idx, f_idx, :] = emb.astype(np.float16)
    return out, present


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ml/configs/default.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=None, help="encode only first N molecules (smoke test)")
    parser.add_argument("--output-dir", default=None, help="override paths.embeddings_dir")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config).apply_overrides(args.overrides)
    out_dir = args.output_dir or cfg.paths.embeddings_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[base] loading {cfg.paths.base_parquet}")
    columns = _load_base_columns(cfg.paths.base_parquet, cfg.embedding.metadata_fields, args.limit)
    n = len(columns["smiles"])
    print(f"[base] {n} molecules")

    mol_emb = compute_molformer(columns["smiles"], cfg)
    meta_emb, meta_present = compute_metadata(columns, cfg)

    mol_path = os.path.join(out_dir, "molformer_emb.npy")
    meta_path = os.path.join(out_dir, "metadata_emb.npy")
    present_path = os.path.join(out_dir, "metadata_present.npy")
    np.save(mol_path, mol_emb)
    np.save(meta_path, meta_emb)
    np.save(present_path, meta_present)

    manifest = {
        "n_molecules": n,
        "limit": args.limit,
        "molformer_model": cfg.embedding.molformer_model,
        "text_encoder_model": cfg.embedding.text_encoder_model,
        "mol_emb_shape": list(mol_emb.shape),
        "metadata_emb_shape": list(meta_emb.shape),
        "metadata_present_shape": list(meta_present.shape),
        "metadata_fields": list(cfg.embedding.metadata_fields),
        "metadata_present_rate": [round(float(x), 4) for x in meta_present.mean(axis=0)],
        "base_parquet": cfg.paths.base_parquet,
        "base_parquet_sha256": _file_sha256(cfg.paths.base_parquet),
        "dtype": "float16",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[done] wrote {mol_path} {mol_emb.shape}, {meta_path} {meta_emb.shape}, "
          f"{present_path} {meta_present.shape}")


if __name__ == "__main__":
    main()
