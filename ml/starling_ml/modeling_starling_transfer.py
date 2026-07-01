"""Self-contained end-to-end Starling transfer model (trust_remote_code).

Bundles the frozen encoders (MolFormer-XL for SMILES, all-MiniLM-L6-v2 per metadata field) with the
trained siamese MLPs + residual-SwiGLU head, so

    m = AutoModel.from_pretrained(repo, trust_remote_code=True)
    logits = m(smiles_a=[...], smiles_b=[...], metadata_a=[{...}], metadata_b=[{...}],
               source_value=[...]).logits   # source_value = molecule A's raw oral_bioavailability_value

runs the whole pipeline on raw inputs. This file is intentionally standalone (no project imports) so it
works when loaded from the Hub. It mirrors `starling_ml/model.py` exactly; the head state_dict keys match
`TransferPairModel` so the trained weights (incl. the learned per-field `missing_emb`) load directly.
"""
from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput


# --------------------------------------------------------------------------------------------------
# MolFormer-XL compat shims (its 4.x remote code under transformers 5.x); math-neutral. Mirrors
# starling_ml/precompute_embeddings.py.
# --------------------------------------------------------------------------------------------------
def _ensure_transformers_compat() -> None:
    try:
        import transformers.onnx  # noqa: F401
    except Exception:
        mod = types.ModuleType("transformers.onnx")
        mod.__file__ = "<shim>"

        def __getattr__(name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return type(name, (object,), {})

        mod.__getattr__ = __getattr__  # type: ignore[attr-defined]
        sys.modules["transformers.onnx"] = mod

    import transformers.pytorch_utils as pu

    if not hasattr(pu, "find_pruneable_heads_and_indices"):

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


def _patch_molformer(model, device=None) -> None:
    if not hasattr(model, "get_head_mask"):

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                raise NotImplementedError("head_mask not supported")
            return [None] * num_hidden_layers

        model.get_head_mask = types.MethodType(get_head_mask, model)
    if not hasattr(model, "warn_if_padding_and_no_attention_mask"):
        model.warn_if_padding_and_no_attention_mask = types.MethodType(lambda self, *a, **k: None, model)

    # Rebuild rotary caches (non-persistent buffers; absent/NaN/meta after from_config or meta-load).
    # Reassign on a real device — inv_freq itself may be on meta after a meta-context load.
    if device is None:
        device = next((p.device for p in model.parameters() if p.device.type != "meta"), torch.device("cpu"))
    for module in model.modules():
        if hasattr(module, "_set_cos_sin_cache") and hasattr(module, "inv_freq"):
            dim, base = module.dim, module.base
            inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            module.register_buffer("inv_freq", inv, persistent=False)
            module._set_cos_sin_cache(module.max_position_embeddings, device, torch.float32)


def _mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


# --------------------------------------------------------------------------------------------------
# Head modules — must match starling_ml/model.py exactly (param names) so trained weights load.
# --------------------------------------------------------------------------------------------------
class MetaFieldProjection(nn.Module):
    def __init__(self, n_fields: int, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_fields, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(n_fields, out_dim))
        self.missing_emb = nn.Parameter(torch.randn(n_fields, out_dim) * 0.02)
        nn.init.xavier_uniform_(self.weight)
        self.out_features = n_fields * out_dim

    def forward(self, x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        proj = torch.einsum("bfd,fdo->bfo", x, self.weight) + self.bias
        present = present.unsqueeze(-1).to(proj.dtype)
        proj = present * proj + (1.0 - present) * self.missing_emb
        return proj.flatten(1)


class SwiGLUBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, layerscale_init: float = 0.0):
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


class StarlingTransferConfig(PretrainedConfig):
    model_type = "starling_transfer"

    def __init__(
        self,
        molformer_model: str = "ibm-research/MoLFormer-XL-both-10pct",
        text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        metadata_fields: list[str] | None = None,
        mol_emb_dim: int = 768,
        text_emb_dim: int = 384,
        mol_hidden: int = 1024,
        mol_out: int = 768,
        meta_field_proj: int = 64,
        d_model: int = 1024,
        d_ff: int = 4096,
        n_blocks: int = 32,
        dropout: float = 0.1,
        layerscale_init: float = 0.0,
        use_source_value: bool = False,
        source_value_scale: float = 100.0,
        max_smiles_tokens: int = 202,
        max_text_tokens: int = 256,
        **kwargs,
    ):
        self.molformer_model = molformer_model
        self.text_encoder_model = text_encoder_model
        self.metadata_fields = list(metadata_fields) if metadata_fields else []
        self.mol_emb_dim = mol_emb_dim
        self.text_emb_dim = text_emb_dim
        self.mol_hidden = mol_hidden
        self.mol_out = mol_out
        self.meta_field_proj = meta_field_proj
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.layerscale_init = layerscale_init
        self.use_source_value = use_source_value
        self.source_value_scale = source_value_scale
        self.max_smiles_tokens = max_smiles_tokens
        self.max_text_tokens = max_text_tokens
        super().__init__(**kwargs)


class StarlingTransferModel(PreTrainedModel):
    config_class = StarlingTransferConfig
    main_input_name = "smiles_a"

    def __init__(self, config: StarlingTransferConfig):
        super().__init__(config)
        _ensure_transformers_compat()
        n_fields = len(config.metadata_fields)

        # Frozen encoders. Built from CONFIG (no nested from_pretrained — that breaks under the
        # meta-device context of an outer from_pretrained). Their weights live in this model's own
        # checkpoint: `build_for_export` loads pretrained encoder weights before save_pretrained, and
        # `from_pretrained` restores them; rotary caches are rebuilt post-load via `_patch_molformer`.
        self.mol_tokenizer = AutoTokenizer.from_pretrained(config.molformer_model, trust_remote_code=True)
        mol_cfg = AutoConfig.from_pretrained(
            config.molformer_model, trust_remote_code=True, deterministic_eval=True
        )
        self.molformer = AutoModel.from_config(mol_cfg, trust_remote_code=True)
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_model)
        self.text_encoder = AutoModel.from_config(AutoConfig.from_pretrained(config.text_encoder_model))
        for p in self.molformer.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        # Trained head (names match TransferPairModel).
        self.mol_mlp = nn.Sequential(
            nn.Linear(config.mol_emb_dim, config.mol_hidden),
            nn.LayerNorm(config.mol_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mol_hidden, config.mol_out),
        )
        self.meta_proj = MetaFieldProjection(n_fields, config.text_emb_dim, config.meta_field_proj)
        head_in = 2 * config.mol_out + 2 * self.meta_proj.out_features
        if config.use_source_value:
            head_in += 1
        self.input_ln = nn.LayerNorm(head_in)
        self.in_proj = nn.Linear(head_in, config.d_model)
        self.blocks = nn.ModuleList(
            SwiGLUBlock(config.d_model, config.d_ff, config.dropout, config.layerscale_init)
            for _ in range(config.n_blocks)
        )
        self.out_ln = nn.LayerNorm(config.d_model)
        self.out = nn.Linear(config.d_model, 1)
        self.post_init()  # sets up tied-weights tracking that from_pretrained relies on

    def _init_weights(self, module):  # encoders/head already initialized; nothing to do
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        _ensure_transformers_compat()
        model = super().from_pretrained(*args, **kwargs)
        _patch_molformer(model.molformer)  # rotary caches are non-persistent → rebuild after load
        return model

    @classmethod
    def build_for_export(cls, config: "StarlingTransferConfig") -> "StarlingTransferModel":
        """Construct with pretrained encoder weights loaded — used before `save_pretrained` so the
        exported checkpoint is self-contained (encoder weights included)."""
        _ensure_transformers_compat()
        model = cls(config)
        mol = AutoModel.from_pretrained(config.molformer_model, trust_remote_code=True, deterministic_eval=True)
        model.molformer.load_state_dict(mol.state_dict(), strict=False)
        txt = AutoModel.from_pretrained(config.text_encoder_model)
        model.text_encoder.load_state_dict(txt.state_dict(), strict=False)
        _patch_molformer(model.molformer)
        return model.eval()

    # ---- encoders ----
    @torch.no_grad()
    def _encode_molecule(self, smiles: list[str]) -> torch.Tensor:
        enc = self.mol_tokenizer(
            list(smiles), padding=True, truncation=True,
            max_length=self.config.max_smiles_tokens, return_tensors="pt",
        ).to(self.device)
        hidden = self.molformer(**enc).last_hidden_state
        return _mean_pool(hidden.float(), enc["attention_mask"])

    @torch.no_grad()
    def _encode_metadata(self, metadata: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        n, fields = len(metadata), self.config.metadata_fields
        emb = torch.zeros(n, len(fields), self.config.text_emb_dim, device=self.device)
        present = torch.zeros(n, len(fields), device=self.device)
        for fi, field in enumerate(fields):
            vals = [(m or {}).get(field) for m in metadata]
            idx = [i for i, v in enumerate(vals) if v is not None and str(v).strip() != ""]
            present[idx, fi] = 1.0
            if not idx:
                continue
            enc = self.text_tokenizer(
                [str(vals[i]) for i in idx], padding=True, truncation=True,
                max_length=self.config.max_text_tokens, return_tensors="pt",
            ).to(self.device)
            pooled = _mean_pool(self.text_encoder(**enc).last_hidden_state.float(), enc["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=-1)  # precompute (SentenceTransformer) L2-normalizes
            emb[torch.tensor(idx, device=self.device), fi] = pooled.to(emb.dtype)
        return emb, present

    def _encode_pair_side(self, smiles, metadata) -> torch.Tensor:
        h = self.mol_mlp(self._encode_molecule(smiles))
        meta_emb, present = self._encode_metadata(metadata)
        m = self.meta_proj(meta_emb, present)
        return torch.cat([h, m], dim=-1)

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(self.input_ln(x))
        for block in self.blocks:
            x = block(x)
        return self.out(self.out_ln(x)).squeeze(-1)

    def forward(self, smiles_a, smiles_b, metadata_a, metadata_b, source_value=None, labels=None):
        za = self._encode_pair_side(smiles_a, metadata_a)
        zb = self._encode_pair_side(smiles_b, metadata_b)
        parts = [za, zb]
        if self.config.use_source_value:
            if source_value is None:
                raise ValueError("source_value (molecule A's raw oral_bioavailability_value) is required")
            sv = torch.as_tensor(source_value, dtype=za.dtype, device=za.device) / self.config.source_value_scale
            parts.append(sv.reshape(-1, 1))
        logits = self._head(torch.cat(parts, dim=-1))
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, torch.as_tensor(labels, dtype=logits.dtype, device=logits.device))
        return ModelOutput(loss=loss, logits=logits)
