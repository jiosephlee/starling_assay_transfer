"""Export the best checkpoint as a self-contained end-to-end model and (optionally) upload to the Hub.

    python -m starling_ml.export_upload --run ml/artifacts/runs/ssv2_srcval_10ep \
        --repo-id jiosephlee/starling-transfer-ssv2-srcval --public

Reads ``<run>/best_val_macro_f1/model.safetensors`` (or legacy ``<run>/best``), wraps it with the frozen
MolFormer + MiniLM encoders into ``StarlingTransferModel``, ``save_pretrained``s a trust_remote_code repo,
and uploads it. The HF repo then loads with ``AutoModel.from_pretrained(repo, trust_remote_code=True)``.
Requires `huggingface-cli login` (or HF_TOKEN) before uploading.
"""
from __future__ import annotations

import argparse
import os

from safetensors.torch import load_file

from .benchmark_spec import BEST_VAL_CHECKPOINT_DIR
from .config import Config
from .modeling_starling_transfer import StarlingTransferConfig, StarlingTransferModel

_README = """---
license: mit
library_name: transformers
tags:
- molecular-property-prediction
- oral-bioavailability
- chemistry
---

# Starling oral-bioavailability transfer model

Given **two molecules** (SMILES + study metadata) and **molecule A's measured oral bioavailability**,
this model predicts whether oral-bioavailability behavior **transfers** from A to B — i.e. whether the
two molecules behave similarly under the given study context. It is **self-contained**: the frozen
encoders are bundled with the trained head, so it runs end-to-end on raw inputs.

## Architecture

Per molecule (siamese — the same encoders + projections are applied to A and B; only the head is
position-aware):

- **Molecule encoder** — `{molformer}` (MolFormer-XL), **frozen**: SMILES → mean-pooled token
  embedding → **{mol_emb}-d**, then a 2-layer MLP ({mol_emb}→{mol_hidden}→{mol_out}).
- **Metadata encoder** — `{text_encoder}` (MiniLM), **frozen**: each of the **{n_fields} metadata
  fields** is embedded **separately** (mean-pooled, L2-normalized) → **{text_emb}-d**, then a learned
  per-field projection → {proj}-d ({n_fields}×{proj} = **{meta_vec}-d** total).
  A **missing/empty field uses a learned per-field "missing" embedding** instead of the text embedding,
  so absent metadata is handled gracefully and distinctly from any real value.
- Per molecule = `[mol_mlp ({mol_out}) | metadata ({meta_vec})]` = **{z_per_mol}-d**.

Pair head:

- Concatenate `{head_inputs}` → **{head_in}-d** input.
- A **pre-norm residual SwiGLU MLP** ({n_blocks} blocks, width {d_model}, FFN {d_ff}) → one logit.
- `sigmoid(logit)` = P(transfer). ~{n_params}M trainable params; encoders frozen.

## Metadata fields (order matters)

`{fields}`

Pass a dict per molecule keyed by these names. **Omit a key, or pass `None`/`""`, for a missing field**
— the model then uses its learned per-field "missing" embedding.

## Usage

```python
from transformers import AutoModel
m = AutoModel.from_pretrained("{repo}", trust_remote_code=True).eval()

out = m(
    smiles_a=["CC(=O)Oc1ccccc1C(=O)O"],          # molecule A (bioavailability known)
    smiles_b=["CCO"],                            # molecule B (candidate)
    metadata_a=[{{"species_or_population": "human", "dose": "325 mg", "oral_exposure_mode": "tablet"}}],
    metadata_b=[{{"species_or_population": "human"}}],   # missing fields are fine
{source_value_usage}
)
p_transfer = out.logits.sigmoid()                # batched: pass parallel lists for many pairs
```

{source_value_note}Inputs are batched lists of equal length.

## Training & performance

Trained on the `same_species_v2` oral-bioavailability transfer split (~338M molecule pairs; the frozen
embeddings are precomputed once and the head is trained on top). The label is `|value_A - value_B|`
thresholded.{training_value_note}

- same_species_v2 validation: AUROC ~0.87, accuracy ~0.83, macro-F1 ~0.79
- tianang (cross-dataset) validation: AUROC ~0.95, accuracy ~0.91, macro-F1 ~0.89 (test: AUROC ~0.95)
"""


def _infer_use_source_value(head: dict, cfg: Config) -> bool:
    base = 2 * cfg.model.mol_out + 2 * (cfg.model.meta_field_proj * len(cfg.embedding.metadata_fields))
    return int(head["input_ln.weight"].shape[0]) == base + 1


def build_export_model(cfg: Config, best_dir: str) -> StarlingTransferModel:
    head = load_file(os.path.join(best_dir, "model.safetensors"))
    use_sv = _infer_use_source_value(head, cfg)
    em, mc = cfg.embedding, cfg.model
    sc = StarlingTransferConfig(
        molformer_model=em.molformer_model, text_encoder_model=em.text_encoder_model,
        metadata_fields=list(em.metadata_fields), mol_emb_dim=em.mol_emb_dim, text_emb_dim=em.text_emb_dim,
        mol_hidden=mc.mol_hidden, mol_out=mc.mol_out, meta_field_proj=mc.meta_field_proj,
        d_model=mc.d_model, d_ff=mc.d_ff, n_blocks=mc.n_blocks, dropout=mc.dropout,
        layerscale_init=mc.layerscale_init, use_source_value=use_sv, source_value_scale=mc.source_value_scale,
        max_smiles_tokens=em.max_smiles_tokens,
    )
    model = StarlingTransferModel.build_for_export(sc)
    missing, unexpected = model.load_state_dict(head, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected head keys not in export model: {list(unexpected)[:8]}")
    print(f"[export] use_source_value={use_sv}; loaded {len(head)} head tensors "
          f"({sum('molformer' in m or 'text_encoder' in m for m in missing)} encoder keys kept from pretrained)")
    return model.eval()


def _source_value_readme_parts(use_source_value: bool, scale: int) -> dict[str, str]:
    if use_source_value:
        return {
            "head_inputs": "[z_A, z_B] **+ molecule A's bioavailability scalar** "
            f"(`value_A / {scale}`)",
            "source_value_usage": (
                "    source_value=[68.0],                         # molecule A's RAW "
                "oral_bioavailability_value (e.g. percent)"
            ),
            "source_value_note": (
                "`source_value` is molecule A's **raw** `oral_bioavailability_value`; "
                f"the model scales it internally by {scale}. "
            ),
            "training_value_note": (
                " The model uses A's known value as an **anchor** and learns to estimate B's "
                "bioavailability from its structure + metadata."
            ),
        }
    return {
        "head_inputs": "[z_A, z_B]",
        "source_value_usage": "    # no source_value argument for this variant",
        "source_value_note": "This variant does not take a `source_value` input. ",
        "training_value_note": (
            " This no-source-value variant learns transfer from molecule structure and metadata only."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help=f"run dir containing {BEST_VAL_CHECKPOINT_DIR}/")
    ap.add_argument("--config", default="ml/configs/default.yaml")
    ap.add_argument("--out", default=None, help="export dir (default <run>/export)")
    ap.add_argument("--repo-id", default=None, help="HF model repo to upload to (omit = export only)")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    best_dir = os.path.join(args.run, BEST_VAL_CHECKPOINT_DIR)
    if not os.path.exists(best_dir):
        best_dir = os.path.join(args.run, "best")
    out = args.out or os.path.join(args.run, "export")

    model = build_export_model(cfg, best_dir)
    StarlingTransferConfig.register_for_auto_class()
    StarlingTransferModel.register_for_auto_class("AutoModel")
    model.save_pretrained(out)
    repo = args.repo_id or "<your-repo>"
    mcfg = model.config
    n_fields = len(mcfg.metadata_fields)
    meta_vec = mcfg.meta_field_proj * n_fields
    z_per_mol = mcfg.mol_out + meta_vec
    head_in = 2 * z_per_mol + (1 if mcfg.use_source_value else 0)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    source_value_parts = _source_value_readme_parts(
        bool(mcfg.use_source_value), int(mcfg.source_value_scale)
    )
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write(_README.format(
            repo=repo, fields=", ".join(mcfg.metadata_fields),
            molformer=mcfg.molformer_model, text_encoder=mcfg.text_encoder_model,
            mol_emb=mcfg.mol_emb_dim, mol_hidden=mcfg.mol_hidden, mol_out=mcfg.mol_out,
            text_emb=mcfg.text_emb_dim, proj=mcfg.meta_field_proj, n_fields=n_fields,
            meta_vec=meta_vec, z_per_mol=z_per_mol, head_in=head_in,
            n_blocks=mcfg.n_blocks, d_model=mcfg.d_model, d_ff=mcfg.d_ff,
            scale=int(mcfg.source_value_scale), n_params=round(n_params),
            **source_value_parts,
        ))
    print(f"[export] saved trust_remote_code model -> {out}")

    if args.repo_id:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
        url = api.upload_folder(folder_path=out, repo_id=args.repo_id, repo_type="model")
        print(f"[upload] -> {url}")


if __name__ == "__main__":
    main()
