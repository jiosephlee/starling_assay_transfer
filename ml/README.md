# `ml/` — MolFormer pair transfer-classification model

Self-contained training code for the oral-bioavailability **transfer** task: given two
molecules (SMILES + study metadata), predict whether bioavailability behavior transfers
between them (binary). Decoupled from the PyArrow/RDKit data pipeline in `../scripts/`.

## Key idea

Only ~82K unique molecules back the ~1.4B training pairs, so we **encode each molecule once**
(frozen MolFormer + per-field MiniLM), cache the embeddings, hold them as GPU buffers, and
make the training DataLoader ship only integer indices. A training step is then an
index-gather + a few small matmuls — which is what makes billions of pairs tractable.

## Architecture

```
mol_emb[a] (768) ─ mol_mlp (siamese 2-layer) ─ h_a (256) ┐
meta_emb[a] (7×384) ─ per-field proj (siamese) ─ m_a (224)├─► residual SwiGLU head ─► logit
mol_emb[b] (768) ─ mol_mlp ───────────────────  h_b (256)│        (8 blocks, d_model=768)
meta_emb[b] (7×384) ─ per-field proj ─────────── m_b (224)┘   BCE(pos_weight) | focal
```

- **Frozen**: `ibm-research/MoLFormer-XL-both-10pct` (mean-pooled, 768-d) and
  `sentence-transformers/all-MiniLM-L6-v2` (per metadata field, 384-d).
- **Trained**: siamese branch MLPs + residual SwiGLU head (~25M params, Mid tier).
- **Loss**: configurable (`loss.kind: bce|focal`, plus optional `label_smoothing`).

## Setup

Target env is conda `openrlhf` (torch 2.10+cu128, transformers 5.7, accelerate, pyarrow,
8×A100-80GB). Only `sentence-transformers` is typically missing:

```bash
pip install -r ml/requirements.txt        # or: pip install sentence-transformers
```

### MolFormer / transformers 5.x compatibility

MolFormer-XL's remote modeling code was written for transformers 4.x. `precompute_embeddings.py`
applies four small, **math-neutral** shims so it runs under transformers 5.7 (see
`_ensure_transformers_compat` / `_patch_model_compat`): an `onnx` stub (unused export class),
`find_pruneable_heads_and_indices` (only used by head-pruning), `get_head_mask` (returns
`[None]*num_layers` for the `head_mask=None` inference path), and rebuilding the rotary
`cos/sin` caches that the meta-device load leaves as NaN. None alter the forward pass — they
restore exactly the 4.x behavior. (Alternative: run precompute in a dedicated transformers 4.x
env; since embeddings are cached to `.npy` and the train/eval path never imports MolFormer, the
precompute env is fully decoupled.)

## Run

All commands run from the repo root.

```bash
# 1. Precompute frozen embeddings (once; ~minutes on one A100). Smoke test with --limit.
python -m starling_ml.precompute_embeddings --config ml/configs/default.yaml
python -m starling_ml.precompute_embeddings --limit 256        # quick check

# 2. Train (builds compact (a,b,label) memmaps on first run). Multi-GPU via torchrun.
torchrun --nproc_per_node=8 -m starling_ml.train --config ml/configs/default.yaml
# tiny single-GPU sanity run on the val split as stand-in train set:
python -m starling_ml.train --set train.bf16=true train.torch_compile=false \
    train.max_steps=200 train.per_device_batch_size=4096 train.dataloader_num_workers=2 \
    --train-split validation --eval-split validation

# 3. Evaluate the final model (reports AUROC/acc/F1 + tanimoto baseline).
#    No periodic checkpoints are saved; only one final save at paths.output_dir.
python -m starling_ml.evaluate --config ml/configs/default.yaml \
    --checkpoint ml/artifacts/runs/default --split test
```

Helper wrappers: `ml/scripts/run_precompute.sh`, `ml/scripts/run_train.sh`.

## Config

`ml/configs/default.yaml` mirrors `starling_ml/config.py:Config`. Override any key inline:
`--set model.n_blocks=12 model.d_model=1024 loss.kind=focal train.learning_rate=2e-4`.

### Scaling ladder (when train≈val, climb; when train≫val, you've hit the frozen-embedding ceiling)
| Tier | d_model | d_ff | n_blocks | ~head params |
|------|---------|------|----------|--------------|
| Baseline | 512 | 1024 | 4 | ~5M |
| Mid (default) | 768 | 2048 | 8 | ~25M |
| Large | 1024 | 3072 | 12 | ~90M |

## Artifacts (git-ignored under `ml/artifacts/`)
- `embeddings/` — `molformer_emb.npy`, `metadata_emb.npy`, `manifest.json`
- `memmap/<split>/` — `a.u32`, `b.u32`, `label.i8`, `meta.json`
- `runs/<name>/` — Trainer logs + single final model (no periodic checkpoints)
