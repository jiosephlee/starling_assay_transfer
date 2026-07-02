# Handoff: running the KNN eval callback on a fresh machine

This repo tracks **all code** and the **small seed datasets** needed to reproduce
the Starling oral-bioavailability KNN evaluation. The large generated artifacts
(training pairs, memmaps, embeddings, model checkpoints, run logs) are **not** in
git — they are regenerable from the seed data, or downloadable from HuggingFace.

_Last updated: 2026-07-01. Supersedes the earlier SLURM "compact splits"
handoff (that workflow and its scripts were removed; see git history if needed)._

## TL;DR — can you run the KNN callback after a fresh clone?

Yes, with some setup. The record-level KNN eval (`RecordKnnEvalCallback` in
`ml/starling_ml/train.py`) reads **local parquet** from
`datasets/starling_eval/condition_key_v3_record_splits_hf/`, which is **now
committed to git** — so the eval input is present on clone. What you still need
to produce locally is the **training data** and a **model**: the callback embeds
molecules with the live model during a training run, so you exercise it by
running `train.py`.

## What IS in git (present on a fresh clone)

- All code: `ml/starling_ml/`, `ml/scripts/`, `ml/configs/*.yaml`, `scripts/`,
  `configs/oral_bioavailability_v3.yaml`, templates.
- `ml/requirements.txt` — Python dependencies.
- **Raw TDC seed**: `tdc/official_tianang/` (incl. `Bioavailability_Ma.jsonl`).
- **Cleaned base**: `datasets/base/Oral_bioavailability_cleaned/` and
  `.../Oral_bioavailability_cleaned_v3/`.
- **Record-split eval set (the KNN callback input)**:
  `datasets/starling_eval/condition_key_v3_record_splits_hf/`.
- **Exclusions/reference**: `datasets/exclusions/tdc_bioavailability_ma_v3/`.

## What is NOT in git (missing pieces to fill in)

| Missing artifact | Path (gitignored) | Size | How to fill in |
|---|---|---|---|
| Training pair datasets | `datasets/pairs_split_full/`, `datasets/pairs_split_hf/`, `datasets/pairs_compact/` | 100s of GB | Regenerate via `scripts/run_oral_bioavailability_pipeline.py` |
| Memmap training tensors | `ml/artifacts/memmap_*` | ~25 GB | Built by `train.py` from the pair datasets |
| Precomputed embeddings | `ml/artifacts/embeddings_*` | ~0.3 GB | `ml/starling_ml/precompute_embeddings.py` |
| KNN candidate cache | `ml/artifacts/record_knn_eval_cache/` | ~0.6 GB | Auto-rebuilt on first eval (Tanimoto); or `--cache-only` |
| Trained model checkpoints | `ml/artifacts/runs/**/model.safetensors` | ~28 GB | Retrain, or copy a checkpoint. Not currently pushed to a known public HF repo |
| W&B logs | `wandb/` | — | Regenerated at runtime |

## Data on HuggingFace

- `jiosephlee/starling-oral-bioavailability-cleaned-aligned-with-assay-tool`
  — **public**. Same record-split dataset now committed under
  `datasets/starling_eval/...`. To fetch instead of using the git copy:
  ```bash
  huggingface-cli download --repo-type dataset \
    jiosephlee/starling-oral-bioavailability-cleaned-aligned-with-assay-tool \
    --local-dir datasets/starling_eval/condition_key_v3_record_splits_hf
  ```
- `jiosephlee/starling-transfer-shared-eval-*` (source / no-source variants)
  — **private** (needs an auth token with access). Used by
  `ml/scripts/run_shared_eval_benchmark.py`.

## Reproduce from scratch (git only, no HF)

1. Install deps: `pip install -r ml/requirements.txt`
   (needs `torch`, `transformers`, `datasets`, `rdkit`, `huggingface_hub`, etc.).
2. Build datasets from the tracked TDC seed:
   ```bash
   python scripts/run_oral_bioavailability_pipeline.py \
     --config configs/oral_bioavailability_v3.yaml
   ```
   (produces base_v3, the pair splits, and the record splits).
3. Train, which runs the KNN callback against the committed record splits:
   ```bash
   cd ml && python -m starling_ml.train \
     --config ml/configs/shared_eval_same_species_v2.yaml
   ```
   The callback reads `datasets/starling_eval/condition_key_v3_record_splits_hf`.

## SLURM

Ready-to-adapt sbatch templates are in `slurm/`. They wrap the current
entrypoints; **edit the cluster-specific bits** (`--partition`, `--account`,
`--gres`, env activation, paths) for the target cluster before submitting.

| Template | Job type | Purpose | Starting resources |
|---|---|---|---|
| `slurm/pipeline_cpu.sbatch` | CPU | Regenerate datasets from the TDC seed (base → pairs → record splits). Pair enumeration is CPU-bound. | 64 CPUs, 256G, 24h |
| `slurm/precompute_embeddings_gpu.sbatch` | GPU | Frozen MolFormer + MiniLM embeddings for the ~82K base molecules (run once). | 1 GPU, 64G, 2h |
| `slurm/train_gpu.sbatch` | GPU | Train the model — **this runs the record-KNN eval callback**. Multi-GPU via torchrun. | 2 GPUs, 128G, 24h |

CPU resource specs come from the original Genoa/EPYC runs (64 CPUs, 128–256G).
GPU specs are conservative defaults — tune to your node type. Typical order:

```bash
sbatch slurm/pipeline_cpu.sbatch                  # 1. build data (once)
sbatch slurm/precompute_embeddings_gpu.sbatch     # 2. embeddings (once)
sbatch slurm/train_gpu.sbatch                     # 3. train + KNN callback
```

Override the config without editing the file, e.g.
`CONFIG=ml/configs/shared_eval_condition_key.yaml sbatch slurm/train_gpu.sbatch`.
Each template writes logs to `logs/<job>_%j.out|err` (gitignored).

## Run the record-KNN eval standalone (needs a trained checkpoint)

```bash
python -m starling_ml.record_knn_eval \
  --config ml/configs/shared_eval_same_species_v2.yaml \
  --checkpoint <path/to/model.safetensors> \
  --dataset-dir datasets/starling_eval/condition_key_v3_record_splits_hf \
  --split validation_1
```

Add `--cache-only` to build the Tanimoto candidate cache without a model.

## Config pointers (defaults in `ml/starling_ml/config.py`)

- `record_knn_eval_dataset_dir = datasets/starling_eval/condition_key_v3_record_splits_hf`
- `record_knn_eval_dataset_config = full_metadata`
- `record_knn_eval_splits = [validation_1]`
- `record_knn_eval_cache_dir = ml/artifacts/record_knn_eval_cache/...`
- `base_parquet = datasets/base/Oral_bioavailability_cleaned_v2/train.parquet`
  — note: only `Oral_bioavailability_cleaned` and `_v3` are committed; the `_v2`
  base is a pipeline output. Point configs at `_v3` or rebuild `_v2` first.

## gitignore policy (so the handoff stays reproducible)

`.gitignore` keeps all files >100 MB out of git. The only exceptions carved back
in are the small seed dirs above (`datasets/base/`, `datasets/starling_eval/`,
`datasets/exclusions/`). If you add a new small seed input under `datasets/`,
add a matching `!datasets/<dir>/` negation so it ships with the repo.
