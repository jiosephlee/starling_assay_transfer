# Starling Assay Transfer Scripts

This workspace contains a generic molecular-transfer data pipeline plus one
Starling-specific preprocessing adapter.

## Pipeline

1. Build the cleaned Starling dataset:

```bash
python scripts/preprocess_starling_oral_bioavailability.py \
  --output-dir datasets/base/starling_oral_bioavailability_numeric
```

Add `--repo-id owner/name` to upload the cleaned dataset. The output keeps the
original Starling columns and replaces only `oral_bioavailability_value` with a
numeric percent value.

2. Create compact generic transfer pairs from one numeric dataset:

```bash
python scripts/create_transfer_pairs_compact_parquet.py \
  --input datasets/base/starling_oral_bioavailability_numeric \
  --output-dir datasets/pairs_compact/starling_oral_bioavailability_pairs \
  --enumerate-all
```

3. Create molecule-disjoint compact splits:

```bash
python scripts/create_splits_from_compact_pairs.py \
  --input-dir datasets/pairs_compact/starling_oral_bioavailability_pairs \
  --eval-pairs-per-split 30000 \
  --similarity-buckets 6 \
  --similarity-thresholds 0.10 0.20 0.40 0.60 0.80 \
  --output-dir datasets/pairs_split_compact/starling_oral_bioavailability_pair_splits
```

For the full oral-bioavailability artifact on a 64-CPU EPYC/Genoa node:

```bash
sbatch scripts/run_oral_bioavailability_splits_compact_epyc.sbatch
```

The full-run SLURM wrapper is restart-aware: it reuses
`_bucketed_input/`, keeps that bucketed compact table on disk, and writes
phase checkpoints under `checkpoints/`.

4. Materialize full split pairs after molecule-overlap discard:

```bash
python scripts/materialize_full_pairs_from_splits.py \
  --base-input datasets/base/starling_oral_bioavailability_numeric \
  --split-dir datasets/pairs_split/starling_oral_bioavailability_pair_splits \
  --output-dir datasets/pairs_split_full/starling_oral_bioavailability_pair_splits_full
```

5. Render HF Parquets:

```bash
python scripts/create_hf_parquets_from_splits.py \
  --split-dir datasets/pairs_split_full/starling_oral_bioavailability_pair_splits_full \
  --template templates/generic_transfer_classification.jinja \
  --output-dir datasets/pairs_split_hf/starling_oral_bioavailability_transfer_hf
```

6. Tokenize for TRL:

```bash
python scripts/tokenize_hf_for_trl.py \
  --input-dir datasets/pairs_split_hf/starling_oral_bioavailability_transfer_hf \
  --tokenizer Qwen/Qwen3-8B \
  --output-dir datasets/pairs_split_hf/tokenized/starling_oral_bioavailability/qwen3_8b
```

7. Upload any local artifact folder:

```bash
python scripts/upload_hf_dataset.py \
  --folder datasets/pairs_split_hf/starling_oral_bioavailability_transfer_hf \
  --repo-id owner/name
```

## Contracts

- Pair creation is generic: it accepts one dataset and column names.
- Compact pair artifacts are lightweight by default: row indices, labels,
  value difference, similarity, and null/not-null metadata flags for
  stratification.
- Starling-specific joining/cleanup is isolated to
  `preprocess_starling_oral_bioavailability.py`.
- Splits are molecule-disjoint. Validation is selected first, test second, and
  train keeps only pairs internal to the remaining molecules.
- Validation/test selection is pair-first and proportional by stratum, 30k
  pairs each by default. Strata are label + similarity bucket + null/not-null
  flags for configured metadata columns. Within each needed stratum, pairs are
  selected by molecule reuse count: two molecules already in the split, then
  one, then zero. This prioritizes stratum proportions while still encouraging
  compact molecule sets.
- Similarity buckets can use fixed thresholds. The full oral-bioavailability
  SLURM job currently uses `0.10 0.20 0.40 0.60 0.80`.
- Long split runs checkpoint major phases: quantiles, bucketed input, source
  strata, eval selections, test-eligible strata, and final write stats.
- Full metadata is reattached only after split selection with
  `materialize_full_pairs_from_splits.py`. By default, materialized prompt
  metadata includes `support_text`, `molecule_name`, and study context fields,
  but excludes `pmid`.
