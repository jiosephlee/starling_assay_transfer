# Starling Assay Transfer Handoff

This repo contains the generic transfer-pair pipeline and Starling oral
bioavailability adapter. The immediate next task is to run compact split
generation on a machine with available CPUs.

## Current State

- No SLURM job is currently running. Pending job `6633940` was cancelled before
  it started.
- The code path for full split generation has been converted to a non-SLURM
  local runner:

```bash
bash scripts/run_oral_bioavailability_splits_compact_local.sh
```

- The current fixed similarity thresholds are:

```text
0.10 0.20 0.40 0.60 0.80
```

This creates six buckets:

```text
0: similarity <= 0.10
1: 0.10 < similarity <= 0.20
2: 0.20 < similarity <= 0.40
3: 0.40 < similarity <= 0.60
4: 0.60 < similarity <= 0.80
5: similarity > 0.80
```

## Required Data Artifacts

The source repo intentionally ignores `datasets/` in git. Copy these artifacts
to the new server separately if you want to continue from the existing pair
enumeration:

```text
datasets/pairs_compact/oral_bioavailability_pairs_full/
```

Expected compact pair artifact:

```text
records/part-*.parquet    256 shards
metadata.json
pairs_written             2,455,662,084
candidate_pairs_seen      3,402,753,760
disk size                 about 21G
```

The cleaned base dataset is also useful to keep:

```text
datasets/base/Oral_bioavailability_cleaned/
```

It was uploaded to Hugging Face as:

```text
jiosephlee/Oral_bioavailability_cleaned
```

Do not rely on the old split output directory unless you intentionally want the
previous quantile-bucket experiment:

```text
datasets/pairs_split_compact/oral_bioavailability_pair_splits/
```

The next clean run should overwrite/rebuild that output using fixed thresholds.

## Run Command

From the repo root:

```bash
bash scripts/run_oral_bioavailability_splits_compact_local.sh
```

Useful overrides:

```bash
ARROW_NUM_THREADS=64 OMP_NUM_THREADS=64 \
PROGRESS_EVERY_SECONDS=60 \
bash scripts/run_oral_bioavailability_splits_compact_local.sh
```

To use custom paths:

```bash
INPUT_DIR=/path/to/oral_bioavailability_pairs_full \
OUTPUT_DIR=/path/to/oral_bioavailability_pair_splits \
bash scripts/run_oral_bioavailability_splits_compact_local.sh
```

The local runner calls:

```bash
python scripts/create_splits_from_compact_pairs.py \
  --input-dir datasets/pairs_compact/oral_bioavailability_pairs_full \
  --output-dir datasets/pairs_split_compact/oral_bioavailability_pair_splits \
  --eval-pairs-per-split 30000 \
  --similarity-buckets 6 \
  --similarity-thresholds 0.10 0.20 0.40 0.60 0.80 \
  --batch-size 250000 \
  --row-group-size 250000 \
  --bucket-file-row-limit 10000000 \
  --parquet-compression zstd \
  --progress-every-seconds 300 \
  --keep-bucketed-input \
  --overwrite
```

## What The Split Script Does

1. Builds `_bucketed_input/` by copying compact pairs and replacing
   `similarity_bucket` using the fixed thresholds.
2. Counts source strata.
3. Selects validation pairs, then test pairs, each proportional by stratum.
4. Enforces no molecule overlap across validation/test/train.
5. Writes compact Parquet outputs:

```text
train/
validation/
test/
metadata.json
checkpoints/
similarity_quantiles.json
_bucketed_input/
```

## Checkpointing

The script writes checkpoints under:

```text
OUTPUT_DIR/checkpoints/
```

Major checkpoint files:

```text
bucketed_input.json
source_strata.json
validation_selection.json
test_available_strata.json
test_selection.json
write_stats.json
```

For a clean fixed-threshold rebuild, use the local runner as-is with
`--overwrite`.

If a run is interrupted after `_bucketed_input/` is complete and you want to
resume from that bucket, run:

```bash
python scripts/create_splits_from_compact_pairs.py \
  --input-dir datasets/pairs_compact/oral_bioavailability_pairs_full \
  --output-dir datasets/pairs_split_compact/oral_bioavailability_pair_splits \
  --eval-pairs-per-split 30000 \
  --similarity-buckets 6 \
  --similarity-thresholds 0.10 0.20 0.40 0.60 0.80 \
  --reuse-bucketed-input \
  --resume-checkpoints \
  --keep-bucketed-input \
  --overwrite
```

Only use `--reuse-bucketed-input` if the existing bucket was built with the same
thresholds.

## Optimizations Already Made

- Bucket assignment is vectorized with NumPy/Arrow:

```python
np.searchsorted(thresholds, weighted_tanimoto)
```

- Bucketed Parquet files are rolled at about 10M rows per file to avoid
  thousands of tiny files.
- Stratum counting is vectorized with encoded integer stratum IDs and
  `numpy.bincount`.
- Final train writing uses vectorized Arrow batch filters. Validation/test are
  written from selected-row checkpoints.

## Expected Runtime Notes

The old row-by-row bucket writer took about 2h45m on one 64-core EPYC node for
2.456B pairs. The optimized bucket writer should be faster, but the true runtime
on the new server should be judged from progress logs.

Progress lines are written to stderr every `PROGRESS_EVERY_SECONDS` seconds and
include rows processed, percent complete, rows/sec, elapsed time, and ETA.

## Dependencies

The environment used here had:

- Python 3
- pyarrow
- numpy
- rdkit for pair generation and preprocessing scripts
- datasets / huggingface_hub for HF upload and dataset loading paths

For split generation from existing compact pairs, the critical packages are:

```text
pyarrow
numpy
```

## Validation

A smoke test passed with the fixed thresholds:

```bash
python scripts/create_splits_from_compact_pairs.py \
  --input-dir datasets/pairs_compact/smoke_300 \
  --output-dir datasets/pairs_split_compact/smoke_fixed_0102040608 \
  --eval-pairs-per-split 100 \
  --similarity-buckets 6 \
  --similarity-thresholds 0.10 0.20 0.40 0.60 0.80 \
  --batch-size 5000 \
  --row-group-size 5000 \
  --bucket-file-row-limit 10000 \
  --keep-bucketed-input \
  --overwrite
```

Result:

```text
validation rows: 100
test rows:       100
molecule overlap errors: 0
```
