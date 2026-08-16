# Starling Assay Transfer Scripts

The canonical dataset and model architecture for generalizing beyond oral
bioavailability is maintained in
[`docs/assay_transfer_design.md`](docs/assay_transfer_design.md). The earlier
[`docs/ASSAY_TRANSFER_GENERALIZATION_PLAN.md`](docs/ASSAY_TRANSFER_GENERALIZATION_PLAN.md)
is retained as the exploratory analysis record.

This workspace contains a generic molecular-transfer data pipeline plus one
Starling-specific preprocessing adapter.

## Assay Transfer V12/V12.1

The active contract for new TxAgent-backed builds is
[`docs/assay_transfer_design_v12.md`](docs/assay_transfer_design_v12.md). V12 uses the corrected
task gold lineages, molecule-disjoint validation/test splits grouped by TxAgent normalized-parent
identity, train-only bucket eligibility and CDF calibration, same-parent training pairs, and
24-candidate validation/test ranking lists. V12.1 freezes all V12 membership and changes only its
target-derived fields using an exact train-pair distance CDF. V11/V11.1 remain frozen historical
releases.

Build and verify one task with:

```bash
python -m scripts.build_v12_release --task bioavailability_ma --rebuild-source
python -m scripts.verify_v12_release \
  --root datasets/hf_parquet/assay_transfer_raw_pair_v12/bioavailability_ma/with_categorical \
  --source datasets/eligible/assay_transfer_starling_txagent_v12/bioavailability_ma/records.parquet
```

## Assay Transfer V4

The active V4 contract is documented in
[`docs/assay_transfer_design_v4.md`](docs/assay_transfer_design_v4.md). It trains an LM
against empirical transfer, non-transfer, and ambiguous vote fractions while retaining
the frozen hard A/B validation and test sets.

Build the V4 dataset with:

```bash
python pipeline/stages/run.py \
  --config configs/builds/assay_transfer_soft_evidence_v4.yaml
```

The HF output retains `prompt`, modal `completion`, and `metadata`.
Training must use `metadata.target_distribution`; the modal completion is not a hard target.

## Assay Transfer V5

The active A/B-only variance-temperature contract is documented in
[`docs/assay_transfer_design_v5.md`](docs/assay_transfer_design_v5.md). Build both the
standard and Intern artifacts from the frozen V4 membership with:

```bash
python pipeline/stages/run.py \
  --config configs/builds/assay_transfer_variance_soft_v5.yaml
```

V5 stores a two-state target distribution, its evidence-derived temperature, and an
explicit tie-anchor flag. Exact count and model-logit ties deterministically choose A.

## Assay Transfer V6 Designs

The active V6/V6.5 research contracts replace query-value aggregates with directed raw
record-to-record supervision inside each condition key:

- [`docs/assay_transfer_design_v6_intern.md`](docs/assay_transfer_design_v6_intern.md) defines Intern soft A/B prediction plus
  four-candidate ListNet training and frozen 20,000-comparison validation/test rankings.
- [`docs/assay_transfer_design_v6_mlp.md`](docs/assay_transfer_design_v6_mlp.md) defines the shared SMILES/assay encoder design,
  cached-embedding 100M soft-prediction MLP comparison on the V6.5 universe.

Both contracts exclude same-molecule pairs and prohibit cross-condition labels, negatives,
and ranking lists.

The V6.5 MLP artifact is built with `scripts/build_v6_mlp_raw_pair.py`. Its 100M
MoLFormer/PubMedBERT ablation is prepared and launched with the
`ml/scripts/*v6_mlp_100m*` commands. Official metrics use W&B project
`assay-transfer-soft` and group `assay-transfer-raw-pair-v6-5-soft`.

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
