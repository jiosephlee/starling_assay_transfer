#!/usr/bin/env sh
set -eu
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] start same_species_v2 intern HF render/upload"
mkdir -p datasets/pairs_split_hf

python scripts/create_oral_bioavailability_hf.py \
  --universe same_species_v2 \
  --variant source_value \
  --split-version v3_v2 \
  --template-family intern \
  --workers 19 \
  --batch-size 1000 \
  --parquet-row-group-size 50000 \
  --progress-every-seconds 300 \
  --overwrite

echo "[$(date -Is)] upload source_value"
hf upload jiosephlee/starling-transfer-shared-eval-same-species-v2-source-value \
  datasets/pairs_split_hf/oral_bioavailability_same_species_v2_source_value_intern_v3_v2 \
  . \
  --repo-type dataset \
  --commit-message "Upload same_species_v2 source_value intern v3_v2 parquets"

python scripts/create_oral_bioavailability_hf.py \
  --universe same_species_v2 \
  --variant no_source_value \
  --split-version v3_v2 \
  --template-family intern \
  --workers 11 \
  --batch-size 1000 \
  --parquet-row-group-size 50000 \
  --progress-every-seconds 300 \
  --validate-unidirectional-train \
  --overwrite

echo "[$(date -Is)] upload no_source_value"
hf upload jiosephlee/starling-transfer-shared-eval-same-species-v2-no-source-value \
  datasets/pairs_split_hf/oral_bioavailability_same_species_v2_no_source_value_intern_v3_v2 \
  . \
  --repo-type dataset \
  --commit-message "Upload same_species_v2 no_source_value intern v3_v2 parquets"

echo "[$(date -Is)] done same_species_v2 intern HF render/upload"
