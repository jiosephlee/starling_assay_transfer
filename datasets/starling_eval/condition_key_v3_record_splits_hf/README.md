---
pretty_name: Starling Oral Bioavailability Cleaned Aligned With Assay Tool
tags:
- chemistry
- molecular-property-prediction
- tabular
configs:
- config_name: full_metadata
  default: true
  data_files:
  - split: train
    path: data/full_metadata/train/*.parquet
  - split: validation_1
    path: data/full_metadata/validation_1/*.parquet
  - split: validation_2
    path: data/full_metadata/validation_2/*.parquet
  - split: test
    path: data/full_metadata/test/*.parquet
- config_name: smiles_only
  data_files:
  - split: train
    path: data/smiles_only/train/*.parquet
  - split: validation_1
    path: data/smiles_only/validation_1/*.parquet
  - split: validation_2
    path: data/smiles_only/validation_2/*.parquet
  - split: test
    path: data/smiles_only/test/*.parquet
---

# Starling Oral Bioavailability Cleaned Aligned With Assay Tool

`full_metadata` contains cleaned single-record oral-bioavailability rows with
`row_index` and binary label `Y`, where `Y = 1` means
`oral_bioavailability_value >= 20.0`.

`smiles_only` contains exact-SMILES deduplicated rows with only `smiles` and
`Y`. Duplicate SMILES labels are computed from the median raw
`oral_bioavailability_value` across all records for that SMILES.

## Split alignment note

These splits are derived from the condition-key v3_v2 assay-transfer pair
split. The split unit for this record dataset is exact SMILES, so
`full_metadata` and `smiles_only` use the same molecule assignment. To keep
record-level evaluation sizes manageable, some molecules that were present in
the original pair-split held-out molecule pools are intentionally moved back to
train. The row cap policy is recorded in `metadata/split_manifest.json`.
