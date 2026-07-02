---
dataset_info:
  features:
  - name: pmid
    dtype: large_string
  - name: support_text
    dtype: large_string
  - name: molecule_name
    dtype: large_string
  - name: oral_bioavailability_value
    dtype: double
  - name: bioavailability_report_type
    dtype: large_string
  - name: species_or_population
    dtype: large_string
  - name: dose
    dtype: large_string
  - name: oral_exposure_mode
    dtype: large_string
  - name: qualifying_conditions
    dtype: large_string
  - name: comparator
    dtype: large_string
  - name: extra_details
    dtype: large_string
  - name: smiles
    dtype: large_string
  splits:
  - name: train
    num_examples: 82496
---

# Starling Oral Bioavailability Numeric

This dataset keeps the original `starling-labs/Oral_Bioavailability` columns and replaces `oral_bioavailability_value` with the cleaned numeric percent value from `Kiria-Nozan/Starling-bioavailability-clean`.
