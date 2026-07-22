---
pretty_name: Hepatic Cleaned
tags:
- chemistry
- pharmacokinetics
- tabular
---

# Hepatic Cleaned

This dataset contains 17,637 accepted finite scalar measurement children from the
versioned canonical-base pipeline. A parent row with several unambiguous measurements
can produce several child rows. Rejected, malformed, ambiguous, bounded, range-only,
TDC-overlapping, and endpoint-unassignable records are not published here.

## Cleaning

The following source columns are replaced in place with validated canonical or lexical
normalizations:

- `smiles` → `canonical_smiles`
- `metric_type` → `canonical_endpoint_key`
- `assay_system` → `assay_system_normalized`
- `species` → `species_normalized`
- `molecular_form` → `molecular_form_normalized`
- `reported_units` → `unit_normalized`
- `enzyme_or_pathway` → `enzyme_or_pathway_normalized`
- `qualifying_conditions` → `qualifying_conditions_normalized`
- `reported_value` → `scalar_value`

The public column names on the left are retained; the names on the right identify the
validated internal values used to replace them. Columns omitted from this dataset:

- `extraction_id`: `internal_identifier`
- `global_identifier`: `internal_identifier`
- `direction`: `all_null_in_dataset`
- `auc_window`: `all_null_in_dataset`

`canonical_endpoint_key` is the endpoint identity field. No `condition_key` is added.
`species_exact` is populated only for explicit, unambiguous species references. Narrative
and other source fields without a validated normalization remain source text.
