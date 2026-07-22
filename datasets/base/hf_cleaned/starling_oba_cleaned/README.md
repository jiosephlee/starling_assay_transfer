---
pretty_name: Starling Oba Cleaned
tags:
- chemistry
- pharmacokinetics
- tabular
---

# Starling Oba Cleaned

This dataset contains 27,640 accepted finite scalar measurement children from the
versioned canonical-base pipeline. A parent row with several unambiguous measurements
can produce several child rows. Rejected, malformed, ambiguous, bounded, range-only,
TDC-overlapping, and endpoint-unassignable records are not published here.

## Cleaning

The following source columns are replaced in place with validated canonical or lexical
normalizations:

- `smiles` → `canonical_smiles`
- `bioavailability_report_type` → `bioavailability_report_type_normalized`
- `species_or_population` → `species_or_population_mechanical_normalized`
- `dose` → `dose_normalized`
- `oral_exposure_mode` → `oral_exposure_mode_normalized`
- `qualifying_conditions` → `qualifying_conditions_normalized`
- `comparator` → `comparator_normalized`
- `oral_bioavailability_value` → `scalar_value`

The public column names on the left are retained; the names on the right identify the
validated internal values used to replace them. Columns omitted from this dataset:

- `direction`: `all_null_in_dataset`
- `target`: `all_null_in_dataset`
- `kinetic_parameter`: `all_null_in_dataset`
- `auc_window`: `all_null_in_dataset`
- `defining_timepoint`: `all_null_in_dataset`
- `variation_value`: `all_null_in_dataset`
- `variation_type`: `all_null_in_dataset`
- `accompanying_interval_lower`: `all_null_in_dataset`
- `accompanying_interval_upper`: `all_null_in_dataset`

`canonical_endpoint_key` is the endpoint identity field. No `condition_key` is added.
`species_exact` is populated only for explicit, unambiguous species references. Narrative
and other source fields without a validated normalization remain source text.
