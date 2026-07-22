---
pretty_name: Intestinal Absorption Cleaned
tags:
- chemistry
- pharmacokinetics
- tabular
---

# Intestinal Absorption Cleaned

This dataset contains 7,897 accepted finite scalar measurement children from the
versioned canonical-base pipeline. A parent row with several unambiguous measurements
can produce several child rows. Rejected, malformed, ambiguous, bounded, range-only,
TDC-overlapping, and endpoint-unassignable records are not published here.

## Cleaning

The following source columns are replaced in place with validated canonical or lexical
normalizations:

- `smiles` → `canonical_smiles`
- `endpoint_category` → `canonical_endpoint_key`
- `assay_system` → `assay_system_normalized`
- `reported_units` → `unit_normalized`
- `condition_medium` → `condition_medium_normalized`
- `biological_context` → `biological_context_normalized`
- `formulation_or_solid_form` → `formulation_or_solid_form_normalized`
- `qualifying_conditions` → `qualifying_conditions_normalized`
- `reported_value` → `scalar_value`

The public column names on the left are retained; the names on the right identify the
validated internal values used to replace them. Columns omitted from this dataset:

- `extraction_id`: `internal_identifier`
- `global_identifier`: `internal_identifier`
- `target`: `all_null_in_dataset`
- `kinetic_parameter`: `all_null_in_dataset`
- `auc_window`: `all_null_in_dataset`

`canonical_endpoint_key` is the endpoint identity field. No `condition_key` is added.
`species_exact` is populated only for explicit, unambiguous species references. Narrative
and other source fields without a validated normalization remain source text.
