---
pretty_name: Gut Wall Cleaned
tags:
- chemistry
- pharmacokinetics
- tabular
---

# Gut Wall Cleaned

This dataset contains 1,490 accepted finite scalar measurement children from the
versioned canonical-base pipeline. A parent row with several unambiguous measurements
can produce several child rows. Rejected, malformed, ambiguous, bounded, range-only,
TDC-overlapping, and endpoint-unassignable records are not published here.

## Cleaning

The following source columns are replaced in place with validated canonical or lexical
normalizations:

- `smiles` → `canonical_smiles`
- `gut_wall_process` → `canonical_endpoint_key`
- `transporter_or_enzyme` → `transporter_or_enzyme_normalized`
- `substrate_status` → `substrate_status_normalized`
- `assay_system` → `assay_system_normalized`
- `intestinal_site` → `intestinal_site_normalized`
- `qualifying_conditions` → `qualifying_conditions_normalized`
- `measured_value` → `scalar_value`

The public column names on the left are retained; the names on the right identify the
validated internal values used to replace them. Columns omitted from this dataset:

- `extraction_id`: `internal_identifier`
- `global_identifier`: `internal_identifier`
- `auc_window`: `all_null_in_dataset`
- `defining_timepoint`: `all_null_in_dataset`

`canonical_endpoint_key` is the endpoint identity field. No `condition_key` is added.
`species_exact` is populated only for explicit, unambiguous species references. Narrative
and other source fields without a validated normalization remain source text.
