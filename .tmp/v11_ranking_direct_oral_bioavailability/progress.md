# V11 ranking endpoint audit

- 2026-08-07: Started audit.
- Goal: determine how many endpoints selected for V11 ranking are direct oral-bioavailability endpoints, with explicit count and denominator.
- Traced the contract to `configs/assay_transfer/v11/prompt_projection.json` and audited the materialized multitask ranking Parquets.
- Confirmed every ranking query has exactly 16 rows and a single task, source, and canonical endpoint key.
- Strict direct definition: canonical endpoint key is `hf_bioavailability / oral_bioavailability / %` and ends in `direct` (absolute, extent_f, or systemic_availability).
- Validation: 10 direct anchors / 80 Bioavailability_Ma anchors = 12.5%; broad `hf_bioavailability` source is 20 / 80 = 25.0%.
- Test: 15 direct anchors / 136 Bioavailability_Ma anchors = 11.03%; broad `hf_bioavailability` source is 30 / 136 = 22.06%.
- Combined: 25 direct anchors / 216 Bioavailability_Ma anchors = 11.57%; broad source is 50 / 216 = 23.15%. Direct is 25 / 50 = 50% of broad-source anchors.
- Across all three v11 tasks: 25 / 356 anchors = 7.02%.
- Distinct endpoint-key view: 3 direct keys / 44 keys represented in Bioavailability_Ma ranking = 6.82%.
- Complete.
