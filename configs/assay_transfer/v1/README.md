# Assay-transfer v1 configuration

Status: design scaffold; not yet consumed by the current pipeline.

These files are the versioned control plane for constructing the generalized Q2-Q4
assay-transfer dataset. The species contract additionally covers oral bioavailability
and Q1 so species identities are normalized consistently across every source. The
files deliberately separate three questions:

1. `condition_keys.yaml`: are two assay records eligible to be compared?
2. `metric_thresholds.yaml`: how close must values of this metric type be to transfer?
3. `label_policy.yaml`: how are record-level outcomes aggregated into a pair target?

`endpoints.yaml` assigns canonical endpoints to metric types and condition-key
schemas. `species.yaml` defines explicit-only species recovery for the nonuniform oral
and Q1-Q4 raw schemas. `release.yaml` pins the files into one reproducible contract.

The implemented structured resolver is
`scripts/internal/assay_species_normalization.py`.
`scripts/internal/species_normalization.py` is its oral-compatible facade. New oral
builds and Q1-Q4 emit only nullable `species_exact` under `species_v1`;
`species_group`, `species_basis`, and `species_status` are not part of the contract.
The resolver recognizes literal exact-species designations in allowlisted structured
fields but does not infer species from population terms or cell-line origin.
Already-materialized oral v3 artifacts remain historical and are not rewritten. Run
`python scripts/audit_assay_species.py` to audit oral and Q1-Q4 coverage without
changing any dataset.

## Invariants

- The query contributes structure only at model inference.
- The retrieval record contributes its normalized metadata and optionally its value.
- Condition-key buckets define comparable settings; membership is not automatically a
  positive transfer label.
- Thresholds are attached to canonical metric types, never to realized condition-key
  buckets.
- The same metric threshold applies in `same_endpoint`,
  `same_species_same_endpoint`, and `most_specific`.
- A future context-specific tolerance must be represented as a scientifically distinct
  metric subtype and a new version, not as an ad hoc bucket override.
- Unknown never equals unknown in a condition key.
- Molecules are split before pair targets are constructed.

## Intended build order

```text
raw oral and Q1-Q4 records
-> normalized canonical records
-> molecule identities and split assignments
-> record-first labels and pairs within each split
-> ML-ready datasets
```

The a priori metric thresholds are frozen before this build. Training-only sensitivity
analyses may create alternative dataset versions, but model performance must not tune
the primary label thresholds.
