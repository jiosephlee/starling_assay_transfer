# Scope: extending v6.5 transfer pairs to categorical measurements

Status: scoping doc (no code changes yet)
Last updated: 2026-07-21

## Goal

Let categorical measurements — e.g. P‑gp **substrate / non‑substrate**
(`q3.transporter_substrate_status`, `metric_type: binary_category`) — participate as transfer
pairs in the v6.5 line, alongside the numeric endpoints. "Transfer" for a categorical pair means
the retrieval record's **category** is expected to hold for the query molecule/setting.

## Current state — why categorical data never reaches v6.5

Upstream **does** retain categorical measurements as a typed value:

- `pipeline/source_normalization/measurement.py::ParsedMeasurement.categorical_value` and
  `parse_measurement(value, categorical)` carry a validated category.
- `pipeline/source_normalization/normalize.py::_categorical_value` matches raw text against an
  allowlist via `approved_category(value, allowlist)`
  (`measurement.py::approved_category`), e.g. `q3_substrate_status_v1: [inconclusive, not_substrate, substrate]`
  (`configs/source_normalization_v1.yaml:41`).
- `configs/assay_transfer/v1/endpoints.yaml:112` marks `q3.transporter_substrate_status` as
  `canonical_unit: categorical`, `metric_type: binary_category`.

But categorical data is absent from the v6.5 build for **two independent reasons**:

1. **Not emitted as a measured endpoint in the v3 base.** In `datasets/base/canonical_endpoints_v3/*`,
   `substrate_status` exists **only as a context field** (`context_substrate_status`); there is no
   `transporter_substrate_status` `endpoint_subtype`. Confirmed: no substrate `endpoint_subtype` in
   any base, and the v6.5 eligible records contain no `binary_category` metric.
2. **The v3/v6.5 path is numeric-only.**
   - `configs/assay_transfer/v6_5/metrics.yaml` (and `v3/metrics.yaml`) has only numeric bands
     (`bounded_percentage`, `bounded_fraction`, `dimensionless_ratio`, `positive_scalar`, `time_hours`).
   - `pipeline/v3_policy.py::MetricSpec` supports transforms `identity`/`log10`/`time_hours` and a
     two‑threshold **absolute-distance** deadband only — no `equality`/categorical distance.
   - `pipeline/stages/compose_v3.py::_eligible_row` calls `metric.transform_value(scalar_value, unit)`;
     a categorical record has no numeric scalar, so `metric_for` → `None` → rejected `unsupported_metric`.
   - `pipeline/v6_intern.py::target_for` computes `abs(comparison_value_r − comparison_value_q)`
     through a sigmoid — there is no categorical branch.

## Reusable machinery (port, don't reinvent)

The categorical concept already exists in the **v1** pipeline and can be ported forward:

- `pipeline/policy.py::MetricThreshold` supports `distance: "equality"` and
  `transform: "canonical_category"` (the v3 loader dropped these).
- `configs/assay_transfer/v1/metric_thresholds.yaml` defines a `binary_category` metric.
- The upstream `categorical_value` + allowlist validation is already produced by
  `source_normalization` — it just isn't carried into `canonical_endpoints_v3` as an endpoint.

## Work items (ordered)

1. **Emit the categorical endpoint in the v3 base.** The `canonical_endpoints_v3` producer must
   output `q3.transporter_substrate_status` (and any other categorical endpoints) as a *measured*
   record carrying `categorical_value`, not just as `context_substrate_status`. (Upstream of compose.)
2. **Add a categorical band to the v6.5 metric policy.** In `configs/assay_transfer/v6_5/metrics.yaml`
   add a `binary_category` (equality-distance) band and a `rules` entry mapping the substrate
   endpoint family/subtype to it. Bump the metrics `version`.
3. **Extend the policy loader + compose.**
   - `pipeline/v3_policy.py::MetricSpec`: add a categorical/`equality` path — a `distance`/`transform`
     that consumes a string category, produces a categorical `comparison_value`, and a `vote()` that
     returns transfer iff categories are equal.
   - `pipeline/stages/compose_v3.py::_eligible_row`: when the metric is categorical, read the
     `categorical_value` instead of `scalar_value`, and write a categorical `comparison_value`
     (+ set `transfer_max`/`not_transfer_min` sentinels or a categorical marker).
4. **Add a categorical branch to `target_for`** (`pipeline/v6_intern.py`): for categorical pairs,
   `d = 0` if categories match else `1`; map to the smoothed A/B target (`q_A ≈ 0.9` on match,
   `≈ 0.1` on mismatch). Guard so numeric pairs keep the sigmoid path unchanged.
5. **Concept assignment + rebuild.** Add the categorical endpoint to `concepts.yaml` (its own concept
   or an existing one), then rebuild eligible → v6.5.x intern pairs and verify.

## Open design questions (decide before building)

- **Target semantics vs. label smoothing.** Categorical transfer is inherently binary (match / no
  match), so the graded soft target has no within-class ordering. Do we just use the ε/1−ε caps, or
  add a confidence dimension (e.g. `inconclusive` as a third state)?
- **The `inconclusive` category.** Allowlist includes `inconclusive` — treat as ineligible (drop),
  its own class, or a "soft" middle?
- **Eval interaction.** How do `binary_category` pairs coexist with numeric pairs in the ranking
  benchmark (which orders by `target_z`) and the per-concept slices? Likely a **separate categorical
  slice** with accuracy/F1 rather than NDCG.
- **Volume.** How many categorical records exist once emitted upstream? If small, it may warrant its
  own concept and its own eval rather than diluting the numeric ranking metrics.

## Files touched (when implemented)
`canonical_endpoints_v3` producer (upstream), `configs/assay_transfer/v6_5/metrics.yaml`,
`pipeline/v3_policy.py`, `pipeline/stages/compose_v3.py`, `pipeline/v6_intern.py`,
`configs/assay_transfer/v3/concepts.yaml`, then the standard build/verify path
(`scripts/build_v6_intern_raw_pair.py`, `scripts/verify_v6_intern_raw_pair.py`).
