# Assay Transfer Design V6.5: tmax Re-band + ListNet Removal + Target Smoothing

Status: active design contract (delta over V6)
Last updated: 2026-07-20

V6.5 is a targeted revision of [V6](assay_transfer_design_v6_intern.md). Everything in the V6
contract holds unchanged **except** the three deltas below. Where V6.5 is silent, the V6
document governs.

## 1. Why V6.5 exists

Auditing the V6 metric-normalization policy against the built targets showed exactly one
mis-specified endpoint: `oral_exposure.tmax`. All other endpoints (cmax, auc, clint, papp,
solubility, half_life, bioavailability, fraction_absorbed, efflux_ratio, …) are
well-calibrated — ~32–45% of same-condition pairs land in the "transfer" class.

`tmax` (time-to-peak) fell through the wildcard rule `[oral_exposure, "*"]` into
`positive_scalar` (log10, within-2-fold / 5-fold). Fold-change is the wrong geometry for a
timing parameter with a compressed natural range, so **62.5% of tmax pairs were labelled
"transfer"** — nearly double every other endpoint, and in the saturated direction — polluting
the `oral_exposure` eval slice.

## 2. Delta 1 — `oral_exposure.tmax` re-banded to a time metric

A new metric band `time_hours` is added to the frozen policy
(`configs/assay_transfer/v6_5/metrics.yaml`, `version: metric_thresholds_v6_5`):

```yaml
time_hours: {transform: time_hours, domain: positive, transfer_max: 1.0, not_transfer_min: 3.0,
             display: "within 1 hour / at least 3 hours apart"}
rules:
  time_hours:
    - [oral_exposure, tmax]
```

`[oral_exposure, tmax]` is an exact rule, which beats the `[oral_exposure, "*"]` wildcard in
`V3Policies.metric_for`, so cmax/auc/etc. are untouched. `tmax` **stays inside the
`oral_exposure` assay concept** (re-band in place; no concept split).

The `time_hours` transform (`pipeline/v3_policy.py`, `MetricSpec.transform_value`)
canonicalizes the raw value to hours before the identity absolute-distance comparison:

```text
hour -> value        minute -> value / 60        day -> value * 24
```

Any other unit (`percent`, `ha`, `hb`, `dimensionless_ratio`, `hours_later`) is not a time and
returns `None` → the record is rejected at compose with reason `invalid_metric_domain`. This
drops the **19 garbage-unit tmax rows** that V6 silently admitted under `positive_scalar`.

`transform_value` now takes an optional `unit_basis`; `compose_v3._eligible_row` passes
`row["unit_basis"]`. Everything downstream is unchanged — the compose stage bakes the new
`comparison_value` (hours) and `transfer_max=1.0` / `not_transfer_min=3.0` onto each record, and
`target_for` reads those baked columns exactly as in V6.

### Frozen counts

Dropping the 19 garbage rows (all in train; no molecule fully removed) changes the source
eligible counts:

| quantity | V6 | V6.5 |
| --- | --- | --- |
| eligible records | 138,806 | **138,787** |
| molecules | 14,982 | 14,982 (unchanged) |
| train records | 116,060 | **116,041** |
| validation records | 11,276 | 11,276 (unchanged) |
| test records | 11,470 | 11,470 (unchanged) |

The molecule universe (14,982) and split-seed prefix are unchanged, so the split assignment is
**≥99.8% identical to V6** (validation 1,249/1,250 preserved; test 1,247/1,250; train
12,480/12,482). The handful that reshuffle do so because `_heldout_subset` solves an exact
per-split *record* quota by subset-sum over per-molecule record weights, and dropping the 19
garbage tmax rows nudged a few molecules' weights. Split sizes stay exact
(12,482 / 1,250 / 1,250 molecules) and no molecule appears in two splits. A perfectly controlled
V6-vs-V6.5 eval should intersect the two held-out sets rather than assume identity.

## 3. Delta 2 — ListNet removed; train is a flat SFT dataset

V6.5 does not use the ListNet loss. The training objective is the full-vocabulary soft A/B
cross-entropy at the decision token plus weighted formatting/EOS cross-entropy — the
`L_soft_AB` term of the V6 objective, with `lambda_listnet = 0`.

Consequently:

- The **train split is a flat HF dataset**: one document per admitted directed pair, carrying
  only the model-facing prompt/completion and the soft target (`target_distribution`,
  `target_z`, `target_a/target_b`, `distance`) plus provenance (`pair_id`, record/endpoint
  ids). There are **no** `listnet_*` group columns.
- There is **no offline packing plan**. `packed_chunk_index` / `optimizer_batch_*` /
  `templated_token_count` columns, the 4,096-token BFD group packing, the group-indivisibility
  constraint, and `scripts/apply_v6_packing_plan.py` are all removed from the V6.5 flow.
  **SFT performs its own packing** at train time.

Pair *selection* still reuses the V6 anchor round-robin + relevance-spread sampler
(`iter_list_groups`), so the training distribution (concept balance, relevance coverage)
matches V6; the output is simply flattened and the group bookkeeping stripped.

The eval benchmarks are unchanged: the ordinary 2,000-comparison set and the
20,000-comparison ranking set per split are evaluation artifacts, not training loss, and are
kept exactly as in V6 (§7 of the V6 contract).

## 4. Delta 3 — label smoothing instead of hard clipping

V6 formed the A/B target as `q_A = clip(σ(z), 0.1, 0.9)`. The hard clip collapses every
strong transfer to exactly 0.9:

```text
σ(z):     0.99   0.95   0.91
clipped:  0.90   0.90   0.90   <- ranking information destroyed
```

Because ListNet is gone (Delta 2), the pointwise soft target is now the *only* place ranking
strength is taught, so that flattening is especially harmful. V6.5 replaces the clip with a
monotonic affine squash (`target_for` in `pipeline/v6_intern.py`):

```text
q_A = eps + (1 - 2*eps) * sigmoid(z / T)     # eps = TARGET_SMOOTHING = 0.1, T = TARGET_TEMPERATURE = 1
q_B = 1 - q_A
```

```text
σ(z):      0.99    0.95    0.91
smoothed:  0.892   0.860   0.828   <- bounded ~[0.1, 0.9] AND still ordered
```

Properties: bounds stay ≈[0.1, 0.9]; the map is strictly monotonic in `z`, so stronger and
weaker transfers keep their order; and `σ(z)=0.5 → q_A=0.5`, so the z=0 crossover — hence the
`(A)/(B)` completion label and all pair *selection* (which keys on `distance`/`target_z`, never
`target_a`) — is unchanged. Only `target_a`/`target_b`/`target_distribution` shift; `target_z`
and benchmark membership are identical. `z/T` is clamped to ±700 before the exponential to avoid
overflow on extreme distances.

## 4. Build & verification

- Compose: `pipeline.stages.compose_v3` with `configs/assay_transfer/v6_5/release.yaml`
  → `datasets/eligible/assay_transfer_soft_evidence_v6_5/records.parquet`
  (expect `eligible_records: 138787`, 19 new `invalid_metric_domain` rejections).
- Build: `scripts/build_v6_intern_raw_pair.py` (flat by default; `--listnet` restores the V6
  path) with `--expected-records 138787 --expected-split train=116041,validation=11276,test=11470`
  → `datasets/hf_parquet/assay_transfer_raw_pair_v6_5_intern`.
- Verify: `scripts/verify_v6_intern_raw_pair.py` (packing/group invariants gated behind
  `--listnet`; the default flat path checks per-row correctness, unique `pair_id`, no
  cross-subset pair reuse, and the 2,000 / 20,000 benchmark cardinalities).

Acceptance signal for the fix: tmax's per-endpoint "% transfer" drops from **62.5% → ~33–40%**,
in line with cmax/auc, while every other endpoint's fraction is unchanged versus V6.

## 5. V6.5.1 — per-concept prompt templates + 4M train

V6.5.1 is a prompt-only + size revision of V6.5; the **eligible source records, targets,
`target_z`, pair selection, and benchmark membership are identical to V6.5**. Two changes:

1. **Per-concept Jinja prompt templates.** `render_prompt` (`pipeline/v6_intern.py`) now renders
   one template per assay concept from `templates/assay_transfer_v6_5_1_intern/{concept}.jinja`
   (loaded via `jinja2` `FileSystemLoader` + `StrictUndefined`), replacing the single hard-coded
   string. Each template frames the concept explicitly and surfaces only that concept's relevant
   assay-context fields. Molecule A (retrieved) shows its value + context; **Molecule B (query)**
   shows SMILES + the same concept-relevant context with the numeric value hidden (the templates
   never reference the query value, so it cannot leak). Only the `prompt` column text changes.
2. **Train grows to 4,000,000 flat pairs** (`--train-pairs 4000000`); eval splits unchanged.

Published as a distinct public dataset `jiosephlee/assay-transfer-raw-pair-v6.5.1-intern`. The
V6.5 prompt format is superseded; already-published V6/V6.5 artifacts are untouched.

## 6. V6.5.2 — full context restored + threshold line removed

V6.5.1's per-concept templates trimmed context to ~7–8 of 16 fields, which dropped populated,
transfer-relevant fields (e.g. `species_or_population` 75% on Fa, `measured_process` 100% on
Fa/Fh, `extra_details` 57% on oral_exposure) and regressed quality. Since the query value is
hidden, the model relies on the query-vs-retrieval **context comparison**, so that loss matters.

V6.5.2 keeps the per-concept template files but (a) shows **all 16 context fields** on both
Molecule A and Molecule B (query value still hidden), and (b) removes the constant, non-actionable
`Classify transfer when values are …` line. Train is **2,000,000** pairs to match V6.5 for a clean
prompt-only comparison. `render_prompt`'s `TEMPLATE_DIR` now points at
`templates/assay_transfer_v6_5_2_intern`. Published as
`jiosephlee/assay-transfer-raw-pair-v6.5.2-intern`; source records/targets/`target_z`/benchmark
membership identical to V6.5.
