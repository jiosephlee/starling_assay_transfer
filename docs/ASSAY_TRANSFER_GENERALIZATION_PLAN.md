# Assay-Conditioned Transfer Similarity: Analysis and Plan

Status: historical exploration and dataset-analysis record. The canonical architecture
is now `docs/assay_transfer_design.md`; where the two conflict, the latter governs.

## 1. Objective

The project learns a better assay-aware transfer score than Morgan fingerprint
similarity. It does not directly predict the query molecule's assay value.

The model estimates whether a concrete retrieved measurement is transferable to a query
molecule:

```text
score(retrieved measurement record -> query molecule)
```

Inputs available at inference are:

- the query molecule structure;
- the retrieved molecule structure;
- all metadata for the retrieved measurement;
- optionally the retrieved measurement value;
- the assay/endpoint request represented by the retrieval context.

Query measurement records and query-side metadata are unavailable at inference and must
never be model features.

## 2. Non-goals and retired behavior

- Do not train a model whose primary output is the query assay value.
- Do not assume the transfer score is a symmetric mathematical metric. Metadata is
  available only on the retrieval side, making the desired score directional even when
  the source value is omitted.
- Do not run KNN evaluation, choose KNN checkpoints, or log KNN metrics during training.
  Retrieval/KNN evaluation may remain a standalone post-training benchmark.
- Do not compare records from different canonical endpoint types, even in the most
  liberal pairing baseline.

## 3. Canonical inference and label contract

For pairing mode `M`, let `K_M(record)` be the selected condition key. For every concrete
retrieval record `r` and different query molecule `q`:

```text
query_group(q, r, M) = {
    x : molecule(x) = q
        and endpoint_key(x) = endpoint_key(r)
        and K_M(x) = K_M(r)
}

query_target = mean(transformed_value(x) for x in query_group)
distance     = abs(transformed_value(r) - query_target)
label        = endpoint-specific threshold_policy(distance)
```

All query-record variation beyond the selected key is deliberately marginalized. A
single query record would introduce arbitrary dose, formulation, protocol, or other
context unavailable at inference.

For `same_species_same_endpoint`, a monkey retrieval record is compared to the mean of
all monkey records for the query molecule measuring the same canonical endpoint. Dose,
formulation, and other context not included in that baseline are averaged over. Bird
records cannot enter that query target.

The query mean, group size, group variance, and contributing record identifiers are
label provenance only. They may support auditing or loss weighting but must not enter
the model.

### Aggregation by value type

- Bounded percentages/fractions: arithmetic mean on the canonical scale.
- Positive continuous values using a log transform: average on the log scale, equivalent
  to a geometric mean in native units. Compare against median/trimmed-log-mean during the
  replicate audit.
- Binary categorical values: aggregate the class distribution. Use a query target only
  when consensus exceeds a calibrated minimum (provisionally 80%); otherwise drop it.
- Ordinal categories: aggregate an encoded ordered distribution and require adequate
  consensus rather than inventing a precise continuous value.
- Multi-parameter records: split into separate canonical measurements before aggregation.
- Ranges/censored values: preserve interval information; do not replace every interval by
  its first number.

### Split safety

- Assign every record for a canonical molecule to one split.
- Compute query aggregates using only records belonging to that molecule's split.
- Fit parsers, unit mappings, and threshold calibration on training records only.
- Never allow validation/test query records to influence training aggregates.
- Deduplicate repeated extractions of the same source passage before estimating noise.

## 4. Full datasets

The Parquets under `datasets/starling_assays/datasets/` are canonical. CSV exports under
`datasets/starling_assays/unpacked/` are for inspection only.

| Dataset | Scientific role | Rows | Unique SMILES | Missing SMILES |
|---|---|---:|---:|---:|
| Q1 | Oral systemic exposure | 119,192 | 3,442 | 1,120 |
| Q2 | Fa: absorption-related factors | 85,061 | 5,520 | 7,769 |
| Q3 | Fg: gut-wall escape factors | 27,713 | 3,182 | 3,095 |
| Q4 | Fh: hepatic escape factors | 67,943 | 6,984 | 4,862 |

This phase uses Q2-Q4. Q1 is excluded from the new endpoint analysis.

### Raw category cleanup required

The data contain values outside their extraction-guidance enums:

- Q2: 1,230 rows, mainly 1,131 `caco2_mdck_permeability` rows plus spelling variants.
- Q3: 14 rows across spelling variants and adjacent transport categories.
- Q4: 26 rows across adjacent clearance/metabolism categories.

Map obvious variants with an audited alias table. Quarantine semantically ambiguous
categories rather than silently coercing them.

## 5. Pairing baseline names and semantics

Endpoint identity is mandatory in every universe. The previous name `no_constraints`
is misleading for a multi-endpoint dataset because it must never mean cross-endpoint
pairing.

### `same_endpoint`

```text
key = canonical endpoint key
```

This is the most liberal baseline. It averages all query records for the same endpoint,
regardless of species or assay context.

### `same_species_same_endpoint`

```text
key = canonical endpoint key + normalized species
```

Rows with unresolved species do not qualify. Physicochemical/acellular endpoints such as
solubility, dissolution, and PAMPA generally do not have a meaningful species and should
not be forced into this universe.

The existing `same_species_v2` normalizer actually produces coarse groups such as
`rodent`, not exact species. We must make one explicit choice before implementation:

- retain v2 grouping and use the technically precise name
  `same_species_group_same_endpoint`; or
- normalize rat, mouse, hamster, etc. separately and use
  `same_species_same_endpoint`.

Until that decision is made, this document uses `same_species_same_endpoint` as the
conceptual name and records the normalization version in every artifact.

### `most_specific_condition_key`

```text
key = canonical endpoint key + normalized endpoint-specific context fields
```

This baseline is not a raw-text exact match. Free-text assay descriptions would create
thousands of singleton keys. Each included field must be normalized into a scientific
equivalence class. Missing optional fields use an explicit `unspecified` value; missing
mandatory fields make the row ineligible.

All raw retrieval metadata is still passed to the model. A field's absence from the
pairing key means query-side variation in that field is marginalized, not that retrieval
metadata is discarded.

## 6. Canonical endpoint key

Every comparable measurement receives an endpoint key containing at least:

```text
source collection
canonical endpoint family
measurement subtype/parameter
canonical unit or normalization basis
value transform
directionality convention, when applicable
```

Examples that must remain distinct:

- permeability versus efflux ratio;
- A-to-B Papp versus B-to-A Papp;
- microsomal intrinsic clearance per mg protein versus hepatocyte clearance per million
  cells versus in-vivo clearance per kg;
- percent remaining at 30 minutes versus metabolic half-life;
- Km versus Vmax;
- transporter substrate status versus a numeric transport rate.

## 7. Threshold policy framework

Thresholds attach to canonical measurement subtypes, not raw extraction categories or
unit strings.

### 7.1 Provisional policies for pipeline development

| Value type | Canonical transform | Provisional transfer | Provisional not-transfer |
|---|---|---:|---:|
| Bounded percentage | identity on 0-100 | difference <= 10 points | difference >= 30 points |
| Bounded fraction | identity on 0-1 | difference <= 0.10 | difference >= 0.30 |
| Positive continuous amount/rate/time | log10 | within 2-fold | at least 10-fold apart |
| Dimensionless kinetic/efflux ratio | log10 | within 1.5- to 2-fold | at least 3- to 5-fold apart |
| Binary categorical | none | same reliable class | opposite reliable classes |
| Ordinal categorical | ordered bins | same bin | at least two bins apart |
| Censored/ranged value | interval on canonical scale | guaranteed close | guaranteed far |

The area between thresholds is a deadband and is dropped. These defaults are for smoke
datasets only, not final scientific thresholds.

A percent sign alone does not select the percentage policy. Percent inhibition, percent
remaining, fraction absorbed, enzyme contribution, and percent change are different
measurement subtypes.

### 7.2 Empirical threshold calibration

For each canonical endpoint subtype:

1. Normalize units and transform values.
2. Deduplicate identical publication/passage/condition measurements.
3. Form cross-record replicate groups with the same molecule and
   `most_specific_condition_key`.
4. Estimate the within-molecule distribution by comparing each concrete record to the
   aggregate of the other records in its group (leave-one-record-out). This mirrors the
   retrieval-record-to-query-aggregate label contract.
5. Estimate the different-molecule distance distribution within the same key.
6. Set the transfer threshold near the 90th-95th percentile of credible replicate noise,
   subject to a scientific minimum tolerance.
7. Set the not-transfer threshold where overlap with replicate noise is low, requiring a
   meaningful gap from the transfer threshold and adequate negative-pair supply.
8. Validate label stability across papers and condition-key ablations.
9. Run sensitivity analyses around both thresholds and select using held-out pairwise
   ranking/transfer metrics, not downstream KNN during training.

Sparse subtypes borrow a prior from the same value type (percentage, log rate, bounded
ratio) through hierarchical shrinkage. They do not receive thresholds derived from an
unrelated endpoint merely to create more pairs.

### 7.3 Preliminary calibration capacity

The following counts are deliberately pre-normalization upper bounds. “Repeated
molecules” means molecules with at least two numeric-like records in the raw category;
counts will fall after measurement subtyping and strict condition keys.

| Dataset | Raw endpoint | Numeric-like records with SMILES | Molecules | Repeated molecules |
|---|---|---:|---:|---:|
| Q2 | fraction absorbed | 7,426 | 1,327 | 780 |
| Q2 | intestinal absorption | 4,551 | 816 | 540 |
| Q2 | human intestinal absorption | 258 | 138 | 49 |
| Q2 | intestinal effective permeability | 5,960 | 751 | 511 |
| Q2 | Caco-2/MDCK/PAMPA permeability | 17,191 | 2,392 | 1,533 |
| Q2 | solubility | 13,883 | 2,010 | 1,101 |
| Q2 | dissolution | 7,230 | 727 | 556 |
| Q2 | GI stability | 2,036 | 633 | 342 |
| Q3 | efflux/secretory transport | 5,203 | 1,133 | 550 |
| Q3 | uptake/absorptive transport | 2,910 | 730 | 396 |
| Q3 | bidirectional permeability | 3,361 | 1,453 | 512 |
| Q3 | intestinal metabolism | 1,348 | 372 | 173 |
| Q3 | gut-wall extraction/first pass | 435 | 125 | 64 |
| Q3 | oral-exposure change due to gut wall | 1,428 | 275 | 136 |
| Q3 | other intestinal transport | 97 | 71 | 13 |
| Q4 | intrinsic clearance | 22,402 | 4,331 | 2,676 |
| Q4 | hepatic clearance | 5,241 | 899 | 546 |
| Q4 | extraction ratio | 1,791 | 507 | 279 |
| Q4 | microsomal stability | 1,116 | 597 | 237 |
| Q4 | hepatocyte stability | 211 | 116 | 34 |
| Q4 | S9 stability | 160 | 82 | 38 |
| Q4 | metabolic half-life | 2,660 | 931 | 503 |
| Q4 | substrate depletion | 740 | 302 | 152 |
| Q4 | CYP metabolism | 6,469 | 1,260 | 770 |
| Q4 | UGT metabolism | 3,418 | 615 | 402 |
| Q4 | hepatic first-pass metabolism | 528 | 205 | 93 |
| Q4 | parent-drug metabolic stability | 130 | 56 | 20 |

This indicates strong calibration potential for permeability, solubility, intrinsic
clearance, and several absorption endpoints. Hepatocyte/S9 stability and parent-drug
stability will likely need value-type priors or broader but still valid normalized keys.

### 7.4 Ranges, inequalities, and qualitative values

Parse every measurement into value, unit, lower/upper bounds, qualifier, and provenance.
For intervals `A` and `B`:

- label transfer only when their maximum plausible distance is within the transfer
  threshold;
- label not-transfer only when their minimum plausible distance exceeds the
  not-transfer threshold;
- otherwise drop the pair.

For qualitative values, first normalize an endpoint-specific vocabulary. `substrate`
versus `not_substrate` is usable; `inconclusive`, vague comparisons, and unsupported
terms are dropped. Ordinal terms such as low/medium/high use an explicit ordered policy.

### 7.5 Measurement types and what distance means

Raw extraction categories are not threshold units. Each record must first become one of
the following canonical measurement types:

| Measurement type | Examples in Q2-Q4 | Canonical representation | Pair distance |
|---|---|---|---|
| Bounded extent | fraction absorbed, Fg, extraction ratio, percent remaining, enzyme contribution | fraction 0-1 or percentage 0-100 with an explicit semantic direction | absolute difference |
| Positive continuous scalar | solubility, Papp/Peff, intrinsic/hepatic clearance, half-life, Km, Vmax | canonical unit followed by log10 | absolute log difference, interpretable as fold difference |
| Dimensionless ratio | efflux ratio, exposure ratio, relative change | positive ratio followed by log10 | absolute log-ratio difference |
| Binary category | transporter/enzyme substrate vs not-substrate, stable vs unstable when genuinely binary | normalized class distribution | class agreement/disagreement |
| Ordinal category | low/medium/high permeability, poor/moderate/good absorption | ordered class distribution | bin separation |
| Timepoint endpoint | percent dissolved or remaining at 30/60/120 minutes | value plus matched timepoint or normalized curve | pointwise difference at matched time or profile distance |
| Kinetic parameter family | Km, Vmax, depletion rate, intrinsic dissolution rate | separate endpoint key per parameter and normalization basis | log-fold distance within that parameter only |
| Censored/ranged measurement | `<10%`, `>90%`, `3-5`, mean plus interval | lower/upper bounds on canonical scale | minimum and maximum possible distance |
| Relative intervention effect | fold/percent AUC change after inhibitor, knockout, formulation, or diet | signed effect on log or percentage scale | difference only under the same effect metric and intervention class |

The semantic direction must be normalized. For example, gut escape `Fg=0.8` and gut
extraction `E=0.2` describe the same direction only after converting one convention.
Likewise, “percent metabolized” and “percent parent remaining” cannot share a key until
one is converted to the other's convention.

### 7.6 Recommended threshold estimator

The recommended final policy is a repeatability-calibrated two-cutoff model with a
deadband, not a fixed threshold chosen only from intuition.

For canonical endpoint `e`, define:

```text
D_repeat(e) = distances from a concrete record to the leave-one-out aggregate
              of other records for the same molecule and strict condition key

D_between(e) = distances from a concrete record to aggregates for different
               molecules under the same strict condition key
```

Estimate these distributions using training molecules only and bootstrap at the molecule
level. Then select:

```text
tau_transfer(e) = a high quantile of D_repeat(e), provisionally q90 or q95

tau_not_transfer(e) = the smallest larger distance where repeat-like pairs are rare,
                      subject to a minimum separation and sufficient negative supply
```

One operational definition for the second cutoff is the first distance where the
estimated posterior probability of repeat-like behavior falls below 5%. A simpler robust
fallback is:

```text
tau_not_transfer >= max(2-3 * tau_transfer, a value-type scientific effect floor)
```

Select between candidate cutoffs with held-out pairwise metrics and threshold-sensitivity
curves. Pair count or class balance may constrain an unusable cutoff, but must not be the
primary scientific objective.

Report for every selected threshold:

- molecule-bootstrap confidence intervals;
- number of repeat groups and independent publications;
- fraction of repeat pairs retained as transfer;
- positive/deadband/negative pair supply;
- label agreement under mean, median, and trimmed-mean query aggregation;
- stability across strict-key and species-key variants.

### 7.7 Soft labels and ranking as threshold complements

Hard labels are robust at the extremes but discard information. The preferred training
strategy is:

1. Use high-confidence binary transfer/not-transfer labels outside the deadband.
2. Weight each label by query-group consistency, parser confidence, and threshold
   uncertainty.
3. Add a ranking objective: for the same retrieval context, a query aggregate with a
   smaller canonical assay distance should score above one with a larger distance.

An advanced alternative is to fit a hierarchical noise model and train on a soft target:

```text
P(behavior transfers | observed distance, endpoint, replicate noise)
```

This avoids a sharp boundary and naturally expresses uncertainty. It is more complex and
should follow a validated hard-label baseline. Raw assay distance regression alone is not
recommended because it gives noisy middle-range differences the same authority as clear
positive and negative examples.

### 7.8 Sparse endpoint fallback

When an endpoint lacks enough strict-key replicate groups:

1. Pool only with subtypes having the same measurement type and transform.
2. Estimate a hierarchical value-type prior, such as a log-rate repeatability prior.
3. Shrink the endpoint estimate toward that prior in proportion to its evidence.
4. Retain wider uncertainty and require a larger deadband.
5. Defer the endpoint if the resulting labels remain unstable.

Do not choose endpoint thresholds solely to force a desired number of pairs.

## 8. What `most_specific_condition_key` includes

The strict key is assembled in layers. It always starts with the canonical endpoint key,
then adds only conditions that materially alter that measurement.

### Common normalized dimensions

| Dimension | Why it matters | Example normalized values |
|---|---|---|
| Species/source organism | In-vivo and biological-system behavior differs by organism | human, rat, mouse, dog, monkey |
| Assay-system family | Different experimental systems are not automatically exchangeable | Caco-2, MDCK, PAMPA, Ussing chamber, liver microsomes, hepatocytes, in vivo |
| Anatomical site/tissue | Permeability and metabolism vary along the intestine and by tissue | duodenum, jejunum, ileum, colon, liver |
| Direction | Transport direction changes the meaning of permeability | A-to-B, B-to-A, mucosal-to-serosal |
| Medium/matrix and pH | Solubility, dissolution, permeability, and stability depend strongly on environment | water, FaSSIF, FeSSIF, SGF, pH bucket |
| Formulation/solid/molecular form | Salt, polymorph, formulation, and parent/metabolite form change measured behavior | free base, hydrochloride salt, tablet, amorphous, parent drug |
| Timepoint/profile definition | Percent dissolved/remaining is meaningless without time | 30 min, 60 min, 120 min, t50, t90 |
| Transporter/enzyme | Substrate status and kinetics are target-specific | P-gp/ABCB1, BCRP/ABCG2, CYP3A4, UGT2B7 |
| Parameter and normalization basis | Numerically similar values may have incompatible denominators | Papp, efflux ratio, Km, Vmax, µL/min/mg, µL/min/10^6 cells |
| Modifier/qualifying-condition class | Inhibitors, knockout models, disease, food, and enhancers intentionally perturb the assay | control, inhibitor-present, knockout, fed, disease model |
| Concentration/dose bucket | Saturable transport/metabolism can depend on concentration | low/sub-Km, near-Km, high/saturating |
| Temperature/cofactor/scaling method | Important for physicochemical and metabolic assays | 25 C, 37 C, NADPH present, PBPK-scaled |

Raw free text is never concatenated directly into the key. Normalizers produce these
equivalence classes, and every key records its normalizer/ontology version.

### Worked strict-key examples

```text
Q2 solubility:
  equilibrium_solubility | mol/L | log10
  + medium=FaSSIF + pH=6.5 + temperature=37C
  + solid_form=free_base + formulation=unformulated

Q2 permeability:
  Papp_A_to_B | cm/s | log10
  + system=Caco-2 + clone=standard + medium=HBSS + pH=7.4
  + modifier=control

Q3 transporter status:
  substrate_status | categorical
  + transporter=P-gp/ABCB1 + system=Caco-2
  + direction=efflux + modifier=control

Q4 intrinsic clearance:
  CLint_microsomal | uL/min/mg_protein | log10
  + species=human + system=liver_microsomes
  + cofactor=NADPH_present + modifier=control
```

The strict baseline is intentionally conservative. Its purpose is to estimate what can
transfer when measurement context is closely aligned. Comparing it with `same_endpoint`
and `same_species_same_endpoint` reveals whether looser marginalization improves useful
pair supply or merely adds contradictory supervision.

## 9. Q2 / Fa endpoint plan

Q2 has 66,767 numeric-like and 18,270 qualitative values across its primary categories.
At least 1,060 raw unit spellings require normalization.

| Raw endpoint | Canonical subtypes | Threshold plan | `most_specific_condition_key` additions |
|---|---|---|---|
| `fraction_absorbed` | fraction/percent absorbed; qualitative extent | 10/30 percentage points; bounded-fraction equivalent; categorical policy for normalized qualitative extent | species, in-vivo/perfusion system family, route or intestinal site, formulation class, exceptional-condition class |
| `intestinal_absorption` | absorption extent %, absorption rate constant, amount transported, qualitative/ordinal absorption | percentage policy for extent; log-fold policy for rates/amounts after unit normalization; ordinal policy for qualitative values | species, assay-system family, intestinal site, rate-vs-extent subtype, medium/pH class, formulation class, exceptional-condition class |
| `human_intestinal_absorption` | human absorption extent %, rate, qualitative extent | same subtype policies as intestinal absorption; keep raw category until merge validation | human species, in-vivo/perfusion system family, intestinal site, formulation class, exceptional-condition class |
| `intestinal_effective_permeability` | Peff by direction; occasionally relative/fold permeability | log10 canonical cm/s with 2-fold/10-fold provisional thresholds; relative results separated | species, perfusion/Ussing/tissue system family, intestinal site, transport direction, medium/pH class, modifier class |
| `caco2_mdck_pampa_permeability` | Papp by direction, efflux ratio, percent transported, qualitative permeability | log-fold Papp; ratio policy for efflux; percentage policy only for percent-transport subtype; ordinal qualitative policy | Caco-2/MDCK/PAMPA system, clone if known, direction, medium/pH class, concentration bucket when necessary, transporter/inhibitor modifier class |
| `solubility` | thermodynamic, kinetic/apparent, intrinsic, qualitative solubility | convert mass concentration to molar where molecular form permits, then log10 mol/L with 2-fold/10-fold provisional thresholds; ordinal qualitative policy | solubility method/type, medium class, pH bucket, temperature bucket, salt/solid-form class, formulation class |
| `dissolution` | percent dissolved/released at time, t50/t90, intrinsic dissolution rate, qualitative dissolution | 10/30 points at matched time; log-fold for t50/t90 and normalized rates; ordinal qualitative policy | dissolution method/apparatus family, medium/pH class, timepoint/profile key, formulation/solid-form class, dose/loading bucket, exceptional-condition class |
| `gi_stability` | percent remaining at time, degradation half-life/rate, qualitative stable/unstable | 10/30 points at matched time; log-fold for half-life/rate; binary/ordinal qualitative policy | gastric/intestinal fluid or matrix, pH class, timepoint, temperature bucket, species/model when biological, formulation class, modifier class |

### Q2 condition-key cautions

- Species resolution is high for fraction absorbed and intestinal permeability but low
  for solubility, dissolution, and many cell/acellular assays.
- Caco-2, MDCK, and PAMPA must not be treated as interchangeable in the strict key.
- Solubility pairing without pH and solid/salt form can create incorrect labels.
- Dissolution and stability percentages are comparable only at matched timepoints.
- Using the current species parser, resolvable-species coverage is approximately 88% for
  fraction absorbed, 80% for intestinal absorption, 78% for effective permeability, 19%
  for cell/membrane permeability, 7% for solubility, 5% for dissolution, and 25% for GI
  stability. Low coverage is often scientifically appropriate because acellular
  endpoints do not have a species.

## 10. Q3 / Fg endpoint plan

Q3 contains 17,152 numeric-like measured values plus structured substrate-status labels.
Its `measured_value` field frequently contains several parameters in one string, so one
raw record may need to become multiple normalized measurements.

| Raw process | Canonical subtypes | Threshold plan | `most_specific_condition_key` additions |
|---|---|---|---|
| `efflux_or_secretory_transport` | transporter substrate status, efflux ratio, directional Papp/Peff, transport amount/rate | exact/opposite substrate classes; ratio policy for efflux; log-fold for Papp/rates | normalized transporter, species/source system, cell/tissue assay family, intestinal site, direction, inhibitor/knockout/modifier class |
| `uptake_or_absorptive_transport` | uptake-transporter substrate status, uptake rate/amount, Peff/Papp | categorical status or log-fold continuous policy | normalized transporter, species, assay-system family, intestinal site/membrane, direction, concentration bucket when saturable, modifier class |
| `bidirectional_permeability` | A-to-B Papp, B-to-A Papp, Peff, efflux ratio | direction-specific log-fold Papp/Peff; ratio policy for efflux | species/source, Caco-2/other system family, cell clone, intestinal site, direction, medium/pH, transporter/modifier when specified |
| `intestinal_metabolism` | enzyme substrate status, fraction metabolized, Km, Vmax, intrinsic metabolic clearance, qualitative metabolism | categorical status; 10/30 points for fractions; separate log-fold policies for each kinetic parameter | normalized enzyme/pathway, species, intestinal microsome/enterocyte/tissue system, intestinal site, kinetic parameter, cofactor/modifier class |
| `gut_wall_extraction_or_first_pass` | Fg/gut escape fraction, extraction fraction, Fa·Fg, qualitative first-pass extent | orient all values to a declared escape or extraction convention, then bounded 0-1/percentage policy | species, estimation/experimental method, intestinal site, endpoint convention, transporter/enzyme, modifier class |
| `oral_exposure_change_due_to_gut_wall` | fold/percent change in AUC, Cmax, bioavailability, or concentration under an intervention | log-fold effect policy or percentage-change subtype; do not compare different exposure measures | species, exposure metric, transporter/enzyme, intervention/modifier identity, study design/dose-regimen class |
| `other_intestinal_transport` | unknown until remapped | no threshold; quarantine or map to a supported subtype | none until manual/automatic remapping is validated |

### Q3 condition-key cautions

- P-gp, BCRP, MRP, PEPT1, CYP, and UGT names need an alias/ontology layer.
- Two molecules sharing `substrate` labels are comparable only for the same normalized
  transporter or enzyme.
- `inconclusive`, literal `null`, and out-of-schema `inhibited` substrate statuses are not
  ordinary binary labels.
- Intervention-driven exposure changes require matching the modifier and exposure metric;
  they should be a later-phase endpoint.
- Current heuristic species resolution ranges from roughly 15% for bidirectional
  permeability and 25% for efflux to 61% for gut-wall extraction and 74% for
  intervention-driven oral exposure. Assay-system normalization must distinguish a
  genuinely missing species from an acellular/cell-line system.

## 11. Q4 / Fh endpoint plan

Q4 contains 49,481 numeric-like measurements and at least 1,769 unit spellings. The
largest repeat pools are intrinsic clearance, hepatic clearance, half-life, and CYP/UGT
measurements, but raw categories still mix multiple parameter types.

| Raw metric | Canonical subtypes | Threshold plan | `most_specific_condition_key` additions |
|---|---|---|---|
| `intrinsic_clearance` | microsomal CLint per protein, hepatocyte CLint per cell count, in-vivo/scaled CLint per kg, enzyme-normalized CLint | separate log10 subtypes with 2-fold/10-fold provisional thresholds | species, microsome/hepatocyte/in-vivo system, normalization basis, enzyme/pathway where specific, cofactor/scaling method, molecular form, modifier class |
| `hepatic_clearance` | in-vivo hepatic clearance, perfused-liver clearance, scaled/model clearance | log10 within a canonical normalization basis | species, experimental/scaled system, blood/plasma convention, normalization basis, route/model, molecular form, modifier class |
| `extraction_ratio` | hepatic extraction fraction or percent | normalize to 0-1 with declared direction; 0.10/0.30 provisional differences | species, in-vivo/perfused/scaled system, blood/plasma convention, molecular form, modifier class |
| `microsomal_stability` | percent remaining at time, half-life, turnover/clearance, qualitative stability | percentage-at-matched-time, log half-life/rate, or categorical policy | species, microsomal system, timepoint, cofactor presence, concentration bucket if needed, molecular form, modifier class |
| `hepatocyte_stability` | percent remaining, half-life, clearance, qualitative stability | same subtype-specific stability policies | species, hepatocyte system, timepoint, cell normalization, concentration bucket, molecular form, modifier class |
| `s9_stability` | percent remaining, half-life/rate, qualitative stability | same subtype-specific stability policies | species, S9 system, timepoint, cofactor condition, concentration bucket, molecular form, modifier class |
| `metabolic_half_life` | time in min/h, endpoint-specific qualitative claims | convert time to one unit and use log-fold thresholds | species, assay-system family, matrix, concentration bucket when saturable, molecular form, modifier class |
| `substrate_depletion` | percent remaining at time, depletion rate constant, derived clearance, qualitative turnover | percentage-at-matched-time or parameter-specific log-fold policy | species, assay-system family, timepoint, parameter subtype, enzyme/cofactor, concentration bucket, molecular form |
| `cyp_metabolism` | CYP substrate/pathway status, isoform contribution %, Km, Vmax, clearance/rate | categorical status; 10/30 points for contribution fraction; parameter-specific log-fold policy | exact normalized CYP isoform, species, microsome/hepatocyte/recombinant system, parameter subtype, cofactor, molecular form, modifier class |
| `ugt_metabolism` | UGT substrate/pathway status, isoform contribution %, Km, Vmax, clearance/rate | same typed policy as CYP; never compare different kinetic parameters | exact normalized UGT isoform, species, assay system, parameter subtype, cofactor, molecular form, modifier class |
| `hepatic_first_pass_metabolism` | hepatic escape/extraction fraction, percent first-pass loss, fold change, qualitative extent | orient to escape or extraction; bounded policy for fractions; log-fold effect policy | species, in-vivo/scaled method, endpoint convention, route, enzyme/pathway, molecular form, modifier class |
| `parent_drug_metabolic_stability` | percent parent remaining, half-life/rate, qualitative stability | stability subtype policies | species, assay-system family, timepoint, cofactor, molecular form, modifier class |

### Q4 condition-key cautions

- `µL/min/mg protein`, `µL/min/10^6 cells`, and `mL/min/kg` are different endpoint
  normalizations, not unit aliases to merge blindly.
- Rat and mouse are currently collapsed to `rodent` by v2 normalization; whether that is
  acceptable must be tested as a pairing-policy ablation.
- A CYP/UGT category is not itself a numerical endpoint. Parameter and isoform extraction
  are mandatory before pairing.
- Percent remaining values need matched incubation time; percentage contribution to an
  enzyme uses a different endpoint key.
- Q4 has a structured species field and current resolution is much higher: approximately
  75-95% across primary endpoints. This makes it the strongest dataset for comparing
  exact-species and coarse-species-group pairing policies.

## 12. Initial endpoint rollout

### Tier 1: build and calibrate first

- Q2 fraction absorbed percentage.
- Q2 Papp/Peff after direction and unit normalization.
- Q2 solubility after molar conversion and pH/solid-form keys.
- Q3 transporter substrate status for normalized transporter identities.
- Q3 efflux ratio and directional Papp.
- Q3 gut-wall escape/extraction fraction.
- Q4 intrinsic clearance split by normalization basis.
- Q4 metabolic half-life.
- Q4 extraction ratio.

### Tier 2: after structured parsing

- Q2 dissolution and GI stability timepoint endpoints.
- Q3 intestinal metabolism kinetic parameters.
- Q4 microsomal/hepatocyte/S9 stability.
- Q4 CYP/UGT kinetic and contribution endpoints.

### Deferred until stronger normalization

- Intervention-specific oral-exposure changes.
- Free-text `other_*` categories.
- Multi-point profiles that cannot be reduced to a matched parameter/timepoint.
- Relative/qualitative comparisons without a stable normalized vocabulary.

## 13. Model and artifact implications

### Pair artifact

Store:

- retrieval record index and full retrieval metadata lookup;
- query canonical molecule identifier only;
- canonical endpoint key and condition-key mode/version;
- optional transformed retrieval value;
- query aggregate target, count, dispersion, and contributing-record digest as
  label-only provenance;
- canonical distance, binary label/deadband decision, thresholds, and threshold version;
- parser, unit-normalization, and endpoint-ontology versions.

### Model

- Use a retrieval-record branch with molecule embedding plus all retrieval metadata.
- Use a query branch with molecule embedding only.
- Never gather metadata using an arbitrary query record index.
- Keep source-value and no-source-value variants, but both follow the same directional
  retrieval-to-query contract.
- Condition the model on endpoint identity/description so one model can learn different
  assay similarities.

### Training

- Balance sampling by endpoint, condition key, molecule, and label rather than raw pair
  count.
- Use binary transfer classification as the primary robust objective.
- Consider ordinal-distance or ranking loss as auxiliary supervision.
- Track pairwise validation metrics and per-endpoint macro averages.
- Do not register KNN callbacks or use KNN metrics for checkpoint selection.

## 14. Implementation plan

1. Create audited raw-category alias maps for Q2-Q4.
2. Implement a canonical measurement parser that preserves values, bounds, qualifiers,
   units, and source text.
3. Implement endpoint subtyping and unit/normalization-basis registries.
4. Implement normalized species, system, site, direction, medium/pH, formulation,
   transporter/enzyme, modifier, timepoint, and molecular-form fields.
5. Produce a normalization audit with retained/quarantined counts and representative
   examples for every endpoint subtype.
6. Decide exact-species versus v2 species-group semantics and freeze the universe name.
7. Build molecule-disjoint record splits before query aggregation.
8. Estimate replicate-noise distributions and publish proposed thresholds with
   confidence intervals and pair-supply tables.
9. Generate the three pair universes using the asymmetric retrieval-to-query contract.
10. Refactor the model so query metadata cannot enter the query branch.
11. Remove training-time KNN callbacks, overrides, checkpoint selection, and reporting.
12. Train Tier 1 endpoints with endpoint-balanced sampling and pairwise validation.
13. Run Morgan versus learned-score retrieval as a standalone post-training benchmark.

## 15. Required analysis artifacts

Every dataset build should publish:

- category alias audit;
- endpoint-subtype and value-kind counts;
- parser success/failure and censoring counts;
- raw-to-canonical unit conversion audit;
- species/system/context normalization coverage;
- molecules and records per endpoint/key;
- repeat-group counts and within-group distance distributions;
- selected thresholds and sensitivity alternatives;
- positive/deadband/negative pair supply by endpoint and universe;
- molecule-overlap and aggregate-leakage checks.

## 16. Open decisions

- Exact species versus coarse v2 species groups.
- Mean versus a robust mean for noisy query aggregates; arithmetic mean remains the
  current contract until compared against median/trimmed mean on replicate groups.
- Minimum query-group size and whether singleton targets receive lower loss weight.
- Whether thresholds should be global per canonical subtype or partially conditioned on
  assay-system family when replicate noise differs materially.
- Which assay-description representation best supports generalization across endpoint
  families.
