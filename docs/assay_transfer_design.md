# Assay Transfer Dataset and Model Architecture

Status: architecture draft and canonical design reference
Last updated: 2026-07-15

This document defines the dataset, labels, model inputs, threshold policy, and
evaluation protocol for learning an assay-transfer similarity function. It supersedes
the design portions of `ASSAY_TRANSFER_GENERALIZATION_PLAN.md`; that file remains a
historical exploration and dataset-analysis record.

## 1. Objective

The goal is to learn a better molecular similarity function than a fixed structural
similarity such as Morgan/Tanimoto. The learned score answers a directional question:

> Given a measured retrieval molecule in a specified assay setting, how safely should
> that measurement transfer to a query molecule whose structure is known?

The primary output is a transfer score or probability, not a prediction of the query
molecule's assay value. Assay-value differences are supervision for molecular
similarity. A continuous distance head may be trained as an auxiliary or alternative
objective, but direct property prediction is not the project objective.

### Non-goals

- Predicting an assay value from the query structure alone.
- Treating records from different biological endpoints as interchangeable.
- Using query-side assay metadata at inference time.
- Calling repeated same-molecule measurements technical replicates when their assay
  contexts differ.
- Tracking KNN retrieval metrics inside the training loop. Retrieval evaluation is a
  separate, post-training benchmark.

## 2. Inference contract

At inference time, the system receives:

- query molecule structure `x_B`;
- retrieval molecule structure `x_A`;
- the retrieval record's full normalized assay metadata `Z_A`;
- the retrieval value `y_A`; and
- an explicit shared transfer setting `K` derived from the retrieval assay context.

It never receives query molecule assay metadata or a query assay value.

```text
Query branch                         Retrieval branch
-------------                        ----------------
query structure x_B                  retrieval structure x_A
        |                            retrieval metadata Z_A
        v                            retrieval value y_A
  query encoder                              |
        |                                    v
        +---------- pair scorer <---- retrieval encoder
                         ^
                         |
              shared setting encoder K
                         |
                         v
             transfer score and/or distance
```

`K` is not molecule-specific query metadata. It is the setting of the transfer
question. For example, the question can be "transfer within the same species and
endpoint" even though no measurement record is supplied for the query molecule.
During training, query records are used only to construct supervision under that
setting; they are never query-branch features.

The score is directional:

```text
score(retrieval record A -> query molecule B | shared setting K)
```

Swapping `A` and `B` changes the available metadata and can change the score.

The asymmetry between the branches is deliberate and mirrors deployment. The retrieval
record is a fully specified anchor: its complete context `Z_A` and value `y_A` are
known at inference and are always provided. The query side is marginalized only because
the query's nuisance context is genuinely unknown at inference; only the setting `K` is
given. The learned target is therefore a point-to-distribution transfer quantity — a
concrete retrieval measurement against the compatible-query distribution under `K`, as
formalized by `E_{C_B}[·]` in Sections 7 and 8. The marginalization lives entirely in
label construction; the model inputs are identical in training and inference, so
exposing full `Z_A` to the retrieval branch introduces no train/inference skew.
Marginalizing or stripping the retrieval branch to only `K` is explicitly rejected: it
would discard information available at inference and force distinct retrieval records to
collapse into a single aggregate, replacing this point-to-distribution question with a
weaker distribution-to-distribution one.

## 3. Formal data model

A normalized assay record is represented as:

```text
r = (molecule_id, structure, endpoint_key, value, unit, K, Z, provenance)
```

where:

- `endpoint_key` identifies the biological measurement and its canonical scale;
- `value` is the parsed numeric, categorical, interval, or censored result;
- `K` is the shared condition key used to define the transfer universe;
- `Z` contains all remaining normalized record metadata; and
- `provenance` preserves the source collection, raw fields, parser version, and row ID.

For retrieval record `r_A` and query molecule `B`, define the compatible query-record
set:

```text
R_B(K_A) = {r_B : molecule(r_B) = B and K_B = K_A}
```

Records in `R_B(K_A)` are label-construction evidence. The model never receives any
individual `r_B`, `Z_B`, or `y_B` as a query-side input.

## 4. Dataset scope

Canonical source files live under the stage-first source layout (one folder per source,
named for its dominant domain; the `q1`-`q4` tag is retained as each record's `source_id`):

```text
datasets/sources/{oral_bioavailability,intestinal_absorption,gut_wall,hepatic}/extractions.parquet
```

The current mapping and snapshot are:

| Collection | Domain | Rows | Unique SMILES | Missing values | Phase |
|---|---|---:|---:|---:|---|
| Q1 | Oral bioavailability/exposure | 119,192 | 3,442 | 1,120 | Species normalization included; endpoint pairing deferred |
| Q2 | `Fa`, intestinal absorption | 85,061 | 5,520 | 7,769 | Included |
| Q3 | `Fg`, gut-wall escape | 27,713 | 3,182 | 3,095 | Included |
| Q4 | `Fh`, hepatic escape | 67,943 | 6,984 | 4,862 | Included |
| Starling OBA | In-house oral bioavailability | — | — | — | Species normalization included; endpoint pairing deferred; primary deployment set |

These counts describe the inspected snapshot and should be regenerated in every
dataset manifest. Raw categories outside the current schema were observed in Q2, Q3,
and Q4, so normalization must retain unknown categories rather than silently map or
discard them.

Q1 remains deferred from the first generalized endpoint-pairing release because oral
bioavailability was the initial special case. It is not excluded from canonical
normalization: Q1 and the separate oral-bioavailability base both use the same
explicit-only `species_v1` contract as Q2-Q4. Existing Q1 labels remain useful as a
backward-compatibility benchmark, not as the schema for all endpoint families.

### 4.1 Oral bioavailability as an endpoint family, not a special case

Q1 and the in-house Starling oral-bioavailability dataset are not architecturally
special. Oral bioavailability `F` is a bounded fraction `[0,1]` (or percent `[0,100]`),
so it flows through the same normalization, split, nuisance-cell, pair, and artifact
pipeline as every other endpoint and uses the **same** bounded-fraction metric policy
(Section 9). It is not given a bespoke tolerance: a fixed value-distance standard is the
entire product promise — "the query molecule reads out a value close to this measured
record" — and per-endpoint thresholds would make "transfer" mean different things in
different endpoints. The cross-endpoint firewall in Section 5.1 keeps `F`, `Fa`, `Fg`,
and `Fh` in separate transfer universes; unifying them means one pipeline and one shared
encoder, not pairs across endpoints.

The composite nature of `F` (approximately `Fa * Fg * Fh`, plus dissolution, dose,
formulation, and first-pass effects) does not change the tolerance. It surfaces instead
as noisier data — more records in the deadband and lower decisive coverage (Sections
7-8) — and, because `F` and its mechanistic components now live in one framework, as a
composability check: pairs that transfer on all of `Fa`, `Fg`, `Fh` should tend to
transfer on `F`.

Before `F` records can pair, "oral bioavailability" must be split into its distinct
canonical endpoints under Section 5.1. It commonly lumps absolute `F`, relative or
comparative `F`, and `%F` derived from an AUC ratio, which are different quantities and
must not share a transfer universe merely because all are percentages. The Starling set
and Q1 also only share a universe once their `F` definition and species are canonicalized
onto the same endpoint key. Rollout is deferred to a later tier (Section 15): the
machinery is validated on the cleaner mechanistic endpoints Q2-Q4 first, then the
oral-bioavailability family is folded in.

## 5. Endpoint and value normalization

A percent sign, unit, or raw category name does not uniquely identify an endpoint.
For example, percent absorbed, percent remaining, percent inhibition, and percent
transport are biologically different quantities and must not be paired merely because
all are percentages.

### 5.1 Canonical endpoint key

Every row is mapped to a versioned canonical endpoint key containing at least:

```text
source collection
biological endpoint family
measurement subtype
canonical unit or normalization basis
value transform
direction, when applicable
```

Examples:

```text
Q2 | fraction_absorbed | percent | percentage_points | identity
Q2 | apparent_permeability | cm/s | log10 | A_to_B
Q2 | equilibrium_solubility | mol/L | log10 | none
Q3 | efflux_ratio | dimensionless | log10 | B_to_A_over_A_to_B
Q4 | intrinsic_clearance | uL/min/mg_protein | log10 | none
Q4 | metabolic_half_life | min | log10 | none
```

Cross-endpoint pairs are prohibited in every transfer universe.

### 5.2 Measurement types

| Measurement type | Examples | Canonical representation | Pairwise distance |
|---|---|---|---|
| Bounded percent | fraction absorbed, percent remaining | `[0, 100]`, percentage points | absolute percentage-point difference |
| Bounded fraction | `Fa`, `Fg`, `Fh` on `[0,1]` | `[0, 1]` | absolute fraction difference |
| Positive continuous | Papp, Peff, solubility, clearance, half-life | canonical unit, then `log10` | absolute log difference / fold difference |
| Dimensionless ratio | efflux ratio, exposure ratio | positive ratio, usually `log10` | absolute log-ratio difference |
| Binary categorical | substrate/non-substrate, stable/unstable | normalized class plus qualifier | equal/not equal when classes are reliable |
| Ordinal categorical | low/medium/high | ordered bins | bin separation |
| Timepoint/profile | percent remaining at time | value plus explicit time definition | only within compatible timepoint/profile keys |
| Kinetic parameter | `Km`, `Vmax`, typed CYP/UGT values | parameter-specific unit/basis | log distance within the same parameter type |
| Censored/range | `<x`, `>x`, `[a,b]` | bound or interval object | interval-aware compatible/incompatible/unknown |
| Intervention effect | fold change after inhibitor/inducer | effect direction and ratio | endpoint-specific; initially deferred |

Values are never compared before unit conversion and endpoint-specific transformation.
Rows that cannot be parsed without guessing remain quarantined with their raw values.

### 5.3 Endpoint families by collection

Q2 / `Fa` includes:

- fraction absorbed, intestinal absorption, and HIA;
- effective permeability (`Peff`);
- Caco-2, MDCK, and PAMPA apparent permeability (`Papp`), efflux ratio,
  and percent transport;
- equilibrium or kinetic solubility;
- dissolution; and
- gastrointestinal stability.

Q3 / `Fg` includes:

- efflux/secretory and uptake transport;
- bidirectional permeability;
- intestinal metabolism;
- gut-wall extraction or escape;
- intervention-specific exposure changes; and
- ambiguous "other" records that require quarantine or manual mapping.

Q4 / `Fh` includes:

- intrinsic and hepatic clearance;
- hepatic extraction ratio;
- microsomal, hepatocyte, and S9 stability;
- metabolic half-life and substrate depletion;
- typed CYP/UGT kinetic parameters;
- hepatic first-pass measurements; and
- parent-compound stability.

Raw categories within these lists can still contain multiple endpoint subtypes. The
normalizer must split them before pair construction.

## 6. Transfer universes and shared condition keys

We compare three nested definitions of the transfer setting. Each definition is
versioned and encoded explicitly as `K`.

A condition key answers whether two records are scientifically eligible to compare.
It does not assert that their molecular values transfer, and it does not define the
numeric tolerance. Once records are eligible, the metric-level policy in Section 9
determines transfer, non-transfer, or the deadband. The same metric policy is used in
every condition-key profile and every realized condition bucket.

### 6.1 Same endpoint

```text
K = canonical_endpoint_key
```

This is the broadest universe. It asks whether transfer is safe across all biological
and protocol contexts represented for that endpoint.

### 6.2 Same species and same endpoint

```text
K = (canonical_endpoint_key, species_exact)
```

`species_exact` must be non-null on both records and equal. Generic groups such as
`monkey`, `rodent`, or `fish` do not qualify, and two null values never match. This
design does not create a coarse species-group transfer universe.

All query records used for the target match the endpoint and species setting. Other
query-record conditions are marginalized by the context-balancing procedure in
Section 7.

### 6.3 Most specific condition key

`most_specific_condition_key` adds normalized fields that are scientifically required
for equivalence within an endpoint subtype. It is an endpoint-specific schema, not one
global concatenation of every raw column.

Candidate field categories are:

- explicit exact species, when the endpoint requires it;
- assay-system family: in vivo, microsome, hepatocyte, cell line, PAMPA, S9, and so on;
- anatomical site or tissue;
- transport direction;
- medium, matrix, and pH;
- formulation, solid state, salt, or molecular form;
- timepoint or profile definition;
- transporter or enzyme identity;
- parameter type and normalization basis;
- modifier class: inhibitor, inducer, control, cofactor;
- concentration or dose bucket;
- temperature, cofactor, and scaling method where material.

Example versioned keys:

```text
solubility | mol/L | log10 | FaSSIF | pH_6.5 | 37C | free_base
Papp | cm/s | log10 | Caco-2 | HBSS | pH_7.4 | A_to_B | control
transporter_status | categorical | P-gp | Caco-2 | efflux | control
CLint | uL/min/mg | log10 | human | liver_microsome | NADPH
```

Only normalized equivalence classes enter a key. Free text, unbounded continuous
values, and source-specific spellings do not. Continuous conditions use documented,
scientifically meaningful bins when exact equality would be too sparse.

Missing values are not evidence of equality. A record missing a required field is
excluded from that most-specific universe or evaluated under an explicitly coarser
key; two unknown values are not automatically considered a match.

The normalized artifact contains one nullable column: `species_exact`. Oral
bioavailability and Q4 first use their dedicated raw species fields. When no dedicated
field exists, Q1 may recover a literal species designation from `study_context`, Q2
from `biological_context` or `assay_system`, and Q3 from `intestinal_site` or
`assay_system`. Q4 may use `assay_system` only when its dedicated `species` field is
missing. These allowlisted fields are versioned in `species.yaml`; broad evidence text
such as `support_text` and `extra_details` is not parsed.

"Explicitly derived" means lexical normalization of a species designation actually
present in an allowlisted field. It does not mean biological inference. For example,
`rat jejunum` may produce `rat`, but Caco-2 does not produce `human`, MDCK does not
produce `dog`, and `healthy volunteers` does not produce `human`. Generic taxonomic
groups, missing values, and text naming multiple exact species all produce null. If a
dedicated species field is populated but generic or ambiguous, it remains null rather
than being overridden by an assay-system guess.

There is no `species_basis`, `species_status`, or `species_group` column in the
normalized contract. Species-independent endpoints such as PAMPA therefore have null
`species_exact`: they remain eligible for `same_endpoint` and endpoint-specific
`most_specific` schemas that do not require species, but are excluded from
`same_species_same_endpoint`. Raw source fields are retained for auditing. Coverage
must be regenerated under this explicit-only resolver before enabling the
same-species universe; coverage numbers from the earlier inference-enabled resolver
are not comparable.

`species_exact` is a join and label-construction field, not a replacement for the
retrieval assay metadata. Every normalized base record retains the original species
or context value unchanged. During training and inference, the model receives that
original retrieval-side value as part of the full retrieval metadata; it does not
receive a nulled or rewritten substitute. The derived `species_exact` value is used
offline to select the compatible query records whose measurements are marginalized.
The query still contributes structure only.

For example, a retrieval record containing raw `human and rat` keeps exactly that text
for model input while its `species_exact` is null, so it is excluded only from the
`same_species_same_endpoint` universe. It can still participate in `same_endpoint` or
another profile that does not require species. Likewise, raw `healthy volunteers`
remains visible to the model even though the explicit-only species mapping is null.

On the inspected snapshot, explicit exact-species coverage is 52.68% for the separate
oral base, 42.46% for Q1, 35.19% for Q2, 33.77% for Q3, and 87.83% for Q4. The lower
Q1-Q3 rates are deliberate: volunteer-only descriptions and cell-line-only systems
are no longer assigned an inferred organism. The audit is stored with the task
progress and must be regenerated in each released dataset manifest.

## 7. Query marginalization and context balancing

The central training problem is that query-side measurements exist in the dataset but
will not exist at inference. The target must therefore summarize the empirical
distribution of compatible records without leaking a particular query record's
metadata into the model.

### 7.1 Flat nuisance-context cells

For a chosen `K`, collect all remaining normalized query conditions into a nuisance
tuple:

```text
C_B = tuple(normalized query conditions not included in K)
```

Group compatible query records by the complete tuple. These are flat cells, not an
arbitrary hierarchy. If the observed cells are `ABB`, `ABC`, and `ACB`, each cell gets
one macro weight:

```text
aggregate = (cell_ABB + cell_ABC + cell_ACB) / 3
```

`ABB` and `ABC` are not first averaged into an `AB` branch unless a scientifically
declared hierarchy intentionally defines that weighting. Flat cells prevent common
experimental contexts from dominating merely because they have more records.

Within a cell, records summarize repeated observations in that context. Across cells,
the default is a uniform macro-average. An observation-frequency-weighted average may
be retained as an ablation, but it estimates the source dataset's sampling process and
not a context-balanced transfer question.

### 7.2 Why same-molecule records are not replicates

The datasets do not appear to contain enough exact technical duplicates to estimate a
replicate error distribution. Multiple records for one molecule under a broad `K`
usually vary in nuisance context. Their spread measures a combination of protocol,
biology, curation, and measurement effects; it must not be labeled "replicate noise."

These records are still useful for estimating transfer across the contexts allowed by
`K`, which is why context-balanced aggregation is part of the target definition.

## 8. Pairwise target construction

Let `m` be the canonical metric type and `d_m(y_A, y_B)` its distance after unit and
value canonicalization. Labels are constructed only after molecule splitting and only
from query records in the same split. `T_transfer(m)` and `T_not_transfer(m)` are read
from the frozen metric policy; neither depends on the condition-key profile or the
realized bucket.

### 8.1 Preferred binary target: compare first, aggregate second

For every compatible query record `j`, apply a two-threshold deadband:

```text
d_j <= T_transfer(m)       -> decisive transfer outcome 1
d_j >= T_not_transfer(m)   -> decisive non-transfer outcome 0
otherwise               -> ambiguous; omit from the binary target
```

where `T_transfer(m) < T_not_transfer(m)`.

For each nuisance-context cell `c`, compute:

```text
p_c = transfer outcomes in c / decisive outcomes in c
```

Then macro-average cells that contain decisive evidence:

```text
P_transfer(A -> B | K) = mean_c(p_c)
```

The pair artifact also stores:

- total and decisive record counts;
- total and decisive context-cell counts;
- decisive record coverage;
- decisive context coverage; and
- the full per-cell summary needed to audit the aggregate.

This record-first construction is preferred to comparing `y_A` with one average query
value. It preserves whether transfer succeeds in most eligible contexts and does not
allow incompatible high and low values to cancel before classification.

Two binary label variants are supported:

1. **Soft binary:** train directly against `P_transfer`.
2. **Hard consensus:** label transfer when `P_transfer >= P_high`, non-transfer when
   `P_transfer <= P_low`, and drop the middle.

A starting hard-consensus policy can use `P_high = 0.8` and `P_low = 0.2`, but these
are versioned configuration values, not permanent constants. Pairs must also satisfy
minimum decisive-record, decisive-cell, and coverage requirements. One extreme record
surviving a large deadband must not create a high-confidence label.

### 8.2 Continuous target: expected distance

For continuous supervision, compute a distance for every compatible record and
aggregate within and across nuisance-context cells:

```text
D_c = mean_j(d_j) within cell c
D_expected(A -> B | K) = mean_c(D_c)
```

Robust within-cell alternatives such as the median can be tested for outlier-heavy
endpoints. A high quantile such as `q90` can represent conservative or worst-case
transfer.

The expected distance is not the same as distance to an average value:

```text
distance to mean = |y_A - E[Y_B]|
expected distance = E[|y_A - Y_B|]
```

If query values are 10 and 90 while the retrieval value is 50, distance to the mean is
zero but expected distance is 40. For a similarity function under unknown query
context, expected distance is the primary continuous target. Distance to the
context-balanced query mean remains a documented baseline for comparison.

### 8.3 Categorical and censored targets

- Binary categorical records transfer only when both labels are reliable and equal;
  opposite reliable classes are non-transfer. Unknown or weakly qualified classes are
  omitted.
- Ordinal records use endpoint-declared bin separation; arbitrary string ordering is
  forbidden.
- Censored values and ranges are compared as intervals. If every possible pairwise
  distance lies below `T_transfer`, the result is transfer; if every possible distance
  lies above `T_not_transfer`, it is non-transfer; overlapping ambiguous cases are
  omitted from the binary label.

## 9. Metric-level a priori threshold policy

Thresholds answer a scientific equivalence question: how different can two values of
a metric be before transfer is no longer useful? In v1, this standard is attached to a
canonical metric type, not to an endpoint's condition-key profile or realized bucket:

```text
condition key -> whether the records may be compared
metric type   -> how their values are compared and thresholded
label policy  -> how record outcomes become a pair target
```

For example, a CLint comparison uses the same CLint threshold in `same_endpoint`,
`same_species_same_endpoint`, and a human-liver-microsome most-specific bucket. Making
the tolerance depend on each bucket would change the meaning of transfer across
training settings and ask the model to learn context-specific standards rather than a
consistent molecular similarity function.

The dataset does not contain sufficient exact technical replicates to estimate assay
noise. The primary thresholds are therefore transparent, scientifically interpretable
a priori policies. They are frozen before model training and are not selected to
maximize model performance.

### 9.1 Provisional v1 policies

| Canonical metric type | Transfer | Non-transfer | Middle |
|---|---:|---:|---|
| Bounded percentage `[0,100]` | at most 10 percentage points | at least 30 points | drop |
| Bounded fraction `[0,1]` | at most 0.10 | at least 0.30 | drop |
| Papp, Peff, solubility, CLint, clearance, half-life | within 2-fold | at least 5-fold | drop |
| `Vmax` | within 2-fold | at least 5-fold | drop |
| `Km` | within 2-fold | at least 5-fold | drop |
| Dimensionless ratio | within 1.5-fold | at least 3-fold | drop |
| Binary categorical | same reliable class | opposite reliable class | unknown/inconclusive |
| Ordinal categorical | same bin | at least two bins apart | adjacent bin |

Positive scalar and ratio metrics are compared by absolute distance after `log10`
transformation. A 2-fold threshold is `log10(2) ~= 0.301`; a 5-fold threshold is
`log10(5) ~= 0.699`. Values must still have compatible canonical units, bases,
directions, and endpoint identity before the metric policy is applied.

A common percentage policy does not permit cross-endpoint pairs. Percent absorbed,
percent parent remaining, and percent dissolved use the same numeric tolerance only
after each has been assigned a distinct canonical endpoint and normalized direction.

### 9.2 Sensitivity analysis and future revisions

The primary policy is preregistered in `metric_thresholds.yaml`. Sensitivity dataset
versions may use plausible alternatives such as `5/20`, `10/30`, and `15/40`
percentage points, or `1.5x/3x`, `2x/5x`, and `3x/10x` for positive log metrics. These
variants test whether conclusions are brittle; they do not tune the primary labels to
the winning model.

Coverage, deadband size, class balance, and bootstrap label stability are reported for
every policy. They are diagnostics rather than threshold-selection objectives. A
future scientific exception must create a new canonical metric subtype and a new
policy version. Ad hoc threshold overrides for a condition-key bucket are prohibited.

## 10. Model interfaces

The model has three conceptual inputs:

1. **Query branch:** molecular structure only.
2. **Retrieval branch:** molecular structure, full normalized retrieval metadata, and
   the measured retrieval value `y_A`.
3. **Shared-setting branch:** the versioned condition key `K` that defines the transfer
   question.

The retrieval value `y_A` is always an input. It is known for every reference record at
inference, so withholding it only handicaps the model. There is no no-source-value
variant: the labels are anchored on a specific `y_A`, so hiding it would force the model
to additionally average over `y_A | x_A, Z_A, K` — a different and weaker estimand that
collapses distinct retrieval records — for no deployment benefit.

The scorer can emit:

- a scalar transfer logit/probability;
- a non-negative expected-distance estimate; and/or
- an embedding used for a conditioned pairwise similarity score.

Query-record aggregates are labels and diagnostics only. They must not be serialized
as inference features. Missing retrieval metadata uses explicit missingness indicators,
not query-side imputation.

Morgan/Tanimoto is a baseline under the same candidate pairs, splits, and evaluation
labels. It is not used as ground truth.

## 11. Modular training objectives

Binary deadband and continuous granularity are independent components. The initial
ablation matrix is:

| Variant | Binary loss | Continuous loss | Treatment of distance middle band |
|---|---|---|---|
| A | Yes | No | Dropped |
| B | No | Yes | Retained |
| C | No | Yes | Dropped from continuous loss |
| D | No | Yes | Downweighted in continuous loss |
| E | Yes | Yes | Dropped for binary; retained for continuous |
| F | Yes | Yes | Dropped for both |
| G | Yes | Yes | Dropped for binary; downweighted for continuous |

The total objective is configuration-driven:

```text
L = lambda_binary * L_binary
  + lambda_continuous * L_continuous
  + lambda_ranking * L_ranking
```

- `L_binary` uses hard labels or soft `P_transfer` targets.
- `L_continuous` predicts context-balanced expected distance, preferably with a robust
  regression loss.
- `L_ranking` is optional and requires the model to rank a closer query molecule above
  a farther one for the same retrieval context.

All loss variants use the same split and frozen target policy. KNN evaluation is not a
training callback, checkpoint criterion, or early-stopping signal.

## 12. Split, leakage, and sampling rules

### 12.1 Molecule-first splitting

- Assign molecules to splits before constructing record sets, aggregates, or pairs.
- Every record for a query molecule belongs to one split.
- Canonicalize structures before molecule identity and split assignment.
- Fit value transforms, buckets, vocabularies, and candidate-policy choices on the
  training split only.
- Construct validation and test labels using frozen training decisions.

The default benchmark must report exactly which side is disjoint:

- **Query-disjoint deployment split:** query molecules are unseen; retrieval records
  may come from a known reference library.
- **Strict molecule-disjoint split:** neither query nor retrieval molecule appears in
  training.

The strict split is the stronger generalization test; the query-disjoint split may
better match deployment. Results from the two must not be merged without a label.

The in-house Starling oral-bioavailability set is the concrete instance of the
query-disjoint deployment scenario: the reference library is the known public and
historical records, and the queries are new Starling structures to be scored. It is
therefore held out of training pairs and used as the primary deployment benchmark by
default. Pooling Starling molecules into training is reserved for the case where data is
too scarce to hold all of them out; even then a clean held-out Starling slice remains
the deployment test. Because oral-bioavailability molecules recur across Q1, the public
sources, and Starling, split assignment must be **joint across all sources** — a molecule
lands in one split everywhere, or retrieval-side records leak the held-out queries.

### 12.2 Sampling

The number of possible pairs grows quadratically and is dominated by frequent
molecules, endpoints, and contexts. Training batches should be balanced over:

- endpoint family and canonical endpoint;
- transfer-universe/key version;
- query and retrieval molecules;
- binary class or soft-target region; and
- evidence/coverage strata.

Sampling weights and random seeds are stored in the manifest. Pair count alone must
not determine an endpoint's influence.

## 13. Common evaluation protocol

Every training variant is evaluated against one frozen binary test target built with
the preferred record-first, context-balanced procedure. This gives binary, continuous,
ranking, Morgan, and other baselines a common decision problem.

Models are converted to binary decisions using validation data only:

- binary model: probability cutoff;
- continuous model: predicted-distance cutoff;
- ranking/similarity model: score cutoff;
- Morgan baseline: Tanimoto cutoff.

Each method receives the same calibration protocol and the cutoff is frozen before
test evaluation.

Raw accuracy is reported, but it is not sufficient under class or endpoint imbalance.
The primary common metrics are:

- balanced accuracy;
- macro F1;
- raw accuracy;
- AUROC and AUPRC; and
- transfer-class precision and recall.

Metrics are reported per canonical endpoint, per endpoint family, per transfer
universe, and as a macro-average across endpoints. Continuous distance correlation or
error and pairwise ranking accuracy are secondary diagnostics. Binary evaluation can
remain the winner-selection criterion, but these diagnostics reveal whether it favors
a model trained directly on the frozen boundary.

The post-training retrieval benchmark can then measure whether the learned score
selects genuinely transferable neighbors. It is run from frozen checkpoints and does
not influence training checkpoints.

## 14. Materialized artifact contract

Every normalized-record artifact stores:

- canonical molecule identifier and structure;
- raw and canonical endpoint fields;
- parsed value, canonical unit, transform, qualifiers, bounds, and censoring;
- normalized condition fields and key versions;
- raw source row identifier and provenance;
- ontology, parser, and schema versions.

Every pair artifact stores at least:

- retrieval record index and query molecule identifier;
- direction of transfer;
- endpoint key and shared `K` with their versions;
- retrieval value `y_A`;
- number of eligible and decisive query records;
- number of eligible and decisive nuisance-context cells;
- record and context coverage;
- per-cell summaries or a stable reference to them;
- `P_transfer`, hard label if assigned, and expected distance;
- canonical metric type, metric-policy version, and resolved transfer/non-transfer
  thresholds;
- condition-key profile, realized bucket, and condition-key version;
- consensus and minimum-evidence rules;
- aggregator, label-policy, ontology, and parser versions; and
- split and sampling provenance.

Artifacts must make it possible to reconstruct why a pair received its label without
loading model code.

## 15. Endpoint rollout

### Tier 1: establish the generalized pipeline

- Q2: fraction absorbed/intestinal absorption, `Papp`/`Peff`, and solubility.
- Q3: transporter status, efflux ratio/`Papp`, and direct gut-wall escape measures.
- Q4: intrinsic clearance, metabolic half-life, and extraction ratio.

These provide bounded, log-continuous, ratio, and categorical cases while retaining
interpretable biology.

### Tier 2: expand after Tier 1 validation

- Q2: dissolution and gastrointestinal stability.
- Q3: intestinal metabolism kinetics.
- Q4: microsomal/hepatocyte/S9 stability and typed CYP/UGT kinetics.
- Oral-bioavailability family: Q1 and the in-house Starling set as one endpoint family
  (Section 4.1), under the uniform bounded-fraction metric policy, once "oral
  bioavailability" is split into its distinct canonical endpoints. The Starling set is
  the primary query-disjoint deployment benchmark (Section 12.1); Q1's legacy labels are
  retained as the frozen backward-compatibility benchmark.

### Deferred or quarantined

- intervention-specific exposure changes without a stable intervention schema;
- vague `other` categories;
- multi-point profiles that cannot be reduced without losing the endpoint definition;
- poorly qualified categorical labels; and
- records whose units, directions, or normalization bases cannot be resolved.

## 16. Implementation sequence

The v1 pre-pipeline control plane is under `configs/assay_transfer/v1/`:

- `endpoints.yaml`: canonical endpoint-to-metric and condition-schema assignments;
- `species.yaml`: source-specific species recovery and missingness behavior;
- `condition_keys.yaml`: liberal, same-species, group, and most-specific eligibility;
- `metric_thresholds.yaml`: a priori thresholds by canonical metric type;
- `label_policy.yaml`: record-first deadband, context balancing, and pair aggregation;
  and
- `release.yaml`: the pinned configuration contract.

The implementation sequence is:

1. Inventory raw endpoint aliases, units, qualifier fields, and condition coverage.
2. Validate and freeze the configuration/ontology version for the build.
3. Normalize values, endpoints, species, assay systems, and condition fields into a
   canonical record table.
4. Canonicalize molecule identities and assign molecules to splits.
5. Audit condition-key coverage and metric-policy label sensitivity on training data;
   changes create a new version rather than mutating v1.
6. Construct nuisance-context cells, record-level outcomes, and pair targets separately
   within each split.
7. Materialize pair artifacts and ML-ready datasets with complete policy provenance.
8. Train Morgan, binary-only, continuous-only, combined, and optional ranking
   baselines without KNN callbacks.
9. Calibrate all scores on validation and run the frozen common binary test.
10. Run standalone retrieval evaluation for selected checkpoints.

## 17. Required configuration versions

At minimum, each experiment names:

```text
dataset_snapshot_version
assay_transfer_release_version
molecule_canonicalization_version
split_version
endpoint_ontology_version
endpoint_assignment_version
value_parser_version
species_normalization_version
condition_key_version
nuisance_context_version
metric_threshold_policy_version
label_aggregation_version
sampling_version
model_input_contract_version
evaluation_protocol_version
```

Changing any of these creates a new dataset or benchmark version. Results are only
directly comparable when the relevant versions match.

## 18. Open decisions

The architecture is fixed enough to implement, but the following choices require
explicit review and versioning:

- expert review of the provisional metric-level a priori thresholds and sensitivity
  variants;
- hard-consensus cutoffs and minimum decisive evidence;
- the normalized fields and buckets required by each most-specific key;
- mean versus robust within-cell continuous aggregation;
- uniform context weighting versus observation-frequency weighting as an ablation;
- the canonical-endpoint split of "oral bioavailability" — absolute `F`, relative or
  comparative `F`, and `%F` from an AUC ratio are distinct endpoints and must not share
  a transfer universe (Section 4.1);
- the bounded-fraction distance function — absolute difference versus a logit or fold
  distance, which changes how `[0,1]` differences at the low and high ends are weighted;
  this is a global metric-level choice for `Fa`, `Fg`, `Fh`, and `F`, not an
  oral-bioavailability override;
- whether the in-house Starling set is held out purely as the deployment benchmark or
  partially pooled into training when data is scarce (Section 12.1);
- the relative binary, continuous, and ranking loss weights; and
- whether query-disjoint or strict molecule-disjoint evaluation is the primary
  deployment benchmark.

These are experimental decisions within the architecture, not reasons to change the
inference contract. In all variants, the query contributes structure only, `K` defines
the shared transfer setting, and query records are used solely to construct
context-marginalized supervision.
