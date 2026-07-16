# Assay Transfer Dataset and Model Architecture, Version 2

Status: architecture draft and proposed canonical design reference  
Last updated: 2026-07-15

## 1. Purpose and relationship to version 1

This document defines the version 2 dataset, target, sampling, artifact, model-input,
and evaluation contracts for assay-transfer learning. It is a standalone design rather
than a change log. Once reviewed and frozen, it supersedes the architectural decisions
in `assay_transfer_design.md`.

Version 2 retains the central inference contract from version 1 but sharpens the unit
of supervision. The dataset is not a collection of randomly sampled assay-record
pairs. Each row is a directed, retrieval-anchored transfer example:

```text
one measured retrieval record
    -> one query molecule
    -> one distribution of compatible records for that query molecule
```

The compatible query records are internal evidence used to construct the target. They
are not independently sampled training rows. Version 2 also makes these decisions
explicit:

- molecules are assigned globally across all included sources and endpoints;
- each split constructs examples only within its assigned molecule universe;
- binary targets use record votes followed by a strict majority over all eligible
  records;
- continuous targets use the mean of individual retrieval-to-query-record distances;
- context-cell macro-averaging is removed from the canonical target;
- validation and test contain only examples with hard binary labels;
- sampling strata are assay concept and low/high Tanimoto similarity;
- high-degree molecules are controlled during selection; and
- the initial 200,000-example training release is an expandable prefix of a
  reproducible training candidate universe.

## 2. Objective

The goal is to learn a molecular similarity function that is more useful for assay
transfer than a fixed structural similarity such as Morgan/Tanimoto.

The model answers a directional question:

> Given a measured retrieval record in a specified assay setting, how safely should
> that measurement transfer to a particular query molecule whose structure is known?

The primary training objective in version 2 is continuous distance prediction. A
binary transfer head may be added as an auxiliary loss on the subset of examples that
receive a hard binary label.

The project does not primarily predict the query molecule's assay value. Query assay
values are available only during offline target construction and are never query-side
model inputs.

### 2.1 Non-goals

- Predicting an assay value from query structure alone.
- Treating records from different canonical biological endpoints as comparable.
- Supplying query-side assay metadata or query values at inference time.
- Treating every source row as an independently sampled training example.
- Using Tanimoto similarity as ground truth or as part of the label definition.
- Selecting checkpoints using retrieval/KNN evaluation.

## 3. Inference contract

At inference time, the system receives:

- query molecule structure `x_B`;
- retrieval molecule structure `x_A`;
- the retrieval record's normalized assay metadata `Z_A`;
- the retrieval value `y_A`; and
- an explicit shared transfer setting `K_A` derived from the retrieval context.

It does not receive a query record, query assay metadata, or a query assay value.

```text
Query branch                         Retrieval branch
-------------                        ----------------
query structure x_B                  retrieval structure x_A
        |                            retrieval metadata Z_A
        |                            retrieval value y_A
        v                                    |
  query encoder                              v
        +---------- pair scorer <---- retrieval encoder
                         ^
                         |
              shared setting encoder K_A
                         |
                         v
            distance and optional transfer score
```

The score is directional because the retrieval record supplies information that is
not available for the query:

```text
score(A -> B | K_A) != score(B -> A | K_B) in general
```

## 4. Canonical terminology and data model

### 4.1 Normalized assay record

A normalized assay record is:

```text
r = (
    record_id,
    molecule_id,
    structure,
    source,
    canonical_endpoint_key,
    metric_type,
    value,
    unit,
    K,
    Z,
    provenance
)
```

The fields have the following meanings:

- `record_id` is a stable normalized-record identifier.
- `molecule_id` is the canonical molecule identity used for global splitting.
- `source` identifies Q1, Q2, Q3, Q4, Starling, or another collection.
- `canonical_endpoint_key` identifies the biological quantity and comparable scale.
- `metric_type` selects the distance and threshold policy.
- `K` contains the declared fields required for transfer eligibility.
- `Z` contains the remaining normalized retrieval-side metadata.
- `provenance` preserves raw row identity, parser version, and source information.

Source is provenance and a sampling/reporting attribute. It is not automatically part
of canonical endpoint identity. Source-specific endpoint names map onto a shared
canonical endpoint only when their biological definition, unit basis, transformation,
and direction are genuinely compatible.

### 4.2 Directed transfer example

The materialized learning unit is:

```text
E = (r_A, B, K_A)
```

where:

- `r_A` is one concrete retrieval record;
- `B` is one different canonical query molecule; and
- `K_A` is the shared setting derived from `r_A` under the selected condition-key
  profile.

The retrieval side remains a record rather than only a molecule because its value and
metadata are model inputs. Degree control may nevertheless operate at both the
retrieval-record and retrieval-molecule levels.

### 4.3 Query evidence distribution

The evidence distribution for `E` is:

```text
R_B(K_A) = {
    r_B : molecule_id(r_B) = B
          and canonical_endpoint_key(r_B) = canonical_endpoint_key(r_A)
          and K(r_B) = K_A
}
```

`R_B(K_A)` contains multiple records for one query molecule, not multiple query
molecules. The model receives `x_B` once. The records in `R_B(K_A)` are used only to
construct targets and evidence diagnostics.

### 4.4 Internal record comparison

For each `r_B` in the evidence distribution, target construction computes a distance
between `y_A` and `y_B`. This internal comparison is not a dataset row and is not an
independently sampled molecular pair. It is one vote and one continuous-distance
observation within the retrieval-anchored transfer example.

## 5. Dataset scope and assay concepts

Version 2 groups the current sources into four assay concepts for sampling and
reporting:

| Assay concept | Included collections or endpoint families |
|---|---|
| Oral bioavailability | Q1 and Starling oral bioavailability |
| `Fa` | Q2 intestinal absorption and related eligible endpoints |
| `Fg` | Q3 gut-wall escape and related eligible endpoints |
| `Fh` | Q4 hepatic escape and related eligible endpoints |

An assay concept is a coarse sampling stratum, not an eligibility key. Records may
only compare within the same canonical endpoint even when two endpoints belong to the
same concept.

For example, absolute oral bioavailability, relative bioavailability, and an AUC-ratio
percentage may all belong to the oral-bioavailability concept while remaining distinct
canonical endpoints. Similarly, `Fa`, `Papp`, and solubility may be reported under the
Q2/`Fa` concept but do not form cross-endpoint transfer examples.

## 6. Endpoint and value normalization

### 6.1 Canonical endpoint identity

A canonical endpoint key must resolve at least:

- biological endpoint family;
- measurement subtype;
- canonical unit or normalization basis;
- value transformation;
- direction, when applicable;
- timepoint or profile definition, when intrinsic to the endpoint; and
- parameter identity for typed kinetic measurements.

Cross-endpoint examples are prohibited. Unit compatibility alone is insufficient.

### 6.2 Metric types and distances

| Metric type | Canonical representation | Distance |
|---|---|---|
| Bounded percentage | `[0,100]` | absolute percentage-point difference |
| Bounded fraction | `[0,1]` | absolute fraction difference |
| Positive continuous | canonical unit, then `log10` | absolute log difference |
| Positive ratio | usually `log10` | absolute log-ratio difference |
| Binary categorical | normalized reliable class | equal or unequal |
| Ordinal categorical | ordered normalized bins | bin separation |
| Censored or interval | bound or interval object | interval-aware distance state |

Values are compared only after canonicalization. Unparseable or scientifically
ambiguous records remain quarantined.

### 6.3 Protocol-defining fields and free text

Condition-key schemas declare which normalized metadata fields define compatible
protocols. Designated text-derived protocol fields may participate when necessary, but
their use must be explicit and versioned. Arbitrary raw free-text equality is not added
to every key by default.

Raw free text is always retained as retrieval metadata and provenance. When a text
field is used for eligibility, the manifest identifies the exact field, normalization
rule, missingness behavior, and comparison policy.

Version 2 does not create nuisance-context cells from metadata outside `K`. Each
eligible query record contributes one observation to the evidence distribution.

## 7. Transfer universes and condition keys

A condition key answers whether records are scientifically eligible to compare. It
does not define the numeric transfer threshold.

### 7.1 Same endpoint

```text
K = canonical_endpoint_key
```

This is the broadest eligible universe and marginalizes over all other observed query
conditions.

### 7.2 Same species and endpoint

```text
K = (canonical_endpoint_key, species_exact)
```

Both species values must be non-null and equal. Two missing values do not match.
Generic groups such as `rodent` or `monkey` do not satisfy an exact-species key.

### 7.3 Most-specific condition key

The most-specific profile uses endpoint-specific normalized fields such as:

- species;
- assay-system family;
- tissue or anatomical site;
- direction;
- medium, matrix, or pH;
- formulation or molecular form;
- timepoint;
- transporter or enzyme identity;
- parameter type and normalization basis;
- modifier class;
- dose or concentration bucket; and
- protocol-defining normalized text fields when explicitly configured.

Missing required values do not match. Continuous protocol conditions use frozen,
scientifically meaningful buckets rather than unversioned exact comparisons.

### 7.4 Separation of responsibilities

```text
canonical endpoint + K -> scientific eligibility
metric type             -> distance and record-vote thresholds
assay concept           -> sampling and reporting stratum
Tanimoto bucket         -> sampling and reporting stratum
```

Neither assay concept nor Tanimoto similarity may override endpoint or condition-key
eligibility.

## 8. Global molecule splitting

### 8.1 Canonicalization before splitting

All selected records from all included sources are composed before molecule assignment.
Structures are canonicalized first, and each canonical molecule receives one split
assignment across the entire build.

```text
canonical molecule -> exactly one of train, validation, or test
```

Every record for that molecule follows the same assignment, regardless of source,
endpoint, or whether the molecule appears in multiple databases.

### 8.2 Independent molecule universes

Version 2 constructs each split using only molecules assigned to that split:

```text
train examples      use train molecules only
validation examples use validation molecules only
test examples       use test molecules only
```

Both retrieval and query molecules must belong to the example's split. This is the
canonical strict molecule-disjoint benchmark. A future query-disjoint deployment
benchmark with a shared retrieval library must be built and named separately.

### 8.3 Generous evaluation reservations

Validation and test molecule pools must be deliberately larger than the minimum number
suggested by their final example quotas. A reserved molecule may fail to yield a
selected example because it lacks:

- a different eligible partner molecule;
- a compatible canonical endpoint and `K` bucket;
- valid comparable values;
- a binary majority label; or
- capacity under the degree-control policy.

Molecules assigned to validation or test never return to training merely because they
do not yield selected examples. Split assignment is frozen before final selection.

## 9. Candidate transfer-example construction

### 9.1 Eligibility

A candidate `E = (r_A, B, K_A)` is created only when:

1. `r_A` and `B` belong to the same split;
2. `molecule_id(r_A) != B`;
3. `R_B(K_A)` is non-empty;
4. all comparisons use the same canonical endpoint and metric type;
5. retrieval and query values are comparable under the frozen value policy; and
6. the example meets the configured minimum evidence requirements.

The example is directional. Its reverse, if eligible, is a separate candidate with a
different retrieval record, evidence distribution, and potentially different target.

### 9.2 Candidate identity

Every candidate receives a stable identifier derived from versioned fields such as:

```text
retrieval_record_id
query_molecule_id
canonical_endpoint_key
condition_key_profile
realized_K
split_version
target_policy_version
```

Stable identity is required for deduplication, deterministic sampling, and later
expansion of the training dataset.

### 9.3 Tanimoto bucket

Tanimoto similarity is computed between the canonical retrieval and query structures.
Each candidate is assigned to exactly one configured bucket:

```text
low Tanimoto
high Tanimoto
```

The fingerprint implementation, parameters, molecule representation, boundary, and
invalid-structure behavior are versioned. Tanimoto affects sampling only.

## 10. Target construction

### 10.1 Record-level distance

For every eligible `r_B` in `R_B(K_A)`, compute:

```text
d_j = d_m(y_A, y_Bj)
```

where `m` is the canonical metric type. The distance uses the frozen canonical units,
transforms, directions, and interval rules.

### 10.2 Record-level binary vote

Each comparable query record contributes one of three vote states:

```text
d_j <= T_transfer(m)       -> transfer vote
d_j >= T_not_transfer(m)   -> non-transfer vote
otherwise                  -> ambiguous vote
```

The thresholds are attached to the metric type and do not vary by source, assay
concept, condition-key profile, Tanimoto bucket, or realized `K` bucket.

An internal vote is evidence, not a materialized training example. Ambiguous votes are
not discarded from the evidence denominator.

### 10.3 Strict-majority binary target

For one candidate transfer example, define:

```text
N             = number of all eligible comparable records in R_B(K_A)
n_transfer    = number of transfer votes
n_nontransfer = number of non-transfer votes
n_ambiguous   = number of ambiguous votes

N = n_transfer + n_nontransfer + n_ambiguous
```

The hard binary target is:

```text
binary_label = 1     if n_transfer    > N / 2
binary_label = 0     if n_nontransfer > N / 2
binary_label = null  otherwise
```

This is a strict majority of all eligible records. Opposing and ambiguous votes both
count against either side obtaining a majority. Ties and distributions without a
majority receive no binary label.

The majority is retrieval-anchored. The same query molecule and `K` may receive a
different label when paired with a different retrieval record or retrieval value.

### 10.4 Continuous target

For numeric examples with defined scalar distances, the primary continuous target is:

```text
D_expected(A -> B | K_A) = (1 / N) * sum_j(d_j)
```

This is the mean of individual distances. It is not the distance between `y_A` and an
average query value:

```text
mean_j(|y_A - y_Bj|) != |y_A - mean_j(y_Bj)| in general
```

All defined distances contribute, including records whose binary votes are ambiguous.
Therefore an example may have a continuous target while its binary label is null.

Robust summaries such as the median, high quantile, standard deviation, and maximum
may be stored as diagnostics or future ablations, but they do not replace the canonical
mean target in version 2.

### 10.5 Removal of context-balanced aggregation

Version 2 does not construct nuisance-context cells or macro-average context means.
Every eligible query record contributes one equal observation.

This reflects the current data model: records are unique when their complete
protocol-defining metadata is considered, so context cells would generally be
singletons and context-macro averaging would collapse to ordinary record averaging.

### 10.6 Evidence and confidence fields

Every materialized example stores at least:

- `N`, `n_transfer`, `n_nontransfer`, and `n_ambiguous`;
- transfer, non-transfer, and ambiguous fractions over `N`;
- nullable hard binary label;
- majority side and majority margin;
- continuous mean distance when defined;
- optional dispersion summaries;
- canonical metric type and resolved thresholds; and
- evidence exclusion counts and reasons.

Minimum evidence requirements remain configurable and versioned. A one-record majority
and a narrow majority over many records are both legal under the mathematical rule but
remain distinguishable in the artifact.

## 11. Provisional metric threshold policy

The version 1 a priori thresholds remain the provisional starting point:

| Canonical metric type | Transfer vote | Non-transfer vote | Ambiguous vote |
|---|---:|---:|---|
| Bounded percentage `[0,100]` | at most 10 points | at least 30 points | between |
| Bounded fraction `[0,1]` | at most 0.10 | at least 0.30 | between |
| Papp, Peff, solubility, clearance, half-life | within 2-fold | at least 5-fold | between |
| `Vmax` | within 2-fold | at least 5-fold | between |
| `Km` | within 2-fold | at least 5-fold | between |
| Dimensionless ratio | within 1.5-fold | at least 3-fold | between |
| Binary categorical | same reliable class | opposite reliable class | unknown |
| Ordinal categorical | same bin | at least two bins apart | adjacent bin |

Positive scalar and ratio comparisons use absolute log distance after canonicalization.
Thresholds are frozen before model training. Sensitivity variants create new policy
versions rather than mutating a released dataset.

## 12. Sampling and fixed dataset sizes

### 12.1 Canonical sampling strata

The only primary selection strata in version 2 are:

```text
assay concept x Tanimoto bucket
```

With four assay concepts and two Tanimoto buckets, the build has eight primary strata:

| Assay concept | Low Tanimoto | High Tanimoto |
|---|---:|---:|
| Oral bioavailability | one stratum | one stratum |
| `Fa` | one stratum | one stratum |
| `Fg` | one stratum | one stratum |
| `Fh` | one stratum | one stratum |

Endpoint, source, condition-key profile, label, and evidence counts are reported but do
not create additional sampling strata unless a future sampling-policy version says
otherwise.

### 12.2 Default release quotas

The default version 2 release targets are:

```text
training:   200,000 transfer examples
validation:   2,000 hard-binary-labeled transfer examples
test:         2,000 hard-binary-labeled transfer examples
```

The allocation of each total across the eight primary strata must be declared in the
sampling configuration. Underfilled strata are reported explicitly; eligibility,
split isolation, or label rules are never weakened silently to meet a quota.

### 12.3 Training selection

Training selection requires a valid continuous target. A hard binary label is not
required. Each selected row stores its nullable binary label so that the same release
supports continuous-only and auxiliary-binary ablations.

The manifest reports separately:

- total selected training rows;
- rows with continuous targets;
- rows with hard binary labels;
- rows without binary labels; and
- counts by the eight primary strata.

### 12.4 Validation and test selection

Validation and test select only candidates whose hard binary label is non-null. They
are frozen after selection and are used for the common binary decision task.

The release also reports binary-label coverage before selection:

```text
hard-binary-labeled candidates / all otherwise eligible candidates
```

Coverage is reported overall and by assay concept, Tanimoto bucket, endpoint, and
condition-key profile. This prevents binary metrics from being interpreted as
performance on query distributions that lack a majority label.

### 12.5 Degree control

Uniform sampling over all candidates is prohibited because a molecule or retrieval
record with many eligible connections could dominate the release.

Selection uses deterministic round-robin or configured caps over:

- query molecule;
- retrieval molecule;
- retrieval record, when necessary; and
- primary sampling stratum.

The exact cap values, exhaustion behavior, and deterministic tie-breaking rule are
part of the sampling-policy version. Degree distributions before and after selection
are stored in the manifest.

### 12.6 Inclusion probabilities and reproducibility

Every selected example stores its stratum, deterministic priority, and selection
policy. The manifest stores the random seed when randomness is used. Selection must be
reproducible from the frozen candidate universe and policy.

## 13. Expandable training artifact

### 13.1 Requirement

The initial 200,000-example training release must be expandable without changing split
assignments, relabeling existing examples, or sampling them twice.

The expansion artifact represents the remaining candidate universe, not a collection
of unused query records. All compatible query records remain label evidence even when
their associated candidates have not yet been selected as training rows.

### 13.2 Required contents

A self-sufficient expansion bundle contains:

- normalized training records required to reconstruct every evidence distribution;
- the frozen global molecule split assignment;
- all bound normalization, condition-key, metric, target, and sampling policy versions;
- stable candidate identifiers and deterministic stratum-local priorities;
- identifiers of the initial 200,000 selected examples;
- a cursor, consumed-prefix marker, or equivalent selection ledger for every stratum;
- exclusion reasons and candidate/evidence counts;
- fingerprints and Tanimoto bucket assignments, or the complete versioned recipe to
  reproduce them; and
- schema and implementation version hashes.

The artifact may be a versioned directory with manifests and sharded tables. “Single
artifact” means one self-contained versioned bundle, not necessarily one physical file.

### 13.3 Nested expansion

Training releases are nested prefixes under a fixed policy:

```text
train_200k subset train_400k subset train_800k subset full_selected_universe
```

Expansion takes the next eligible candidates from each stratum according to the frozen
ordering and allocation policy. It does not recompute validation or test and does not
move molecules between splits.

## 14. Model objectives

### 14.1 Primary continuous loss

The canonical training objective predicts `D_expected`. A robust regression loss may
be used, but the target remains the arithmetic mean of record distances.

```text
L_primary = L_continuous(predicted_distance, D_expected)
```

### 14.2 Optional binary auxiliary loss

An auxiliary binary head may be trained only where `binary_label` is non-null:

```text
L = lambda_continuous * L_continuous
  + lambda_binary * binary_label_mask * L_binary
```

Binary-only, continuous-only, and combined variants use the same training release and
split assignments. The binary auxiliary head is an ablation, not a requirement for a
valid training row.

### 14.3 Optional future objectives

Ranking or uncertainty objectives may be added in later model-policy versions. They
must not change the frozen target or evaluation examples in place.

## 15. Evaluation protocol

### 15.1 Validation calibration

Validation is used to convert each model's score into a binary decision:

- continuous model: select a predicted-distance cutoff;
- binary head: select a probability cutoff;
- ranking or similarity score: select a score cutoff;
- Morgan/Tanimoto baseline: select a Tanimoto cutoff.

The selected cutoff is frozen before test evaluation.

### 15.2 Primary binary metrics

The primary test metrics are:

- macro F1; and
- accuracy.

Balanced accuracy, per-class precision and recall, AUROC, and AUPRC are supporting
diagnostics. Metrics are reported overall, per assay concept, and per Tanimoto bucket.
Endpoint-level and condition-key-profile results are reported when sample size permits.

### 15.3 Interpretation boundary

The binary benchmark evaluates performance conditional on the empirical query-record
distribution producing a strict-majority hard label. It does not directly evaluate
examples with a null binary target.

Continuous error and correlation can be reported over the broader set of examples with
valid continuous targets. Binary-label coverage accompanies every binary result.

### 15.4 Dependence-aware uncertainty

Because examples can share retrieval or query molecules, example rows are not fully
independent. Confidence intervals should resample or cluster by canonical molecule
rather than treating all 2,000 rows as independent observations.

### 15.5 Retrieval benchmark

KNN or retrieval evaluation remains a post-training benchmark. It uses frozen
checkpoints and does not influence early stopping, checkpoint selection, or target
construction.

## 16. Materialized artifact contract

### 16.1 Normalized-record artifact

Every normalized record artifact stores:

- stable record and canonical molecule identifiers;
- canonical structure;
- source and raw row provenance;
- raw and canonical endpoint fields;
- parsed value, canonical unit, transform, qualifiers, bounds, and censoring;
- normalized condition fields;
- raw retrieval metadata, including retained free text;
- condition-key realizations and versions; and
- ontology, parser, schema, and dataset snapshot versions.

### 16.2 Transfer-example artifact

Every transfer-example artifact stores at least:

- stable candidate identifier;
- retrieval record identifier and retrieval molecule identifier;
- query molecule identifier;
- canonical retrieval and query structures or stable references;
- direction of transfer;
- retrieval value and retrieval metadata required by the model;
- canonical endpoint and assay concept;
- condition-key profile, realized `K`, and versions;
- metric type, distance policy, and thresholds;
- evidence record identifiers or a stable reference to the evidence distribution;
- evidence counts, vote counts, vote fractions, and exclusion reasons;
- continuous mean distance and optional dispersion diagnostics;
- nullable hard binary label and majority margin;
- Tanimoto value, bucket, and fingerprint-policy version;
- split, stratum, selection priority, and sampling-policy version; and
- all normalization and target-policy provenance.

Artifacts must make the target auditable without loading model code.

## 17. Implementation sequence

1. Inventory source endpoint aliases, units, protocol fields, and value qualifiers.
2. Freeze the endpoint, condition-key, metric, target, fingerprint, and sampling
   policies.
3. Normalize all included source records into canonical record artifacts.
4. Canonicalize molecule identities across the composed source union.
5. Assign molecules globally to generous train, validation, and test pools.
6. Enumerate directed candidate transfer examples within each split.
7. Resolve each candidate's query-record evidence distribution.
8. Compute record distances, three-state votes, strict-majority binary labels, and
   continuous mean-distance targets.
9. Assign assay-concept and low/high-Tanimoto strata.
10. Select fixed validation and test examples from hard-binary-labeled candidates with
    degree control.
11. Select the initial 200,000 continuous-target training examples and write the
    expansion bundle.
12. Materialize ML-ready artifacts and complete manifests.
13. Train continuous-only and optional binary-auxiliary ablations.
14. Calibrate score cutoffs on validation and evaluate once on frozen test.
15. Run standalone retrieval evaluation for selected frozen checkpoints.

## 18. Required configuration and manifest versions

Every released build names at least:

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
metric_threshold_policy_version
record_vote_policy_version
majority_label_policy_version
continuous_target_version
fingerprint_version
tanimoto_bucket_version
sampling_strata_version
degree_control_version
sampling_version
model_input_contract_version
evaluation_protocol_version
artifact_schema_version
```

Changing any item creates a new build or benchmark version. Released artifacts are
immutable.

## 19. Audits required for every build

Before release, report:

- molecule and record counts by split, source, assay concept, and endpoint;
- cross-source molecule overlaps and proof of globally consistent assignment;
- candidate counts before and after evidence and label filtering;
- continuous-target and binary-label coverage;
- transfer, non-transfer, and null-label distributions;
- vote-count and majority-margin distributions;
- Tanimoto distributions and low/high bucket counts;
- counts and underfill by the eight primary sampling strata;
- retrieval-record, retrieval-molecule, and query-molecule degree distributions;
- threshold deadband coverage by metric type;
- evidence-distribution sizes;
- validation/test molecule disjointness; and
- reproducibility checks for deterministic candidate and selection identifiers.

## 20. Remaining decisions requiring explicit configuration

The following choices are not fixed by the architectural decisions above:

- the fingerprint implementation and low/high Tanimoto boundary;
- equal versus proportional allocation across the eight primary strata;
- the number of molecules reserved for validation and test before example construction;
- minimum eligible-record evidence for training, validation, and test;
- degree-cap values and round-robin exhaustion behavior;
- expert review of the provisional metric thresholds;
- endpoint-specific most-specific condition-key fields;
- exact use and normalization of protocol-defining free-text fields;
- treatment of scalar distances for censored and interval-valued records;
- the continuous regression loss and optional robust diagnostics;
- the auxiliary binary-loss weight; and
- the schedule and quotas for expanded training prefixes.

These are versioned implementation or experimental decisions. They do not change the
core version 2 contract: globally split molecules, directed retrieval-anchored transfer
examples, one query molecule per evidence distribution, equal record contributions,
strict-majority binary labels, continuous mean-distance supervision, binary-only fixed
evaluation sets, and assay-concept/Tanimoto sampling strata.
