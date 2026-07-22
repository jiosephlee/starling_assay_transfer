# Assay Transfer: Deferred Research Extensions to Soft-Evidence Supervision

Status: deferred research and design outline
Last updated: 2026-07-19

## 1. Purpose

Version 5 freezes the variance-temperature binary design: evidence-count logits,
deadband-inflated temperature, A/B soft-target cross-entropy, and the unchanged hard
A/B benchmark.

This document preserves ideas deliberately excluded from V5. None is an implementation
requirement or a permitted silent change to the V5 target.

## 2. Final-decision probability targets

Neither V4's empirical distribution nor V5's variance-soft binary projection claims to
be the posterior probability of an overall majority conclusion.

A future design may instead model latent evidence proportions:

```text
theta ~ Dirichlet(n_transfer + alpha_T,
                  n_nontransfer + alpha_N,
                  n_ambiguous + alpha_A)

q_A_final = P(theta_T > 0.5)
q_B_final = P(theta_N > 0.5)
q_C_final = 1 - q_A_final - q_B_final
```

This distinguishes 6/10 support from 60/100 support, but it introduces a prior and an
independence assumption. Prior selection, sensitivity, and scientific interpretation
must be resolved before promotion.

## 3. Calibration extensions

Candidate extensions include global temperature scaling, vector or Dirichlet
calibration, and sufficiently supported concept-specific calibration. Every calibrator
must be fitted on validation only, frozen before test, and evaluated for both soft
distribution calibration and hard A/B behavior.

These methods remain deferred until the fixed V5 temperature establishes an uncalibrated
reference on molecule-disjoint validation.

## 4. Correlated evidence and effective support

Raw `N` may overstate evidence strength when several records share a publication,
experiment, assay, or duplicated measurement. Future alternatives include:

- one vote per provenance or study cluster;
- hierarchical aggregation within and then across clusters;
- an effective evidence count distinct from raw `N`; and
- source-balanced or protocol-balanced voting.

Any grouping policy must use explicit normalized provenance, remain reconstructable, and
be compared with the V5 equal-record reference.

## 5. Scalar projections

Alternative one-number projections may be useful for ablations but are not canonical in V5.

### 5.1 Majority confidence

```text
q_A = n_transfer / N         for hard transfer rows
q_A = 1 - n_nontransfer / N for hard non-transfer rows
```

This assigns the winning answer its observed majority fraction but is undefined as a
single coherent estimand for no-majority rows without an additional rule.

### 5.2 Ambiguity-neutral binary projection

```text
q_A = (n_transfer + 0.5 * n_ambiguous) / N
q_B = 1 - q_A
```

This treats ambiguity as neutral but collapses conflict and deadband evidence into one
number. The `0.5` allocation is a modeling convention rather than observed truth.

## 6. Continuous distance targets

Vote fractions discard distance magnitude within each threshold region. Deferred target
families include:

- metric-normalized mean distance;
- median, upper-quantile, and worst-case distance;
- per-record graded transfer curves followed by aggregation; and
- joint vote-distribution and distance-summary supervision.

Raw distances cannot share one loss across metric families. Any promoted target requires
frozen metric normalization, monotonicity, treatment of censoring, and sensitivity tests
for interpolation or quantile choices.

## 7. Alternative model objectives

Deferred model designs include:

- constrained numerical generation;
- a scalar regression head with Huber or squared loss;
- multitask binary, evidence-distribution, and distance objectives; and
- non-LM architectures such as MLP evidence heads.

Numerical text generation must address tokenization, formatting failures, and numeric
error separately from token cross-entropy. Non-LM heads remain outside the present
LM-only strategy unless scope is explicitly reopened.

## 8. Promoted V6 model designs

The active raw record-to-record V6 designs are canonical contracts at:

- [Intern soft A/B plus ListNet](assay_transfer_design_v6_intern.md); and
- [MLP contrastive and soft-prediction variants](assay_transfer_design_v6_mlp.md).

Those contracts supersede the earlier aggregate-query ranking proposal. This document now
retains only alternatives and research extensions that remain deferred.

## 9. Sampling, weighting, and curriculum

Future experiments may balance assay concepts, label mass, similarity strata, query
degree, retrieval degree, or provenance clusters. They may also stage unanimous,
conflicting, and deadband-heavy examples as a curriculum.

Weighting must remain separate from target definition. Raw evidence count must not become
an automatic loss weight, and comparisons must control training examples or tokens so
that gains are not caused only by additional compute.

## 10. Promotion requirements

An extension may leave V6 deferral only when:

- its estimand and relationship to V4/V3 are mathematically explicit;
- new priors, grouping rules, mappings, or calibration data are versioned;
- molecule-disjoint validation shows improvement on preregistered metrics;
- sensitivity and subgroup behavior are acceptable;
- V3 binary, V4 empirical-distribution, and V5 variance-soft artifacts remain usable unchanged; and
- implementation, schema, and evaluation contracts are frozen before test access.
