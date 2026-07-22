# Assay Transfer Version 5: Variance-Soft Binary Supervision

Status: active immutable contract
Last updated: 2026-07-17

## Purpose

V5 converts the complete V4 evidence membership into an A/B-only target. It preserves
the frozen V4 candidate identities and held-out binary labels while expressing greater
confidence for larger consistent evidence sets and less confidence for deadband-heavy
sets. V4 artifacts and their A/B/C empirical targets remain unchanged.

## Target policy

For transfer, non-transfer, and ambiguous record counts, define:

```text
N = n_transfer + n_nontransfer + n_ambiguous
T = sqrt(N) * (1 + n_ambiguous / N)
q = softmax([n_transfer, n_nontransfer] / T)
```

Raw counts are logits. The temperature scale is fixed at `1.0`; evidence count does not
weight row loss. Deadband records affect uncertainty but are not an output class.

The target policy is `variance_temperature_binary_v5`, and the artifact schema is
`assay_transfer_variance_soft_v5`.

## Dataset contract

V5 retains all V4 training candidates, including conflicting and all-deadband rows.
Validation and test retain their ordered candidate IDs and binary labels. Prompts expose
only `(A) transfer` and `(B) not transfer`.

Each row stores the existing evidence counts and fractions plus:

- `metadata.target_distribution.{transfer,nontransfer}`
- `metadata.target_temperature`
- `metadata.completion_is_tie_anchor`
- `metadata.target_policy_version`

A unique transfer/non-transfer count winner is serialized as A/B. Exact count ties,
including all-deadband rows, store `[0.5, 0.5]`, serialize A, and set the tie-anchor flag.
The serialized decision token receives no hard formatting loss. Inference also chooses A
on an exact A/B model-logit tie.

## Training and evaluation

Training uses full-vocabulary soft cross-entropy at the A/B decision token. Every
decision token is excluded from formatting CE; shared formatting and EOS retain the
weighted hard loss. BFD packing, padding-free global decision offsets, document-reset
position IDs, Liger formatting CE, and PEFT fallbacks follow the V4 implementation.

Report binary accuracy, binary macro-F1, soft NLL, Brier score, and binned reliability
against the A/B distribution. V5 has no C prediction-rate metric.

## Build and publication

The build consumes the frozen V4 materialized dataset directly:

```bash
python pipeline/stages/run.py \
  --config configs/builds/assay_transfer_variance_soft_v5.yaml
```

It writes standard and Intern artifacts. The public Intern dataset is
`jiosephlee/assay-transfer-variance-soft-intern`. Rebuilds must be byte-deterministic,
and publication is accepted only when remote Parquet SHA-256 values match locally.

Research extensions not included in this contract are tracked in
`assay_transfer_research_extensions_deferred.md`.
