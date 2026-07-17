# Assay Transfer Version 4: LM Soft-Evidence Supervision

Status: implemented design contract
Last updated: 2026-07-17

## 1. Purpose and scope

Version 4 retains the version 3 retrieval-anchored task, metric policies, record votes,
and molecule-disjoint evaluation benchmark. It replaces hard-only LM supervision with
the empirical distribution of transfer, non-transfer, and ambiguous record votes.

V4 remains an LM-only design. It does not add an MLP, scalar regression head, numerical
completion, metric-distance target, Bayesian target, or post-hoc calibration. Those
extensions are deferred to version 5.

The intended benefit is calibrated answer strength. A model may prefer transfer with
probability `0.60` for mixed evidence and `0.90` for stronger evidence instead of
receiving the same one-hot transfer target for both examples.

## 2. Inherited V3 contract

V4 does not change:

- canonical endpoint composition or metric thresholds;
- the directed candidate identity `(retrieval record A, query molecule B, endpoint K)`;
- the equal contribution of every eligible query record;
- global molecule split isolation;
- prompt exclusion of query values, query metadata, evidence counts, and labels; or
- the frozen hard-labelled validation and test selections.

Each eligible query record retains the V3 vote rule:

```text
distance <= transfer threshold       -> transfer
distance >= non-transfer threshold   -> non-transfer
otherwise                            -> ambiguous
```

With `N = n_transfer + n_nontransfer + n_ambiguous`, the reference binary label remains:

```text
binary_label = 1     if n_transfer    > N / 2
binary_label = 0     if n_nontransfer > N / 2
binary_label = null  otherwise
```

## 3. Canonical V4 target

The canonical target is the empirical three-state evidence distribution:

```text
q_A = n_transfer / N
q_B = n_nontransfer / N
q_C = n_ambiguous / N
q_A + q_B + q_C = 1
```

The three states mean:

- `A`: an eligible query measurement supports transfer;
- `B`: an eligible query measurement supports non-transfer; and
- `C`: an eligible query measurement lies in the ambiguous threshold deadband.

This is a distribution over record-level evidence outcomes. `q_C` is not the probability
that the final example has no majority. An example can lack a majority because transfer
and non-transfer conflict even when `q_C` is small.

The integer counts are the canonical source of truth. Stored fractions are deterministic
`float32` conveniences and are never rounded for rendering.

## 4. Training and evaluation membership

The V4 training pool contains every eligible candidate assigned to the training molecule
split, including candidates whose `binary_label` is null. A no-majority candidate does
not receive a hard C target; it receives its complete soft A/B/C distribution.

Validation and test remain the frozen V3 hard A/B selections. No null-labelled candidate
is added to either evaluation split. Their empirical distributions are retained so soft
loss and calibration metrics can be measured without changing evaluation membership.

This separation lets no-majority examples teach grey cases during training without
changing the canonical binary benchmark.

## 5. Prompt and serialized completion

The V4 prompt retains the V3 transfer-classification framing and adds only an ambiguous
choice:

```text
(A) transfer
(B) not transfer
(C) ambiguous
```

`completion` is a serialization and inspection aid, not the training target:

```text
unique argmax(q_A, q_B, q_C) -> corresponding A, B, or C
any tie for the maximum      -> C
```

The C tie fallback indicates that there is no unique modal evidence outcome. It does not
replace the stored distribution, and training code must never apply hard completion
cross-entropy to V4 rows.

## 6. Hugging Face dataset contract

Each V4 split has four top-level columns:

| Column | Arrow type | Purpose |
|---|---|---|
| `prompt` | `large_string` | Model input with A/B/C choices |
| `completion` | `string` | Modal serialization with C tie fallback |
| `target_distribution` | fixed named `struct` | Soft LM target |
| `metadata` | fixed nested `struct` | Audit, identity, context, and provenance |

`target_distribution` contains exactly:

```text
transfer: float32
nontransfer: float32
ambiguous: float32
```

Metadata retains `n_records`, all three integer counts, all three fractions, the nullable
V3 binary label, metric and threshold fields, and `target_policy_version`. The policy
version for this contract must be frozen before publication.

## 7. LM objective

Let `s_A`, `s_B`, and `s_C` be the LM scores for the three answer choices. Normalize only
over those choices:

```text
p = softmax([s_A, s_B, s_C])
loss = -(q_A log p_A + q_B log p_B + q_C log p_C)
```

This is soft-target cross-entropy, also called label-distribution learning. It is not
teacher-model distillation, although its logit-level behavior is similar.

If each answer identifier is one token, its next-token logit is its score. Otherwise the
score is the sum of its complete answer-sequence log probabilities. Tokenization must be
validated for every supported model. Raw full-vocabulary token probability is not a
valid confidence because it includes formatting alternatives and unrelated tokens.

Every training row uses the soft objective, including rows that have a V3 hard label.
The loss is not multiplied by raw `N`, because records need not be independent.

## 8. Prediction and evaluation

The public decision rule mirrors the V3 strict-majority rule:

```text
predict A if p_A > 0.5
predict B if p_B > 0.5
predict C otherwise
```

This is not argmax. For example, `[0.45, 0.45, 0.10]` predicts C because neither A nor B
has a strict majority, even though C is not the largest component.

On the frozen A/B validation and test benchmark, every C prediction is counted as
incorrect. Report:

- accuracy and macro F1 against the frozen V3 binary label;
- C prediction rate;
- soft-target cross-entropy and Brier score against the empirical distribution;
- reliability by predicted-confidence band; and
- each metric by assay concept, endpoint, metric type, and Tanimoto bucket.

No validation-fitted temperature or metric-specific recalibration is part of V4.

## 9. Reproducibility and implementation order

1. Version the target policy and A/B/C template.
2. Extend selection so all eligible training candidates, including null binary labels,
   are retained while validation and test remain frozen.
3. Render counts, fractions, target distribution, and modal completion deterministically.
4. Add the three-choice soft loss and prohibit fallback to hard-completion CE.
5. Add strict-majority prediction and the required evaluation metrics.
6. Publish a new immutable HF dataset version without mutating V3 artifacts.

Every manifest must bind the source candidate hashes, target policy, template, schema,
row counts by split and binary-label status, completion counts, and output hashes.

## 10. Historical binary-V4 names

The historical `assay_transfer_binary_v4` and `assay_transfer_binary_v4_fg_v3` tracked
configs are retained under `binary_v4_legacy` and `binary_v4_fg_v3_legacy` paths. Their
internal build identifiers remain unchanged so existing generated datasets and manifests
remain reproducible.

The active V4 build identifier is `assay_transfer_soft_evidence_v4`. Generated datasets,
tables, manifests, and published artifacts with historical binary-V4 identifiers are not
renamed in place.

## 11. Acceptance checks

- Every target component is finite, within `[0, 1]`, and sums to one within float32
  tolerance.
- Fractions reconstruct exactly from stored counts before float casting.
- Unique maxima serialize to their matching completion; every maximum tie serializes C.
- No hard-completion loss is applied during V4 training.
- All eligible no-majority rows occur only in training.
- Validation and test candidate IDs and binary labels match the frozen V3 benchmark.
- Strict-majority prediction uses `> 0.5`, not argmax or `>= 0.5`.
- C predictions are counted as incorrect in binary evaluation.
- Prompt leakage, molecule isolation, schema, manifest, and deterministic-hash tests pass.
- No touched function exceeds 60 lines.
