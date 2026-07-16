# Assay Transfer Version 4: Deferred Continuous-Target Design

Status: deferred research and design outline  
Last updated: 2026-07-15

## 1. Purpose

This document preserves the requirements and unresolved decisions for a future
continuous assay-transfer target. It is deliberately not an implementation plan or a
frozen target policy.

Version 3 establishes a binary-first dataset and benchmark. Version 4 may add richer
continuous supervision only after a common, scientifically defensible target is chosen
for all supported metric types.

## 2. Desired outcome

The future model should express transferability on an interpretable continuous scale,
ideally:

```text
0.00 to 100.00 transfer percent
```

The scale should have consistent semantics across bounded percentages, bounded
fractions, log-fold distances, ratios, and other supported metric types. A prediction
should be serializable to the hundredth place and should support an interpretable
binary evaluation without replacing the version 3 benchmark prematurely.

The intended future workflow is:

```text
retrieval record + query molecule + K
    -> continuous transfer prediction
    -> frozen conversion or calibration rule
    -> binary prediction
    -> comparison with the strict-majority binary benchmark
```

## 3. Why continuous prediction is deferred

### 3.1 Raw distances are not comparable

Current metric distances live on different numerical scales:

| Metric family | Example distance scale |
|---|---|
| Bounded percentage | percentage points, commonly `0–100` |
| Bounded fraction | absolute difference on `0–1` |
| Positive continuous | absolute `log10` difference |
| Positive ratio | absolute log-ratio difference |
| Categorical | discrete equality or separation |

A shared regression loss on raw distances would give larger-scale metric families more
influence and would make one predicted number mean different things across endpoints.

### 3.2 Mean distance does not determine the binary label

The version 3 binary label is a strict majority over record-level votes. Mean distance
does not retain the full vote distribution.

For transfer/non-transfer thresholds of `10/30`, these evidence distributions have the
same mean distance but opposite hard labels:

| Record distances | Mean | Strict-majority label |
|---|---:|---|
| `[0, 0, 60]` | `20` | transfer |
| `[0, 30, 30]` | `20` | non-transfer |

Therefore no threshold on mean distance can reproduce every version 3 label.

### 3.3 One transfer percentage may hide ambiguity

The most direct percentage is:

```text
p_transfer = 100 * n_transfer / N
```

Its complement is not necessarily non-transfer because ambiguous evidence is retained:

```text
transfer:      60%
non-transfer:  10%
ambiguous:     30%
```

Reducing these three values to one scalar loses information. A low transfer percentage
could mean strong non-transfer evidence or mostly ambiguous evidence.

### 3.4 Numerical LM generation is not automatically regression

A causal LM trained to emit `"72.43"` still optimizes token cross-entropy. Number
tokenization, formatting failures, and rounding can make nearby numerical targets behave
like unrelated strings. A true scalar head with a numerical loss has different behavior
and must be compared explicitly.

## 4. Requirements for an acceptable continuous target

Any version 4 target must satisfy or explicitly reject each requirement below.

### 4.1 Common semantics

The same numerical value must convey the same transfer meaning across metric types.
For example, `80.00` should not mean strong transfer for one metric and weak transfer
for another.

### 4.2 Bounded and interpretable range

The preferred public scale is:

```text
0.00   -> strongest non-transfer evidence
100.00 -> strongest transfer evidence
```

The behavior of ambiguous, censored, missing, or conflicting evidence must be explicit.

### 4.3 Monotonicity

For a fixed metric and retrieval record, increasing assay-value disagreement must not
increase the record's transfer score.

### 4.4 Metric-policy grounding

The transformation must use frozen metric metadata, including:

- metric type;
- canonical transformation and units;
- transfer threshold;
- non-transfer threshold; and
- interval/censoring behavior.

Ad hoc endpoint-specific rescaling is prohibited unless it creates a new, scientifically
defined metric subtype and policy version.

### 4.5 Evidence-distribution awareness

The target must specify how multiple records for one query molecule contribute and how
opposing, ambiguous, and censored evidence affect the result.

### 4.6 Relationship to binary truth

The design must state whether the continuous output:

1. reproduces the strict-majority binary label exactly;
2. approximates it through validation calibration; or
3. represents a distinct scientific estimand evaluated only secondarily against the
   binary benchmark.

This relationship cannot remain implicit.

### 4.7 Reproducibility

The target must be reconstructable from normalized records and versioned policy without
model code. Rounding is a rendering decision and must not destroy the full-precision
canonical target.

## 5. Candidate target families

The following candidates require comparison before version 4 is frozen.

### 5.1 Transfer-vote percentage

```text
score = 100 * n_transfer / N
```

Advantages:

- simple and directly tied to record votes;
- transfer majority corresponds exactly to `score > 50`; and
- naturally bounded on `0–100`.

Limitations:

- a non-transfer majority cannot be inferred from `score <= 50` because ambiguity may
  dominate;
- distance magnitude within a vote region is discarded; and
- one scalar cannot distinguish non-transfer from ambiguity.

### 5.2 Two-probability evidence target

```text
p_transfer    = n_transfer / N
p_nontransfer = n_nontransfer / N
p_ambiguous   = 1 - p_transfer - p_nontransfer
```

Advantages:

- preserves the complete three-state vote distribution;
- reproduces the strict-majority label exactly; and
- remains interpretable as percentages.

Limitations:

- requires two model outputs rather than one;
- is closer to distribution prediction than scalar regression; and
- still discards within-region distance magnitude.

This is the strongest candidate when exact compatibility with version 3 is required.

### 5.3 Metric-normalized expected distance

```text
z = mean_record_distance / T_not_transfer(metric_type)
```

It can be converted to a bounded transfer score through a frozen monotone transform.

Advantages:

- retains continuous distance magnitude;
- normalizes common threshold meaning across metrics; and
- supports de-normalization back to the canonical metric scale.

Limitations:

- does not reproduce the majority label;
- remains sensitive to outliers and multimodal evidence; and
- requires a justified bounded mapping from normalized distance to `0–100`.

### 5.4 Per-record graded transfer followed by averaging

Define a metric-specific monotone function:

```text
s_m(d) in [0, 100]
```

with anchors such as:

```text
d <= T_transfer      -> near 100
d >= T_not_transfer  -> near 0
middle               -> graded interpolation
```

Then aggregate:

```text
score(A -> B | K) = mean_j s_m(d_j)
```

Candidate interpolation functions include linear, logistic, and scientifically defined
piecewise curves.

Advantages:

- produces the desired bounded scalar directly;
- uses every distance; and
- gives metric thresholds an explicit role.

Limitations:

- the interpolation is a new scientific policy requiring justification;
- ambiguous records can change the score in ways that do not match strict majority;
  and
- different plausible curves may yield materially different supervision.

### 5.5 Quantile or robust-distance targets

Median, upper quantile, and worst-case distances may better represent conservative
transfer than the mean.

Advantages:

- reduced sensitivity to outliers for median targets; or
- explicit conservative behavior for high-quantile targets.

Limitations:

- even-sample median conventions complicate exact majority equivalence;
- quantile choice is another policy decision; and
- one summary still loses evidence-distribution shape.

## 6. Recommended research order

Version 4 research should proceed in this order:

1. Treat the version 3 hard labels and full vote summaries as the frozen reference.
2. Measure coverage and joint distributions of `p_transfer`, `p_nontransfer`, ambiguity,
   mean distance, median distance, and metric-normalized distance.
3. Quantify how often the same scalar summary maps to opposite hard labels.
4. Compare one-score and two-probability targets for information loss.
5. Propose candidate metric-to-`0–100` mappings before training any model.
6. Test sensitivity of labels and rankings to mapping choices.
7. Select one primary continuous estimand through scientific review, not downstream
   model performance alone.
8. Freeze the target policy and create a new immutable dataset version.

## 7. Future dataset contract

Version 3 artifacts must preserve enough information for version 4 to add fields such
as:

```text
continuous_target_raw
continuous_target_normalized
transfer_percentage
nontransfer_percentage
ambiguous_percentage
continuous_target_policy_version
```

Not all fields will necessarily become canonical. They illustrate the information that
must remain derivable.

The future HF dataset may retain the binary fields and add a numerical completion:

```json
{
  "prompt": "...",
  "completion": "72.43",
  "continuous_target": 72.43,
  "binary_label": 1,
  "metric_type": "positive_log_continuous",
  "transfer_threshold": 0.301,
  "nontransfer_threshold": 0.699
}
```

The canonical target remains full precision, preferably `float32`. The text completion
is rounded to two decimal places only for LM rendering.

## 8. Future model alternatives

### 8.1 Scalar regression head

Attach a numerical head and optimize Huber, MSE, or another frozen regression loss.
This is the preferred implementation for genuine regression.

### 8.2 Multi-output evidence head

Predict transfer, non-transfer, and ambiguous proportions using two logits plus a
simplex constraint. This best preserves the strict-majority semantics.

### 8.3 Generative numerical completion

Constrain the LM to emit one valid number with exactly two decimal places. Evaluation
must report invalid-format and parsing-failure rates. Token loss and numeric error are
both reported.

### 8.4 Multitask binary and continuous model

Retain the version 3 binary head while adding the frozen version 4 continuous objective.
Loss weights and checkpoint criteria must be selected using validation only.

## 9. Evaluation requirements

A version 4 release must report both continuous and binary behavior.

### 9.1 Continuous metrics

- MAE and RMSE on the canonical continuous scale;
- Spearman correlation;
- calibration or reliability by predicted-score band;
- error by assay concept, metric type, endpoint, and Tanimoto bucket; and
- invalid-output rate for generative numerical models.

### 9.2 Binary interpretation

If a continuous prediction is converted to a binary decision:

1. choose or calibrate the conversion using validation only;
2. freeze it before test;
3. report macro F1 and accuracy against version 3 hard labels; and
4. report abstention coverage if the conversion has a middle region.

Metric-specific calibration is permitted only when validation support is sufficient and
the policy is versioned. Otherwise use a global rule on a normalized common scale.

### 9.3 Information-loss diagnostics

Report cases where identical or nearly identical continuous targets correspond to
opposite binary labels. These are irreducible errors for a one-dimensional conversion
and must not be attributed solely to model quality.

## 10. Promotion gates

Continuous prediction may move from deferred design to implementation only when:

- the 0–100 quantity has a precise mathematical definition;
- ambiguity has an explicit representation;
- cross-metric comparability is demonstrated;
- the relationship to strict-majority labels is documented;
- sensitivity to transformation choice is acceptable;
- a canonical full-precision dtype and two-decimal rendering rule are frozen;
- the training loss and output architecture are selected;
- validation/test conversion rules are preregistered; and
- version 3 binary artifacts remain usable without migration.

Until these gates are satisfied, version 3 binary classification remains the canonical
training and evaluation task.
