# CDF soft-target semantics

Status: active V11/V11.1 target contract
Last updated: 2026-08-13

## Meaning of the target

The A/B soft target is a graded value-equivalence score. It asks whether two measurements in
the same pair bucket are effectively the same enough to transfer. Although the normalized A/B
values are commonly called probabilities, they do not estimate a population frequency of
successful transfer events.

## V11: individual-value CDF separation

For continuous values, V11 maps each value to its exact empirical midrank percentile within the
pair bucket:

```text
u_query     = F_X(value_query)
u_retrieval = F_X(value_retrieval)
delta       = abs(u_query - u_retrieval)
target_a    = clip(1 - delta, 1e-6, 1 - 1e-6)
target_b    = 1 - target_a
```

Ordinal categories use the same construction with ordered-category midrank coordinates. This
geometry is location-sensitive: equal raw differences can receive different targets when they
span different amounts of the bucket's observed value distribution.

Binary targets are categorical rather than prevalence-calibrated: matching categories receive
`target_a = 0.95`, and mismatches receive `target_a = 0.05`.

## V11.1: percentile-distance CDF

V11.1 retains V11's value-percentile coordinates, then converts their distance through a second
within-bucket empirical CDF. Let `D` be the reference distribution of cross-molecule percentile
distances and let its empirical midrank be:

```text
M(d) = (count(D < d) + 0.5 * count(D = d)) / count(D)
```

The endpoint-anchored distance percentile is:

```text
R_D(d) = (M(d) - M(d_min)) / (M(d_max) - M(d_min))
target_a = clip(1 - R_D(delta), 1e-6, 1 - 1e-6)
target_b = 1 - target_a
```

Thus the minimum reference distance maps to near-certain equivalence, the maximum maps to
near-certain non-equivalence, and intermediate scores express how small the location-sensitive
distance is relative to admissible pairs in that bucket. Binary targets retain V11's 0.95/0.05
agreement rule.

The reference CDF uses the full eligible bucket before scaffold reservation or source sampling.
It enumerates all unordered cross-molecule pairs through 100,000 pairs and otherwise uses a
deterministic 100,000-pair cross-molecule sample. The calibration artifact freezes the support,
counts, fitting scope, tie convention, and sampling contract.

### Frozen V11 membership

The canonical V11.1 release is a controlled target-only rebuild. It preserves the exact ordered
V11 train pairs, train component boundaries, ranking pairs, ranking groups, member indices, and
ranking families. Only target-derived fields are recomputed: the A/B targets and distributions,
completion, decisiveness, target metadata, and percentile-distance audit fields. This prevents
the nonlinear second CDF from changing the examples selected for either split.

## Output contract

`target_a` and `target_b` are the complete scientific soft target. V11 and V11.1 do not require
or emit a separate `target_z`; ranking selection and evaluation use `target_a` directly. Public
V11 artifacts created before this cleanup retain a historical `target_z` column but are not
rebuilt or republished.
