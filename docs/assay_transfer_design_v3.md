# Assay Transfer V3: Canonical Base to Binary Hugging Face

Status: implemented design contract  
Last updated: 2026-07-15

## 1. Scope

Version 3 starts from the immutable `canonical_endpoints_v1` base artifacts and ends in
a binary MCQA Hugging Face dataset. Raw-source normalization, continuous targets, and
model/MLP changes are outside this release. The single supported condition profile is
`same_endpoint`.

The nominal release target is 200,000 training, 2,000 validation, and 2,000 test examples.
Every released row has a non-null strict-majority binary label. Sparse high-similarity
strata use the explicit availability policy in section 4 rather than synthetic backfill.

## 2. Base composition

The build validates each base manifest, artifact schema, Parquet hash, and required
canonical columns before composing records. `child_id` is the unique scalar-record
identity. `canonical_smiles` is accepted directly without recanonicalization.

The five sampling concepts are:

| Source records | Assay concept |
|---|---|
| Q1 `oral_bioavailability` family | `oral_bioavailability` |
| Q1 `oral_exposure` family | `oral_exposure` |
| Starling oral bioavailability | `oral_bioavailability` |
| Q2 | `Fa` |
| Q3 | `Fg` |
| Q4 | `Fh` |

Concept is not an eligibility key. Q1 and Starling can cross sources only when their
exact canonical endpoint keys match. Q1 oral-exposure records never compare with
Starling records.

### 2.1 Metric policies

- Bounded percentages use absolute percentage-point distance with 10/30 thresholds.
- Bounded fractions use absolute distance with 0.10/0.30 thresholds.
- Dimensionless ratios use log distance with 1.5-fold/3-fold thresholds.
- Positive scalar measurements use log distance with 2-fold/5-fold thresholds.

Bounded values outside their domain and non-positive log-metric values are rejected with
an explicit audit reason. Accepted approximate central estimates remain eligible; their
variation metadata does not alter the vote.

## 3. Split and supervision unit

Molecules are split globally by canonical SMILES with seed 17 and fixed 70/15/15
train/validation/test proportions. Every record for a molecule follows that assignment,
and both molecules in an example must belong to the same split.

One directed candidate is:

```text
(retrieval scalar child A, query molecule B, canonical endpoint K)
```

The query evidence distribution contains every eligible scalar child for molecule B at
the exact endpoint K. The retrieval molecule cannot equal B.

Each query record independently votes:

```text
distance <= transfer threshold       -> transfer
distance >= non-transfer threshold   -> not transfer
otherwise                            -> ambiguous
```

With `N = transfer + non-transfer + ambiguous`:

```text
label = 1     if transfer     > N / 2
label = 0     if non-transfer > N / 2
label = null  otherwise
```

There is no record deduplication, protocol-level macro averaging, context balancing, or
pre-vote value averaging. Ambiguous and opposing records remain in the denominator.
Null candidates are audited but excluded from every released split.

## 4. Candidate enumeration and selection

Morgan/feature-weighted Tanimoto similarity is a sampling attribute only. Similarity
below 0.4 is `low`; similarity at least 0.4 is `high`.

Candidate queries are deterministically hash-ranked separately inside each similarity
bucket. Initial per-retrieval caps are 4 for oral bioavailability, 4 for oral exposure,
16 for Fa, 128 for Fg, and 48 for Fh. A capacity pass doubles only deficient concept caps
until every stratum has 25% headroom or the query universe is exhausted. Resolved caps
are frozen in the build manifest.

The primary strata are the five assay concepts crossed with low/high similarity. The
standard allocation selects 20,000 training and 200 validation/test rows from each of ten
strata. Fa-high, Fg-high, and Fh-high are frozen training exceptions that take every
available hard-labeled candidate. For Fa-high and Fg-high, validation and test additionally
take the smaller of their two available counts so held-out split sizes match. Fh-high can
meet the normal held-out quota and therefore remains at 200 rows in each held-out split.
The release does not backfill missing rows from other strata.

Deterministic round-robin ordering over query molecules prevents a single high-degree
query from consuming a small quota. Label is reported but is not a sampling stratum.

An underfilled standard stratum blocks the release. A sparse stratum is complete when it
meets its resolved availability target. Splits, endpoints, thresholds, and majority rules
are never relaxed to meet quotas.

## 5. Materialized and HF artifacts

Materialization joins retrieval context through `child_id`. Query-side model input is
only the canonical query SMILES. The prompt may contain the retrieved SMILES, known
value, unit, assay metadata, endpoint description, and public threshold rule. It never
contains query values or metadata, evidence counts, labels, similarity, support text, or
provenance identifiers.

Five concept-specific templates preserve the earlier MCQA format:

```text
(A) transfer
(B) not transfer

Answer:
```

Completion is exactly `A` for label 1 and `B` for label 0.

Each HF split has exactly three top-level columns:

| Column | Arrow type |
|---|---|
| `prompt` | `large_string` |
| `completion` | `string` |
| `metadata` | fixed nested `struct` |

The metadata struct retains endpoint and metric dimensions, retrieval/query identities,
the structured binary label, similarity bucket, evidence audit, normalized retrieval
context, and provenance. These fields are not interpolated into the prompt unless the
template contract explicitly allows them.

## 6. Expansion and reproducibility

The release produces a portable train-expansion archive containing eligible training
records, molecule assignments, selected candidate IDs, policy and template snapshots,
resolved enumeration caps, and the ordering seed. It contains no validation or test
records and can regenerate the remaining frozen training universe without the original
canonical-base directories.

Every stage manifest binds input hashes, schema and policy versions, counts, exclusions,
and output hashes. Repeating a build from identical inputs must reproduce candidate IDs,
selections, prompts, completions, and Parquet hashes.

## 7. Acceptance checks

- Exact resolved per-stratum targets, including matched Fa-high/Fg-high held-out counts.
- No null labels or completions outside `A` and `B`.
- No molecule overlap among splits and no cross-endpoint candidates.
- Cross-source oral-bioavailability comparisons only on shared endpoint keys.
- Strict-majority, deadband, domain, prompt-leakage, and schema tests pass.
- The expansion archive regenerates candidates without the original bases.
- No touched function exceeds 60 lines.
