# Assay Transfer Design V6: Intern Record-Ranking Contract

Status: active design contract
Last updated: 2026-07-19

## 1. Goal

Intern V6 ranks retrieved molecule-and-assay records for a query molecule. The model is
trained on raw record-to-record comparisons and never uses an aggregate query value.

The model-facing estimand is:

```text
Will the retrieved value transfer from the retrieval molecule and source assay setting
to the query molecule under the stated target assay setting?
```

During training and evaluation, the target assay setting comes from a real query record.
Deployment-time context substitution is outside this dataset and trainer build contract.

## 2. Raw directed-pair universe

For every split, a directed pair `(q, r)` is scientifically eligible when:

- query record `q` and retrieval record `r` have the same condition key;
- their canonical molecules differ; and
- both values are valid under the same versioned metric policy.

The prompt exposes the query molecule and assay metadata but never its value. It exposes
the retrieval molecule, assay metadata, and value. The target uses the normalized distance
between the two raw record values. Exactly one direction of each eligible unordered pair
is admitted by a SHA-256 coin flip over sorted globally unique record IDs; the reverse
direction is never emitted.

No query-record mean, molecule-level marginal distribution, or other query aggregation is
used. Multiple raw query records remain distinct because their assay contexts and hidden
values are distinct observations.

For condition key `c`, let `V_c` be its record count and `n_mc` the number of records for
molecule `m`. The exhaustive self-excluded unordered-pair count is:

```text
C_c = (V_c^2 - sum_m(n_mc^2)) / 2
```

The condition key remains a scientific firewall. V6 does not construct labels, negatives,
ListNet lists, or ranking metrics across condition keys.

## 3. Continuous A/B target

For an eligible pair, transform both raw values with the metric-specific versioned
transformation and calculate distance:

```text
d_qr = abs(g_c(value_r) - g_c(value_q))
z_qr = h_c(d_qr)
q_A = clip(sigmoid(z_qr), 0.1, 0.9)
q_B = 1 - q_A
```

`h_c` is a frozen monotone-decreasing normalization derived from the scientific transfer
and non-transfer boundaries for the metric family. The exact `g_c` and `h_c` tables belong
to the target-policy artifact and must be frozen before building V6. Fractions are not used
as logits, evidence count does not weight the loss, and the 0.1/0.9 cap prevents unjustified
certainty.

Each row stores the pre-sigmoid `z_qr`, normalized distance, capped A/B distribution, raw
policy versions, and enough record identifiers to reconstruct the target. Query values may
exist in a private audit artifact but must not enter model-facing columns or prompts.

## 4. Intern prompt contract

The prompt contains two explicitly separate context blocks:

1. **Retrieval record:** retrieval SMILES, known value, and real retrieval assay metadata.
2. **Target query:** query SMILES and real query-record assay metadata, with its value hidden.

The question asks whether the retrieval value transfers to the target query under the
target block's setting. Endpoint, units, missingness, and every model-facing context field
must be rendered deterministically. Metadata fields that directly reveal the hidden query
value are prohibited.

The artifact always renders each side's real source-record context. How deployment builds
a target block without a measured query record is deliberately out of scope.

## 5. Intern losses

Every document retains full-vocabulary soft A/B cross-entropy at its decision token and
weighted formatting/EOS cross-entropy elsewhere:

```text
L_soft_AB = -q_A * log P_full_vocab(A) - q_B * log P_full_vocab(B)
```

ListNet is computed over four retrieval candidates sharing one raw query-record anchor and
condition key:

```text
s_i = logit_A_i - logit_B_i
t_i = softmax(z_qri / target_temperature)
p_i = softmax(s_i / model_temperature)
L_listnet = -sum_i(t_i * log(p_i))
L_total = L_soft_AB + lambda_listnet * L_listnet + 0.1 * L_format
```

The candidates must be distinct records from molecules other than the query molecule.
Lists span the available relevance range among 16 deterministic unused proposals.
`lambda_listnet` is 0.1 and both target and model temperatures default to 1.0.

RankNet and embedding-level contrastive losses are not part of the default Intern V6
objective. They remain controlled ablations.

## 6. Offline list construction and packing

List membership is constructed before training. The final Intern chat template and
tokenizer are applied before packing so all length decisions use exact training tokens.

Each four-document ListNet group is indivisible and must fit within one 4,096-token BFD
packed sequence. Oversized groups are deterministically resampled rather than truncated or
split. Complete groups may share unused space in a packed sequence. Document-reset
`position_ids`, padding-free global decision offsets, full-vocabulary decision gradients,
Liger formatting CE, and PEFT fallbacks remain required.

The current effective batch is 256 packed sequences per optimizer update. Every document
stores:

- `query_record_id` and `retrieval_record_id`;
- `canonical_endpoint_key` and stable directed `pair_id`;
- `listnet_query_group_id`, `listnet_group_id`, and `listnet_member_index`;
- `target_z` and `templated_token_count`;
- `packed_chunk_index`; and
- `optimizer_batch_index` and `optimizer_batch_position`.

Every four-member group must be complete within one packed sequence. Sharing only an
optimizer batch is insufficient because gradient-accumulation microsteps and distributed
ranks do not share one computation graph. Runtime shuffling or repacking must preserve the
offline assignments.

The pinned build retains 1,998,840 documents in 249,856 chunks after dropping the final
incomplete optimizer batch. This yields 976 complete optimizer batches of 256 chunks.

## 7. Frozen 20,000-comparison ranking benchmarks

Each of validation and test contains:

```text
X = 1,000 distinct raw query-record anchors
K = 20 retrieval-record candidates per anchor
X * K = 20,000 model comparisons
```

Each anchor must have at least 20 eligible other-molecule records in its condition key.
Candidates are selected by a deterministic hash probe that does not inspect relevance or
model scores. Query anchors, candidate IDs, and hashes are frozen before evaluation. Test
records are never used to tune selection or hyperparameters.

Every ranking list is anchored by one raw query record. Ground-truth order is descending
continuous relevance `z_qr`; model order is descending A/B logit margin. No list mixes
condition keys.

Primary metrics are macro NDCG@10, top-1 regret, best-in-top-10 regret, and mean top-10
relevance. Spearman correlation and the existing binary accuracy, macro-F1, soft NLL,
Brier score, and reliability metrics are secondary. Metrics are averaged per query record
and reported by condition-key and assay-concept slices; those slice summaries do not create
cross-condition preference labels.

## 8. Verification requirements

- Every pair and list is same-condition and different-molecule.
- Every target is finite, monotone in normalized distance, and sums to one.
- No query value appears in a prompt or model-facing feature.
- Every frozen evaluation anchor has exactly 20 distinct eligible candidates.
- Each ranking benchmark contains exactly 20,000 comparisons and rebuilds deterministically.
- Every ListNet group is complete within one 4,096-token packed sequence.
- V4 and V5 artifacts, prompts, memberships, and trainers remain unchanged.
