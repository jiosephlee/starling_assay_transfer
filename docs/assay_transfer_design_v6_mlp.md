# Assay Transfer Design V6.5: Cached-Embedding 100M MLP

Status: active design contract  
Last updated: 2026-07-20

## 1. Goal and data contract

The canonical V6.5 MLP experiment is a direct soft-A/B predictor over the immutable V6.5
raw record-pair dataset. The current build uses 138,787 records, 100,000,000 training
pairs, and the ordinary and ranking
benchmarks defined in
[assay_transfer_design_v6_5_intern.md](assay_transfer_design_v6_5_intern.md). Dataset
membership, pair direction, continuous targets, hidden query values, condition keys, and
held-out molecule splits do not change when encoders or fusion features change.

The completed 20M-row ablation compared three fusion representations while holding the
training examples, optimizer schedule, depth, and parameter count effectively constant.
It selected `difference_product`; the 100M-row production rerun trains that winner from
scratch rather than repeating architecture selection. It does not train a contrastive
retriever and does not add ListNet to the MLP loss.

## 2. Frozen offline embeddings

Preprocessing computes every pretrained embedding once:

```text
MoLFormer(canonical SMILES) -> molecule embedding [768]
PubMedBERT(value-free assay paragraph) -> assay embedding [768]
```

MoLFormer is pinned to revision
`7b12d946c181a37f6012b9dc3b002275de070314`. Complete SMILES are passed without
truncation and `pooler_output` is cached. The implementation's rotary cache expands beyond
the configured 202 positions; the current universe reaches 736 tokens. Molecules exceeding
the pretraining cutoff are retained and audited as length extrapolation.

PubMedBERT is pinned to revision
`b79526d6ef3645e0df4530322e266f24c829f5ef`. It uses attention-mask-aware mean pooling and
a 256-token contract; every current assay paragraph fits.

V6.5 is a strict 19-record subset of the original 138,806-record cache universe. The
14,982-molecule embeddings are therefore reused byte-for-byte, while retained assay
embeddings are reindexed by stable record ID only after verifying identical SMILES, assay
text, molecule indices, and concepts. Any added or changed encoder input forces a full
rebuild.

The immutable float16 embedding cache contains molecule embeddings, per-record assay embeddings,
record-to-molecule and record-to-concept indices, token lengths, and a manifest. The
manifest pins source and output hashes, model revisions, pooling, tokenization, dtype,
shape, and environment versions. Building never overwrites an existing cache.

At training startup, the roughly 225 MiB embedding cache is validated, loaded onto the
selected GPU once, and converted to BF16. The 100M pair arrays stay memory-mapped on the CPU;
only batch indices, targets, and normalized retrieval values are gathered. The trainer
does not instantiate MoLFormer, PubMedBERT, or any tokenizer.

## 3. Eight-block fusion architecture

The molecule adapter is shared between query and retrieval molecules. A separate assay
adapter is shared between query and retrieval assay descriptions:

```text
LayerNorm(768) -> Linear(768,1024) -> SiLU -> Dropout(0.1)
-> Linear(1024,512) -> LayerNorm(512)
```

Only the normalized known retrieval value is appended. The query value is hidden.
`retrieval_is_approximate` is deliberately ignored because it conflates lexical
approximation with reported variation; the immutable dataset column remains untouched.

The ablation modes are:

```text
concat:
  [Mq, Mr, Aq, Ar, value_r]                              # 2,049

difference:
  [Mq, Mr, |Mq-Mr|, Aq, Ar, |Aq-Ar|, value_r]           # 3,073

difference_product:
  [Mq, Mr, |Mq-Mr|, Mq*Mr,
   Aq, Ar, |Aq-Ar|, Aq*Ar, value_r]                      # 4,097
```

Each input is normalized and projected to width 1,024, followed by exactly eight pre-norm
residual SwiGLU blocks. Each block uses dropout 0.1 and LayerScale initialized to `1e-4`.
FFN sizes compensate for different input-projection sizes:

| fusion mode | FFN size | trainable parameters |
|---|---:|---:|
| `concat` | 3,872 | 99,990,020 |
| `difference` | 3,824 | 99,860,228 |
| `difference_product` | 3,792 | 100,123,908 |

The head emits `logit_A` and `logit_B`. Training minimizes full soft A/B cross-entropy:

```text
L = -target_a * log softmax(logits)_A
    -(1-target_a) * log softmax(logits)_B
```

## 4. Ablation and training protocol

A seed-4878 array preassigns all 5,000 optimizer updates, with 128 groups of 40 pairs per
update. Every fusion mode reads the same group indices in the same order. Microbatching may
change memory use but not the effective 5,120-row update.

All runs use AdamW, learning rate `1e-4`, weight decay 0.01, 250-step warmup, a cosine
schedule with a fixed 5,000-update horizon, BF16, and gradient clipping at 1.0. Each mode
stops at update 1,000. Ordinary validation and a fixed, seed-42, concept-stratified subset
of 100 complete `validation_ranking` queries run at step 0, every 50 updates, and at the
end. Each mode retains its best checkpoint by overall NDCG@5, then NDCG@10. The fusion
winner uses those same metrics and then fixed mode order. Test data cannot select a
checkpoint or architecture.

The winner resumes from its exact update-1,000 model, optimizer, scheduler, and RNG state
and continues to update 5,000. Its best-ranking state is preserved across resume, so the
final selected checkpoint can come from any scheduled evaluation in the complete
0-to-5,000 trajectory. Full ranking splits and test artifacts are opened only after this
checkpoint is frozen.

The 100M-row production rerun instead uses seed 4878 to permute all 2,500,000
four-candidate groups once, without replacement. An optimizer update contains 128 complete
groups (5,120 rows). Only complete updates are retained: 19,531 updates consume 2,499,968
groups, or 99,998,720 rows, and the final 32 groups (1,280 rows) are omitted. The selected
`difference_product` model starts from fresh weights and uses the same AdamW, BF16,
clipping, 250-step warmup, and validation protocol, with the cosine schedule set to the
19,531-update horizon. Runtime microbatching does not alter the preassigned group order or
optimizer-update boundaries.

## 5. Evaluation and logging

For ordinary decisive rows, `p_transfer = softmax(logits)_A` and the required metrics are
accuracy, macro-F1, transfer precision, parse rate (always 1.0 for the MLP), and
`mean(abs(p_transfer-target_a))`.

For ranking, the score is `logit_A-logit_B`. Each 20-candidate query reports tie-aware
Spearman against `target_z`, NDCG@5 and NDCG@10 with linear `target_a` gain and logarithmic discount,
top-1 membership in the maximum-`target_z` tie set, and top-10 containment of that set.
Constant-rank Spearman values are NaN and excluded from the macro mean. Metrics are
macro-averaged over queries, never across condition keys.

Every split contains `overall` and the five assay-concept sections. Ranking sections also
contain query counts. W&B uses project `assay-transfer-soft`, group
`assay-transfer-raw-pair-v6-5-soft`, and the Intern-compatible keys:

```text
eval/validation/overall/binary_<metric>
eval/<split>/assay_concept/<concept>/<metric>
eval/ranking_validation/overall/<metric>
```

W&B is restricted to each model's training-time loss, ordinary validation, and fixed
100-query ranking validation. Fusion selection and post-training full validation/test
metrics are never sent to W&B. They are written only to local JSON with the human-facing
metric names and the same overall/concept structure. After selection, the local evaluation
order is ordinary validation, `validation_ranking`, ordinary test, then `test_ranking`.

## 6. Verification and legacy baselines

Verification covers cache hashes and shapes, source-index alignment, full-length SMILES,
query-value absence, zero encoder calls during training, fusion dimensions, exact parameter
counts, gradients, shared schedules, resume equivalence, selection isolation, metric ties,
W&B namespaces, BF16 CUDA forward/backward, and the repository 60-line function limit.

The earlier V6 Morgan-1024 plus assay character-hash models remain registered as lightweight
legacy baselines: direct soft-only, direct soft-plus-ListNet, and graded contrastive. Their
artifacts and results are not overwritten or reinterpreted as this experiment.

## 7. 100M-row reference execution

The seed-4878 production run completed 19,531 updates on 2026-07-20. The fixed-100-query
ranking selector chose step 2,350 (NDCG@5 0.671868; NDCG@10 0.725055). Its local-only test
results were macro-F1 0.508474, soft MAE 0.347727, Spearman 0.359126, and ranking NDCG@10
0.656241. The ordinary-validation macro-F1 curve itself peaked later, at 0.584887 on step
14,450; that checkpoint was not substituted because macro-F1 is not the frozen selection
metric.

This run did not improve the selected 20M-row checkpoint, whose test macro-F1 was 0.517187
and test-ranking NDCG@10 was 0.672515. The result is retained rather than cherry-picked;
follow-up changes to schedule or selection are new experiments.
