# Assay-transfer V12/V12.1 dataset contract

Status: implemented dataset-construction contract  
Last updated: 2026-08-15

V12 corrects the benchmark lineage and removes held-out values from target calibration. V11 and
V11.1 remain frozen historical releases; none of their artifacts are rewritten.

## File-system architecture

The implementation is deliberately split at the ownership boundary:

- TxAgent owns normalized evidence Stage 03/04 and the global Stage 05 measurement SD.
- `pipeline/v12_source.py` owns assay-transfer sampling, molecule splits, bucket eligibility,
  train-only value CDFs, and the train-only V12.1 distance CDF.
- `pipeline/v12_ranking.py` owns validation/test ranking-list construction.
- `scripts/build_v12_release.py` materializes task-local V12 releases.
- `scripts/build_v12_1_from_defined_split.py` changes targets while preserving V12 membership.
- `scripts/build_v12_composed_release.py` concatenates the three task releases without changing
  task-local row order.
- `scripts/verify_v12_release.py` checks lineage hashes, split isolation, calibration counts,
  training degree caps, and ranking topology.

The executable registry is `configs/assay_transfer/v12/prompt_projection.json`. It inherits the
frozen V11 source-to-prompt bindings but replaces lineage, split, calibration, and ranking policy.

## TxAgent lineage

The gold label files used only to identify molecule universes are:

| Task | Gold lineage |
|---|---|
| BBB | `processed_starling_experimental_meaningful_cns_access_v2/BBB_Martins/scaffold` |
| Bioavailability | `processed_starling_record_supported_v2/Bioavailability_Ma/scaffold` |
| Skin Reaction | `processed_starling_record_supported_v2/Skin_Reaction/scaffold` |

Every evidence molecule is normalized with TxAgent `rdkit_fragment_parent.v1`. All records with
the same normalized-parent identity receive the same dataset split. This is described as
“molecule-disjoint splits, grouped by TxAgent normalized-parent identity.”
Records for which TxAgent cannot produce either a parent InChIKey or parent SMILES are rejected
before splitting and counted as `parent_identity_unavailable` in the source manifest.

The test parent universe is the eligible evidence intersection with the gold valid+test parent
union. Validation is a deterministic hash sample from eligible gold-train parents, with target
size `floor(test_parent_count / 2)`. Remaining parents are train. The split manifest freezes the
three parent sets, seed namespace, input paths, and hashes before model training.

## Train-only calibration order

The order is normative:

1. Read TxAgent normalized Stage 03/04 evidence and globally valid Stage 05 measurement metadata.
2. Apply configured source-record sampling.
3. Assign normalized parents to train, validation, or test.
4. Count records in train per exact pair bucket.
5. Retain a bucket only when it has at least 25 train records; V12 does not reuse a global
   value-distribution gate for training eligibility.
6. Fit continuous value CDFs and ordinal category CDFs from train records only.
7. Project train/validation/test records through the frozen train CDF.
8. Fit V12.1's percentile-distance CDF from all unordered distinct-record train pairs.

Same-parent record pairs are included in step 8. Held-out values cannot affect source sampling,
bucket eligibility, CDF support, CDF counts, or targets. TxAgent's global Stage 05 sample SD stays
global and is copied unchanged, including null values, because it is measurement/ranking-error
metadata; it is not rendered in prompts and does not affect eligibility, targets, or loss.

## Pair construction

Training pairs must share task, source, measurement kind, endpoint, scale, and exact pair bucket.
Self-pairs and duplicate directed pairs are forbidden. Distinct records may share a normalized
parent or even the same canonical SMILES. Every row includes:

```text
query_parent_identity_key
retrieval_parent_identity_key
pair_relationship
```

Continuous and categorical components are each capped at 1.25 million pairs. The independent
query/retrieval record degree caps remain 6/6: one record can appear at most six times in each
role, or twelve emitted rows total if it reaches both caps. These are record caps, not molecule
caps.

## Ranking evaluation

V12 emits `validation_ranking` and `test_ranking` only; it does not emit pointwise held-out
evaluation. Query records come from the corresponding held-out molecule split. All 24 candidates
come from train records in the exact same pair bucket.

Every complete list contains at least two normalized retrieval parents, so a bucket supported by
only one train molecule cannot enter ranking evaluation. Continuous lists retain deterministic
value-spanning selection. Categorical candidates start from a deterministic hash shuffle and are
selected without replacement, followed only by deterministic minimal swaps needed to ensure:

- at least three candidates with the query category;
- at least one category mismatch; and
- at least two retrieval parents.

There is no adversarial or exact category composition. The old maximum-two-candidates-per-molecule
rule is removed. Ranking record degree remains capped at 48. Structural quota shortfalls are
recorded and never backfilled using test outcomes. Continuous ranking requires a positive global
Stage 05 SD because that statistic is needed by the regression error metric; this measurement-only
requirement does not remove the bucket from training.

## Release and model-use protocol

Each task and the composed release contain:

```text
train/data.parquet
validation_ranking/data.parquet
test_ranking/data.parquet
calibration/
manifest.json
README.md
```

V12.1 preserves the exact ordered V12 train pair IDs, ranking query IDs, member indices, prompts,
and splits. Only target-derived fields and its additional calibration artifact differ.

Model checkpoints are selected using `validation_ranking` only. `test_ranking` is run after the
checkpoint and retrieval configuration are frozen. A V12/V12.1 model can then rank analogous
assay records for GLM context assembly, but the GLM benchmark must retain its own held-out-parent
filter at retrieval time. Claim-dependent handling is explicitly outside this change.

## Commands

Build a task source and V12 release:

```bash
python -m scripts.build_v12_release \
  --task bioavailability_ma \
  --rebuild-source
```

Build V12.1 over the frozen V12 release:

```bash
python -m scripts.build_v12_1_from_defined_split \
  --task bioavailability_ma \
  --reference-release datasets/hf_parquet/assay_transfer_raw_pair_v12/bioavailability_ma/with_categorical
```

Verify either release (supply `--reference-v12` for V12.1):

```bash
python -m scripts.verify_v12_release \
  --root <release-root> \
  --source datasets/eligible/assay_transfer_starling_txagent_v12/<task>/records.parquet
```
