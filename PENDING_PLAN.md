# Pending Plan: KNN Unification After Dataset Repair

## Summary

Refactor KNN around the canonical record dataset
`datasets/starling_eval/condition_key_v3_record_splits_hf`, after the
full-metadata splits are repaired to be molecule-disjoint. Use
`full_metadata/train` as the source pool and `validation_1`, `validation_2`,
and `test` as query splits. Pair datasets remain for assay-transfer model
training only; active KNN evaluation should not load pair-shaped eval rows.

## Key Changes

- Create and maintain `tmp/knn_unification/progress.md` when this work starts.
- Keep all new and refactored Python functions under 60 lines.
- Remove active pair-based KNN paths:
  - no pair-row auto-detection in active KNN loaders
  - no pair-derived record-eval builder in active KNN workflows
  - no KNN config fields that point to pair split roots
- Split KNN into focused modules:
  - `knn_data.py`: load record splits, normalize split names, validate schema
    and molecule disjointness
  - `knn_retrieval.py`: RDKit Morgan and feature-Morgan top `X%` or `top_n`
    candidate retrieval
  - `knn_scorers.py`: scorer protocol plus Tanimoto and MLP assay-transfer
    scorer implementations
  - `knn_pipeline.py`: evaluate split(s), aggregate votes, compute metrics,
    and write outputs
- Canonicalize split names to `validation_1`, `validation_2`, and `test`.
  Accept `val_1` and `val_2` only at CLI/config boundaries and normalize them
  before cache or result keys are built.
- Keep retrieval, scoring, and aggregation separate:
  - load source/query records
  - retrieve candidates once at the maximum requested candidate count
  - score candidates with the selected scorer
  - aggregate with explicit `k`, vote weighting, and tie policy
- Use split-derived metric and artifact names, for example
  `record_knn_validation_1_macro_f1` and
  `best_record_knn_validation_1_macro_f1`.

## Scorer Interface

The shared scorer contract should expose:

- `name`
- `requires_source_value`
- `validate(dataset, context)`
- `prepare(split)`
- `score_candidates(batch) -> scores aligned to candidate positions`
- `cache_identity()`

The MLP assay-transfer scorer uses source record as `a_idx` and query record as
`b_idx`. It validates checkpoint weight hash, model config hash,
`use_source_value`, `source_value_scale`, embedding manifest hash,
`base_parquet_sha256`, and direction `source_to_query`.

Future LLM, alternate MLP, and no-source-value scorers should plug into the same
interface without changing retrieval.

## Cache Design

Retrieval cache identity should include only retrieval-relevant state:

- dataset id: `condition_key_v3_record_splits_hf`
- config: `full_metadata`
- schema version
- canonical split name
- source/query ordered key-table hashes
- row counts and schema
- RDKit version
- canonicalization policy
- Morgan and feature-Morgan parameters
- similarity weights
- invalid-SMILES policy
- top policy: `top_fraction` or `top_n`
- retrieval backend version

Write retrieval caches as:

- `candidates_<cache_key_sha>.npz`
- `candidates_<cache_key_sha>.json`

Reject stale caches from `condition_key_v3_record_eval` unconditionally. Do not
include embedding manifest identity in RDKit retrieval cache keys; embedding
identity belongs to scorer validation and scorer cache identity.

## Entrypoints And Training Integration

- Replace scalar `record_knn_eval_split` with list config:
  - `record_knn_eval_splits=["validation_1"]`
  - `record_knn_final_splits=["validation_1","validation_2","test"]`
- Add shared helpers:
  - `record_knn_metric_prefix(split)`
  - `ensure_record_knn_caches(...)`
  - `evaluate_record_knn_splits(...)`
- Standalone KNN CLI and `RecordKnnEvalCallback` must call the same pipeline.
- Benchmark finals should prebuild caches serially before distributed
  training/eval.
- Live model selection uses `validation_1`; final reporting evaluates the main
  best checkpoint on `validation_1`, `validation_2`, and `test`.
- TDC KNN can share retrieval/scorer primitives where practical, but its
  external-query embedding construction remains a separate adapter.

## Test Plan

- Validate the repaired `full_metadata` and `smiles_only` split memberships are
  molecule-disjoint before running KNN tests.
- Run `python -m py_compile` on changed Python modules.
- Enforce the 60-line function limit.
- Build retrieval caches for `validation_1`, `validation_2`, and `test`.
- Verify stale pair-derived caches are rejected.
- Smoke test Tanimoto-only KNN on a small `max_queries`.
- Smoke test MLP rerank KNN on `validation_1`.
- Confirm CLI and callback produce matching predictions for the same
  split/model/query subset.
- Confirm no active metrics, cache keys, or result files use `val_1`, `val_2`,
  or `record_knn_val1`.

## Assumptions

- This work starts only after the dataset repair in `PLAN.md` is complete.
- `full_metadata` is the assay-aware KNN dataset.
- `smiles_only` remains for non-assay-aware baselines.
- Pair split artifacts are only for training the assay-transfer model, not for
  KNN evaluation.
