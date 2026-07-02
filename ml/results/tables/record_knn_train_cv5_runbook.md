# Record KNN Train 5-Fold CV Runbook

## Current State

- The train-CV job was canceled on request.
- No final TSVs were modified by the canceled run.
- Six partial row JSONs were preserved in:
  - `ml/results/tables/eval_tracking_rerun_v1_train_cv_rows`
- The stable runner lives at:
  - `ml/scripts/run_record_knn_train_cv5.py`
- The canceled temp helper also remains at:
  - `tmp/record_knn_train_cv_work/train_cv.py`
- The five prebuilt train-fold candidate caches are still available in:
  - `ml/artifacts/record_knn_eval_cache/condition_key_v3_record_splits_hf_train_cv5`

## What This Computes

- Five-fold CV within the training records from `condition_key_v3_record_splits_hf`.
- Dataset config is `full_metadata`.
- For each fold, the held-out train fold is used as queries and the remaining train folds are sources.
- Retrieval uses the prebuilt weighted-Tanimoto top-10% candidate cache.
- The checkpoint model reranks those cached candidates live.
- Metrics are averaged across the five folds:
  - `train_cv5_macro_f1`
  - `train_cv5_accuracy`

The `universe` column is the training/data-selection regime for the checkpoint, not a different evaluation dataset. All universes are evaluated against the same condition-key-derived train-fold records.

## Checkpoints Covered

The driver evaluates 24 tasks:

- 12 final runs at `best_val_macro_f1`
- 12 final runs at `best_record_knn_validation_1_macro_f1`

The manager skips any row JSON that already exists, so the preserved six rows can be reused when resuming.

## Resume Command

From the repo root:

```sh
PYTHONPATH=ml /data1/joseph/miniconda3/envs/openrlhf/bin/python ml/scripts/run_record_knn_train_cv5.py
```

Expected behavior:

- Uses GPUs `0` through `7` by default.
- Writes per-task logs to `tmp/record_knn_train_cv_work/logs`.
- Writes per-task durable JSON rows to `ml/results/tables/eval_tracking_rerun_v1_train_cv_rows`.
- After all 24 row JSONs exist, writes CV-only TSVs and appends the two CV columns to the main metrics TSVs.

To use a smaller GPU set:

```sh
PYTHONPATH=ml /data1/joseph/miniconda3/envs/openrlhf/bin/python \
  ml/scripts/run_record_knn_train_cv5.py --gpu-ids 0,1
```

## Non-Compute Commands

List all 24 tasks and show which row JSONs already exist:

```sh
PYTHONPATH=ml /data1/joseph/miniconda3/envs/openrlhf/bin/python \
  ml/scripts/run_record_knn_train_cv5.py --dry-run
```

Validate existing row JSONs against the expected task identities and metric schema:

```sh
PYTHONPATH=ml /data1/joseph/miniconda3/envs/openrlhf/bin/python \
  ml/scripts/run_record_knn_train_cv5.py --validate-rows
```

Once all 24 row JSONs exist, write the CV-only TSVs and append the CV columns without launching workers:

```sh
PYTHONPATH=ml /data1/joseph/miniconda3/envs/openrlhf/bin/python \
  ml/scripts/run_record_knn_train_cv5.py --finalize-only
```

`--finalize-only` also validates the row JSONs before writing any TSVs.

## Fresh Rerun

If you want to recompute all 24 rows instead of resuming from the six preserved rows, move the existing row JSONs aside first:

```sh
mkdir -p ml/results/tables/eval_tracking_rerun_v1_train_cv_rows_canceled_snapshot
mv ml/results/tables/eval_tracking_rerun_v1_train_cv_rows/*.json \
  ml/results/tables/eval_tracking_rerun_v1_train_cv_rows_canceled_snapshot/
```

Then run the resume command above.

## Expected Outputs

CV-only TSVs:

- `ml/results/tables/eval_tracking_rerun_v1_best_val_macro_f1_checkpoint_train_cv5_record_knn.tsv`
- `ml/results/tables/eval_tracking_rerun_v1_best_record_knn_val1_checkpoint_train_cv5_record_knn.tsv`

Main TSVs with appended columns:

- `ml/results/tables/eval_tracking_rerun_v1_best_val_macro_f1_checkpoint_metrics.tsv`
- `ml/results/tables/eval_tracking_rerun_v1_best_record_knn_val1_checkpoint_metrics.tsv`

The appended columns should be:

- `train_cv5_macro_f1`
- `train_cv5_accuracy`

## Monitoring

Count completed row JSONs:

```sh
find ml/results/tables/eval_tracking_rerun_v1_train_cv_rows -type f | wc -l
```

Check active workers:

```sh
pgrep -af 'run_record_knn_train_cv5.py'
```

Check GPU usage:

```sh
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

Check fold progress:

```sh
for f in tmp/record_knn_train_cv_work/logs/*.log; do
  printf '%s\t' "$(basename "$f")"
  rg -c '^\[score\]' "$f"
done | sort
```

Scan for failures:

```sh
rg -n "Traceback|RuntimeError|CUDA out of memory|Error|failed" \
  tmp/record_knn_train_cv_work/logs \
  ml/results/tables/eval_tracking_rerun_v1_train_cv_rows
```

## Verification

After completion:

```sh
find ml/results/tables/eval_tracking_rerun_v1_train_cv_rows -type f | wc -l
wc -l ml/results/tables/eval_tracking_rerun_v1_*_train_cv5_record_knn.tsv
head -n 1 ml/results/tables/eval_tracking_rerun_v1_best_val_macro_f1_checkpoint_metrics.tsv
head -n 1 ml/results/tables/eval_tracking_rerun_v1_best_record_knn_val1_checkpoint_metrics.tsv
```

Expected:

- 24 row JSONs.
- Each CV-only TSV has 13 lines: one header plus 12 runs.
- Both main TSV headers include `train_cv5_macro_f1` and `train_cv5_accuracy`.
