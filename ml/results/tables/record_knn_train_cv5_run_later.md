# Record KNN Train CV5 Run Later

This is the short run-later note for the canceled train-fold record-KNN CV job. The fuller context is in `ml/results/tables/record_knn_train_cv5_runbook.md`.

Last checked: 2026-07-02 01:07 EDT.

## Current State

- The cross-fold job is not running; no `run_record_knn_train_cv5.py` process was found at the latest check.
- Six completed row JSONs are preserved in `ml/results/tables/eval_tracking_rerun_v1_train_cv_rows`.
- The stable runner is `ml/scripts/run_record_knn_train_cv5.py`.
- Fold candidate caches are in `ml/artifacts/record_knn_eval_cache/condition_key_v3_record_splits_hf_train_cv5`.
- Row outputs are skipped if they already exist, so rerunning resumes from the preserved rows.
- No final TSVs need to be regenerated until all 24 row JSONs exist.
- If other training runs are using the GPUs, wait for those to finish or pass a restricted `--gpu-ids` list.

## Resume

From the repo root:

```sh
PYTHONPATH=ml /data1/joseph/miniconda3/envs/openrlhf/bin/python \
  ml/scripts/run_record_knn_train_cv5.py
```

Default behavior:

- Uses all available GPUs unless `--gpu-ids` is provided.
- Writes worker logs under `tmp/record_knn_train_cv_work/logs`.
- Writes durable row JSONs under `ml/results/tables/eval_tracking_rerun_v1_train_cv_rows`.
- Finalizes automatically after all 24 expected row JSONs exist.

To restrict GPUs:

```sh
PYTHONPATH=ml /data1/joseph/miniconda3/envs/openrlhf/bin/python \
  ml/scripts/run_record_knn_train_cv5.py --gpu-ids 0,1
```

## Check Before Running

List expected tasks and existing rows:

```sh
PYTHONPATH=ml /data1/joseph/miniconda3/envs/openrlhf/bin/python \
  ml/scripts/run_record_knn_train_cv5.py --dry-run
```

Validate preserved rows:

```sh
PYTHONPATH=ml /data1/joseph/miniconda3/envs/openrlhf/bin/python \
  ml/scripts/run_record_knn_train_cv5.py --validate-rows
```

## Finalize Without Workers

After all 24 row JSONs exist:

```sh
PYTHONPATH=ml /data1/joseph/miniconda3/envs/openrlhf/bin/python \
  ml/scripts/run_record_knn_train_cv5.py --finalize-only
```

Expected TSVs:

- `ml/results/tables/eval_tracking_rerun_v1_best_val_macro_f1_checkpoint_train_cv5_record_knn.tsv`
- `ml/results/tables/eval_tracking_rerun_v1_best_record_knn_val1_checkpoint_train_cv5_record_knn.tsv`

The finalize step also appends `train_cv5_macro_f1` and `train_cv5_accuracy` to:

- `ml/results/tables/eval_tracking_rerun_v1_best_val_macro_f1_checkpoint_metrics.tsv`
- `ml/results/tables/eval_tracking_rerun_v1_best_record_knn_val1_checkpoint_metrics.tsv`

## Fresh Rerun

If you want to recompute the six preserved rows instead of resuming, move them aside first:

```sh
mkdir -p ml/results/tables/eval_tracking_rerun_v1_train_cv_rows_canceled_snapshot
mv ml/results/tables/eval_tracking_rerun_v1_train_cv_rows/*.json \
  ml/results/tables/eval_tracking_rerun_v1_train_cv_rows_canceled_snapshot/
```

Then run the resume command above.

## Monitoring

```sh
find ml/results/tables/eval_tracking_rerun_v1_train_cv_rows -type f | wc -l
pgrep -af 'run_record_knn_train_cv5.py'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

Expected final row count:

```sh
find ml/results/tables/eval_tracking_rerun_v1_train_cv_rows -type f | wc -l
# 24
```

Scan logs and rows for failures:

```sh
rg -n "Traceback|RuntimeError|CUDA out of memory|Error|failed" \
  tmp/record_knn_train_cv_work/logs \
  ml/results/tables/eval_tracking_rerun_v1_train_cv_rows
```
