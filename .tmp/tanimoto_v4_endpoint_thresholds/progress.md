# V4-on-V3 endpoint-threshold weighted-Tanimoto baseline

## Progress

- Inspected V4-on-V3 selected artifacts: 477 train endpoints.
- Each endpoint will tune all 100 thresholds using only its train rows.
- Validation and test contain endpoints absent from train, so they will use a separately tuned global train threshold as a deterministic fallback.
- Implemented a dedicated endpoint-threshold baseline with train-only tuning, global fallback, and standard held-out slices.
- Wrote 47,700 sweep rows (477 endpoints x 100 thresholds), 477 selected endpoint thresholds, and held-out reports.
- Verified validation/test reports include all standard slices and respectively use 22/17 fallback rows for endpoints absent from train.
- Tests pass; all functions remain at most 60 lines.
- Endpoint-specific tuning did not outperform the global cutoff, consistent with sparse per-endpoint support.
