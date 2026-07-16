# Assay Transfer V3 Weighted-Tanimoto Baseline

The stored weighted Tanimoto score is thresholded as transfer when score `>= t`.
Exactly 100 thresholds from 0 through 1 were evaluated on train; the cutoff was
selected by macro-F1, then accuracy, transfer precision, and the smaller threshold.

Selected threshold: `0.3030303030`
(train macro-F1 `0.603887`, accuracy `0.632334`).

| split | n | macro-F1 | accuracy | transfer precision | transfer recall |
|---|---:|---:|---:|---:|---:|
| validation | 1703 | 0.599304 | 0.618321 | 0.481638 | 0.546474 |
| test | 1703 | 0.619375 | 0.641221 | 0.484419 | 0.580645 |

Detailed overall and slice metrics are in `assay_transfer_v3_tanimoto_baseline.tsv`.
