# Assay Transfer V4-on-V3 Weighted-Tanimoto Baseline

The stored weighted Tanimoto score is thresholded as transfer when score `>= t`.
Exactly 100 thresholds from 0 through 1 were evaluated on train; the cutoff was
selected by macro-F1, then accuracy, transfer precision, and the smaller threshold.

Selected threshold: `0.1414141414`
(train macro-F1 `0.525691`, accuracy `0.647902`).

| split | n | macro-F1 | accuracy | transfer precision | transfer recall |
|---|---:|---:|---:|---:|---:|
| validation | 2000 | 0.572589 | 0.573000 | 0.465236 | 0.701164 |
| test | 2000 | 0.560148 | 0.561000 | 0.435919 | 0.712121 |

Detailed overall and slice metrics are in `assay_transfer_v4_fg_v3_tanimoto_baseline.tsv`.
