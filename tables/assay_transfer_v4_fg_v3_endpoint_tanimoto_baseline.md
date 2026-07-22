# Assay Transfer V4-on-V3 Endpoint-Threshold Weighted-Tanimoto Baseline

Each canonical endpoint selects its own weighted-Tanimoto cutoff from exactly 100
thresholds spanning 0 through 1 on its train rows. Selection uses macro-F1, then
accuracy, transfer precision, and the smaller threshold.

Endpoint-specific thresholds: `477`.
Global fallback for endpoints absent from train: `0.1414141414`.

| split | n | endpoints | global fallbacks | macro-F1 | accuracy | transfer precision | transfer recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 2000 | 135 | 22 | 0.571916 | 0.572000 | 0.465388 | 0.721863 |
| test | 2000 | 138 | 17 | 0.560297 | 0.560500 | 0.437855 | 0.742424 |

Detailed metrics are in `assay_transfer_v4_fg_v3_endpoint_tanimoto_baseline.tsv`.
