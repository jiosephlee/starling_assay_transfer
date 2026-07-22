# Assay Transfer V4-on-V3 Support-Gated Endpoint Weighted-Tanimoto Baseline

Each endpoint is tuned on 25 thresholds spanning 0 through 1 using train rows only.
A validation-selected minimum train support determines which endpoint cutoffs are used;
all remaining rows use the global cutoff tuned on 100 train thresholds.

Selected minimum train rows: `50` (176 endpoint-specific thresholds).
Global fallback threshold: `0.1414141414`.

| split | n | endpoints | global fallbacks | macro-F1 | accuracy | transfer precision | transfer recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 2000 | 135 | 53 | 0.575434 | 0.575500 | 0.468386 | 0.728331 |
| test | 2000 | 138 | 49 | 0.557887 | 0.558000 | 0.436393 | 0.746556 |

Detailed metrics are in `assay_transfer_v4_fg_v3_endpoint_shrinkage_tanimoto_baseline.tsv`.
