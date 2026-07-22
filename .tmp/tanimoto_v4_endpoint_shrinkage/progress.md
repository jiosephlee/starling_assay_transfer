# V4-on-V3 support-gated endpoint weighted-Tanimoto baseline

## Progress

- Confirmed support gates of 25–1,000 train rows retain 208–56 endpoint-specific thresholds and cover 98–91% of held-out rows.
- Will tune 25 endpoint cutoffs on train and select the support gate from validation only.
- Endpoints below the selected support gate, including unseen endpoints, will use the 100-grid global train cutoff.
- Implemented the support-gated baseline and evaluated 11,925 endpoint-threshold candidates (477 endpoints x 25 thresholds).
- Validation selected the 50-row gate: 176 endpoint cutoffs, with 53 validation and 49 test rows using the global fallback.
- Validation macro-F1 improved to 0.575434, but test macro-F1 was 0.557887, below the global baseline.
- Tests pass and all functions are at most 60 lines.
