# Raw source to canonical base report

## Result

All **349,073** pinned raw parents reconcile exactly through the parent ledger. The pipeline retained **116,112** finite scalar children with canonical endpoint keys.

| Source | Parents | Scalars | Base records | Keys | Partial parents |
|---|---:|---:|---:|---:|---:|
| oral_bioavailability | 119,192 | 75,213 | 61,448 | 839 | 0 |
| intestinal_absorption | 85,061 | 32,279 | 7,897 | 97 | 0 |
| gut_wall | 27,713 | 4,380 | 1,490 | 54 | 504 |
| hepatic | 67,943 | 24,804 | 17,637 | 768 | 142 |
| starling_oba | 49,164 | 49,133 | 27,640 | 2 | 0 |

Only dedicated structured measurement fields were parsed. Narrative support and detail fields were never inspected. Generated Parquets are reproducible local artifacts and are not committed.
