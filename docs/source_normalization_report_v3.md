# Raw source to canonical base report

## Result

All **349,073** pinned raw parents reconcile exactly through the parent ledger. The pipeline retained **139,965** finite scalar children with canonical endpoint keys.

| Source | Parents | Scalars | Base records | Keys | Partial parents |
|---|---:|---:|---:|---:|---:|
| oral_bioavailability | 119,192 | 95,407 | 77,214 | 972 | 0 |
| intestinal_absorption | 85,061 | 43,598 | 10,803 | 116 | 0 |
| gut_wall | 27,713 | 5,634 | 1,982 | 68 | 591 |
| hepatic | 67,943 | 31,779 | 22,326 | 955 | 189 |
| starling_oba | 49,164 | 49,133 | 27,640 | 2 | 0 |

Only dedicated structured measurement fields were parsed, except for the pinned Q3 Fg support-text re-extraction artifact. Other narrative support and detail fields were never inspected. Generated Parquets are reproducible local artifacts and are not committed.
