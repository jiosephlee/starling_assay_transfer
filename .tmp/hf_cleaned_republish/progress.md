# HF cleaned republish progress

- [x] Inspected repository instructions, working tree, export implementation, and tests.
- [x] Refined source-facing schemas and omission reporting.
- [x] Added validation and deterministic regression coverage.
- [x] Rebuilt all five staged datasets twice; all 15 core artifact hashes matched.
- [x] Ran full tests and function-length check: 94 passed, 5 skipped, 13 subtests passed.
- [x] Published five public repositories and verified each Hub Parquet SHA-256 against local staging.

Published commits:

- oral_bioavailability_cleaned: `f6c95462a2627c5c381b4a7471e5783030fd933c`
- intestinal_absorption_cleaned: `b7990204533eda27db9cb4dd49d7b9a0eb5aedd1`
- gut_wall_cleaned: `f5a0ce16e46c833691137e9e47b84ba472d748e5`
- hepatic_cleaned: `3a4ac04a67dfe1542c47251b85225bcd69b29857`
- starling_oba_cleaned: `4261ad97a2a43a9f7890298b11bd7a9e1b9efe48`

## Column-order refinement

- [x] Chosen scientific-first, narrative-last public ordering.
- [x] Implemented and tested deterministic column ordering.
- [x] Regenerated twice and validated all five datasets; hashes were deterministic.
- [x] Republished and verified all Hub Parquet files byte-for-byte.

Column-order commits:

- oral_bioavailability_cleaned: `fef03eca7c94329e9ef549c5904e0d5d0b14c9b7`
- intestinal_absorption_cleaned: `a9c8940282795600b477138f76682930d8331282`
- gut_wall_cleaned: `0443b21b1e712e0269789e52e9863d4cde0eb4ee`
- hepatic_cleaned: `eaff3469ac7f60b34e27dba645c08bce6a677497`
- starling_oba_cleaned: `db1ed61990242a6e978e04c03e9a7a67dfa85a25`

Existing working-tree changes predate this task and will be preserved.
