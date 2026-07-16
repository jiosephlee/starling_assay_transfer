# Source normalization report

## Round-one result

The five pinned inputs reconcile exactly across **349,073** source rows: **204,547 accepted** and **144,526 rejected**. Structures in Q1--Q4 come only from the authoritative global-identifier mapping; Starling uses its source SMILES. All accepted structures were RDKit parsed, rejected when wildcard-bearing, and screened against both raw and canonical forms of all 640 TDC molecules.

| Source | Input | Accepted | Rejected | TDC removed | Species | Combinations |
|---|---:|---:|---:|---:|---:|---:|
| q1 | 119,192 | 75,213 | 43,979 | 33,991 | 32,761 | 68,059 |
| q2 | 85,061 | 39,745 | 45,316 | 23,942 | 13,474 | 30,079 |
| q3 | 27,713 | 11,908 | 15,805 | 8,461 | 3,848 | 9,654 |
| q4 | 67,943 | 28,548 | 39,395 | 20,714 | 26,372 | 16,230 |
| starling | 49,164 | 49,133 | 31 | 0 | 32,624 | 30,341 |

## Round-two handoff

Each source directory contains a lossless normalized record table, all rejected rows with ordered reasons, and every observed post-filter combination of mechanically normalized structured fields. No canonical endpoint keys were assigned, and no endpoint-specific unit conversions, validity ranges, thresholds, pair logic, or models were changed in this round.

### q1

Authoritative structures resolved for 118,072/119,192 rows and 112,571 were structurally usable before later gates. Exported/source comparison: exact_match=118,072. TDC removed 33,991; accepted overlap is zero.

Measurement parsing accepted 75,213 rows; 5,274 rows lacked a measurement and 0 had a present but unapproved/unparseable value.

Observed 105 endpoint aliases and 786 lexical units. Top aliases: `cmax` (22,704), `tmax` (15,084), `auc0 inf` (9,114), `auc0 t` (7,882), `auc` (7,666).

There are 68,059 observed combinations. Largest structured-field missing counts: `categorical_value`=75,213, `qualifying_conditions_normalized`=63,739, `comparator_exposure_normalized`=63,180.

Duplicate annotation found 132 multirow groups covering 267 rows; evidence rows were not collapsed.

Species coverage is 32,761/75,213; 3,966 publications contribute records and the top ten account for 1.6%.

### q2

Authoritative structures resolved for 77,292/85,061 rows and 74,382 were structurally usable before later gates. Exported/source comparison: exact_match=77,292. TDC removed 23,942; accepted overlap is zero.

Measurement parsing accepted 39,745 rows; 29 rows lacked a measurement and 16,770 had a present but unapproved/unparseable value.

Observed 25 endpoint aliases and 655 lexical units. Top aliases: `caco2 mdck pampa permeability` (11,482), `solubility` (9,286), `fraction absorbed` (4,976), `dissolution` (4,055), `intestinal effective permeability` (3,667).

There are 30,079 observed combinations. Largest structured-field missing counts: `categorical_value`=37,994, `qualifying_conditions_normalized`=32,746, `species_exact`=26,271.

Duplicate annotation found 174 multirow groups covering 361 rows; evidence rows were not collapsed.

Species coverage is 13,474/39,745; 4,733 publications contribute records and the top ten account for 2.0%.

### q3

Authoritative structures resolved for 24,618/27,713 rows and 24,039 were structurally usable before later gates. Exported/source comparison: exact_match=24,618. TDC removed 8,461; accepted overlap is zero.

Measurement parsing accepted 11,908 rows; 1,051 rows lacked a measurement and 4,609 had a present but unapproved/unparseable value.

Observed 10 endpoint aliases and 109 lexical units. Top aliases: `efflux or secretory transport` (5,685), `uptake or absorptive transport` (1,970), `bidirectional permeability` (1,742), `intestinal metabolism` (1,482), `oral exposure change due to gut wall` (777).

There are 9,654 observed combinations. Largest structured-field missing counts: `species_exact`=8,060, `intestinal_site_normalized`=6,428, `qualifying_conditions_normalized`=5,782.

Duplicate annotation found 66 multirow groups covering 192 rows; evidence rows were not collapsed.

Species coverage is 3,848/11,908; 2,996 publications contribute records and the top ten account for 2.9%.

### q4

Authoritative structures resolved for 63,081/67,943 rows and 61,135 were structurally usable before later gates. Exported/source comparison: exact_match=63,081. TDC removed 20,714; accepted overlap is zero.

Measurement parsing accepted 28,548 rows; 10,586 rows lacked a measurement and 9,575 had a present but unapproved/unparseable value.

Observed 13 endpoint aliases and 1,046 lexical units. Top aliases: `intrinsic clearance` (15,201), `cyp metabolism` (3,443), `hepatic clearance` (2,600), `ugt metabolism` (2,217), `metabolic half life` (1,843).

There are 16,230 observed combinations. Largest structured-field missing counts: `categorical_value`=27,885, `qualifying_conditions_normalized`=25,149, `molecular_form_normalized`=22,440.

Duplicate annotation found 677 multirow groups covering 1,479 rows; evidence rows were not collapsed.

Species coverage is 26,372/28,548; 3,997 publications contribute records and the top ten account for 2.6%.

### starling

Authoritative structures resolved for 49,164/49,164 rows and 49,133 were structurally usable before later gates. Exported/source comparison: exact_match=49,164. TDC removed 0; accepted overlap is zero.

Measurement parsing accepted 49,133 rows; 0 rows lacked a measurement and 0 had a present but unapproved/unparseable value.

Observed 1 endpoint aliases and 1 lexical units. Top aliases: `oral bioavailability` (49,133).

There are 30,341 observed combinations. Largest structured-field missing counts: `categorical_value`=49,133, `qualifying_conditions_normalized`=35,541, `comparator_normalized`=24,864.

Duplicate annotation found 1,127 multirow groups covering 3,112 rows; evidence rows were not collapsed.

Species coverage is 32,624/49,133; 17,159 publications contribute records and the top ten account for 0.8%.
