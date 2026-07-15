# Dataset System and Pipeline Layout

Status: architecture draft and canonical reference for the dataset build system
Last updated: 2026-07-15

This document defines how raw assay collections become training-ready datasets: the
on-disk stage layout, the notion of a *build*, how builds compose one or more sources,
and how the pipeline code is organized. It is the operational companion to
`assay_transfer_design.md`, which defines the labels, inference contract, condition
keys, and metric policy. Where this document says "record", "endpoint_key", "K", "Z", or
"pair artifact", the meaning is the one fixed in `assay_transfer_design.md` (Sections 3,
6, 8, 14).

## 1. Goals

- Treat every raw assay collection (`q1`-`q4`, the in-house Starling set, and future
  additions) as one interchangeable **source**.
- Normalize each source once into clean **records**, reusable across every dataset that
  draws on it.
- Compose one or more sources — or filtered subsets of them — into a named, versioned
  **build** that flows through split -> pairs -> final dataset without special-case code.
- Keep the pipeline observable: each stage is its own directory, so the progress of any
  build is visible at a glance.
- Retire the oral-bioavailability-specific scripts in favor of a source-generic pipeline
  driven by a build config and the `TaskSpec` registry.

## 2. Stage-first directory layout

`datasets/` is organized by pipeline stage. Every stage directory holds one subdirectory
per unit at that stage. Data flows top to bottom.

```text
datasets/
  sources/       # raw extractions, immutable                 unit: one source
  base/          # normalized clean records                   unit: one source (reusable)
  splits/        # molecule-first split assignment            unit: one build
  pairs/         # candidate pairs enumerated within the split unit: one build
  parquet/       # final materialized build (proper dataset)  unit: one build
  hf_parquet/    # HF-templated rendering of the final build  unit: one build
```

The two ends are the fixed points:

- **`base/` is the normalized clean version of `sources/`**, one folder per source. A
  source is normalized once and reused by every build that references it.
- **`parquet/` is the final build** — the materialized proper dataset in the pair-artifact
  schema of `assay_transfer_design.md` Section 14, ready to consume. `hf_parquet/` is only
  that same final build rendered through a prompt template into Hugging Face
  `train`/`validation`/`test` splits.

Everything between — `splits/`, `pairs/`, `parquet/`, `hf_parquet/` — is keyed by
**build name**. Because a build's name appears as a subdirectory in each of those stages,
build progress is directly visible: if `fa_plus_oba_human_v1` exists under `splits/` and
`pairs/` but not `parquet/`, the build stopped after pairing.

### 2.1 Mapping from the current layout

| New | Current | Notes |
|---|---|---|
| `datasets/sources/` | `datasets/starling_assays/datasets/` | rename `q1`-`q4` to semantic names |
| `datasets/base/` | `datasets/base/` | keep; becomes strictly per-source normalized records |
| `datasets/splits/` | split half of `datasets/pairs_split_full/` | molecule assignment only |
| `datasets/pairs/` | `datasets/pairs_compact/` | candidate pairs, compact form |
| `datasets/parquet/` | materialized half of `datasets/pairs_split_full/` | final proper dataset |
| `datasets/hf_parquet/` | `datasets/pairs_split_hf/` | templated HF dataset |

## 3. Sources

A **source** is a raw extraction batch exactly as delivered — messy strings, qualifiers,
free-text species, source-specific columns. It is immutable input and is never edited in
place. Sources carry provenance forever, because normalization is versioned and must be
reproducible from the raw rows (`assay_transfer_design.md` Sections 14, 17).

A source is *not* a single endpoint. Per `assay_transfer_design.md` Section 5.3, one
collection (for example `q2`) holds fraction absorbed, permeability, and solubility rows
simultaneously. Semantic identity lives on the per-record `endpoint_key`, not on the
source name.

### 3.1 Rename

Source directories are renamed to their dominant domain for readability, while each
record keeps a `source_id` provenance field carrying the original `q1`-`q4` tag.

| Old | Source name | Dominant endpoint(s) |
|---|---|---|
| `q1` | `oral_bioavailability` | `oral_bioavailability_F` (bounded fraction) |
| `q2` | `intestinal_absorption` | `fraction_absorbed` (Fa), `papp`, `solubility` |
| `q3` | `gut_wall` | `gut_wall_escape` (Fg), `efflux_ratio`, `transporter_status` |
| `q4` | `hepatic` | `hepatic_escape` (Fh), `clint`, `half_life` |
| — | `starling_oba` | `oral_bioavailability_F` (in-house, primary deployment set) |

The domain name is a convenience label. If a source later sprawls across many unrelated
endpoints, a provenance-neutral name is acceptable; `source_id` and `endpoint_key`
remain the load-bearing identifiers either way.

## 4. Base: normalized records

`base/<source>/` holds the canonical, normalized records for one source: the
`r = (molecule_id, structure, endpoint_key, value, unit, K, Z, provenance)` schema of
`assay_transfer_design.md` Section 3, with parsed numeric values, canonical endpoint
keys, normalized species, condition keys, and explicit missingness indicators.

Base is:

- **per source and reusable** — `intestinal_absorption` is normalized once and consumed
  by every build that references it; builds never re-normalize;
- **versioned** — regenerated when a parser, species ontology, endpoint assignment, or
  condition-key version changes, without re-touching the raw source; and
- **not yet composed or split** — no build identity exists at this stage.

Normalization is performed by a source's `TaskSpec` (Section 7). A source that
contributes multiple endpoints emits multiple `endpoint_key` values within its single
base table.

## 5. Builds

A **build** is a named, versioned dataset instance — for example
`fa_plus_oba_human_v1`. It is not a directory type; it is a name plus a manifest that
threads through `splits/`, `pairs/`, `parquet/`, and `hf_parquet/`. Its purpose is
reproducibility and progress tracking.

A build is fully described by its **build config** (Section 6), which declares:

- which sources and `endpoint_key`s to include;
- optional filters (species, value range, coverage, source subset);
- which shared policy versions to bind (metric thresholds, label policy, condition keys,
  species ontology — from `configs/assay_transfer/v1/`);
- the split policy (query-disjoint vs. strict molecule-disjoint, seed, sizes); and
- the pair modes (`K` profiles) and the HF prompt template variant.

### 5.1 Composition

Combining sources is nothing more than a build config that lists more than one source or
`endpoint_key`. There is **no separate composition directory**. Composition first
materializes at the `splits/` stage: the split stage reads the declared union of records
from `base/` and assigns molecules. Single-source builds are the degenerate one-item
case.

Composition **unions records; it never creates cross-endpoint pairs.** The pairs stage
partitions by `endpoint_key` and `K` (`assay_transfer_design.md` Section 5.1), so a build
that mixes `fraction_absorbed` and `oral_bioavailability_F` trains one model — one shared
encoder — on both endpoints' pairs while every pair stays within a single endpoint.
"Combine subsets" is the same operation with filters applied *before* the split, so the
molecule set the splitter sees is already the intended subset.

### 5.2 Joint splitting across a build

Because a molecule can appear in several sources and endpoints, split assignment is made
once per molecule over the whole composed build (`assay_transfer_design.md` Section 12).
Feeding the splitter the composed union guarantees a molecule lands in one split
everywhere; otherwise retrieval-side records would leak held-out query molecules.

## 6. Configuration layering

Two config axes, kept separate so "same policy, different data" is provable from the
manifest:

- **Shared policy** — `configs/assay_transfer/v1/` (`endpoints.yaml`, `species.yaml`,
  `condition_keys.yaml`, `metric_thresholds.yaml`, `label_policy.yaml`, `release.yaml`).
  Dataset-independent, versioned. A build *references* a policy version rather than
  embedding it.
- **Build config** — `configs/builds/<build>.yaml`. Declares composition, filters, split
  policy, pair modes, HF template, and the bound policy versions.

A reusable fragment of composition (a frequently reused source/endpoint/filter selection)
can be factored into `configs/subdatasets/<name>.yaml` and referenced by build configs.
This is an optional convenience, not a new on-disk artifact.

## 7. Pipeline code

The pipeline code is promoted out of loose `scripts/` into an importable package. It is
implementation code, not utilities, and it must not reuse the `starling_assays` name
(which already denotes the raw data directory).

```text
pipeline/
  taskspecs/     # per-source/endpoint TaskSpec classes + registry   (from scripts/task_specs)
  normalize/     # species, value parsers, condition-key builders     (from scripts/internal)
  stages/
    prepare.py       # raw source            -> base records
    split.py         # composed base records -> molecule split assignment
    pairs.py         # split + base records  -> candidate pairs (within endpoint/K)
    materialize.py   # pairs + labels        -> final build parquet (Section 14 schema)
    render_hf.py     # final build           -> templated HF dataset
  run.py         # DAG runner: reads a build config, executes the stages in order
```

The `TaskSpec` abstraction already exists (`scripts/task_specs/base.py`,
`registry.py`, per-endpoint specs) and is the model to keep: each spec owns its metadata
columns, matching (`same_columns`) keys per `K` mode, value parse/transform, column
normalization, and the record-level label rule. The redesign generalizes the *stage
runners* around that registry and a build config, replacing the hardcoded
`create_oral_bioavailability_*` scripts.

### 7.1 Stage order: split before pairs

Molecules are assigned to splits *before* pairs are enumerated
(`assay_transfer_design.md` Section 12.1). This is why `splits/` precedes `pairs/`. It is
the leakage guarantee (a molecule is train or eval, never both), it avoids enumerating
the quadratic global pair pool only to discard most of it, and it makes joint
cross-endpoint splitting fall out naturally from feeding the splitter the composed
molecule set.

## 8. Manifests and reproducibility

Every build writes a `manifest.json` (present alongside each stage output) pinning the
version axes of `assay_transfer_design.md` Section 17: dataset snapshot, release,
molecule canonicalization, split, endpoint ontology and assignment, value parser, species
normalization, condition key, nuisance context, metric threshold policy, label
aggregation, sampling, model input contract, and evaluation protocol — plus the resolved
source/endpoint/filter selection and the split seed. Two builds are directly comparable
only when the relevant versions match. Changing any pinned version creates a new build,
not an in-place mutation.

## 9. Templated final datasets

"Templated" means two separate things that must stay decoupled:

- **A fixed data schema.** Every `parquet/` build conforms to the one pair-artifact schema
  of `assay_transfer_design.md` Section 14, so `ml/starling_ml/data.py` reads any build
  identically.
- **A swappable prompt renderer.** `hf_parquet/` applies a per-build jinja template
  (`templates/*.jinja`) on top of that fixed schema. Since the no-source-value variant is
  removed (`assay_transfer_design.md` Sections 2, 10), the retrieval value `y_A` is always
  present, and the no-source-value template is dropped.

## 10. Migration path

The current tree already implements most stages under different names, so migration is
staged and low-risk:

1. Finish `TaskSpec` coverage for `fraction_absorbed`, `gut_wall`, and `hepatic`
   endpoints (fg/fh specs).
2. Generalize the four `create_oral_bioavailability_*` scripts into `pipeline/stages/*`
   reading a build config; keep behavior identical for the existing oral-bioavailability
   build as a regression anchor.
3. Add composition (multi-source selection resolved at the split stage) and flip the
   stage order to split -> pairs.
4. Introduce `configs/builds/` and bind the shared `configs/assay_transfer/v1/` policy.
5. Rename `datasets/` stage directories to `sources/base/splits/pairs/parquet/hf_parquet`
   and rename the `q1`-`q4` source directories, preserving `source_id`. This is a
   mechanical last step with git-history cost, done only once the pipeline reads from the
   new locations.
