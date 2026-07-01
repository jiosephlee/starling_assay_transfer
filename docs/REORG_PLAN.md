# Repo Reorganization Plan

Status: proposal for review. Nothing has been moved yet.

## Problem

`ml/results/` has 24 flat sibling directories whose names each encode 5–6
orthogonal dimensions. The same dimension matrix is then repeated again inside
every `runs/<run_name>/` folder. This makes the tree impossible to scan and
guarantees combinatorial growth at the top level.

Dimensions currently flattened into directory names:

- run type: `shared_eval` | `smoke` | `debug` | `knn`
- pairing/constraint: `condition_key` | `no_constraints` | `same_species_v2`
- source value: default (`srcval`) | `no_source_value`
- dataset version: `v3_v2` (older ones have no suffix)
- model tier: `small_lt10m_v1` | `large_400m_v1`
- campaign tag: `eval_tracking_rerun_v1` | `ssv2_v3_v2` | base

Example: `shared_eval_no_source_value_step_logging_300_macro_f1_large_400m_v1_eval_tracking_rerun_v1`
decodes to `{run_type=shared_eval, srcval=off, model=large_400m, campaign=eval_tracking_rerun_v1}`.

Note: `ml/results/` is NOT git-tracked, so moves have no history cost. The only
risk is scripts that read/write these paths.

## Target structure for `ml/results/`

```
ml/results/
  _smoke/                         # throwaway smoke + debug runs, isolated
  _knn/                           # knn eval outputs
  <campaign>/                     # eval_tracking_rerun_v1, v3_v2_baseline, ...
    <condition>/                  # condition_key | no_constraints | same_species_v2
      <srcval|no_source_value>/
        <small_lt10m|large_400m>/
          runs/<lr..._bs..._ga...>/metrics.csv   # short names; path holds the rest
          hp_sweep_winners.{csv,md}
    tables/                       # cross-run aggregates for this campaign
```

Benefits:
- scan one axis at a time with plain `ls`
- run names shrink to just the hyperparameters that vary
- a new campaign/condition/model tier does not add a new top-level folder

## Proposed mapping of current dirs

| current | campaign | condition | srcval | model |
|---|---|---|---|---|
| shared_eval_condition_key_v3_v2 | v3_v2_baseline | condition_key | srcval | mixed (in runs) |
| shared_eval_condition_key_no_source_value_v3_v2 | v3_v2_baseline | condition_key | no_source_value | mixed |
| shared_eval_no_constraints_v3_v2 | v3_v2_baseline | no_constraints | srcval | mixed |
| shared_eval_no_constraints_no_source_value_v3_v2 | v3_v2_baseline | no_constraints | no_source_value | mixed |
| shared_eval_same_species_v2_v3_v2 | v3_v2_baseline | same_species_v2 | srcval | mixed |
| shared_eval_same_species_v2_no_source_value_v3_v2 | v3_v2_baseline | same_species_v2 | no_source_value | mixed |
| shared_eval_*_eval_tracking_rerun_v1 | eval_tracking_rerun_v1 | (from name) | (from name) | (from name) |
| shared_eval_*_ssv2_v3_v2 | v3_v2_baseline | same_species_v2 | srcval | (from name) |
| shared_eval_condition_key (legacy) | legacy_pre_v3 | condition_key | srcval | mixed |
| shared_eval_same_species_v2 (legacy) | legacy_pre_v3 | same_species_v2 | srcval | mixed |
| smoke_* / debug_* | -> _smoke/ | | | |
| knn | -> _knn/ | | | |
| tables | -> per-campaign tables/ | | | |

(Model tier for `shared_eval_*` dirs currently lives in the individual run
names; when splitting, group runs by their `small_lt10m` / `large_400m` token.)

## Root cleanup (optional, second phase)

- Move `GOAL.md`, `HANDOFF.md`, `PENDING_PLAN.md` into `docs/`.
  Keep `README.md` and `AGENTS.md` at the root.
- Add `tmp/` (esp. `tmp/triton_cache/`, ~166M, 1063 dirs) to `.gitignore`, and
  set `TRITON_CACHE_DIR` outside the repo so the compiler cache stops accreting.
- Add a "Repo layout" section to `README.md` documenting the intentional split:
  - root `scripts/` + `configs/` = data pipeline (PyArrow/RDKit)
  - `ml/scripts/` + `ml/configs/` = model training/eval
- Confirm `archive/` (276G) and `datasets/` (235G) contents are still needed;
  candidates to drop: `archive/2026-06-29_*_invalid`, `*_interrupted_*`.

## Execution options (pick when ready)

1. Results only: restructure `ml/results/` into the hierarchy above.
2. Results + root cleanup.
3. Full reorg including an `archive/` + `datasets/` staleness audit.

For any of these, decide whether to also:
- update the scripts that write these paths (recommended), or
- move existing files only and leave compatibility symlinks.

## Scripts that reference result/artifact paths (to update if scripts are changed)

- `ml/scripts/run_shared_eval_benchmark.py`
- `ml/scripts/continue_shared_eval_after_kill.py`
- `ml/scripts/summarize_shared_eval_models_and_baselines.py`
- `ml/scripts/knn/run_knn_eval.py`
- `ml/scripts/sweep_lr_bs.sh`, `ml/scripts/sweep_head_capacity.sh`

(Verify exact write paths in each before moving.)
