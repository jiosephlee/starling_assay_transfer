# MLP fix for the multi-endpoint dataset (DEFERRED — design note)

Status: design note, **not implemented**. Captured for later; no MLP work is planned for now.
Last updated: 2026-07-15

This documents how to make the structured MLP (`ml/starling_ml/model.py`) work correctly on the
multi-endpoint v2 dataset. It is a design record only — nothing here is wired up.

## Problem

The v2 primary objective is continuous distance regression, but two defects make it ill-posed
across endpoints:

1. **Incomparable target scale.** `continuous_target = mean_j |y_A − y_Bj|` lives on each
   metric's *transformed* scale. A distance of `0.3` means 0.3 percentage points for
   `fraction_absorbed`, a third of the range for `extraction_ratio`, and ~2-fold for
   `metabolic_half_life`. A single Huber/MSE head over the raw target is dominated by the
   large-scale endpoints (`%` distances up to 100 vs log distances ~0–2), so the loss is
   endpoint-imbalanced and the predicted number is meaningless across endpoints.
2. **No endpoint awareness.** The head input is `[h_a, m_a, h_b, y_A]` — it never sees which
   canonical endpoint / metric it is scoring, so it cannot apply endpoint-specific similarity
   behavior or (before fix 1) even know the target's scale.

The binary auxiliary head is already endpoint-comparable (votes are thresholded per metric) and
needs no change.

## Fix

**(a) Predict a metric-normalized, scale-free target**, and **(b) condition the head on the
canonical endpoint.**

- **Normalization** by the metric's `not_transfer_min` (available on `MetricThreshold`,
  `pipeline/policy.py`): `continuous_target_normalized = continuous_target / not_transfer_min`.
  Then `1.0` = "at the non-transfer boundary" for **every** metric; the transfer boundary lands
  near `transfer_max / not_transfer_min` (~0.3–0.5) uniformly. This becomes the training signal;
  raw `continuous_target` is retained for interpretable / per-endpoint de-normalization.
- **Endpoint conditioning** on `canonical_endpoint_id` (richer than `metric_type`) via a learned
  `nn.Embedding`, gated by a `use_endpoint_conditioning` flag so it can be ablated.

Bonus: the normalized target makes the head's output directly interpretable — `<~0.35` ≈ "within
the transfer band", `≥1.0` ≈ "at/over the non-transfer band", uniformly across endpoints — which
also gives a natural endpoint-agnostic default cutoff for the §15 binary calibration.

## Changes (by file)

### 1. Dataset: emit the normalized target — `pipeline/stages/pairs.py`
In `build()`, where the candidate dict is assembled, add
`continuous_target_normalized = d_expected / metric.not_transfer_min` (guard
`not_transfer_min > 0`; true for every enabled metric). Add the column to the `_FLOAT` set.
`materialize.py` is schema-agnostic and carries it through; add `continuous_target_normalized`
to the explicit column list in `render_hf.py`.

### 2. Precompute: endpoint vocabulary — `ml/starling_ml/precompute_embeddings.py`
In `build_v2_tables`, after the SMILES/metadata vocabularies, build
`endpoint_to_id = {canonical_endpoint_id: contiguous int}` (sorted, deterministic) and write
`endpoint_index.json`; add `n_endpoints` to the manifest. No embedding table — the model holds a
learned `nn.Embedding`; precompute only fixes the id map + count.

### 3. Data layer: carry endpoint id + normalized target — `ml/starling_ml/data.py`
- `build_split_memmap(..., endpoint_to_id, ...)`: write a new `endpoint.u16` memmap
  (`endpoint_to_id[r["canonical_endpoint_id"]]`), and set the `target.f32` memmap from
  `continuous_target_normalized` (the training signal). Keep the raw as optional `target_raw.f32`
  for eval de-norm.
- `PairDataset` + `collate_pairs`: add `endpoint_id` (int64) to the batch alongside
  `{a_idx, b_idx, meta_a_idx, source_value, distance, labels}`.

### 4. Model: endpoint conditioning — `ml/starling_ml/model.py`
- `TransferPairModel.__init__(..., n_endpoints)`: add
  `self.endpoint_emb = nn.Embedding(n_endpoints, model_cfg.endpoint_emb_dim)` and, when
  `model_cfg.use_endpoint_conditioning`, `head_in += endpoint_emb_dim`.
- `forward(a_idx, b_idx, meta_a_idx, source_value, endpoint_id, distance=None, labels=None)`:
  concat `endpoint_emb(endpoint_id)` into the trunk input before `_trunk`. The distance head now
  predicts the **normalized** target; the loss is unchanged (Huber/MSE, now comparable across
  endpoints).
- `build_model` (`model.py:267`): read `n_endpoints` from `embeddings_dir/endpoint_index.json`
  and pass it in.

### 5. Config + export mirror
- `ml/starling_ml/config.py` `ModelConfig`: add `endpoint_emb_dim: int = 32` and
  `use_endpoint_conditioning: bool = True`.
- `ml/starling_ml/modeling_starling_transfer.py`: mirror the `endpoint_emb` + `endpoint_id`
  input (param names must match for weight loading); `StarlingTransferConfig` carries
  `n_endpoints` + `endpoint_emb_dim`.

### 6. Train / eval glue
- `ml/starling_ml/train.py`: `_build_split_memmap_for_config` loads `endpoint_index.json`
  (alongside the smiles/meta maps) and passes `endpoint_to_id`. `endpoint_id` flows through
  `collate_pairs` → `forward` automatically (all collate keys are model inputs). Regression
  `metric_set` unchanged (now on the normalized target).
- `ml/starling_ml/evaluate.py`: pass `endpoint_id`; report regression metrics on the normalized
  target and, optionally, per-endpoint de-normalized RMSE via `target_raw`.

## Reuse
- `MetricThreshold.not_transfer_min` (`pipeline/policy.py`).
- The `smiles_index` / `meta_index` map pattern in `precompute_embeddings.build_v2_tables`.
- The memmap / collate pattern in `ml/starling_ml/data.py`.

## Verification (when implemented)
1. Unit test `continuous_target_normalized == continuous_target / not_transfer_min`; run
   `pairs.py` on real q2+q3+q4 and confirm the normalized target sits on a comparable ~[0, few]
   range across endpoints (vs raw spanning 0–100 for `%` and 0–2 for log).
2. Extend the synthetic `ml/tests` so `build_v2_tables` emits `endpoint_index.json`,
   `build_split_memmap` writes `endpoint.u16`, `collate` includes `endpoint_id`, and
   `TransferPairModel(..., n_endpoints)` forward + `loss.backward()` run with the endpoint
   embedding; assert `use_endpoint_conditioning=False` still works (head_in without endpoint dims).
3. Run `tests/` and `ml/tests/` under `/data1/joseph/miniconda3/envs/tuning/bin/python`.
4. End-to-end training needs GPU + real MoLFormer/MiniLM (out of scope for static checks); the
   shapes/plumbing are covered by the synthetic contract tests.

## Alternative (not chosen)
An LM-in-prompt classifier (`templates/assay_transfer_classification.jinja` + `render_hf.py`)
sidesteps the scale problem by stating the endpoint/units/`K` as text, but fights the continuous
primary objective (LMs are unreliable at numeric regression) and uses SMILES-as-text (a weaker
structural prior than MolFormer). Reasonable only if the objective is reframed as binary
classification with a chemistry-pretrained LM.
</content>
