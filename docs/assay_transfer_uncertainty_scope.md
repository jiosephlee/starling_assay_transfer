# Scope: restoring measurement uncertainty into v6.5 record representation

Status: scoping doc + diagnostic (no code changes yet)
Last updated: 2026-07-21

## Goal

Carry each measurement's **uncertainty** (the `± spread` from `8 ± 0.5`, and ranges) into the record
representation used to build transfer pairs and render prompts, so the model sees how noisy a
retrieval value is — not just the point mean.

## Where the uncertainty is dropped (diagnostic)

Upstream fully decomposes uncertainty (`pipeline/source_normalization/scalar.py:24`,
`(?P<mean>…)±(?P<variation>…)`) into `scalar_value` (mean), `variation_value`, `variation_type`
(currently only `"unspecified_variation"`), `scalar_is_approximate`, and
`accompanying_interval_lower/upper`. These fields **are present in the `canonical_endpoints_v3`
base records**.

The drop happens at **compose** (`pipeline/stages/compose_v3.py`): its `REQUIRED` tuple (line 17)
keeps `scalar_value` and `scalar_is_approximate` but **not** `variation_value`, `variation_type`,
or `accompanying_interval_lower/upper`, and `_eligible_row` only copies `REQUIRED` + `context_*` +
metric columns. So:

| field | in base | in v6.5 eligible | in prompt |
| --- | --- | --- | --- |
| `scalar_value` (mean) | ✓ | **kept** | shown (retrieval), hidden (query) |
| `scalar_is_approximate` | ✓ | **kept** | **not rendered** |
| `variation_value` (± magnitude) | ✓ | **dropped** | — |
| `variation_type` | ✓ | dropped | — |
| `accompanying_interval_lower/upper` | ✓ | dropped | — |

So the **mean already enters v6.5**; the missing piece is the spread. Note `scalar_is_approximate`
is already on the record but never templated.

### How much uncertainty exists (base records)

| base | records | scalar_is_approximate / variation_value | intervals |
| --- | --- | --- | --- |
| intestinal_absorption (Fa) | 10,803 | ~39% / ~36% | ~0% |
| hepatic (Fh) | 22,326 | ~28% / ~28% | ~1% |
| gut_wall (Fg) | 1,982 | ~27% / ~27% | ~0% |
| oral_bioavailability | 77,214 | 0% | 0% |
| starling_oba | 27,640 | 0% | 0% |

Uncertainty is concentrated on the **in-vitro mechanistic concepts (Fa/Fg/Fh, ~27–39%)** and is
absent on the human PK endpoints. `accompanying_interval_*` is negligible (~0–1%) — the actionable
signal is `variation_value`.

## Inclusion path (simpler than categorical — fields already parsed)

1. **Compose:** add `variation_value`, `variation_type` (and optionally
   `accompanying_interval_lower/upper`) to `compose_v3`'s kept columns so they land in the eligible
   records next to `scalar_value`. `scalar_is_approximate` is already kept.
   - The spread is on the **native** scale (same as `scalar_value`/`unit_basis`), so it renders
     naturally next to the native known value; the transformed `comparison_value` is unaffected.
2. **Record representation / template:** surface it on **Molecule A (retrieval)** only —
   e.g. `known value: 8 ± 0.5 percent` (or an "uncertainty:" line, and an "(approximate)" marker
   when `scalar_is_approximate` and no magnitude). **Molecule B (query) hides it**, exactly like the
   scalar value, so the measurement scale can't leak. This is a template edit in
   `templates/assay_transfer_v6_5_intern/default.jinja` (and any per-concept variant), plus making
   the fields available to `render_prompt` (already are — they'd be on the record dict).
3. **Optional target use (design decision, not default):** whether `target_for` should consume the
   uncertainty — e.g. widen the transfer band or down-weight the soft target for high-variance
   retrieval values, so a noisy measurement is treated as weaker evidence. Flag for discussion;
   keep the default target unchanged unless chosen.
4. **Rollout:** a `v6.5.x` rebuild + re-upload through the existing build/verify path once approved.
   Since only compose's kept columns and the template change, the pair selection, `target_z`, and
   benchmark membership stay identical (as with the earlier prompt-only revisions).

## Open questions
- Render format: inline `mean ± spread` vs. a separate `uncertainty:` line vs. also showing
  `variation_type` (currently uninformative — single value `unspecified_variation`).
- Whether to render `scalar_is_approximate` as an explicit "(approximate)" marker even when no
  magnitude is available (~all of the extra coverage beyond `variation_value`).
- Whether the query side should show a bare "(approximate)" flag (doesn't leak the value) or stay
  fully uncertainty-free.

## Files touched (when implemented)
`pipeline/stages/compose_v3.py` (kept columns), `templates/assay_transfer_v6_5_intern/default.jinja`
(+ per-concept templates if used), optionally `pipeline/pair_core.py::target_for`, then the standard
build/verify/upload path. Recompose is required (eligible schema gains the columns).
