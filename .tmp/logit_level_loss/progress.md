# Logit-level loss implementation

- [x] Traced TRL dataset preparation, column filtering, and collation.
- [x] Chose three-way soft cross-entropy at the first A/B/C divergent token.
- [x] Add opt-in dataset extraction, collator, and trainer.
- [x] Add tests and smoke-check the SFT entrypoint.
- [x] Document supported launch configuration.

## Full-vocabulary hybrid revision

- [x] Use the full vocabulary denominator for the soft A/B/C decision.
- [x] Retain 0.1-weighted hard CE for shared formatting and EOS.
- [x] Preserve Liger backbone kernels and fuse formatting CE when compatible.
- [x] Add PyTorch and PEFT-safe fallbacks.
- [x] Complete final regression and end-to-end checks.

## BFD packing and padding-free revision

- [x] Locate the active trainer, collator, launcher validation, TRL implementation, and tests.
- [x] Preserve per-document soft targets through TRL BFD packing.
- [x] Emit global decision positions for flattened padding-free batches.
- [x] Gather every decision state and exclude every decision from formatting CE.
- [x] Replace TRL's internal collator after padding-free trainer initialization.
- [x] Update the V4 launch documentation and packing-strategy validation.
- [x] Run focused regressions, function-length checks, and the CUDA Intern smoke test.

Validation results:

- 14 soft-target unit tests pass, including CUDA Liger fused CE.
- Real V4 rows packed into one flattened Intern-tokenized batch with ordered targets,
  correct `seq_lengths`, and one `position_ids` reset per document.
- CUDA Flash Attention 2 forward/backward and a PEFT-wrapped-head backward pass succeed.
- The SFT entrypoint compiles and exposes BFD as the default packing strategy.
- All functions in the soft-target implementation and focused test module are at most 60 lines.
