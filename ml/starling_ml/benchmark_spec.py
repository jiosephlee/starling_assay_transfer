"""Shared benchmark policy for Oral Bioavailability transfer runs."""

from __future__ import annotations

from dataclasses import dataclass


HP_SWEEP_PROJECT = "oral_bioavailability_transfer_hp_sweep"
FULL_RUN_PROJECT = "oral_bioavailability_transfer"
LEARNING_RATES = ("1e-4", "2e-4", "4e-4")
EFFECTIVE_BATCH_SIZES = ("8192", "16384", "32768")
SWEEP_MAX_STEPS = 300
SWEEP_EVAL_STEPS = 50
FULL_TDC_EVAL_STEPS = 250
HP_SELECT_METRIC = "eval_val_no_overlap_macro_f1"
BEST_VAL_CHECKPOINT_DIR = "best_val_macro_f1"
BEST_TDC_CHECKPOINT_DIR = "best_tdc_train_macro_f1"
BEST_RECORD_KNN_CHECKPOINT_DIR = "best_record_knn_validation_1_macro_f1"

EVAL_SUBSET_ALIASES = {
    "no_overlap": "double_unseen",
    "a_seen_only": "query_unseen",
    "both_seen": "both_seen",
}


@dataclass(frozen=True)
class BatchPlan:
    effective_batch_size: int
    per_device_batch_size: int
    n_devices: int
    gradient_accumulation_steps: int


def per_device_batch_size(
    effective_batch_size: int | str,
    *,
    n_devices: int,
    gradient_accumulation_steps: int,
) -> int:
    effective = int(effective_batch_size)
    denom = int(n_devices) * int(gradient_accumulation_steps)
    if denom <= 0:
        raise ValueError("n_devices and gradient_accumulation_steps must be positive")
    if effective % denom != 0:
        raise ValueError(
            f"effective batch size {effective} is not divisible by "
            f"n_devices * gradient_accumulation_steps = {denom}"
        )
    return effective // denom


def batch_plan(
    effective_batch_size: int | str,
    *,
    n_devices: int,
    gradient_accumulation_steps: int,
) -> BatchPlan:
    return BatchPlan(
        effective_batch_size=int(effective_batch_size),
        per_device_batch_size=per_device_batch_size(
            effective_batch_size,
            n_devices=n_devices,
            gradient_accumulation_steps=gradient_accumulation_steps,
        ),
        n_devices=int(n_devices),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
    )
