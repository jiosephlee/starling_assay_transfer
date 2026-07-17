"""Shared target and decision helpers for assay-transfer soft evidence V4."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

STATES = ("transfer", "nontransfer", "ambiguous")
COMPLETIONS = ("A", "B", "C")
COUNT_FIELDS = ("n_transfer", "n_nontransfer", "n_ambiguous")
FRACTION_FIELDS = ("transfer_fraction", "nontransfer_fraction", "ambiguous_fraction")


def evidence_counts(row: Mapping[str, Any]) -> tuple[int, int, int]:
    counts = tuple(int(row[name]) for name in COUNT_FIELDS)
    if any(count < 0 for count in counts):
        raise ValueError(f"evidence counts must be nonnegative: {counts}")
    total = int(row["n_records"])
    if total <= 0 or sum(counts) != total:
        raise ValueError(f"evidence counts {counts} do not sum to n_records={total}")
    return counts


def target_distribution(row: Mapping[str, Any]) -> dict[str, float]:
    counts = evidence_counts(row)
    total = float(sum(counts))
    target = {state: count / total for state, count in zip(STATES, counts)}
    for state, field in zip(STATES, FRACTION_FIELDS):
        stored = row.get(field)
        if stored is not None and not math.isclose(float(stored), target[state], abs_tol=1e-9):
            raise ValueError(f"stored {field}={stored} disagrees with counts ({target[state]})")
    return target


def modal_completion(row: Mapping[str, Any]) -> str:
    counts = evidence_counts(row)
    maximum = max(counts)
    winners = [index for index, count in enumerate(counts) if count == maximum]
    return COMPLETIONS[winners[0]] if len(winners) == 1 else "C"


def validate_probabilities(values: Sequence[float], tolerance: float = 1e-6) -> tuple[float, ...]:
    probabilities = tuple(float(value) for value in values)
    if len(probabilities) != 3 or any(not math.isfinite(value) for value in probabilities):
        raise ValueError("soft-evidence probabilities must contain three finite values")
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError(f"soft-evidence probabilities outside [0, 1]: {probabilities}")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=tolerance):
        raise ValueError(f"soft-evidence probabilities do not sum to one: {probabilities}")
    return probabilities


def strict_majority_completion(values: Sequence[float]) -> str:
    transfer, nontransfer, _ = validate_probabilities(values)
    if transfer > 0.5:
        return "A"
    if nontransfer > 0.5:
        return "B"
    return "C"
