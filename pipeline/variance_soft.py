"""Variance-temperature binary targets for assay-transfer V5."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from pipeline.soft_evidence import evidence_counts

BINARY_STATES = ("transfer", "nontransfer")
BINARY_COMPLETIONS = ("A", "B")


def target_temperature(row: Mapping[str, Any], scale: float = 1.0) -> float:
    """Return sqrt(N) times the deadband inflation factor."""
    transfer, nontransfer, ambiguous = evidence_counts(row)
    total = transfer + nontransfer + ambiguous
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"temperature scale must be positive and finite: {scale}")
    return scale * math.sqrt(total) * (1.0 + ambiguous / total)


def target_distribution(row: Mapping[str, Any], scale: float = 1.0) -> dict[str, float]:
    """Temperature-soften transfer/non-transfer evidence counts over A/B."""
    transfer, nontransfer, _ = evidence_counts(row)
    temperature = target_temperature(row, scale)
    logit_delta = (transfer - nontransfer) / temperature
    probability = 1.0 / (1.0 + math.exp(-logit_delta))
    return {"transfer": probability, "nontransfer": 1.0 - probability}


def completion_is_tie_anchor(row: Mapping[str, Any]) -> bool:
    transfer, nontransfer, _ = evidence_counts(row)
    return transfer == nontransfer


def completion(row: Mapping[str, Any]) -> str:
    """Serialize the unique count winner, anchoring exact ties to A."""
    transfer, nontransfer, _ = evidence_counts(row)
    return "B" if nontransfer > transfer else "A"


def argmax_completion(values: Sequence[float]) -> str:
    """Predict A on an exact two-logit tie."""
    probabilities = tuple(float(value) for value in values)
    if len(probabilities) != 2 or any(not math.isfinite(value) for value in probabilities):
        raise ValueError("V5 probabilities must contain two finite values")
    return "B" if probabilities[1] > probabilities[0] else "A"
