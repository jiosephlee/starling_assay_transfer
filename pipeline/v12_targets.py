"""V12 value-CDF targets with measurement-only nullable global SD metadata."""

from __future__ import annotations

from typing import Any, Mapping

from pipeline.v11_targets import (
    BINARY_MATCH_PROBABILITY, NONTRANSFER_MAX_PROBABILITY, TARGET_EPSILON,
    TRANSFER_MIN_PROBABILITY, validate_pair,
)


TARGET_CONTRACT = "train_only_within_pair_bucket_value_cdf_separation.v12"


def _probability(query: Mapping[str, Any], retrieval: Mapping[str, Any]) -> float:
    if str(query["measurement_kind"]) == "binary":
        same = query["canonical_category_id"] == retrieval["canonical_category_id"]
        return BINARY_MATCH_PROBABILITY if same else 1.0 - BINARY_MATCH_PROBABILITY
    distance = abs(float(query["value_percentile"]) - float(retrieval["value_percentile"]))
    return min(1.0 - TARGET_EPSILON, max(TARGET_EPSILON, 1.0 - distance))


def target_for(query: Mapping[str, Any], retrieval: Mapping[str, Any]) -> dict[str, Any]:
    validate_pair(query, retrieval)
    distance = abs(float(retrieval["geometry_value"]) - float(query["geometry_value"]))
    probability = _probability(query, retrieval)
    return {
        "distance": distance, "absolute_geometry_difference": distance,
        "target_a": probability, "target_b": 1.0 - probability,
        "calibration_sample_standard_deviation": query.get(
            "calibration_sample_standard_deviation"
        ),
    }


def is_decisive(row: Mapping[str, Any], _query: Mapping[str, Any]) -> bool:
    value = float(row["target_a"])
    return value > TRANSFER_MIN_PROBABILITY or value < NONTRANSFER_MAX_PROBABILITY


__all__ = ["TARGET_CONTRACT", "is_decisive", "target_for"]
