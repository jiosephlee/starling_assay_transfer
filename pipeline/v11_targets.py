"""Raw-geometry target calibration shared by every v11 task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


LOWER_PERCENTILE = 5
UPPER_PERCENTILE = 95
TARGET_AT_LOWER = 0.95
TARGET_AT_UPPER = 0.05
TRANSFER_MIN_PROBABILITY = 0.6
NONTRANSFER_MAX_PROBABILITY = 0.4
ANCHOR_LOGIT = math.log(TARGET_AT_LOWER / TARGET_AT_UPPER)


@dataclass(frozen=True)
class RawCalibration:
    pair_bucket_key: str
    measurement_kind: str
    distance_p05: float
    distance_p95: float


def geometry_value(record: Mapping[str, Any]) -> float:
    kind = str(record.get("measurement_kind") or "")
    field = "canonical_category_rank" if kind in {"binary", "ordinal"} else "finite_scalar_value"
    value = record.get(field)
    if value is None:
        raise ValueError(f"{kind or 'unknown'} record lacks {field}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite v11 geometry value: {number}")
    return number


def raw_calibration(pair_bucket_key: str, entry: Mapping[str, Any]) -> RawCalibration:
    if not bool(entry.get("calibration_valid")):
        raise ValueError(f"pair bucket {pair_bucket_key!r} is not calibration-valid")
    calibration = entry.get("distance_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError(f"pair bucket {pair_bucket_key!r} lacks distance calibration")
    sd = float(calibration["sample_standard_deviation"])
    knots = calibration["standardized_distance_percentile_knots"]
    lower = float(knots[LOWER_PERCENTILE]) * sd
    upper = float(knots[UPPER_PERCENTILE]) * sd
    if not all(math.isfinite(value) for value in (sd, lower, upper)) or sd <= 0:
        raise ValueError(f"pair bucket {pair_bucket_key!r} has invalid geometry scale")
    if upper <= lower:
        raise ValueError(f"pair bucket {pair_bucket_key!r} has degenerate p05/p95 anchors")
    return RawCalibration(
        pair_bucket_key=str(pair_bucket_key),
        measurement_kind=str(entry["measurement_kind"]),
        distance_p05=lower,
        distance_p95=upper,
    )


def target_for(query: Mapping[str, Any], retrieval: Mapping[str, Any]) -> dict[str, float]:
    if query["pair_bucket_key"] != retrieval["pair_bucket_key"]:
        raise ValueError("v11 target requires one pair bucket")
    lower = float(query["calibration_distance_p05"])
    upper = float(query["calibration_distance_p95"])
    retrieval_anchors = (
        float(retrieval["calibration_distance_p05"]),
        float(retrieval["calibration_distance_p95"]),
    )
    if retrieval_anchors != (lower, upper):
        raise ValueError("v11 pair records disagree on calibration anchors")
    distance = abs(float(retrieval["geometry_value"]) - float(query["geometry_value"]))
    target_z = ANCHOR_LOGIT * (lower + upper - 2.0 * distance) / (upper - lower)
    if target_z >= 0:
        target_a = 1.0 / (1.0 + math.exp(-min(target_z, 700.0)))
    else:
        exponential = math.exp(max(target_z, -700.0))
        target_a = exponential / (1.0 + exponential)
    return {
        "distance": distance,
        "absolute_geometry_difference": distance,
        "target_z": target_z,
        "target_a": target_a,
        "target_b": 1.0 - target_a,
        "calibration_distance_p05": lower,
        "calibration_distance_p95": upper,
    }


def candidate_label(query: Mapping[str, Any], retrieval: Mapping[str, Any]) -> str | None:
    probability = target_for(query, retrieval)["target_a"]
    if probability > TRANSFER_MIN_PROBABILITY:
        return "transfer"
    if probability < NONTRANSFER_MAX_PROBABILITY:
        return "nontransfer"
    return None


def is_decisive(row: Mapping[str, Any], _query: Mapping[str, Any]) -> bool:
    value = float(row["target_a"])
    return value > TRANSFER_MIN_PROBABILITY or value < NONTRANSFER_MAX_PROBABILITY


__all__ = [
    "NONTRANSFER_MAX_PROBABILITY",
    "RawCalibration",
    "TARGET_AT_LOWER",
    "TARGET_AT_UPPER",
    "TRANSFER_MIN_PROBABILITY",
    "candidate_label",
    "geometry_value",
    "is_decisive",
    "raw_calibration",
    "target_for",
]
