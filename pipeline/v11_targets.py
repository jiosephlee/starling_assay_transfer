"""Individual-value CDF and categorical soft targets for V11 assay transfer."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Mapping


TARGET_EPSILON = 1e-6
BINARY_MATCH_PROBABILITY = 0.95
TRANSFER_MIN_PROBABILITY = 0.6
NONTRANSFER_MAX_PROBABILITY = 0.4
TARGET_CONTRACT = "within_pair_bucket_value_cdf_separation.v1"


@dataclass(frozen=True)
class ContinuousCdf:
    support: tuple[float, ...]
    midranks: tuple[float, ...]
    cumulative_less: tuple[int, ...]
    total: int


@dataclass(frozen=True)
class OrdinalCdf:
    midrank_by_category: Mapping[str, float]


@dataclass(frozen=True)
class ValueCalibration:
    pair_bucket_key: str
    measurement_kind: str
    sample_standard_deviation: float
    standard_deviation_ddof: int
    standard_deviation_value_field: str
    value_cdf: ContinuousCdf | None
    category_cdf: OrdinalCdf | None


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


def _continuous_cdf(payload: Mapping[str, Any]) -> ContinuousCdf:
    support = tuple(float(item) for item in payload["support_values"])
    midranks = tuple(float(item) for item in payload["support_midranks_0_1"])
    counts = tuple(int(item) for item in payload["support_counts"])
    if not support or len({len(support), len(midranks), len(counts)}) != 1:
        raise ValueError("continuous value CDF arrays are empty or misaligned")
    if any(left >= right for left, right in zip(support, support[1:])):
        raise ValueError("continuous value CDF support is not strictly increasing")
    cumulative, running = [], 0
    for count in counts:
        cumulative.append(running)
        running += count
    total = int(payload["total_record_count"])
    if running != total or total <= 0:
        raise ValueError("continuous value CDF counts disagree with total")
    return ContinuousCdf(support, midranks, tuple(cumulative), total)


def _ordinal_cdf(payload: Mapping[str, Any]) -> OrdinalCdf:
    values = {
        str(item["category_id"]): float(item["midrank_0_1"])
        for item in payload["categories"]
    }
    if not values:
        raise ValueError("ordinal category CDF has no categories")
    return OrdinalCdf(values)


def _cdf_payload(
    kind: str, entry: Mapping[str, Any],
) -> tuple[ContinuousCdf | None, OrdinalCdf | None]:
    if kind == "continuous" and not entry.get("value_cdf_valid"):
        raise ValueError("continuous bucket lacks a valid empirical value CDF")
    if kind == "ordinal" and not entry.get("category_cdf_valid"):
        raise ValueError("ordinal bucket lacks a valid empirical category CDF")
    if kind == "binary":
        return None, None
    if kind == "continuous":
        return _continuous_cdf(entry["value_cdf"]), None
    return None, _ordinal_cdf(entry["category_cdf"])


def value_calibration(pair_bucket_key: str, entry: Mapping[str, Any]) -> ValueCalibration:
    if not bool(entry.get("calibration_valid")):
        raise ValueError(f"pair bucket {pair_bucket_key!r} is not calibration-valid")
    kind = str(entry.get("measurement_kind") or "")
    if kind not in {"continuous", "binary", "ordinal"}:
        raise ValueError(f"pair bucket {pair_bucket_key!r} has unknown measurement kind")
    sample_sd = float(entry["observed_sample_standard_deviation"])
    ddof = int(entry["standard_deviation_ddof"])
    value_field = str(entry["standard_deviation_value_field"])
    if not math.isfinite(sample_sd) or sample_sd <= 0 or ddof != 1:
        raise ValueError(f"pair bucket {pair_bucket_key!r} has invalid sample SD")
    value_cdf, category_cdf = _cdf_payload(kind, entry)
    return ValueCalibration(
        str(pair_bucket_key), kind, sample_sd, ddof, value_field, value_cdf, category_cdf
    )


def _continuous_percentile(value: float, cdf: ContinuousCdf) -> float:
    position = bisect.bisect_left(cdf.support, value)
    if position < len(cdf.support) and cdf.support[position] == value:
        return cdf.midranks[position]
    if position == len(cdf.support):
        return 1.0
    return cdf.cumulative_less[position] / cdf.total


def _ordinal_percentile(category_id: str, cdf: OrdinalCdf) -> float:
    try:
        return cdf.midrank_by_category[category_id]
    except KeyError as exc:
        raise ValueError(f"ordinal category {category_id!r} is absent from its CDF") from exc


def record_percentile(record: Mapping[str, Any], calibration: ValueCalibration) -> float | None:
    kind = calibration.measurement_kind
    if str(record.get("measurement_kind")) != kind:
        raise ValueError("record and calibration measurement kinds disagree")
    if kind == "binary":
        return None
    if kind == "continuous":
        if calibration.value_cdf is None:
            raise ValueError("continuous calibration lacks parsed value CDF")
        return _continuous_percentile(geometry_value(record), calibration.value_cdf)
    category_id = str(record.get("canonical_category_id") or "")
    if not category_id:
        raise ValueError("ordinal record lacks canonical_category_id")
    if calibration.category_cdf is None:
        raise ValueError("ordinal calibration lacks parsed category CDF")
    return _ordinal_percentile(category_id, calibration.category_cdf)


def _pair_equivalence_score(query: Mapping[str, Any], retrieval: Mapping[str, Any]) -> float:
    kind = str(query["measurement_kind"])
    if kind == "binary":
        same = str(query["canonical_category_id"]) == str(retrieval["canonical_category_id"])
        return BINARY_MATCH_PROBABILITY if same else 1.0 - BINARY_MATCH_PROBABILITY
    left = float(query["value_percentile"])
    right = float(retrieval["value_percentile"])
    return min(1.0 - TARGET_EPSILON, max(TARGET_EPSILON, 1.0 - abs(left - right)))


def validate_pair(query: Mapping[str, Any], retrieval: Mapping[str, Any]) -> None:
    fields = ("pair_bucket_key", "measurement_kind", "canonical_measurement_scale_id")
    if any(query.get(field) != retrieval.get(field) for field in fields):
        raise ValueError("v11 target requires one bucket, kind, and measurement scale")
    calibration = (
        "calibration_sample_standard_deviation",
        "calibration_standard_deviation_ddof",
        "calibration_standard_deviation_value_field",
    )
    if any(query.get(field) != retrieval.get(field) for field in calibration):
        raise ValueError("v11 pair records disagree on calibration scale")


def target_for(query: Mapping[str, Any], retrieval: Mapping[str, Any]) -> dict[str, float]:
    validate_pair(query, retrieval)
    distance = abs(float(retrieval["geometry_value"]) - float(query["geometry_value"]))
    probability = _pair_equivalence_score(query, retrieval)
    return {
        "distance": distance,
        "absolute_geometry_difference": distance,
        "target_a": probability,
        "target_b": 1.0 - probability,
        "calibration_sample_standard_deviation": float(
            query["calibration_sample_standard_deviation"]
        ),
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
    "BINARY_MATCH_PROBABILITY",
    "ContinuousCdf",
    "NONTRANSFER_MAX_PROBABILITY",
    "OrdinalCdf",
    "TARGET_CONTRACT",
    "TARGET_EPSILON",
    "TRANSFER_MIN_PROBABILITY",
    "ValueCalibration",
    "candidate_label",
    "geometry_value",
    "is_decisive",
    "record_percentile",
    "target_for",
    "validate_pair",
    "value_calibration",
]
