"""Train-only all-distinct-record percentile-distance targets for V12.1."""

from __future__ import annotations

import bisect
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pipeline.v11_targets import (
    BINARY_MATCH_PROBABILITY, NONTRANSFER_MAX_PROBABILITY, TARGET_EPSILON,
    TRANSFER_MIN_PROBABILITY, validate_pair,
)
from pipeline.v12_source import DISTANCE_CALIBRATION_SCHEMA


TARGET_CONTRACT = "train_only_all_distinct_record_percentile_distance_cdf.v12.1"
CALIBRATION_FIELD = "_percentile_distance_calibration"


@dataclass(frozen=True)
class DistanceCdf:
    pair_bucket_key: str
    support: tuple[float, ...]
    counts: tuple[int, ...]
    cumulative_less: tuple[int, ...]
    total: int
    minimum_midrank: float
    maximum_midrank: float


def load_distance_calibration(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != DISTANCE_CALIBRATION_SCHEMA:
        raise ValueError("unexpected V12.1 distance-calibration schema")
    if document.get("fit_split") != "train":
        raise ValueError("V12.1 distance calibration is not train-only")
    return document


def _parse(bucket: str, entry: Mapping[str, Any]) -> DistanceCdf:
    support = tuple(float(value) for value in entry["support_values"])
    counts = tuple(int(value) for value in entry["support_counts"])
    if not support or len(support) != len(counts):
        raise ValueError(f"bucket {bucket!r} has invalid distance support")
    cumulative, running = [], 0
    for count in counts:
        cumulative.append(running)
        running += count
    minimum = 0.5 * counts[0] / running
    maximum = (running - 0.5 * counts[-1]) / running
    return DistanceCdf(
        bucket, support, counts, tuple(cumulative), running, minimum, maximum
    )


def parsed_calibrations(document: Mapping[str, Any]) -> dict[str, DistanceCdf]:
    return {
        str(bucket): _parse(str(bucket), entry)
        for bucket, entry in document["buckets"].items()
    }


def distance_percentile(distance: float, cdf: DistanceCdf) -> float:
    if len(cdf.support) == 1:
        return 0.0 if distance <= cdf.support[0] else 1.0
    position = bisect.bisect_left(cdf.support, distance)
    if position < len(cdf.support) and cdf.support[position] == distance:
        raw = (cdf.cumulative_less[position] + 0.5 * cdf.counts[position]) / cdf.total
    elif position == len(cdf.support):
        raw = 1.0
    else:
        raw = cdf.cumulative_less[position] / cdf.total
    anchored = (raw - cdf.minimum_midrank) / (cdf.maximum_midrank - cdf.minimum_midrank)
    return min(1.0, max(0.0, anchored))


def target_for(query: Mapping[str, Any], retrieval: Mapping[str, Any]) -> dict[str, Any]:
    validate_pair(query, retrieval)
    raw_distance = abs(float(retrieval["geometry_value"]) - float(query["geometry_value"]))
    if str(query["measurement_kind"]) == "binary":
        same = query["canonical_category_id"] == retrieval["canonical_category_id"]
        probability = BINARY_MATCH_PROBABILITY if same else 1.0 - BINARY_MATCH_PROBABILITY
        percentile_distance, percentile = None, None
    else:
        percentile_distance = abs(
            float(query["value_percentile"]) - float(retrieval["value_percentile"])
        )
        calibration = query.get(CALIBRATION_FIELD)
        if not isinstance(calibration, DistanceCdf) or retrieval.get(CALIBRATION_FIELD) != calibration:
            raise ValueError("V12.1 pair lacks one shared train-only distance calibration")
        percentile = distance_percentile(percentile_distance, calibration)
        probability = min(1.0 - TARGET_EPSILON, max(TARGET_EPSILON, 1.0 - percentile))
    return {
        "distance": raw_distance, "absolute_geometry_difference": raw_distance,
        "percentile_distance": percentile_distance, "percentile_distance_cdf": percentile,
        "target_a": probability, "target_b": 1.0 - probability,
        "calibration_sample_standard_deviation": query.get(
            "calibration_sample_standard_deviation"
        ),
    }


def is_decisive(row: Mapping[str, Any], _query: Mapping[str, Any]) -> bool:
    value = float(row["target_a"])
    return value > TRANSFER_MIN_PROBABILITY or value < NONTRANSFER_MAX_PROBABILITY


__all__ = [
    "CALIBRATION_FIELD", "DistanceCdf", "TARGET_CONTRACT", "distance_percentile",
    "is_decisive", "load_distance_calibration", "parsed_calibrations", "target_for",
]
