"""Second-CDF soft targets for the V11.1 assay-transfer release."""

from __future__ import annotations

import bisect
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from pipeline.v11_targets import (
    BINARY_MATCH_PROBABILITY,
    NONTRANSFER_MAX_PROBABILITY,
    TARGET_EPSILON,
    TRANSFER_MIN_PROBABILITY,
    validate_pair,
)


CALIBRATION_SCHEMA_VERSION = "within_pair_bucket_percentile_distance_cdf.v1"
TARGET_CONTRACT = CALIBRATION_SCHEMA_VERSION
REFERENCE_PAIR_LIMIT = 100_000
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


def _admissible_pair_count(rows: list[Mapping[str, Any]]) -> int:
    molecules = Counter(str(row["canonical_smiles"]) for row in rows)
    total = len(rows) * (len(rows) - 1) // 2
    return total - sum(count * (count - 1) // 2 for count in molecules.values())


def _exact_distances(rows: list[Mapping[str, Any]]) -> list[float]:
    distances = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1:]:
            if left["canonical_smiles"] == right["canonical_smiles"]:
                continue
            distances.append(abs(float(left["value_percentile"]) - float(right["value_percentile"])))
    return distances


def _hash_index(seed: str, draw: int, attempt: int, side: str, size: int) -> int:
    payload = f"{seed}:{draw}:{attempt}:{side}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16) % size


def _sample_distances(
    rows: list[Mapping[str, Any]], seed: str, count: int,
) -> list[float]:
    output, size = [], len(rows)
    for draw in range(count):
        attempt = 0
        while True:
            left = rows[_hash_index(seed, draw, attempt, "left", size)]
            right = rows[_hash_index(seed, draw, attempt, "right", size)]
            if left["canonical_smiles"] != right["canonical_smiles"]:
                break
            attempt += 1
            if attempt >= 100_000:
                raise RuntimeError("unable to sample a cross-molecule calibration pair")
        output.append(abs(float(left["value_percentile"]) - float(right["value_percentile"])))
    return output


def _reference_distances(
    rows: list[Mapping[str, Any]], task_id: str, bucket: str, pair_count: int,
) -> tuple[list[float], str]:
    if pair_count <= REFERENCE_PAIR_LIMIT:
        return _exact_distances(rows), "exact_unordered_cross_molecule"
    seed = f"v11.1-percentile-distance-cdf:{task_id}:{bucket}"
    return _sample_distances(rows, seed, REFERENCE_PAIR_LIMIT), "deterministic_hash_sample"


def _calibration_entry(
    rows: list[Mapping[str, Any]], task_id: str, bucket: str,
) -> tuple[dict[str, Any] | None, str | None]:
    pair_count = _admissible_pair_count(rows)
    if pair_count == 0:
        return None, "no_cross_molecule_pairs"
    distances, method = _reference_distances(rows, task_id, bucket, pair_count)
    counts = Counter(distances)
    support = sorted(counts)
    if len(support) < 2:
        return None, "degenerate_percentile_distances"
    return {
        "measurement_kind": str(rows[0]["measurement_kind"]),
        "record_count": len(rows),
        "distinct_molecule_count": len({str(row["canonical_smiles"]) for row in rows}),
        "admissible_unordered_pair_count": pair_count,
        "reference_method": method,
        "reference_pair_count": len(distances),
        "support_values": support,
        "support_counts": [counts[value] for value in support],
    }, None


def build_distance_calibration(
    rows: Iterable[Mapping[str, Any]], task_id: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["measurement_kind"]) in {"continuous", "ordinal"}:
            grouped[str(row["pair_bucket_key"])].append(row)
    buckets, rejected = {}, Counter()
    for bucket, values in sorted(grouped.items()):
        entry, reason = _calibration_entry(values, task_id, bucket)
        if entry is None:
            rejected[str(reason)] += 1
        else:
            buckets[bucket] = entry
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "task_id": task_id,
        "fit_scope": "full_eligible_before_reservation_and_source_sampling",
        "pair_scope": "unordered_cross_molecule_within_pair_bucket",
        "reference_pair_limit": REFERENCE_PAIR_LIMIT,
        "tie_convention": "endpoint_anchored_empirical_midrank",
        "unseen_distance_rule": "reference_distances_strictly_less_than_value",
        "buckets": buckets,
        "summary": {
            "calibrated_buckets": len(buckets),
            "rejected_bucket_reasons": dict(sorted(rejected.items())),
        },
    }


def write_distance_calibration(document: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    temporary.write_bytes(gzip.compress(payload, mtime=0))
    temporary.replace(path)


def load_distance_calibration(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unexpected V11.1 distance-calibration schema")
    return document


def _parse_distance_cdf(bucket: str, entry: Mapping[str, Any]) -> DistanceCdf:
    support = tuple(float(value) for value in entry["support_values"])
    counts = tuple(int(value) for value in entry["support_counts"])
    if len(support) < 2 or len(support) != len(counts):
        raise ValueError(f"bucket {bucket!r} has invalid distance-CDF support")
    cumulative, running = [], 0
    for count in counts:
        cumulative.append(running)
        running += count
    minimum = 0.5 * counts[0] / running
    maximum = (running - 0.5 * counts[-1]) / running
    if maximum <= minimum:
        raise ValueError(f"bucket {bucket!r} has degenerate anchored distance CDF")
    return DistanceCdf(bucket, support, counts, tuple(cumulative), running, minimum, maximum)


def parsed_calibrations(document: Mapping[str, Any]) -> dict[str, DistanceCdf]:
    return {
        str(bucket): _parse_distance_cdf(str(bucket), entry)
        for bucket, entry in document["buckets"].items()
    }


def attach_distance_calibrations(
    rows: Iterable[dict[str, Any]], calibrations: Mapping[str, DistanceCdf],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output, rejected = [], Counter()
    for row in rows:
        if str(row["measurement_kind"]) == "binary":
            output.append(row)
            continue
        calibration = calibrations.get(str(row["pair_bucket_key"]))
        if calibration is None:
            rejected[str(row["measurement_kind"])] += 1
            continue
        row[CALIBRATION_FIELD] = calibration
        output.append(row)
    return output, dict(sorted(rejected.items()))


def distance_percentile(distance: float, cdf: DistanceCdf) -> float:
    position = bisect.bisect_left(cdf.support, distance)
    if position < len(cdf.support) and cdf.support[position] == distance:
        raw = (cdf.cumulative_less[position] + 0.5 * cdf.counts[position]) / cdf.total
    elif position == len(cdf.support):
        raw = 1.0
    else:
        raw = cdf.cumulative_less[position] / cdf.total
    anchored = (raw - cdf.minimum_midrank) / (cdf.maximum_midrank - cdf.minimum_midrank)
    return min(1.0, max(0.0, anchored))


def _binary_probability(query: Mapping[str, Any], retrieval: Mapping[str, Any]) -> float:
    same = str(query["canonical_category_id"]) == str(retrieval["canonical_category_id"])
    return BINARY_MATCH_PROBABILITY if same else 1.0 - BINARY_MATCH_PROBABILITY


def target_for(query: Mapping[str, Any], retrieval: Mapping[str, Any]) -> dict[str, Any]:
    validate_pair(query, retrieval)
    raw_distance = abs(float(retrieval["geometry_value"]) - float(query["geometry_value"]))
    if str(query["measurement_kind"]) == "binary":
        probability = _binary_probability(query, retrieval)
        percentile_distance, percentile = None, None
    else:
        percentile_distance = abs(float(query["value_percentile"]) - float(retrieval["value_percentile"]))
        calibration = query.get(CALIBRATION_FIELD)
        if not isinstance(calibration, DistanceCdf) or retrieval.get(CALIBRATION_FIELD) != calibration:
            raise ValueError("V11.1 pair lacks one shared percentile-distance calibration")
        percentile = distance_percentile(percentile_distance, calibration)
        probability = min(1.0 - TARGET_EPSILON, max(TARGET_EPSILON, 1.0 - percentile))
    return {
        "distance": raw_distance,
        "absolute_geometry_difference": raw_distance,
        "percentile_distance": percentile_distance,
        "percentile_distance_cdf": percentile,
        "target_a": probability,
        "target_b": 1.0 - probability,
        "calibration_sample_standard_deviation": float(query["calibration_sample_standard_deviation"]),
    }


def candidate_label(query: Mapping[str, Any], retrieval: Mapping[str, Any]) -> str | None:
    probability = target_for(query, retrieval)["target_a"]
    if probability > TRANSFER_MIN_PROBABILITY:
        return "transfer"
    if probability < NONTRANSFER_MAX_PROBABILITY:
        return "nontransfer"
    return None


def is_decisive(row: Mapping[str, Any], _query: Mapping[str, Any]) -> bool:
    probability = float(row["target_a"])
    return probability > TRANSFER_MIN_PROBABILITY or probability < NONTRANSFER_MAX_PROBABILITY


__all__ = [
    "CALIBRATION_SCHEMA_VERSION", "DistanceCdf", "REFERENCE_PAIR_LIMIT", "TARGET_CONTRACT",
    "attach_distance_calibrations", "build_distance_calibration", "candidate_label",
    "distance_percentile", "is_decisive", "load_distance_calibration", "parsed_calibrations",
    "target_for", "write_distance_calibration",
]
