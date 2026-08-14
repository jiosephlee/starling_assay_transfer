"""Deterministic hierarchical construction of V11 ranking-validation anchors."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from pipeline.pair_core import pair_id, record_key, stable_hash
from pipeline.v11_targets import target_for


CATEGORICAL_MATCHES = 5


@dataclass(frozen=True)
class RankingAnchor:
    family: str
    query: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]


def _interleave(groups: Mapping[str, Iterable[Any]]) -> list[Any]:
    queues = {key: deque(values) for key, values in groups.items() if values}
    output: list[Any] = []
    while queues:
        for key in sorted(tuple(queues), key=stable_hash):
            output.append(queues[key].popleft())
            if not queues[key]:
                del queues[key]
    return output


def _category_key(row: Mapping[str, Any], family: str) -> str:
    if family == "continuous":
        return "__continuous__"
    category = str(row.get("canonical_category_id") or "")
    if not category:
        raise ValueError("categorical ranking record lacks canonical_category_id")
    return category


def _ordered_queries(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    leaves: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        endpoint = str(row["canonical_endpoint_key"])
        leaves[endpoint][_category_key(row, family)][str(row["pair_bucket_key"])].append(row)
    endpoint_sequences: dict[str, list[dict[str, Any]]] = {}
    for endpoint, categories in leaves.items():
        category_sequences: dict[str, list[dict[str, Any]]] = {}
        for category, buckets in categories.items():
            ordered_buckets = {
                bucket: sorted(values, key=lambda item: stable_hash(record_key(item)))
                for bucket, values in buckets.items()
            }
            category_sequences[category] = _interleave(ordered_buckets)
        endpoint_sequences[endpoint] = _interleave(category_sequences)
    return _interleave(endpoint_sequences)


def _pool_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row["source_id"]), str(row["measurement_kind"]),
        str(row["canonical_endpoint_key"]),
        str(row.get("canonical_measurement_scale_id") or ""),
        str(row["pair_bucket_key"]),
    )


def _candidate_index(rows: Iterable[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    indexed: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        indexed[_pool_key(row)].append(row)
    return {
        key: sorted(values, key=lambda item: stable_hash(record_key(item)))
        for key, values in indexed.items()
    }


def _available_candidates(
    query: dict[str, Any], pool: list[dict[str, Any]], used: set[str],
    degree: Mapping[str, int], cap: int,
) -> list[dict[str, Any]]:
    output = []
    for candidate in pool:
        pair = pair_id(query, candidate)
        if candidate["canonical_smiles"] == query["canonical_smiles"] or pair in used:
            continue
        if degree.get(record_key(candidate), 0) >= cap:
            continue
        output.append(candidate)
    return output


def _continuous_candidates(
    query: dict[str, Any], available: list[dict[str, Any]], width: int,
    target_builder: Callable,
) -> list[dict[str, Any]]:
    if len(available) < width:
        return []
    scored = sorted(
        ((item, target_builder(query, item)["target_a"]) for item in available),
        key=lambda pair: (pair[1], stable_hash(record_key(pair[0]))),
    )
    low, high = scored[0][1], scored[-1][1]
    desired = [low + index * (high - low) / (width - 1) for index in range(width)]
    chosen: list[dict[str, Any]] = []
    molecule_counts: Counter = Counter()
    for target in desired:
        options = [pair for pair in scored if pair[0] not in chosen]
        options = [pair for pair in options if molecule_counts[pair[0]["canonical_smiles"]] < 2]
        if not options:
            return []
        candidate = min(options, key=lambda pair: (abs(pair[1] - target), record_key(pair[0])))[0]
        chosen.append(candidate)
        molecule_counts[candidate["canonical_smiles"]] += 1
    return chosen


def _take_distinct_molecules(
    candidates: Iterable[dict[str, Any]], count: int, molecule_counts: Counter,
) -> list[dict[str, Any]]:
    selected = []
    for candidate in candidates:
        molecule = str(candidate["canonical_smiles"])
        if molecule_counts[molecule] >= 2:
            continue
        selected.append(candidate)
        molecule_counts[molecule] += 1
        if len(selected) == count:
            break
    return selected


def _other_category_candidates(
    candidates: list[dict[str, Any]], query_category: str, count: int,
    molecule_counts: Counter,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        category = str(candidate.get("canonical_category_id") or "")
        if category and category != query_category:
            grouped[category].append(candidate)
    ordered = _interleave(grouped)
    return _take_distinct_molecules(ordered, count, molecule_counts)


def _categorical_candidates(
    query: dict[str, Any], available: list[dict[str, Any]], width: int,
) -> list[dict[str, Any]]:
    query_category = str(query["canonical_category_id"])
    molecule_counts: Counter = Counter()
    same = [item for item in available if str(item.get("canonical_category_id")) == query_category]
    selected_same = _take_distinct_molecules(same, CATEGORICAL_MATCHES, molecule_counts)
    other_count = width - CATEGORICAL_MATCHES
    selected_other = _other_category_candidates(
        available, query_category, other_count, molecule_counts
    )
    if len(selected_same) != CATEGORICAL_MATCHES or len(selected_other) != other_count:
        return []
    return selected_same + selected_other


def _select_candidates(
    family: str, query: dict[str, Any], pool: list[dict[str, Any]], width: int,
    used: set[str], degree: Mapping[str, int], cap: int, target_builder: Callable,
) -> list[dict[str, Any]]:
    available = _available_candidates(query, pool, used, degree, cap)
    if family == "continuous":
        return _continuous_candidates(query, available, width, target_builder)
    return _categorical_candidates(query, available, width)


def _source_anchor(
    family: str, schedule: deque, pools: Mapping[tuple, list[dict[str, Any]]], width: int,
    used: set[str], degree: Mapping[str, int], anchor_molecules: set[str], cap: int,
    target_builder: Callable,
) -> RankingAnchor | None:
    while schedule:
        query = schedule.popleft()
        molecule = str(query["canonical_smiles"])
        remaining = cap - degree.get(record_key(query), 0)
        if molecule in anchor_molecules or remaining < width:
            continue
        candidates = _select_candidates(
            family, query, pools.get(_pool_key(query), []), width,
            used, degree, cap, target_builder,
        )
        if len(candidates) == width:
            return RankingAnchor(family, query, tuple(candidates))
    return None


def _commit_anchor(
    anchor: RankingAnchor, used: set[str], degree: dict[str, int],
    anchor_molecules: set[str],
) -> None:
    anchor_molecules.add(str(anchor.query["canonical_smiles"]))
    degree[record_key(anchor.query)] += len(anchor.candidates)
    for candidate in anchor.candidates:
        degree[record_key(candidate)] += 1
        used.add(pair_id(anchor.query, candidate))


def _family_diagnostics(
    family: str, quotas: Mapping[str, int], eligible: Counter, anchors: list[RankingAnchor],
) -> dict[str, Any]:
    achieved = Counter(str(anchor.query["source_id"]) for anchor in anchors)
    endpoints = Counter(
        (str(anchor.query["source_id"]), str(anchor.query["canonical_endpoint_key"]))
        for anchor in anchors
    )
    categories = Counter(
        (str(anchor.query["source_id"]), str(anchor.query.get("canonical_category_id") or ""))
        for anchor in anchors if family == "categorical"
    )
    shortfalls = {}
    for source, target in quotas.items():
        actual = achieved[source]
        if actual < int(target):
            reason = "no_eligible_rows" if eligible[source] == 0 else "constraints_exhausted"
            shortfalls[source] = {"target": int(target), "achieved": actual, "reason": reason}
    return {
        "target_by_source": dict(quotas), "achieved_by_source": dict(sorted(achieved.items())),
        "eligible_rows_by_source": dict(sorted(eligible.items())),
        "achieved_by_source_endpoint": {f"{a}|{b}": count for (a, b), count in sorted(endpoints.items())},
        "achieved_by_source_query_category": {
            f"{a}|{b}": count for (a, b), count in sorted(categories.items())
        },
        "shortfalls": shortfalls,
    }


def _build_family(
    family: str, rows: list[dict[str, Any]], task: Mapping[str, Any],
    release: Mapping[str, Any], width: int, cap: int, target_builder: Callable,
) -> tuple[list[RankingAnchor], dict[str, Any]]:
    kinds = set(release["ranking_families"][family])
    family_rows = [row for row in rows if str(row["measurement_kind"]) in kinds]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in family_rows:
        by_source[str(row["source_id"])].append(row)
    schedules = {source: deque(_ordered_queries(values, family)) for source, values in by_source.items()}
    pools, anchors = _candidate_index(family_rows), []
    used, anchor_molecules, degree = set(), set(), defaultdict(int)
    quotas = task["construction"]["ranking_anchors"][family]
    achieved: Counter = Counter()
    for phase in task["priority_phases"]:
        active = {source for source in phase if int(quotas[source]) > 0}
        while active:
            for source in sorted(tuple(active), key=stable_hash):
                anchor = _source_anchor(
                    family, schedules.get(source, deque()), pools, width,
                    used, degree, anchor_molecules, cap, target_builder,
                )
                if anchor is None:
                    active.discard(source)
                    continue
                _commit_anchor(anchor, used, degree, anchor_molecules)
                anchors.append(anchor)
                achieved[source] += 1
                if achieved[source] >= int(quotas[source]):
                    active.discard(source)
    eligible = Counter(str(row["source_id"]) for row in family_rows)
    return anchors, _family_diagnostics(family, quotas, eligible, anchors)


def build_ranking_anchors(
    rows: list[dict[str, Any]], task: Mapping[str, Any], release: Mapping[str, Any],
    target_builder: Callable = target_for,
) -> tuple[list[RankingAnchor], dict[str, Any]]:
    construction = task["construction"]
    width = int(construction["ranking_anchor_width"])
    cap = int(construction["ranking_record_degree_cap"])
    anchors, diagnostics = [], {}
    for family in ("continuous", "categorical"):
        selected, report = _build_family(
            family, rows, task, release, width, cap, target_builder
        )
        anchors.extend(selected)
        diagnostics[family] = report
    return anchors, {
        "ranking_anchor_width": width,
        "categorical_same_category_candidates": CATEGORICAL_MATCHES,
        "directed_pair_policy": "each_direction_at_most_once",
        "maximum_unordered_pair_uses": 2,
        "record_degree_definition": "emitted_pair_rows_across_query_and_retrieval_roles",
        "families": diagnostics,
    }


def materialize_ranking_rows(
    anchors: list[RankingAnchor], row_builder: Callable, metadata_builder: Callable,
    decisive: Callable,
) -> list[dict[str, Any]]:
    output = []
    for number, anchor in enumerate(anchors):
        query_id = f"validation-rank-{number:05d}"
        for member, retrieval in enumerate(anchor.candidates):
            row = row_builder(anchor.query, retrieval, "validation_ranking")
            row["is_decisive"] = decisive(row, anchor.query)
            metadata_builder(row)
            row.update({
                "ranking_query_id": query_id,
                "ranking_member_index": member,
                "ranking_family": anchor.family,
                "ranking_query_category_id": anchor.query.get("canonical_category_id"),
            })
            output.append(row)
    return output


__all__ = [
    "CATEGORICAL_MATCHES", "RankingAnchor", "build_ranking_anchors",
    "materialize_ranking_rows",
]
