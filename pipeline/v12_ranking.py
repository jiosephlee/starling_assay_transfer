"""Two-pool V12 ranking construction with held-out queries and train candidates."""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from pipeline.pair_core import record_key, stable_hash
from pipeline.v12_targets import target_for


MINIMUM_SAME_CATEGORY = 3
MINIMUM_RETRIEVAL_PARENTS = 2


@dataclass(frozen=True)
class RankingAnchor:
    family: str
    query: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]


def _pool_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row["source_id"]), str(row["measurement_kind"]),
        str(row["canonical_endpoint_key"]),
        str(row.get("canonical_measurement_scale_id") or ""),
        str(row["pair_bucket_key"]),
    )


def _candidate_index(rows: Iterable[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    output: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[_pool_key(row)].append(row)
    return dict(output)


def _hash_order(rows: Iterable[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: stable_hash(f"{salt}:{record_key(row)}"))


def _queries(rows: Iterable[dict[str, Any]], family: str, split: str) -> list[dict[str, Any]]:
    kinds = {"continuous"} if family == "continuous" else {"binary", "ordinal"}
    selected = [row for row in rows if str(row["measurement_kind"]) in kinds]
    if family == "continuous":
        selected = [row for row in selected if _measurable_sd(row)]
    return _hash_order(selected, f"v12-ranking-query:{split}:{family}")


def _measurable_sd(row: Mapping[str, Any]) -> bool:
    value = row.get("calibration_sample_standard_deviation")
    return value is not None and math.isfinite(float(value)) and float(value) > 0


def _available(
    query: Mapping[str, Any], pool: Iterable[dict[str, Any]], degree: Mapping[str, int], cap: int,
) -> list[dict[str, Any]]:
    return [
        row for row in pool
        if str(row["child_id"]) != str(query["child_id"])
        and degree.get(record_key(row), 0) < cap
    ]


def _two_parents(rows: Iterable[Mapping[str, Any]]) -> bool:
    return len({str(row["normalized_parent_identity_key"]) for row in rows}) >= 2


def _swap_for_two_parents(
    selected: list[dict[str, Any]], ordered: list[dict[str, Any]],
    preserve: Callable[[list[dict[str, Any]]], bool],
) -> list[dict[str, Any]]:
    if _two_parents(selected):
        return selected
    selected_ids = {record_key(row) for row in selected}
    for candidate in ordered:
        if record_key(candidate) in selected_ids:
            continue
        for index in range(len(selected) - 1, -1, -1):
            replaced = [*selected[:index], candidate, *selected[index + 1:]]
            if _two_parents(replaced) and preserve(replaced):
                return replaced
    return []


def _continuous(
    query: dict[str, Any], available: list[dict[str, Any]], width: int,
    target_builder: Callable,
) -> list[dict[str, Any]]:
    if len(available) < width:
        return []
    scored = sorted(
        ((row, float(target_builder(query, row)["target_a"])) for row in available),
        key=lambda item: (item[1], stable_hash(record_key(item[0]))),
    )
    low, high = scored[0][1], scored[-1][1]
    desired = [low + index * (high - low) / (width - 1) for index in range(width)]
    selected: list[dict[str, Any]] = []
    for value in desired:
        options = [item for item in scored if item[0] not in selected]
        selected.append(min(options, key=lambda item: (abs(item[1] - value), record_key(item[0])))[0])
    ordered = [row for row, _ in scored]
    return _swap_for_two_parents(selected, ordered, lambda _rows: True)


def _categorical_constraints(rows: Iterable[Mapping[str, Any]], category: str) -> bool:
    categories = [str(row.get("canonical_category_id") or "") for row in rows]
    return categories.count(category) >= MINIMUM_SAME_CATEGORY and any(
        value != category for value in categories
    )


def _swap_category(
    selected: list[dict[str, Any]], ordered: list[dict[str, Any]], category: str,
) -> list[dict[str, Any]]:
    def count_same(rows):
        return sum(str(row.get("canonical_category_id") or "") == category for row in rows)

    selected_ids = {record_key(row) for row in selected}
    while count_same(selected) < MINIMUM_SAME_CATEGORY:
        candidate = next((
            row for row in ordered
            if record_key(row) not in selected_ids
            and str(row.get("canonical_category_id") or "") == category
        ), None)
        replace = next((
            index for index in range(len(selected) - 1, -1, -1)
            if str(selected[index].get("canonical_category_id") or "") != category
        ), None)
        if candidate is None or replace is None:
            return []
        selected_ids.remove(record_key(selected[replace]))
        selected[replace] = candidate
        selected_ids.add(record_key(candidate))
    if all(str(row.get("canonical_category_id") or "") == category for row in selected):
        candidate = next((
            row for row in ordered
            if record_key(row) not in selected_ids
            and str(row.get("canonical_category_id") or "") != category
        ), None)
        if candidate is None:
            return []
        selected[-1] = candidate
    return selected


def _categorical(
    query: dict[str, Any], available: list[dict[str, Any]], width: int, split: str,
) -> list[dict[str, Any]]:
    category = str(query.get("canonical_category_id") or "")
    ordered = _hash_order(available, f"v12-ranking-candidate:{split}:{record_key(query)}")
    if len(ordered) < width:
        return []
    selected = _swap_category(ordered[:width], ordered, category)
    if not selected:
        return []
    preserve = lambda rows: _categorical_constraints(rows, category)
    return _swap_for_two_parents(selected, ordered, preserve)


def _select(
    family: str, query: dict[str, Any], pool: list[dict[str, Any]], width: int,
    degree: Mapping[str, int], cap: int, split: str, target_builder: Callable,
) -> list[dict[str, Any]]:
    available = _available(query, pool, degree, cap)
    if family == "continuous":
        return _continuous(query, available, width, target_builder)
    return _categorical(query, available, width, split)


def _family(
    family: str, query_rows: list[dict[str, Any]], train_pools: Mapping[tuple, list[dict[str, Any]]],
    task: Mapping[str, Any], split: str, target_builder: Callable,
) -> tuple[list[RankingAnchor], dict[str, Any]]:
    width = int(task["construction"]["ranking_anchor_width"])
    cap = int(task["construction"]["ranking_record_degree_cap"])
    quotas = task["construction"]["ranking_anchors"][split][family]
    by_source: dict[str, deque] = defaultdict(deque)
    for row in _queries(query_rows, family, split):
        by_source[str(row["source_id"])].append(row)
    degree: Counter = Counter()
    achieved: Counter = Counter()
    anchors, used_query_parents = [], set()
    for phase in task["priority_phases"]:
        active = {source for source in phase if int(quotas[source]) > 0}
        while active:
            for source in sorted(tuple(active), key=stable_hash):
                anchor = None
                while by_source[source]:
                    query = by_source[source].popleft()
                    parent = str(query["normalized_parent_identity_key"])
                    if parent in used_query_parents:
                        continue
                    selected = _select(
                        family, query, train_pools.get(_pool_key(query), []), width,
                        degree, cap, split, target_builder,
                    )
                    if len(selected) == width:
                        anchor = RankingAnchor(family, query, tuple(selected))
                        break
                if anchor is None:
                    active.discard(source)
                    continue
                anchors.append(anchor)
                used_query_parents.add(str(anchor.query["normalized_parent_identity_key"]))
                degree[record_key(anchor.query)] += width
                degree.update(record_key(row) for row in anchor.candidates)
                achieved[source] += 1
                if achieved[source] >= int(quotas[source]):
                    active.discard(source)
    shortfalls = {
        source: {"target": int(target), "achieved": achieved[source], "reason": "constraints_exhausted"}
        for source, target in quotas.items() if achieved[source] < int(target)
    }
    return anchors, {
        "target_by_source": dict(quotas), "achieved_by_source": dict(sorted(achieved.items())),
        "shortfalls": shortfalls, "observed_max_record_degree": max(degree.values(), default=0),
    }


def build_ranking_anchors(
    query_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]],
    task: Mapping[str, Any], split: str, target_builder: Callable = target_for,
) -> tuple[list[RankingAnchor], dict[str, Any]]:
    if split not in {"validation", "test"}:
        raise ValueError("ranking split must be validation or test")
    pools = _candidate_index(train_rows)
    anchors, families = [], {}
    for family in ("continuous", "categorical"):
        selected, report = _family(family, query_rows, pools, task, split, target_builder)
        anchors.extend(selected)
        families[family] = report
    return anchors, {
        "ranking_anchor_width": int(task["construction"]["ranking_anchor_width"]),
        "retrieval_candidate_split": "train", "query_split": split,
        "categorical_minimum_same_category": MINIMUM_SAME_CATEGORY,
        "categorical_minimum_mismatch": 1,
        "minimum_distinct_retrieval_parents": MINIMUM_RETRIEVAL_PARENTS,
        "categorical_selection": "deterministic_hash_shuffle_then_minimal_constraint_swaps",
        "continuous_selection": "value_spanning_then_minimal_parent_swap",
        "families": families,
    }


def materialize_ranking_rows(
    anchors: list[RankingAnchor], split: str, row_builder: Callable,
    metadata_builder: Callable, decisive: Callable,
) -> list[dict[str, Any]]:
    output = []
    release_split = f"{split}_ranking"
    for number, anchor in enumerate(anchors):
        query_id = f"{split}-rank-{number:05d}"
        for member, retrieval in enumerate(anchor.candidates):
            row = row_builder(anchor.query, retrieval, release_split)
            row["is_decisive"] = decisive(row, anchor.query)
            metadata_builder(row)
            row.update({
                "ranking_query_id": query_id, "ranking_member_index": member,
                "ranking_family": anchor.family,
                "ranking_query_category_id": anchor.query.get("canonical_category_id"),
            })
            output.append(row)
    return output


__all__ = [
    "MINIMUM_RETRIEVAL_PARENTS", "MINIMUM_SAME_CATEGORY", "RankingAnchor",
    "build_ranking_anchors", "materialize_ranking_rows",
]
