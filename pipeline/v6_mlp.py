"""Deterministic compact dataset construction helpers for V6 MLP models."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pipeline.pair_core import (CONTEXT_FIELDS, eligible, oriented_pair, pair_id,
                                record_key, stable_hash, target_for)


def assay_paragraph(row: dict[str, Any]) -> str:
    """Render a deterministic value-free assay paragraph from one real source record."""
    lines = [f"endpoint: {row['canonical_endpoint_key']}", f"concept: {row['assay_concept']}"]
    for field_name in CONTEXT_FIELDS:
        value = row.get("context_" + field_name) or "not specified"
        lines.append(f"{field_name.replace('_', ' ')}: {value}")
    return "\n".join(lines)


def condition_stats(records: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in records:
        value = float(row["comparison_value"])
        values["endpoint:" + str(row["canonical_endpoint_key"])].append(value)
        values["concept:" + str(row["assay_concept"])].append(value)
    output = {}
    for key, items in values.items():
        mean = sum(items) / len(items)
        variance = sum((item - mean) ** 2 for item in items) / max(1, len(items) - 1)
        output[key] = (mean, max(math.sqrt(variance), 1e-6))
    return output


def normalized_retrieval_value(row: dict[str, Any], stats: dict[str, tuple[float, float]]) -> float:
    endpoint = "endpoint:" + str(row["canonical_endpoint_key"])
    concept = "concept:" + str(row["assay_concept"])
    mean, scale = stats.get(endpoint, stats[concept])
    return max(-10.0, min(10.0, (float(row["comparison_value"]) - mean) / scale))


def model_record(row: dict[str, Any], index: int, molecule_index: int, split: str) -> dict[str, Any]:
    """Create a value-free model-input record; target values remain private to pair building."""
    return {"record_index": index, "record_id": record_key(row), "source_record_id": row["record_id"],
            "input_sha256": row["input_sha256"], "molecule_index": molecule_index,
            "canonical_smiles": row["canonical_smiles"],
            "canonical_endpoint_key": row["canonical_endpoint_key"],
            "assay_concept": row["assay_concept"], "unit_basis": row["unit_basis"],
            "assay_paragraph": assay_paragraph(row), "split": split}


def _probe_parameters(query: dict[str, Any], size: int) -> tuple[int, int]:
    digest = int(stable_hash("4878:v6-mlp-probe:" + record_key(query)), 16)
    start, step = digest % size, (digest // size) % size or 1
    while math.gcd(step, size) != 1:
        step += 1
    return start, step


def _take_proposals(query: dict[str, Any], pool: list[dict[str, Any]], cursor: int,
                    wanted: int) -> tuple[list[dict[str, Any]], int]:
    start, step = _probe_parameters(query, len(pool))
    proposals = []
    while cursor < len(pool) and len(proposals) < wanted:
        candidate = pool[(start + cursor * step) % len(pool)]
        cursor += 1
        if eligible(query, candidate) and oriented_pair(query, candidate):
            proposals.append(candidate)
    return proposals, cursor


def _spread_candidates(query: dict[str, Any], proposals: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    ranked = sorted(proposals, key=lambda row: (target_for(query, row)["target_z"], record_key(row)))
    if len(ranked) < size:
        return []
    positions = [round(slot * (len(ranked) - 1) / max(1, size - 1)) for slot in range(size)]
    return [ranked[position] for position in positions]


@dataclass
class MLPGroupSampler:
    records: list[dict[str, Any]]
    group_size: int = 40
    proposal_factor: int = 2
    pools: dict[str, list[dict[str, Any]]] = field(init=False)
    anchors: dict[str, list[dict[str, Any]]] = field(init=False)
    cursors: dict[str, int] = field(default_factory=dict)
    offsets: dict[str, int] = field(default_factory=dict)
    exhausted: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.pools, self.anchors = defaultdict(list), defaultdict(list)
        for row in self.records:
            self.pools[str(row["canonical_endpoint_key"])].append(row)
            self.anchors[str(row["assay_concept"])].append(row)
        for values in list(self.pools.values()) + list(self.anchors.values()):
            values.sort(key=lambda row: stable_hash(record_key(row)))

    def next_group(self, concept: str) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        anchors = self.anchors[concept]
        for _ in range(len(anchors)):
            offset = self.offsets.get(concept, 0)
            query = anchors[offset % len(anchors)]
            self.offsets[concept] = offset + 1
            key = record_key(query)
            if key in self.exhausted:
                continue
            pool = self.pools[str(query["canonical_endpoint_key"])]
            proposals, cursor = _take_proposals(
                query, pool, self.cursors.get(key, 0), self.group_size * self.proposal_factor)
            self.cursors[key] = cursor
            chosen = _spread_candidates(query, proposals, self.group_size)
            if len(chosen) == self.group_size:
                return query, chosen
            self.exhausted.add(key)
        return None


def compact_pair(query: dict[str, Any], retrieval: dict[str, Any], group: int, member: int,
                 indices: dict[str, int], stats: dict[str, tuple[float, float]]) -> dict[str, Any]:
    target = target_for(query, retrieval)
    return {"group_index": group, "member_index": member,
            "query_record_index": indices[record_key(query)],
            "retrieval_record_index": indices[record_key(retrieval)],
            "pair_id": bytes.fromhex(pair_id(query, retrieval)),
            "retrieval_value": normalized_retrieval_value(retrieval, stats),
            "retrieval_is_approximate": bool(retrieval["scalar_is_approximate"]), **target}


def iter_mlp_groups(records: list[dict[str, Any]], wanted: int, group_size: int = 40):
    sampler, concepts, active = MLPGroupSampler(records, group_size), [], set()
    concepts = sorted(sampler.anchors)
    active.update(concepts)
    group = 0
    while group < wanted and active:
        progressed = False
        for concept in concepts:
            if concept not in active:
                continue
            selected = sampler.next_group(concept)
            if selected is None:
                active.remove(concept)
                continue
            yield group, *selected
            group += 1
            progressed = True
            if group == wanted:
                break
        if not progressed:
            break
    if group != wanted:
        raise RuntimeError(f"MLP sampler produced {group:,} of {wanted:,} requested groups")


def membership_hash(pair_ids: list[str]) -> str:
    payload = "\n".join(pair_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
