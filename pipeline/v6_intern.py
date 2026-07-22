"""Deterministic V6 raw-record pair construction and Intern prompt rendering."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from jinja2 import Environment, FileSystemLoader, StrictUndefined

# v6.5.1: prompts are rendered from per-concept Jinja templates (one file per assay concept)
# instead of a single hard-coded string. The eligible records are unchanged.
CONCEPTS = ("oral_bioavailability", "oral_exposure", "Fa", "Fg", "Fh")
TEMPLATE_DIR = str(Path(__file__).resolve().parents[1] / "templates" / "assay_transfer_v6_5_intern")

CONTEXT_FIELDS = ("species_or_population", "report_or_statistic_type", "dose",
                  "study_or_assay_system", "measured_process", "biological_context",
                  "medium", "formulation_or_solid_form", "transporter_or_enzyme",
                  "substrate_status", "intestinal_site", "molecular_form",
                  "enzyme_or_pathway", "qualifying_conditions", "comparator", "extra_details")
SPLIT_SIZES = {"train": 12482, "validation": 1250, "test": 1250}
SPLIT_RECORDS = {"train": 116060, "validation": 11276, "test": 11470}


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_key(row: dict[str, Any]) -> str:
    """Return the composed artifact's globally unique raw-record identifier."""
    return str(row["child_id"])


def _heldout_subset(weights: dict[str, int], split: str) -> set[str]:
    """Select an exact molecule/record quota using SHA-256 order as tie-breaking priority."""
    count, total = SPLIT_SIZES[split], SPLIT_RECORDS[split]
    excess, best = total - count, [None] * (total - count + 1)
    best[0] = (0, None, None)
    ordered = sorted(weights, key=lambda text: stable_hash("4878:v6-fresh:" + text))
    for smiles in ordered:
        delta = weights[smiles] - 1
        if delta <= 0 or delta > excess:
            continue
        for value in range(excess, delta - 1, -1):
            prior = best[value - delta]
            if prior is not None and prior[0] < count and best[value] is None:
                best[value] = (prior[0] + 1, smiles, value - delta)
    if best[excess] is None:
        raise RuntimeError(f"unable to meet exact {split} record quota")
    selected, cursor = set(), excess
    while cursor:
        _, smiles, cursor = best[cursor]
        selected.add(smiles)
    ones = [text for text in ordered if weights[text] == 1 and text not in selected]
    selected.update(ones[:count - len(selected)])
    if len(selected) != count or sum(weights[text] for text in selected) != total:
        raise RuntimeError(f"invalid exact {split} allocation")
    return selected


def molecule_splits(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Assign exact frozen molecule and record quotas with SHA-256 deterministic priority."""
    weights: dict[str, int] = defaultdict(int)
    for row in records:
        weights[str(row["canonical_smiles"])] += 1
    if len(weights) != sum(SPLIT_SIZES.values()):
        raise ValueError(f"expected 14982 molecules, found {len(weights)}")
    validation = _heldout_subset(weights, "validation")
    remaining = {key: value for key, value in weights.items() if key not in validation}
    test = _heldout_subset(remaining, "test")
    return {smiles: ("validation" if smiles in validation else "test" if smiles in test else "train")
            for smiles in weights}


# v6.5: label-smooth the A/B target instead of hard-clipping. Clipping flattened every strong
# transfer to exactly 0.9 (destroying pointwise ranking signal); the affine squash keeps the
# ~[eps, 1-eps] bounds while remaining monotonic in z, so stronger/weaker transfers stay ordered.
TARGET_SMOOTHING = 0.1   # epsilon: q_A in [eps, 1-eps]
TARGET_TEMPERATURE = 1.0  # T in sigmoid(z / T)


def target_for(query: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, float]:
    """Return the frozen continuous A/B target from real raw record values."""
    a, b = float(query["transfer_max"]), float(query["not_transfer_min"])
    distance = abs(float(retrieval["comparison_value"]) - float(query["comparison_value"]))
    z = math.log(9.0) * (a + b - 2.0 * distance) / (b - a)
    sigmoid = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, z / TARGET_TEMPERATURE))))
    probability = TARGET_SMOOTHING + (1.0 - 2.0 * TARGET_SMOOTHING) * sigmoid
    return {"distance": distance, "target_z": z, "target_a": probability,
            "target_b": 1.0 - probability}


def oriented_pair(query: dict[str, Any], retrieval: dict[str, Any]) -> bool:
    """Choose exactly one stable orientation for an unordered record pair."""
    left, right = sorted((record_key(query), record_key(retrieval)))
    return (stable_hash(f"4878:v6-direction:{left}:{right}")[-1] in "01234567") == (
        record_key(query) == left)


def pair_id(query: dict[str, Any], retrieval: dict[str, Any]) -> str:
    return stable_hash(f"v6:{record_key(query)}:{record_key(retrieval)}")


@lru_cache(maxsize=None)
def _templates(template_dir: str) -> dict[str, Any]:
    env = Environment(loader=FileSystemLoader(template_dir), undefined=StrictUndefined)
    available = {path.name for path in Path(template_dir).glob("*.jinja")}
    # per-concept file when present, otherwise a single shared `default.jinja`
    return {concept: env.get_template(f"{concept}.jinja" if f"{concept}.jinja" in available
                                      else "default.jinja") for concept in CONCEPTS}


def render_prompt(query: dict[str, Any], retrieval: dict[str, Any],
                  template_dir: str = TEMPLATE_DIR) -> str:
    """Render the per-concept Jinja prompt; the query record's value is never templated."""
    template = _templates(template_dir)[str(query["assay_concept"])]
    return template.render(query=query, retrieval=retrieval)


def eligible(query: dict[str, Any], retrieval: dict[str, Any]) -> bool:
    return (query["canonical_endpoint_key"] == retrieval["canonical_endpoint_key"] and
            query["canonical_smiles"] != retrieval["canonical_smiles"])


def row_for(query: dict[str, Any], retrieval: dict[str, Any], split: str) -> dict[str, Any]:
    target = target_for(query, retrieval)
    distribution = {"transfer": target["target_a"], "nontransfer": target["target_b"]}
    return {"prompt": render_prompt(query, retrieval), "completion": "(A)" if target["target_a"] >= .5 else "(B)",
            "split": split, "query_record_id": record_key(query), "retrieval_record_id": record_key(retrieval),
            "query_smiles": query["canonical_smiles"], "retrieval_smiles": retrieval["canonical_smiles"],
            "canonical_endpoint_key": query["canonical_endpoint_key"], "assay_concept": query["assay_concept"],
            "pair_id": pair_id(query, retrieval), "query_source_record_id": query["record_id"],
            "retrieval_source_record_id": retrieval["record_id"],
            "query_input_sha256": query["input_sha256"],
            "retrieval_input_sha256": retrieval["input_sha256"],
            "target_distribution": distribution,
            "metadata": {"target_distribution": distribution, "target_z": target["target_z"]},
            **target}


def condition_index(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        indexed[str(row["canonical_endpoint_key"])].append(row)
    return indexed


def candidate_rows(query: dict[str, Any], pool: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows = [row_for(query, other, split) for other in pool if eligible(query, other) and oriented_pair(query, other)]
    return sorted(rows, key=lambda row: row["pair_id"])


def unordered_pair_id(left: dict[str, Any], right: dict[str, Any]) -> str:
    first, second = sorted((record_key(left), record_key(right)))
    return stable_hash(f"v6-unordered:{first}:{second}")


def propose_records(query: dict[str, Any], pool: list[dict[str, Any]], count: int,
                    used: set[str], salt: str) -> list[dict[str, Any]]:
    """Deterministically probe a condition pool without relevance or model-score selection."""
    if not pool:
        return []
    size = len(pool)
    digest = int(stable_hash(f"4878:v6-propose:{salt}:{record_key(query)}"), 16)
    start, step = digest % size, (digest // size) % size or 1
    while math.gcd(step, size) != 1:
        step += 1
    output = []
    for offset in range(size):
        retrieval = pool[(start + offset * step) % size]
        key = unordered_pair_id(query, retrieval)
        if key not in used and eligible(query, retrieval) and oriented_pair(query, retrieval):
            output.append(retrieval)
            if len(output) == count:
                break
    return output


def _span_four(query: dict[str, Any], proposed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(proposed, key=lambda row: target_for(query, row)["target_z"])
    desired = [0, len(ranked) // 3, 2 * len(ranked) // 3, len(ranked) - 1]
    chosen, molecule_counts = [], defaultdict(int)
    for position in desired:
        order = sorted(range(len(ranked)), key=lambda index: (abs(index - position), index))
        candidate = next((ranked[index] for index in order if ranked[index] not in chosen and
                          molecule_counts[ranked[index]["canonical_smiles"]] < 2), None)
        if candidate is not None:
            chosen.append(candidate)
            molecule_counts[candidate["canonical_smiles"]] += 1
    return chosen


def iter_list_groups(records: list[dict[str, Any]], split: str, wanted: int):
    """Yield balanced four-candidate lists, redistributing exhausted concept capacity."""
    indexed, anchors = condition_index(records), defaultdict(list)
    for row in sorted(records, key=lambda item: stable_hash(record_key(item))):
        anchors[str(row["assay_concept"])].append(row)
    offsets, failures, active, used = defaultdict(int), defaultdict(int), set(anchors), set()
    group = 0
    while group < wanted and active:
        for concept in sorted(tuple(active)):
            values = anchors[concept]
            query = values[offsets[concept] % len(values)]
            offsets[concept] += 1
            proposed = propose_records(query, indexed[query["canonical_endpoint_key"]], 16,
                                       used, f"{group}:{offsets[concept]}")
            chosen = _span_four(query, proposed)
            if len(chosen) != 4:
                failures[concept] += 1
                if failures[concept] >= len(values):
                    active.remove(concept)
                continue
            failures[concept] = 0
            group_id = f"{split}-list-{group:09d}"
            rows = []
            for index, retrieval in enumerate(chosen):
                used.add(unordered_pair_id(query, retrieval))
                item = row_for(query, retrieval, split)
                item.update({"listnet_group_id": group_id, "listnet_group_index": group,
                             "listnet_query_group_id": record_key(query), "listnet_member_index": index})
                rows.append(item)
            yield rows
            group += 1
            if group >= wanted:
                break


def choose_spread(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Choose deterministic candidates across relevance range without target-based ranking selection."""
    if len(candidates) < count:
        return []
    ordered = sorted(candidates, key=lambda row: stable_hash(row["pair_id"]))
    return [ordered[round(index * (len(ordered) - 1) / (count - 1))]
            for index in range(count)]


def list_groups(records: list[dict[str, Any]], split: str, wanted: int) -> list[dict[str, Any]]:
    """Round-robin concept anchors into four-member ListNet lists."""
    return [row for group in iter_list_groups(records, split, wanted) for row in group]
