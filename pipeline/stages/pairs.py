#!/usr/bin/env python3
"""Build directed, retrieval-anchored binary candidates from v3 eligible records."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.normalize.common_transfer import FingerprintCache
from pipeline.v3_policy import V3Policies, resolve_path, stable_hash

INT_FIELDS = {"n_records", "n_transfer", "n_nontransfer", "n_ambiguous"}
FLOAT_FIELDS = {"retrieved_value", "retrieved_comparison_value", "transfer_max",
                "not_transfer_min", "transfer_fraction", "nontransfer_fraction",
                "ambiguous_fraction", "majority_margin", "tanimoto"}


def _base_file(path: Path) -> Path:
    return path / "records.parquet" if path.is_dir() else path


def _load_rows(paths: list[Path], split_dir: Path) -> list[dict[str, Any]]:
    split_table = pq.read_table(split_dir / "molecule_splits.parquet")
    split_map = dict(zip(split_table["canonical_smiles"].to_pylist(), split_table["split"].to_pylist()))
    rows = []
    for path in paths:
        rows.extend(pq.read_table(_base_file(path)).to_pylist())
    for row in rows:
        row["_split"] = split_map.get(row["canonical_smiles"])
    return [row for row in rows if row["_split"]]


def _query_cap(args: argparse.Namespace, concept: str) -> int:
    caps = getattr(args, "query_caps", None)
    if caps:
        return int(caps[concept])
    value = getattr(args, "max_queries", None)
    return int(value or 0)


def _ranked_queries(retrieved: str, queries: list[str]) -> Iterable[str]:
    """Stable pseudo-random permutation without hashing/sorting every molecule pair."""
    count = len(queries)
    if count == 0:
        return
    digest = stable_hash(["query-order", retrieved])
    start = int(digest[:12], 16) % count
    step = 1 if count == 1 else 1 + int(digest[12:24], 16) % (count - 1)
    while math.gcd(step, count) != 1:
        step += 1
    for offset in range(count):
        yield queries[(start + offset * step) % count]


def _bucketed_queries(retrieved: str, queries: list[str], cap: int, cache: FingerprintCache,
                      boundary: float) -> tuple[list[tuple[str, float, str]], dict[str, bool]]:
    selected: dict[str, list[tuple[str, float, str]]] = {"low": [], "high": []}
    left = cache.get(retrieved)
    if left is None:
        return [], {"low": True, "high": True}
    for query in _ranked_queries(retrieved, queries):
        if query == retrieved:
            continue
        right = cache.get(query)
        if right is None:
            continue
        similarity = cache.similarity(left, right)
        bucket = "high" if similarity >= boundary else "low"
        if cap == 0 or len(selected[bucket]) < cap:
            selected[bucket].append((query, similarity, bucket))
        if cap and all(len(values) >= cap for values in selected.values()):
            break
    exhausted = {bucket: cap == 0 or len(values) < cap for bucket, values in selected.items()}
    return selected["low"] + selected["high"], exhausted


def _label(metric: Any, retrieved: float, evidence: list[float]) -> dict[str, Any]:
    votes = [metric.vote(retrieved, value) for value in evidence]
    transfer, nontransfer, total = votes.count("transfer"), votes.count("not_transfer"), len(votes)
    ambiguous = total - transfer - nontransfer
    binary = 1 if transfer * 2 > total else 0 if nontransfer * 2 > total else None
    return {"n_records": total, "n_transfer": transfer, "n_nontransfer": nontransfer,
            "n_ambiguous": ambiguous, "transfer_fraction": transfer / total,
            "nontransfer_fraction": nontransfer / total, "ambiguous_fraction": ambiguous / total,
            "binary_label": binary,
            "majority_margin": max(transfer, nontransfer) / total - 0.5 if binary is not None else None}


def _candidate(row: dict[str, Any], query: str, similarity: float, bucket: str,
               evidence: list[float], evidence_sources: set[str], metric: Any,
               versions: dict[str, str]) -> dict[str, Any]:
    vote = _label(metric, float(row["comparison_value"]), evidence)
    identity = [row["child_id"], query, row["canonical_endpoint_key"], row["_split"], versions]
    result = {"candidate_id": stable_hash(identity), "split": row["_split"],
              "canonical_endpoint_key": row["canonical_endpoint_key"],
              "endpoint_family": row["endpoint_family"], "endpoint_subtype": row["endpoint_subtype"],
              "unit_basis": row["unit_basis"], "assay_concept": row["assay_concept"],
              "k_profile": "same_endpoint", "setting_key": row["canonical_endpoint_key"],
              "metric_type": row["metric_type"], "retrieval_record_id": row["child_id"],
              "retrieved_smiles": row["canonical_smiles"], "retrieved_source_id": row["source_id"],
              "retrieved_measurement_label": row.get("measurement_label"),
              "retrieved_value": float(row["scalar_value"]),
              "retrieved_comparison_value": float(row["comparison_value"]),
              "query_evidence_source_ids": ",".join(sorted(evidence_sources)),
              "query_smiles": query, "transfer_max": float(row["transfer_max"]),
              "not_transfer_min": float(row["not_transfer_min"]),
              "threshold_display": row["threshold_display"], "tanimoto": similarity,
              "tanimoto_bucket": bucket}
    result.update(vote)
    return result


def _arrow_table(rows: list[dict[str, Any]]) -> pa.Table:
    arrays = {}
    for name in rows[0]:
        values = [row.get(name) for row in rows]
        if name == "binary_label":
            arrays[name] = pa.array(values, pa.int8())
        elif name in INT_FIELDS:
            arrays[name] = pa.array(values, pa.int32())
        elif name in FLOAT_FIELDS:
            arrays[name] = pa.array(values, pa.float64())
        else:
            arrays[name] = pa.array(values, pa.string())
    return pa.table(arrays)


def _write_candidates(path: Path, candidates: Iterable[dict[str, Any]]) -> tuple[int, Counter, Counter]:
    writer, batch, total = None, [], 0
    labels, strata = Counter(), Counter()
    for candidate in candidates:
        batch.append(candidate)
        labels[{1: "transfer", 0: "not_transfer"}.get(candidate["binary_label"], "null")] += 1
        strata[(candidate["split"], candidate["assay_concept"], candidate["tanimoto_bucket"])] += 1
        if len(batch) < 50_000:
            continue
        table = _arrow_table(batch)
        writer = writer or pq.ParquetWriter(path, table.schema, compression="zstd")
        writer.write_table(table)
        total += len(batch)
        batch = []
    if batch:
        table = _arrow_table(batch)
        writer = writer or pq.ParquetWriter(path, table.schema, compression="zstd")
        writer.write_table(table)
        total += len(batch)
    if writer:
        writer.close()
    return total, labels, strata


def _candidate_stream(rows: list[dict[str, Any]], policies: V3Policies,
                      args: argparse.Namespace, stats: Counter) -> Iterable[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["_split"], row["canonical_endpoint_key"])].append(row)
    fp = policies.fingerprint
    cache = FingerprintCache(radius=int(fp["radius"]), nbits=int(fp["n_bits"]))
    boundary = float(policies.tanimoto["boundary"])
    for group in groups.values():
        evidence: dict[str, list[float]] = defaultdict(list)
        evidence_sources: dict[str, set[str]] = defaultdict(set)
        for row in group:
            evidence[row["canonical_smiles"]].append(float(row["comparison_value"]))
            evidence_sources[row["canonical_smiles"]].add(row["source_id"])
        if len(evidence) < 2:
            continue
        concept, metric = group[0]["assay_concept"], policies.metric_for(group[0])
        cap, query_cache, molecules = _query_cap(args, concept), {}, sorted(evidence)
        for row in group:
            molecule = row["canonical_smiles"]
            if molecule not in query_cache:
                choices, exhausted = _bucketed_queries(molecule, molecules, cap, cache, boundary)
                query_cache[molecule] = choices
                for bucket, is_exhausted in exhausted.items():
                    state = "exhausted" if is_exhausted else "capped"
                    stats[f"{concept}|{bucket}|{state}_molecules"] += 1
            for query, similarity, bucket in query_cache[molecule]:
                sources = evidence_sources[query]
                if row["source_id"] not in sources:
                    stats[f"cross_source|{row['canonical_endpoint_key']}"] += 1
                yield _candidate(row, query, similarity, bucket, evidence[query], sources,
                                 metric, policies.version_bundle)


def build(args: argparse.Namespace) -> dict[str, Any]:
    policies = V3Policies(resolve_path(args.release))
    rows = _load_rows([Path(path) for path in args.base], args.split_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "pairs.parquet"
    generation_stats: Counter = Counter()
    stream = _candidate_stream(rows, policies, args, generation_stats)
    total, labels, strata = _write_candidates(output, stream)
    if not total:
        raise RuntimeError("no v3 candidates produced")
    manifest = {"stage": "pairs_v3", "schema_version": policies.release["artifact_schema_version"],
                "policy_versions": policies.version_bundle, "query_caps": getattr(args, "query_caps", None),
                "total_candidates": total, "binary_label_counts": dict(labels),
                "hard_binary_coverage": (labels["transfer"] + labels["not_transfer"]) / total,
                "enumeration_saturation": dict(generation_stats),
                "candidates_by_split_concept_bucket": {"|".join(key): value for key, value in strata.items()}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", nargs="+", required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
