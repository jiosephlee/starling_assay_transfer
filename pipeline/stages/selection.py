#!/usr/bin/env python3
"""Select stage: candidate universe -> fixed train / validation / test selections.

Applies the version-2 sampling policy (``docs/assay_transfer_design_v2.md`` 12) over the
candidate universe from the pairs stage:

- **strata** = ``assay_concept x tanimoto_bucket`` (the only primary selection strata);
- **quotas** = train 200k / validation 2k / test 2k (from ``sampling.yaml``);
- **training** retains null binary labels only for a soft-evidence release;
- **validation / test** require a non-null hard binary label and are frozen;
- **degree control**: deterministic round-robin over query molecules within each stratum,
  with optional per-query / per-retrieval-molecule / per-retrieval-record caps, so a
  high-degree molecule cannot dominate; and
- **equal allocation** per stratum, with underfill reported rather than backfilled.

Selection is deterministic from the frozen candidate universe: candidates are ordered by
their ``candidate_id`` within each (stratum, query molecule) bucket. Output:
``selected/{train,validation,test}/selected.parquet`` + a ``manifest.json`` with the 19
required audits (per-stratum counts + underfill, binary-label coverage, degree
distributions).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.policy import load_sampling_policy  # noqa: E402
from pipeline.v3_policy import V3Policies, resolve_path  # noqa: E402

SPLITS = ("train", "validation", "test")
FROZEN_COMPARE_FIELDS = ("split", "retrieval_record_id", "query_smiles",
                         "canonical_endpoint_key", "binary_label")


def _load_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        f = p / "pairs.parquet" if p.is_dir() else p
        rows.extend(pq.read_table(f).to_pylist())
    return rows


def _stratum(row: dict[str, Any]) -> Optional[tuple[str, str]]:
    concept, bucket = row.get("assay_concept"), row.get("tanimoto_bucket")
    if not concept or not bucket:
        return None
    return (concept, bucket)


def _round_robin(by_query: dict[str, list[dict[str, Any]]], sampling: Any) -> list[dict[str, Any]]:
    """Full deterministic round-robin order over query molecules in one stratum.

    Iterates query molecules (sorted) taking one candidate each pass, skipping candidates
    blocked by the degree caps, until the stratum is exhausted. The returned order is
    frozen: any size-``k`` prefix is a valid degree-controlled selection, and a smaller
    prefix is contained in a larger one (nested-prefix expansion, section 13.3).
    """
    q_cap = sampling.max_per_query_molecule
    rm_cap = sampling.max_per_retrieval_molecule
    rr_cap = sampling.max_per_retrieval_record
    q_counts: Counter = Counter()
    rm_counts: Counter = Counter()
    rr_counts: Counter = Counter()
    cursors = {q: 0 for q in by_query}
    queries = sorted(by_query)
    order: list[dict[str, Any]] = []
    progressed = True
    while progressed:
        progressed = False
        for q in queries:
            lst = by_query[q]
            idx = cursors[q]
            while idx < len(lst):
                cand = lst[idx]
                if q_cap and q_counts[q] >= q_cap:
                    idx = len(lst)
                    break
                if rm_cap and rm_counts[cand["retrieved_smiles"]] >= rm_cap:
                    idx += 1
                    continue
                if rr_cap and rr_counts[cand["retrieval_record_id"]] >= rr_cap:
                    idx += 1
                    continue
                break
            cursors[q] = idx
            if idx >= len(lst):
                continue
            cand = lst[idx]
            order.append(cand)
            q_counts[q] += 1
            rm_counts[cand["retrieved_smiles"]] += 1
            rr_counts[cand["retrieval_record_id"]] += 1
            cursors[q] = idx + 1
            progressed = True
    return order


def stratum_orders(pool: list[dict[str, Any]], sampling: Any) -> tuple[dict[tuple, list[dict[str, Any]]], int]:
    """Per-stratum frozen round-robin orders + count of candidates lacking a stratum."""
    by_stratum: dict[tuple, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    excluded = 0
    for r in pool:
        s = _stratum(r)
        if s is None:
            excluded += 1
            continue
        by_stratum[s][r["query_smiles"]].append(r)
    for by_q in by_stratum.values():
        for q in by_q:
            by_q[q].sort(key=lambda r: r["candidate_id"])
    return {s: _round_robin(by_q, sampling) for s, by_q in by_stratum.items()}, excluded


def _equal_allocation(quota: int, n_strata: int) -> tuple[int, int]:
    if n_strata == 0:
        return 0, 0
    return quota // n_strata, quota - (quota // n_strata) * n_strata


def _select_split(
    pool: list[dict[str, Any]], quota: int, sampling: Any, expected_strata: Optional[list[tuple]] = None,
    targets: Optional[dict[tuple, Optional[int]]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Equal-allocation, degree-controlled selection of up to ``quota`` from one pool."""
    orders, excluded_no_stratum = stratum_orders(pool, sampling)
    strata = sorted(expected_strata or orders)
    per_stratum_quota, remainder = _equal_allocation(quota, len(strata))

    selected: list[dict[str, Any]] = []
    per_stratum_selected: Counter = Counter()
    per_stratum_available: Counter = Counter()
    underfilled: dict[str, int] = {}
    for i, s in enumerate(strata):
        order = orders.get(s, [])
        per_stratum_available[s] = len(order)
        target = targets.get(s) if targets is not None else per_stratum_quota + (1 if i < remainder else 0)
        target = len(order) if target is None else target
        take = order[:target]
        selected.extend(take)
        per_stratum_selected[s] = len(take)
        if len(take) < target:
            underfilled[f"{s[0]}|{s[1]}"] = len(order)

    audit = {
        "quota": quota,
        "selected": len(selected),
        "excluded_no_stratum": excluded_no_stratum,
        "per_stratum_selected": {f"{c}|{b}": n for (c, b), n in sorted(per_stratum_selected.items())},
        "per_stratum_available": {f"{c}|{b}": n for (c, b), n in sorted(per_stratum_available.items())},
        "underfilled_strata": underfilled,
    }
    return selected, audit


def _coverage_report(pool: list[dict[str, Any]]) -> dict[str, Any]:
    def cov(rows: list[dict[str, Any]]) -> Optional[float]:
        if not rows:
            return None
        hard = sum(1 for r in rows if r.get("binary_label") is not None)
        return round(hard / len(rows), 4)

    by_concept: dict[str, list] = defaultdict(list)
    by_bucket: dict[str, list] = defaultdict(list)
    by_endpoint: dict[str, list] = defaultdict(list)
    for r in pool:
        by_concept[r.get("assay_concept")].append(r)
        by_bucket[r.get("tanimoto_bucket")].append(r)
        by_endpoint[r.get("canonical_endpoint_key") or r.get("canonical_endpoint_id")].append(r)
    return {
        "overall": cov(pool),
        "by_assay_concept": {k: cov(v) for k, v in by_concept.items()},
        "by_tanimoto_bucket": {k: cov(v) for k, v in by_bucket.items()},
        "by_endpoint": {k: cov(v) for k, v in by_endpoint.items()},
    }


def _eligible_rows(pool: list[dict[str, Any]], split: str, soft_evidence: bool) -> list[dict[str, Any]]:
    if split == "train" and soft_evidence:
        return pool
    return [row for row in pool if row.get("binary_label") is not None]


def _frozen_heldout(pool: list[dict[str, Any]], source: Path, split: str) -> list[dict[str, Any]]:
    path = source / "selected" / split / "selected.parquet"
    frozen = pq.read_table(path).to_pylist()
    current = {row["candidate_id"]: row for row in pool}
    if len(current) != len(pool):
        raise ValueError(f"duplicate candidate IDs in current {split} pool")
    selected = []
    for old in frozen:
        new = current.get(old["candidate_id"])
        if new is None:
            raise ValueError(f"frozen {split} candidate missing: {old['candidate_id']}")
        mismatch = [name for name in FROZEN_COMPARE_FIELDS if old.get(name) != new.get(name)]
        if mismatch:
            raise ValueError(f"frozen {split} candidate changed fields {mismatch}: {old['candidate_id']}")
        selected.append(new)
    return selected


def _apply_frozen_audit(audit: dict[str, Any], rows: list[dict[str, Any]], source: Path) -> None:
    strata = Counter((row["assay_concept"], row["tanimoto_bucket"]) for row in rows)
    audit.update({"quota": len(rows), "selected": len(rows), "underfilled_strata": {},
                  "per_stratum_selected": {f"{c}|{b}": n for (c, b), n in sorted(strata.items())},
                  "frozen_evaluation_source": str(source)})


def _selected_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    query = Counter(row["query_smiles"] for row in rows)
    retrieval = Counter(row["retrieved_smiles"] for row in rows)
    records = Counter(row["retrieval_record_id"] for row in rows)
    labels = Counter(str(row["binary_label"]) for row in rows)
    return {"binary_label_counts": dict(labels), "unique_query_molecules": len(query),
            "unique_retrieval_molecules": len(retrieval), "unique_retrieval_records": len(records),
            "max_query_degree": max(query.values(), default=0),
            "max_retrieval_molecule_degree": max(retrieval.values(), default=0),
            "max_retrieval_record_degree": max(records.values(), default=0)}


def _write_split(rows: list[dict[str, Any]], out_dir: Path, schema: pa.Schema) -> None:
    """Write selected rows under the candidate parquet schema (empty table if none)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        table = pa.Table.from_pylist(rows, schema=schema)
    else:
        table = schema.empty_table()
    pq.write_table(table, out_dir / "selected.parquet", compression="zstd")


def build(args: argparse.Namespace) -> dict[str, Any]:
    release = getattr(args, "release", None)
    policies = V3Policies(resolve_path(release)) if release else None
    sampling = _v3_sampling(policies) if policies else load_sampling_policy()
    quotas = {
        "train": args.train_quota if args.train_quota is not None else sampling.quotas.get("train", 0),
        "validation": args.val_quota if args.val_quota is not None else sampling.quotas.get("validation", 0),
        "test": args.test_quota if args.test_quota is not None else sampling.quotas.get("test", 0),
    }
    candidate_paths = [Path(p) for p in args.candidates]
    candidates = _load_candidates(candidate_paths)
    if not candidates:
        raise RuntimeError("no candidates to select from")
    first = candidate_paths[0]
    schema = pq.read_schema((first / "pairs.parquet") if first.is_dir() else first)

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in candidates:
        by_split[r.get("split")].append(r)

    report: dict[str, Any] = {"stage": "select", "sampling_version": sampling.version,
                              "quotas": quotas, "seed": sampling.seed, "splits": {}}
    configured_targets = getattr(args, "stratum_targets", None)
    soft_evidence = bool(policies and policies.release.get("soft_evidence_primary"))
    frozen_value = policies.release.get("frozen_evaluation_selection") if policies else None
    frozen_source = resolve_path(frozen_value) if frozen_value else None
    for split in SPLITS:
        pool = by_split.get(split, [])
        eligible = _eligible_rows(pool, split, soft_evidence)
        expected = _expected_strata(policies) if policies else None
        targets = _targets_for_split(configured_targets, split)
        selected, audit = _select_split(eligible, quotas[split], sampling, expected, targets)
        if frozen_source and split in ("validation", "test"):
            selected = _frozen_heldout(eligible, frozen_source, split)
            _apply_frozen_audit(audit, selected, frozen_source)
        out_dir = args.output_dir / "selected" / split
        _write_split(selected, out_dir, schema)
        report["splits"][split] = {
            **audit,
            **_selected_audit(selected),
            "eligible_candidates": len(eligible),
            "total_candidates": len(pool),
            "binary_label_coverage": _coverage_report(pool),
            "soft_evidence_training": soft_evidence and split == "train",
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2))
    return report


def _targets_for_split(config: Optional[dict], split: str) -> Optional[dict[tuple, Optional[int]]]:
    if not config or split not in config:
        return None
    targets = {}
    for key, value in config[split].items():
        concept, bucket = key.split("|", 1)
        targets[(concept, bucket)] = None if value is None else int(value)
    return targets


def _expected_strata(policies: V3Policies) -> list[tuple[str, str]]:
    return [(concept, bucket) for concept in policies.concepts["concepts"] for bucket in ("low", "high")]


class _V3Sampling:
    def __init__(self, policies: V3Policies):
        config = policies.sampling
        self.version = config["version"]
        self.quotas = {key: int(value) for key, value in config["quotas"].items()}
        self.seed = int(config["seed"])
        self.max_per_query_molecule = 0
        self.max_per_retrieval_molecule = 0
        self.max_per_retrieval_record = 0


def _v3_sampling(policies: V3Policies) -> _V3Sampling:
    return _V3Sampling(policies)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", required=True, help="pairs.parquet dirs/files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-quota", type=int, default=None)
    parser.add_argument("--val-quota", type=int, default=None)
    parser.add_argument("--test-quota", type=int, default=None)
    parser.add_argument("--release", default=None)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
