#!/usr/bin/env python3
"""Pairs stage: base records + split -> asymmetric transfer pairs, within endpoint & split.

For a chosen condition-key profile ``K`` (``same_endpoint``,
``same_species_same_endpoint``, ``most_specific``) this builds the asymmetric training
pairs of ``docs/assay_transfer_design.md`` sections 2, 8:

- **retrieved** ``A`` = one concrete measurement row (its value ``y_A`` is kept);
- **query** ``B`` = a molecule whose value is *marginalized* over the ``K`` setting — the
  mean of ``B``'s records sharing ``A``'s setting key (incidental context averaged away).

Pairs never cross a ``canonical_endpoint_id`` (the section 5.1 firewall) and are built
**within a single split** so no label uses a held-out molecule. The record label comes
from the metric policy (:meth:`pipeline.policy.MetricThreshold.label`) applied on the
transformed value scale; deadband pairs are dropped from the binary target.

Marginalizing to the query *mean* is the documented distance-to-mean baseline (section
8.2); the preferred record-first ``P_transfer`` aggregation is a later increment. Queries
per retrieved record are sampled (seeded) to bound the quadratic pool (section 12.2).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.endpoints import load_endpoint_resolver  # noqa: E402
from pipeline.policy import load_condition_key_policy, load_metric_policy  # noqa: E402

PROFILES = ("same_endpoint", "same_species_same_endpoint", "most_specific")


def _load_split_map(split_dir: Path) -> dict[str, str]:
    table = pq.read_table(split_dir / "molecule_splits.parquet")
    raw = table.column("smiles").to_pylist()
    split = table.column("split").to_pylist()
    return dict(zip(raw, split))


def _load_base_rows(base_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in base_paths:
        p = path / "base.parquet" if path.is_dir() else path
        rows.extend(pq.read_table(p).to_pylist())
    return rows


def _setting_fields(profile: str, ck: Any, most_specific_schema: Optional[str]) -> list[str]:
    """Join fields beyond the endpoint key that define the K setting."""
    fields = ck.join_fields(profile, most_specific_schema)
    return [f for f in fields if f != "canonical_endpoint_key"]


def _setting_key(row: dict[str, Any], fields: list[str]) -> Optional[tuple]:
    """Setting-key tuple, or None if any required field is missing (exclude the row)."""
    values = []
    for f in fields:
        v = row.get(f)
        if v is None or str(v).strip() == "":
            return None
        values.append(str(v))
    return tuple(values)


def build(args: argparse.Namespace) -> dict[str, Any]:
    resolver = load_endpoint_resolver()
    ck = load_condition_key_policy()
    metric_policy = load_metric_policy()

    split_map = _load_split_map(args.split_dir)
    base_paths = [Path(p) for p in args.base]
    rows = _load_base_rows(base_paths)
    for i, r in enumerate(rows):
        r["_row_index"] = i
        r["_split"] = split_map.get(r.get("smiles"))
        r["_mol"] = r.get("smiles")

    rng = random.Random(args.seed)
    pair_rows: list[dict[str, Any]] = []
    stats: Counter = Counter()
    per_endpoint: Counter = Counter()

    # Partition by (split, canonical_endpoint_id) -> firewall + no cross-split leakage.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["_split"] is None or r.get("canonical_endpoint_id") is None:
            continue
        groups[(r["_split"], r["canonical_endpoint_id"])].append(r)

    for (split, endpoint_id), grp in groups.items():
        schema = resolver.most_specific_schema(endpoint_id)
        try:
            fields = _setting_fields(args.profile, ck, schema)
        except (KeyError, ValueError):
            continue
        metric = metric_policy.for_metric(resolver.metric_type(endpoint_id))

        # Eligible rows carry a valid setting key; index queries by (setting, molecule).
        query_values: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        eligible: list[dict[str, Any]] = []
        for r in grp:
            key = _setting_key(r, fields)
            if key is None:
                stats["excluded_missing_setting"] += 1
                continue
            r["_setting"] = key
            eligible.append(r)
            query_values[key][r["_mol"]].append(float(r["property_value"]))

        # Precompute query marginal (mean) per (setting, molecule).
        query_mean: dict[tuple, dict[str, tuple[float, int, float]]] = defaultdict(dict)
        for key, mols in query_values.items():
            for mol, vals in mols.items():
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                query_mean[key][mol] = (statistics.fmean(vals), len(vals), std)

        # Asymmetric enumeration: retrieved row A -> query molecule B (B != A's molecule),
        # both sharing A's setting key. Sample queries to bound the pool.
        for a in eligible:
            key = a["_setting"]
            candidates = [m for m in query_mean[key] if m != a["_mol"]]
            if not candidates:
                continue
            if len(candidates) > args.max_queries:
                candidates = rng.sample(candidates, args.max_queries)
            y_a = float(a["property_value"])
            for b in candidates:
                q_mean, q_n, q_std = query_mean[key][b]
                label = metric.label(y_a, q_mean)
                if label is None:
                    stats["deadband_dropped"] += 1
                    continue
                pair_rows.append(
                    {
                        "split": split,
                        "canonical_endpoint_id": endpoint_id,
                        "k_profile": args.profile,
                        "setting_key": "|".join(key),
                        "retrieved_row_index": a["_row_index"],
                        "retrieved_smiles": a["_mol"],
                        "retrieved_value": y_a,
                        "retrieved_value_native": float(a["property_value_native"]),
                        "query_smiles": b,
                        "query_value_mean": q_mean,
                        "query_n": q_n,
                        "query_std": q_std,
                        "metric_type": a["metric_type"],
                        "transfer_label": label,
                    }
                )
                stats[label] += 1
                per_endpoint[endpoint_id] += 1

    if not pair_rows:
        raise RuntimeError(
            f"no pairs produced for profile {args.profile!r}; the setting may exclude all rows"
        )

    columns = list(pair_rows[0].keys())
    arrays = {}
    for c in columns:
        vals = [pr[c] for pr in pair_rows]
        if c in ("retrieved_row_index", "query_n"):
            arrays[c] = pa.array(vals, type=pa.int64())
        elif c in ("retrieved_value", "retrieved_value_native", "query_value_mean", "query_std"):
            arrays[c] = pa.array(vals, type=pa.float64())
        else:
            arrays[c] = pa.array([str(v) for v in vals], type=pa.string())
    table = pa.table(arrays)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output_dir / "pairs.parquet", compression="zstd")

    manifest = {
        "stage": "pairs",
        "k_profile": args.profile,
        "base_inputs": [str(p) for p in base_paths],
        "split_dir": str(args.split_dir),
        "seed": args.seed,
        "max_queries_per_retrieved": args.max_queries,
        "condition_key_version": ck.version,
        "metric_threshold_version": metric_policy.version,
        "total_pairs": len(pair_rows),
        "label_counts": {"transfer": stats.get("transfer", 0), "not_transfer": stats.get("not_transfer", 0)},
        "deadband_dropped": stats.get("deadband_dropped", 0),
        "excluded_missing_setting": stats.get("excluded_missing_setting", 0),
        "pairs_by_endpoint": dict(per_endpoint.most_common()),
        "pairs_by_split": dict(Counter(pr["split"] for pr in pair_rows)),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", nargs="+", required=True, help="Base dirs/parquets (composed union).")
    parser.add_argument("--split-dir", type=Path, required=True, help="Split stage output dir.")
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-queries", type=int, default=64, help="Query molecules sampled per retrieved row.")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
