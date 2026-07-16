#!/usr/bin/env python3
"""Pairs stage: base records + split -> directed candidate transfer examples.

Implements the version-2 unit of supervision (``docs/assay_transfer_design_v2.md`` 4, 9,
10): each candidate is ``E = (r_A, B, K_A)`` — one concrete retrieval record ``r_A``, one
different query molecule ``B``, and the shared setting ``K_A``. The target comes from the
*evidence distribution* ``R_B(K_A)`` = every record of ``B`` under the same canonical
endpoint and setting:

- **record distance** ``d_j = |y_A - y_Bj|`` on the transformed metric scale;
- **three-state vote** per record (``MetricThreshold.label`` -> transfer / not_transfer /
  None=ambiguous); ambiguous votes count in the denominator ``N``;
- **strict-majority** nullable binary label (``> N/2``); and
- **continuous primary target** ``D_expected = mean_j(d_j)`` (mean of distances, NOT the
  distance to the mean value), retained even when the binary label is null.

Candidates never cross a ``canonical_endpoint_id`` (the firewall) and are built within a
single split (both molecules in the example's split). Each candidate carries a stable
``candidate_id``, its assay concept, and a low/high Tanimoto bucket (sampling strata only).
Degree control and fixed sizes are applied later in the select stage; this stage
enumerates the (optionally capped) candidate universe.

The condition-key machinery (profiles / setting fields) is consumed unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from pipeline.normalize.common_transfer import FingerprintCache  # noqa: E402
from pipeline.policy import (  # noqa: E402
    load_assay_concepts,
    load_condition_key_policy,
    load_fingerprint_policy,
    load_majority_policy,
    load_metric_policy,
    load_sampling_policy,
    load_tanimoto_policy,
)

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
    fields = ck.join_fields(profile, most_specific_schema)
    return [f for f in fields if f != "canonical_endpoint_key"]


def _setting_key(row: dict[str, Any], fields: list[str]) -> Optional[tuple]:
    values = []
    for f in fields:
        v = row.get(f)
        if v is None or str(v).strip() == "":
            return None
        values.append(str(v))
    return tuple(values)


def _record_id(row: dict[str, Any]) -> str:
    """Stable retrieval-record identity from provenance (falls back to row index)."""
    ext = row.get("extraction_id")
    if ext:
        return str(ext)
    return f"pmid={row.get('pmid')}#row={row.get('_row_index')}"


def _candidate_id(record_id: str, query: str, endpoint: str, profile: str,
                  setting_key: str, split_version: str, target_policy_version: str) -> str:
    payload = "|".join([record_id, query, endpoint, profile, setting_key,
                        split_version, target_policy_version])
    return hashlib.sha1(payload.encode()).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    resolver = load_endpoint_resolver()
    ck = load_condition_key_policy()
    metric_policy = load_metric_policy()
    majority = load_majority_policy()
    concepts = load_assay_concepts()
    tanimoto = load_tanimoto_policy()
    fp_policy = load_fingerprint_policy()
    sampling = load_sampling_policy()

    fp_cache = FingerprintCache(radius=fp_policy.radius, nbits=fp_policy.n_bits)
    target_policy_version = f"{majority.version}+{metric_policy.version}"
    enum_cap = args.max_queries if args.max_queries is not None else sampling.enumeration_cap

    split_map = _load_split_map(args.split_dir)
    base_paths = [Path(p) for p in args.base]
    rows = _load_base_rows(base_paths)
    for i, r in enumerate(rows):
        r["_row_index"] = i
        r["_split"] = split_map.get(r.get("smiles"))
        r["_mol"] = r.get("smiles")

    candidates: list[dict[str, Any]] = []
    stats: Counter = Counter()
    per_endpoint: Counter = Counter()
    label_counts: Counter = Counter()

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
        if not metric.is_numeric:
            continue  # non-numeric endpoints handled by a later categorical target policy

        # Evidence index: (setting_key) -> query molecule -> list of transformed values.
        evidence: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        eligible: list[dict[str, Any]] = []
        for r in grp:
            key = _setting_key(r, fields)
            if key is None:
                stats["excluded_missing_setting"] += 1
                continue
            r["_setting"] = key
            eligible.append(r)
            evidence[key][r["_mol"]].append(float(r["property_value"]))

        for a in eligible:
            key = a["_setting"]
            candidate_mols = [m for m in evidence[key] if m != a["_mol"]]
            if enum_cap and len(candidate_mols) > enum_cap:
                candidate_mols = sorted(candidate_mols)[:enum_cap]
            y_a = float(a["property_value"])
            fp_a = fp_cache.get(a["_mol"])
            for b in candidate_mols:
                y_bs = evidence[key][b]
                n_total = len(y_bs)
                if n_total < majority.min_eligible_records:
                    stats["below_min_evidence"] += 1
                    continue
                distances = [abs(y_a - y) for y in y_bs]
                n_transfer = n_nontransfer = 0
                for y in y_bs:
                    vote = metric.label(y_a, y)
                    if vote == "transfer":
                        n_transfer += 1
                    elif vote == "not_transfer":
                        n_nontransfer += 1
                n_ambiguous = n_total - n_transfer - n_nontransfer

                binary_label = majority.majority_label(n_transfer, n_nontransfer, n_total)
                if binary_label == 1:
                    majority_side = "transfer"
                elif binary_label == 0:
                    majority_side = "not_transfer"
                else:
                    majority_side = None
                margin = (
                    max(n_transfer, n_nontransfer) / n_total - 0.5
                    if binary_label is not None
                    else None
                )
                d_expected = statistics.fmean(distances)

                fp_b = fp_cache.get(b)
                sim = fp_cache.similarity(fp_a, fp_b) if (fp_a and fp_b) else None
                bucket = tanimoto.bucket_for(sim)

                setting_str = "|".join(key)
                record_id = _record_id(a)
                candidates.append(
                    {
                        "candidate_id": _candidate_id(
                            record_id, b, endpoint_id, args.profile, setting_str,
                            args.split_version, target_policy_version,
                        ),
                        "split": split,
                        "canonical_endpoint_id": endpoint_id,
                        "assay_concept": concepts.concept_for(a.get("source_id")),
                        "k_profile": args.profile,
                        "setting_key": setting_str,
                        "metric_type": a["metric_type"],
                        "retrieval_record_id": record_id,
                        "retrieved_row_index": a["_row_index"],
                        "retrieved_smiles": a["_mol"],
                        "retrieved_source_id": a.get("source_id"),
                        "retrieved_value": y_a,
                        "retrieved_value_native": float(a["property_value_native"]),
                        "query_smiles": b,
                        "n_records": n_total,
                        "n_transfer": n_transfer,
                        "n_nontransfer": n_nontransfer,
                        "n_ambiguous": n_ambiguous,
                        "transfer_fraction": n_transfer / n_total,
                        "nontransfer_fraction": n_nontransfer / n_total,
                        "ambiguous_fraction": n_ambiguous / n_total,
                        "binary_label": binary_label,
                        "majority_side": majority_side,
                        "majority_margin": margin,
                        "continuous_target": d_expected,
                        "dist_std": statistics.pstdev(distances) if n_total > 1 else 0.0,
                        "dist_median": statistics.median(distances),
                        "dist_max": max(distances),
                        "tanimoto": sim,
                        "tanimoto_bucket": bucket,
                    }
                )
                per_endpoint[endpoint_id] += 1
                label_counts[{1: "transfer", 0: "not_transfer"}.get(binary_label, "null")] += 1

    if not candidates:
        raise RuntimeError(
            f"no candidates produced for profile {args.profile!r}; the setting may exclude all rows"
        )

    _INT = {"retrieved_row_index", "n_records", "n_transfer", "n_nontransfer", "n_ambiguous"}
    _FLOAT = {"retrieved_value", "retrieved_value_native", "transfer_fraction",
              "nontransfer_fraction", "ambiguous_fraction", "majority_margin",
              "continuous_target", "dist_std", "dist_median", "dist_max", "tanimoto"}
    columns = list(candidates[0].keys())
    arrays = {}
    for c in columns:
        vals = [row[c] for row in candidates]
        if c == "binary_label":
            arrays[c] = pa.array(vals, type=pa.int8())  # nullable 0/1/None
        elif c in _INT:
            arrays[c] = pa.array(vals, type=pa.int64())
        elif c in _FLOAT:
            arrays[c] = pa.array(vals, type=pa.float64())
        else:
            arrays[c] = pa.array([None if v is None else str(v) for v in vals], type=pa.string())
    table = pa.table(arrays)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output_dir / "pairs.parquet", compression="zstd")

    manifest = {
        "stage": "pairs",
        "k_profile": args.profile,
        "base_inputs": [str(p) for p in base_paths],
        "split_dir": str(args.split_dir),
        "split_version": args.split_version,
        "enumeration_cap": enum_cap,
        "condition_key_version": ck.version,
        "metric_threshold_version": metric_policy.version,
        "record_vote_version": "record_vote_v1",
        "majority_label_version": majority.version,
        "fingerprint_version": fp_policy.version,
        "tanimoto_bucket_version": tanimoto.version,
        "assay_concept_version": concepts.version,
        "target_policy_version": target_policy_version,
        "total_candidates": len(candidates),
        "binary_label_counts": {k: label_counts.get(k, 0) for k in ("transfer", "not_transfer", "null")},
        "hard_binary_coverage": round(
            (label_counts.get("transfer", 0) + label_counts.get("not_transfer", 0)) / len(candidates), 4
        ),
        "excluded_missing_setting": stats.get("excluded_missing_setting", 0),
        "below_min_evidence": stats.get("below_min_evidence", 0),
        "candidates_by_endpoint": dict(per_endpoint.most_common()),
        "candidates_by_split": dict(Counter(c["split"] for c in candidates)),
        "candidates_by_tanimoto_bucket": dict(Counter(c["tanimoto_bucket"] for c in candidates)),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", nargs="+", required=True, help="Base dirs/parquets (composed union).")
    parser.add_argument("--split-dir", type=Path, required=True, help="Split stage output dir.")
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-version", default="v2")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Optional enumeration cap per retrieved record (default: sampling.yaml).")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
