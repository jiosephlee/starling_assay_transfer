#!/usr/bin/env python3
"""Expand stage: candidate universe -> nested training prefixes + expansion bundle.

Builds the expandable training artifact (``docs/assay_transfer_design_v2.md`` 13). The
initial 200k training release is the first prefix of a frozen ordering; expansion to 400k,
800k, ... advances a per-stratum cursor without recomputing validation/test, relabeling
examples, or moving molecules between splits.

The bundle is deterministic from the frozen candidate universe:

- a **ledger** (``ledger.parquet``): per stratum, the frozen round-robin order of every
  train-eligible candidate (``stratum, position, candidate_id``) — the shared
  :func:`pipeline.stages.selection.stratum_orders`, so a prefix is a valid
  degree-controlled selection and ``train_200k subset train_400k subset ...`` holds by
  construction;
- ``prefixes.json``: for each prefix size, the per-stratum cursor (first-k) and the
  selected candidate_ids; and
- ``manifest.json``: bound policy versions, exclusion counts, and a nested-containment proof.

A prefix's training rows are the candidates whose ``candidate_id`` is in that prefix's id
set (materialize consumes them exactly like the select stage's train selection).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.policy import load_sampling_policy  # noqa: E402
from pipeline.stages.selection import _equal_allocation, stratum_orders  # noqa: E402


def _load_train_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        f = p / "pairs.parquet" if p.is_dir() else p
        for r in pq.read_table(f).to_pylist():
            # Training candidates: in the train split with a defined continuous target.
            if r.get("split") == "train" and r.get("continuous_target") is not None:
                rows.append(r)
    return rows


def _prefix_ids(orders: dict[tuple, list[dict[str, Any]]], size: int) -> tuple[set[str], dict[str, int]]:
    """Candidate ids in the first-k (equal per-stratum allocation) of the frozen orders."""
    strata = sorted(orders)
    per, remainder = _equal_allocation(size, len(strata))
    ids: set[str] = set()
    cursors: dict[str, int] = {}
    for i, s in enumerate(strata):
        target = per + (1 if i < remainder else 0)
        take = orders[s][:target]
        cursors[f"{s[0]}|{s[1]}"] = len(take)
        ids.update(c["candidate_id"] for c in take)
    return ids, cursors


def build(args: argparse.Namespace) -> dict[str, Any]:
    sampling = load_sampling_policy()
    prefixes = args.prefixes if args.prefixes else sampling.train_prefixes
    candidate_paths = [Path(p) for p in args.candidates]
    pool = _load_train_candidates(candidate_paths)
    if not pool:
        raise RuntimeError("no train-eligible candidates for the expansion bundle")

    orders, excluded_no_stratum = stratum_orders(pool, sampling)

    # Ledger: frozen per-stratum order.
    ledger_rows: list[dict[str, Any]] = []
    for s in sorted(orders):
        for pos, cand in enumerate(orders[s]):
            ledger_rows.append(
                {"stratum": f"{s[0]}|{s[1]}", "position": pos, "candidate_id": cand["candidate_id"]}
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "stratum": pa.array([r["stratum"] for r in ledger_rows], type=pa.string()),
                "position": pa.array([r["position"] for r in ledger_rows], type=pa.int64()),
                "candidate_id": pa.array([r["candidate_id"] for r in ledger_rows], type=pa.string()),
            }
        ),
        args.output_dir / "ledger.parquet",
        compression="zstd",
    )

    # Nested prefixes + containment proof.
    prefix_info: dict[str, Any] = {}
    prev_ids: set[str] = set()
    nested_ok = True
    for size in sorted(prefixes):
        ids, cursors = _prefix_ids(orders, size)
        if not prev_ids <= ids:
            nested_ok = False
        prefix_info[str(size)] = {
            "requested_size": size,
            "selected": len(ids),
            "per_stratum_cursor": cursors,
            "contains_previous_prefix": prev_ids <= ids,
        }
        prev_ids = ids
    (args.output_dir / "prefixes.json").write_text(
        json.dumps({s: {**v, "candidate_ids": sorted(_prefix_ids(orders, int(s))[0])}
                    for s, v in prefix_info.items()}, indent=2)
    )

    manifest = {
        "stage": "expand",
        "candidate_inputs": [str(p) for p in candidate_paths],
        "sampling_version": sampling.version,
        "train_eligible_candidates": len(pool),
        "excluded_no_stratum": excluded_no_stratum,
        "strata": [f"{c}|{b}" for (c, b) in sorted(orders)],
        "prefixes": prefix_info,
        "nested_containment_holds": nested_ok,
        "ledger_rows": len(ledger_rows),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", required=True, help="pairs.parquet dirs/files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefixes", type=int, nargs="*", default=None,
                        help="Prefix sizes (default: sampling.yaml train_prefixes).")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
