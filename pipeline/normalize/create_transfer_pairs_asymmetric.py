#!/usr/bin/env python3
"""Asymmetric transfer-pair builder (retrieved row + marginalized query).

This is the pairing model the task requires: at inference we only have the query
molecule's *structure*, plus a retrieved molecule with a known value measured under some
*setting*. So a training pair is:

- **retrieved** = a concrete measurement row (kept with its full metadata + value);
- **query** = a molecule, whose value is the **mean over that molecule's rows sharing the
  mode's ``same_columns`` setting** (incidental metadata marginalized away). We also carry
  the group's ``n`` and ``std`` so noisy query groups can be down-weighted/dropped later.

The label is ``TaskSpec.label(retrieved_value, query_mean)`` on the transformed value
scale (identity for %, log10 for ratio/clearance — the transform is already applied to
``property_value`` in the prepare stage). The three modes are a spectrum of how much of
the setting is pinned: ``no_constraints`` pins nothing (query mean is the molecule's
global mean); richer modes pin more.

Output is denormalized: each pair row inlines the query aggregate + a ``retrieved_row_index``
into the base table (for retrieved metadata lookup downstream).
"""

from __future__ import annotations

import argparse
import json
import shutil
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

from pipeline.normalize.common_transfer import (  # noqa: E402
    FingerprintCache,
    bucket_for_value,
    finite_float,
    iter_parquet_rows,
    utc_now,
    write_json,
)
from pipeline.taskspecs import get_task  # noqa: E402

LABEL_CODES = {"transfer": 1, "not_transfer": 0}


def setting_key(row: dict[str, Any], columns: list[str]) -> Optional[tuple[str, ...]]:
    """Tuple of the (non-null) setting column values, or None if any is missing."""
    if not columns:
        return ()
    values: list[str] = []
    for column in columns:
        value = row.get(column)
        if value is None or str(value).strip() == "":
            return None
        values.append(str(value).strip())
    return tuple(values)


def load_records(
    base_input: str,
    spec: Any,
    same_columns: list[str],
    value_column: str,
    max_rows: Optional[int],
) -> tuple[list[dict[str, Any]], FingerprintCache]:
    columns = sorted(
        {value_column, "smiles", "property_value_raw", *spec.metadata_columns, *same_columns}
    )
    fp_cache = FingerprintCache()
    canon: dict[str, Optional[str]] = {}
    records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for row_index, row in enumerate(iter_parquet_rows(base_input, columns=columns, max_rows=max_rows)):
        value = finite_float(row.get(value_column))
        smiles = row.get("smiles")
        if value is None:
            stats["missing_value"] += 1
            continue
        if not smiles or not str(smiles).strip():
            stats["missing_smiles"] += 1
            continue
        raw = str(smiles).strip()
        if raw not in canon:
            mol = fp_cache.chem.MolFromSmiles(raw)
            canon[raw] = fp_cache.chem.MolToSmiles(mol) if mol is not None else None
        canonical = canon[raw]
        if canonical is None or fp_cache.get(canonical) is None:
            stats["invalid_smiles"] += 1
            continue
        setting = setting_key(row, same_columns)
        if setting is None:
            stats["missing_setting"] += 1
            continue
        records.append(
            {
                "row_index": row_index,
                "smiles": canonical,
                "value": float(value),
                "value_raw": finite_float(row.get("property_value_raw")),
                "metadata": {c: row.get(c) for c in spec.metadata_columns},
                "setting": setting,
            }
        )
        stats["records_loaded"] += 1
    return records, fp_cache, dict(sorted(stats.items()))


def aggregate_query_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse records into (molecule, setting) query aggregates with mean/n/std."""
    groups: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["smiles"], record["setting"])].append(record)
    aggregates: list[dict[str, Any]] = []
    for (smiles, setting), members in groups.items():
        values = [m["value"] for m in members]
        raws = [m["value_raw"] for m in members if m["value_raw"] is not None]
        aggregates.append(
            {
                "smiles": smiles,
                "setting": setting,
                "mean": statistics.fmean(values),
                "mean_raw": statistics.fmean(raws) if raws else None,
                "n": len(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }
        )
    return aggregates


def build(args: argparse.Namespace) -> dict[str, Any]:
    spec = get_task(args.task)
    same_columns = spec.same_columns_for_mode(args.mode)
    value_column = args.value_column or spec.label_column

    records, fp_cache, load_stats = load_records(
        args.input, spec, same_columns, value_column, args.max_rows
    )
    if len(records) < 2:
        raise RuntimeError("need at least two valid base records")

    query_aggs = aggregate_query_groups(records)
    by_setting: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for agg in query_aggs:
        by_setting[agg["setting"]].append(agg)

    thresholds = list(args.similarity_thresholds or [])

    fields = [
        pa.field("retrieved_row_index", pa.uint32()),
        pa.field("retrieved_smiles", pa.large_string()),
        pa.field("query_smiles", pa.large_string()),
        pa.field("transfer_label", pa.int8()),
        pa.field("source_value", pa.float32()),
        pa.field("query_value_mean", pa.float32()),
        pa.field("query_value_mean_raw", pa.float32()),
        pa.field("query_group_n", pa.int32()),
        pa.field("query_group_std", pa.float32()),
        pa.field("value_difference", pa.float32()),
        pa.field("weighted_tanimoto", pa.float32()),
        pa.field("similarity_bucket", pa.int8()),
    ]
    for column in spec.metadata_columns:
        fields.append(pa.field(f"{column}_present", pa.int8()))
    out_schema = pa.schema(fields)

    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    buffer: list[dict[str, Any]] = []
    pairs_written = 0
    candidates = 0
    label_counts: Counter[str] = Counter()
    drop_reasons: Counter[str] = Counter()
    with pq.ParquetWriter(
        args.output_dir / "records.parquet", out_schema, compression=args.parquet_compression
    ) as writer:
        for retrieved in records:
            r_fp = fp_cache.get(retrieved["smiles"])
            for query in by_setting[retrieved["setting"]]:
                if query["smiles"] == retrieved["smiles"]:
                    continue
                candidates += 1
                label = spec.label(retrieved["value"], query["mean"])
                if label is None:
                    drop_reasons["deadband"] += 1
                    continue
                similarity = fp_cache.similarity(r_fp, fp_cache.get(query["smiles"]))
                row = {
                    "retrieved_row_index": int(retrieved["row_index"]),
                    "retrieved_smiles": retrieved["smiles"],
                    "query_smiles": query["smiles"],
                    "transfer_label": LABEL_CODES[label],
                    "source_value": retrieved["value_raw"],
                    "query_value_mean": float(query["mean"]),
                    "query_value_mean_raw": query["mean_raw"],
                    "query_group_n": int(query["n"]),
                    "query_group_std": float(query["std"]),
                    "value_difference": abs(retrieved["value"] - query["mean"]),
                    "weighted_tanimoto": similarity,
                    "similarity_bucket": bucket_for_value(similarity, thresholds) if thresholds else None,
                }
                for column in spec.metadata_columns:
                    row[f"{column}_present"] = 1 if retrieved["metadata"].get(column) is not None else 0
                buffer.append(row)
                pairs_written += 1
                label_counts[label] += 1
                if len(buffer) >= args.row_group_size:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=out_schema))
                    buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=out_schema))

    if pairs_written <= 0:
        raise RuntimeError("no labeled pairs written")

    metadata = {
        "schema_version": "asymmetric_transfer_pairs_v1",
        "created_at_utc": utc_now(),
        "task": spec.name,
        "mode": args.mode,
        "input": args.input,
        "output_dir": str(args.output_dir),
        "value_column": value_column,
        "same_columns": same_columns,
        "metadata_columns": spec.metadata_columns,
        "records_loaded": len(records),
        "query_groups": len(query_aggs),
        "candidate_pairs": candidates,
        "pairs_written": pairs_written,
        "pairs_by_transfer_label": dict(sorted(label_counts.items())),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "load_stats": load_stats,
        "query_group_size": {
            "singletons": sum(1 for a in query_aggs if a["n"] == 1),
            "multi": sum(1 for a in query_aggs if a["n"] > 1),
            "max_n": max((a["n"] for a in query_aggs), default=0),
        },
        "thresholds": {"transfer": getattr(spec, "transfer_threshold", None),
                       "not_transfer": getattr(spec, "not_transfer_threshold", None)},
    }
    write_json(args.output_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--input", required=True, help="Base parquet dir/file (train.parquet).")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--value-column", default=None)
    parser.add_argument("--similarity-thresholds", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--row-group-size", type=int, default=250_000)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    metadata = build(parse_args())
    print(json.dumps(
        {
            "mode": metadata["mode"],
            "records_loaded": metadata["records_loaded"],
            "query_groups": metadata["query_groups"],
            "candidate_pairs": metadata["candidate_pairs"],
            "pairs_written": metadata["pairs_written"],
            "pairs_by_transfer_label": metadata["pairs_by_transfer_label"],
            "drop_reasons": metadata["drop_reasons"],
            "query_group_size": metadata["query_group_size"],
        },
        indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
