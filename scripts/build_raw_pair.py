#!/usr/bin/env python3
"""Build deterministic raw-pair Parquets from immutable V4 eligible records.

Defaults target the V6/V6.5 artifact; the v7/v7.1/v8 wrappers reuse ``build()`` after
monkey-patching this module's ``_ordinary``/``_ranking``/``_split_records``/``_write_train_flat``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.pair_core import (condition_index, iter_list_groups, molecule_splits,
                                propose_records, record_key, row_for, stable_hash,
                                target_for, unordered_pair_id)
from pipeline.v3_policy import file_sha256


def _records(path: Path, expected_records: int = 138806) -> list[dict]:
    rows = pq.read_table(path).to_pylist()
    if expected_records is not None and len(rows) != expected_records:
        raise ValueError(f"expected {expected_records} eligible records, found {len(rows)}")
    return rows


def _split_records(rows: list[dict]) -> dict[str, list[dict]]:
    assignment = molecule_splits(rows)
    result = defaultdict(list)
    for row in rows:
        result[assignment[row["canonical_smiles"]]].append(row)
    return {name: sorted(values, key=record_key) for name, values in result.items()}


def _is_decisive(row: dict, query: dict) -> bool:
    return row["distance"] <= float(query["transfer_max"]) or row["distance"] >= float(query["not_transfer_min"])


def _candidate_label(query: dict, retrieval: dict) -> str | None:
    distance = target_for(query, retrieval)["distance"]
    if distance <= query["transfer_max"]:
        return "transfer"
    return "nontransfer" if distance >= query["not_transfer_min"] else None


def _ordinary(records: list[dict], split: str, forbidden: set[str]) -> list[dict]:
    indexed, selected = condition_index(records), defaultdict(list)
    query_degree, retrieval_degree = defaultdict(int), defaultdict(int)
    queries = sorted(records, key=lambda row: stable_hash(record_key(row)))
    for pass_index in range(32):
        for query in queries:
            concept = query["assay_concept"]
            if query_degree[record_key(query)] >= 16:
                continue
            candidates = propose_records(query, indexed[query["canonical_endpoint_key"]], 128,
                                         forbidden, f"ordinary:{pass_index}")
            labels = sorted(("transfer", "nontransfer"), key=lambda label: len(selected[concept, label]))
            for wanted_label in labels:
                retrieval = next((item for item in candidates if _candidate_label(query, item) == wanted_label
                                  and retrieval_degree[record_key(item)] < 8), None)
                if retrieval is None or len(selected[concept, wanted_label]) >= 200:
                    continue
                row = row_for(query, retrieval, split)
                row["binary_label"] = 1 if wanted_label == "transfer" else 0
                selected[concept, wanted_label].append(row)
                forbidden.add(unordered_pair_id(query, retrieval))
                query_degree[record_key(query)] += 1
                retrieval_degree[record_key(retrieval)] += 1
                break
        if all(len(selected[concept, label]) == 200 for concept in selected_concepts(records)
               for label in ("transfer", "nontransfer")):
            break
    output = [row for concept in selected_concepts(records) for label in ("transfer", "nontransfer")
              for row in selected[concept, label]]
    if len(output) != 2000:
        counts = {f"{concept}:{label}": len(selected[concept, label])
                  for concept in selected_concepts(records) for label in ("transfer", "nontransfer")}
        raise RuntimeError(f"{split} ordinary benchmark requires 2000 rows: {counts}")
    return output


def selected_concepts(records: list[dict]) -> list[str]:
    return sorted({str(row["assay_concept"]) for row in records})


def _ranking(records: list[dict], split: str, forbidden: set[str]) -> list[dict]:
    indexed, anchors = condition_index(records), []
    for query in records:
        candidates = propose_records(query, indexed[query["canonical_endpoint_key"]], 20,
                                     forbidden, "ranking")
        if len(candidates) == 20:
            anchors.append((query, candidates))
    anchors.sort(key=lambda item: stable_hash(record_key(item[0])))
    selected, used_molecules = [], set()
    for query, candidates in anchors:
        if query["canonical_smiles"] in used_molecules:
            continue
        selected.append((query, candidates))
        used_molecules.add(query["canonical_smiles"])
        if len(selected) == 1000:
            break
    if len(selected) != 1000:
        raise RuntimeError(f"{split} ranking benchmark needs 1000 distinct-molecule anchors")
    output = []
    for number, (query, candidates) in enumerate(selected):
        for index, retrieval in enumerate(candidates):
            row = row_for(query, retrieval, split)
            row.update({"ranking_query_id": f"{split}-rank-{number:04d}", "ranking_member_index": index})
            output.append(row)
    return output


def _table(rows: list[dict], schema) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema) if schema is not None else pa.Table.from_pylist(rows)


def _write(output: Path, name: str, rows: list[dict], schema=None) -> str:
    path = output / name / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_table(rows, schema), path, compression="zstd")
    return str(path)


def _union_split_schema(ordinary_rows: list[dict], ranking_rows: list[dict]) -> pa.Schema:
    """One schema shared by every flat split so HF's single-config loader casts cleanly.

    ordinary rows carry `binary_label`; ranking rows carry `ranking_query_id`/`ranking_member_index`;
    train carries neither. Unifying the two benchmark schemas yields the full column set (with
    types); `from_pylist(rows, schema=...)` then null-fills whichever columns a split lacks.
    """
    ordinary_schema = pa.Table.from_pylist(ordinary_rows[:256]).schema
    ranking_schema = pa.Table.from_pylist(ranking_rows[:256]).schema
    return pa.unify_schemas([ordinary_schema, ranking_schema])


# ListNet group bookkeeping is meaningless once packing is left to SFT; v6_5 strips it.
_LISTNET_COLUMNS = ("listnet_group_id", "listnet_group_index",
                    "listnet_query_group_id", "listnet_member_index")


def _flush(
    path: Path, writer, batch: list[dict], schema=None, compression_level: int | None = None,
):
    table = _table(batch, schema)
    options = {"compression": "zstd"}
    if compression_level is not None:
        options["compression_level"] = compression_level
    writer = writer or pq.ParquetWriter(path, table.schema, **options)
    writer.write_table(table)
    return writer


def _write_train(output: Path, records: list[dict], wanted: int) -> tuple[str, int]:
    path = output / "train" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer, batch, count = None, [], 0
    for group in iter_list_groups(records, "train", wanted):
        batch.extend(group)
        if len(batch) >= 4096:
            writer = _flush(path, writer, batch)
            count += len(batch)
            batch = []
    if batch:
        writer = _flush(path, writer, batch)
        count += len(batch)
    if writer is None:
        raise RuntimeError("no usable V6 training groups")
    writer.close()
    return str(path), count


def _write_train_flat(output: Path, records: list[dict], wanted_pairs: int, schema=None) -> tuple[str, int]:
    """Write one flat row per selected directed pair; SFT handles packing (no ListNet groups)."""
    path = output / "train" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    wanted_groups = -(-wanted_pairs // 4)
    writer, batch, count = None, [], 0
    for group in iter_list_groups(records, "train", wanted_groups):
        for row in group:
            if count >= wanted_pairs:
                break
            for column in _LISTNET_COLUMNS:
                row.pop(column, None)
            batch.append(row)
            count += 1
        if len(batch) >= 4096:
            writer = _flush(path, writer, batch, schema)
            batch = []
        if count >= wanted_pairs:
            break
    if batch:
        writer = _flush(path, writer, batch, schema)
    if writer is None:
        raise RuntimeError("no usable V6_5 training pairs")
    writer.close()
    return str(path), count


def build(source: Path, output: Path, train_volume: int, *, listnet: bool = True,
          expected_records: int = 138806, expected_split: dict | None = None,
          records: list[dict] | None = None) -> dict:
    # Callers that already materialized the eligible rows pass them in rather than making this
    # re-read and re-convert the same parquet (~225k dicts) a second time.
    if records is None:
        rows = _records(source, expected_records)
    else:
        rows = records
        if expected_records is not None and len(rows) != expected_records:
            raise ValueError(f"expected {expected_records} eligible records, found {len(rows)}")
    split_rows = _split_records(rows)
    counts = {name: len(values) for name, values in split_rows.items()}
    expected = expected_split or {"train": 116060, "validation": 11276, "test": 11470}
    if counts != expected:
        raise RuntimeError(f"frozen record split mismatch: {counts} (expected {expected})")
    heldout = {}
    for split in ("validation", "test"):
        used = set()
        ordinary = _ordinary(split_rows[split], split, used)
        ranking = _ranking(split_rows[split], split, used)
        heldout[split] = (ordinary, ranking)
    # Flat (v6_5) splits share one union schema so the HF single-config loader casts cleanly;
    # the listnet (v6) path keeps its historical per-split schemas untouched.
    schema = None if listnet else _union_split_schema(heldout["validation"][0], heldout["validation"][1])
    if listnet:
        train_path, train_rows = _write_train(output, split_rows["train"], train_volume)
    else:
        train_path, train_rows = _write_train_flat(output, split_rows["train"], train_volume, schema)
    paths = {"train": train_path}
    for split, (ordinary, ranking) in heldout.items():
        paths[split] = _write(output, split, ordinary, schema)
        paths[f"{split}_ranking"] = _write(output, f"{split}_ranking", ranking, schema)
    row_counts = {"train": train_rows, "validation": len(heldout["validation"][0]),
                  "test": len(heldout["test"][0]),
                  "validation_ranking": len(heldout["validation"][1]),
                  "test_ranking": len(heldout["test"][1])}
    hashes = {name: file_sha256(Path(path)) for name, path in paths.items()}
    manifest = {"version": "v6-intern-raw-pair", "source": str(source), "record_counts": counts,
                "listnet": listnet, "rows": row_counts,
                "paths": paths, "parquet_sha256": hashes,
                "intern_revision": "fcb667c380ae01f57693a45b4b5c2d331052a107"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _parse_split(text: str) -> dict:
    out = {}
    for item in text.split(","):
        name, _, value = item.partition("=")
        out[name.strip()] = int(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/eligible/assay_transfer_soft_evidence_v4/records.parquet"))
    parser.add_argument("--output", type=Path, default=Path("datasets/hf_parquet/assay_transfer_raw_pair_v6_intern"))
    parser.add_argument("--listnet", action="store_true",
                        help="build indivisible 4-member ListNet groups (v6); default is flat per-pair (v6_5)")
    parser.add_argument("--train-groups", type=int, default=500000, help="ListNet group count (with --listnet)")
    parser.add_argument("--train-pairs", type=int, default=2000000, help="flat training pair count (default)")
    parser.add_argument("--expected-records", type=int, default=138806)
    parser.add_argument("--expected-split", type=_parse_split, default=None,
                        help="e.g. train=116041,validation=11276,test=11470")
    args = parser.parse_args()
    volume = args.train_groups if args.listnet else args.train_pairs
    print(json.dumps(build(args.source, args.output, volume, listnet=args.listnet,
                           expected_records=args.expected_records,
                           expected_split=args.expected_split), indent=2))


if __name__ == "__main__":
    main()
