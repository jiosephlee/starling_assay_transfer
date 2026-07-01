#!/usr/bin/env python3
"""Build HF-ready oral-bioavailability record splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SPLITS = ("train", "validation_1", "validation_2", "test")
HELDOUT_SPLITS = ("validation_1", "validation_2", "test")
HELDOUT_ROW_CAPS = {"validation_1": 250, "validation_2": 250, "test": 500}
SCHEMA_VERSION = "oral_bioavailability_record_splits_hf_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_file(root: Path, split: str) -> Path:
    files = sorted((root / split).glob("*.parquet"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one parquet file under {root / split}, found {len(files)}")
    return files[0]


def read_pair_split(root: Path, split: str) -> pa.Table:
    cols = [
        "record_id_a",
        "record_id_b",
        "row_index_a",
        "row_index_b",
        "eval_subset",
        "source_oral_bioavailability_value",
    ]
    return pq.read_table(split_file(root, split), columns=cols)


def base_with_labels(path: Path) -> pa.Table:
    base = pq.read_table(path)
    values = base.column("oral_bioavailability_value").to_pylist()
    labels = [1 if float(value) >= 20.0 else 0 for value in values]
    row_index = pa.array(range(base.num_rows), type=pa.uint32())
    out = base.append_column("row_index", row_index)
    return out.append_column("Y", pa.array(labels, type=pa.int8()))


def verify_pair_ids(table: pa.Table, split: str) -> None:
    ids_a = table.column("record_id_a").to_pylist()
    ids_b = table.column("record_id_b").to_pylist()
    idx_a = table.column("row_index_a").to_pylist()
    idx_b = table.column("row_index_b").to_pylist()
    bad = [
        i
        for i, (ra, rb, ia, ib) in enumerate(zip(ids_a, ids_b, idx_a, idx_b, strict=True))
        if str(ra) != str(ia) or str(rb) != str(ib)
    ]
    if bad:
        raise ValueError(f"{split}: record_id/row_index mismatch at rows {bad[:5]}")


def verify_source_values(table: pa.Table, base_values: list[Any], split: str) -> None:
    idx_a = table.column("row_index_a").to_pylist()
    source_values = table.column("source_oral_bioavailability_value").to_pylist()
    for pos, (idx, source_value) in enumerate(zip(idx_a, source_values, strict=True)):
        if float(base_values[int(idx)]) != float(source_value):
            raise ValueError(f"{split}: source value mismatch at pair row {pos}")


def eval_record_indices(table: pa.Table) -> set[int]:
    out: set[int] = set()
    subsets = table.column("eval_subset").to_pylist()
    idx_a = table.column("row_index_a").to_pylist()
    idx_b = table.column("row_index_b").to_pylist()
    for subset, left, right in zip(subsets, idx_a, idx_b, strict=True):
        if subset == "no_overlap":
            out.add(int(left))
            out.add(int(right))
        elif subset == "a_seen_only":
            out.add(int(right))
    return out


def median_labels_by_smiles(smiles: list[str], values: list[Any]) -> dict[str, int]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for text, value in zip(smiles, values, strict=True):
        grouped[str(text)].append(float(value))
    return {key: int(statistics.median(vals) >= 20.0) for key, vals in grouped.items()}


def stable_order(seed: int, smiles: str) -> str:
    return hashlib.sha256(f"{seed}|smiles|{smiles}".encode()).hexdigest()


def stable_move_order(seed: int, split: str, smiles: str) -> str:
    return hashlib.sha256(f"{seed}|{split}|{smiles}".encode()).hexdigest()


def split_validation_smiles(val_smiles: set[str], labels: dict[str, int], seed: int) -> tuple[set[str], set[str]]:
    val1: set[str] = set()
    val2: set[str] = set()
    for label in sorted({labels[text] for text in val_smiles}):
        items = sorted((stable_order(seed, text), text) for text in val_smiles if labels[text] == label)
        n_val1 = (len(items) + 1) // 2
        val1.update(text for _key, text in items[:n_val1])
        val2.update(text for _key, text in items[n_val1:])
    return val1, val2


def take_rows(table: pa.Table, indices: set[int]) -> pa.Table:
    ordered = sorted(indices)
    return table.take(pa.array(ordered, type=pa.uint32()))


def write_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def write_full_splits(base: pa.Table, split_indices: dict[str, set[int]], root: Path) -> dict[str, Any]:
    counts = {}
    for split in SPLITS:
        table = take_rows(base, split_indices[split])
        write_table(table, root / "data" / "full_metadata" / split / "part-00000.parquet")
        counts[split] = split_stats(table)
    return counts


def split_stats(table: pa.Table) -> dict[str, Any]:
    labels = table.column("Y").to_pylist()
    smiles = table.column("smiles").to_pylist()
    return {
        "rows": int(table.num_rows),
        "unique_smiles": len(set(smiles)),
        "Y": {str(key): int(value) for key, value in sorted(Counter(labels).items())},
    }


def smiles_table(smiles_values: set[str], labels: dict[str, int]) -> pa.Table:
    ordered = sorted(smiles_values)
    return pa.Table.from_pydict(
        {
            "smiles": ordered,
            "Y": [labels[text] for text in ordered],
        },
        schema=pa.schema([("smiles", pa.large_string()), ("Y", pa.int8())]),
    )


def write_smiles_splits(split_smiles: dict[str, set[str]], labels: dict[str, int], root: Path) -> dict[str, Any]:
    counts = {}
    for split in SPLITS:
        table = smiles_table(split_smiles[split], labels)
        write_table(table, root / "data" / "smiles_only" / split / "part-00000.parquet")
        counts[split] = split_stats(table)
    return counts


def record_split_indices(smiles: list[str], split_smiles: dict[str, set[str]]) -> dict[str, set[int]]:
    out = {split: set() for split in SPLITS}
    for idx, text in enumerate(smiles):
        out[split_for_smiles(str(text), split_smiles)].add(idx)
    return out


def split_for_smiles(smiles: str, split_smiles: dict[str, set[str]]) -> str:
    for split in HELDOUT_SPLITS:
        if smiles in split_smiles[split]:
            return split
    return "train"


def smiles_split_sets(all_smiles: set[str], val_smiles: set[str], test_smiles: set[str], val1: set[str]) -> dict[str, set[str]]:
    val_smiles = val_smiles - test_smiles
    val2 = val_smiles - val1
    train = all_smiles - val_smiles - test_smiles
    return {"train": train, "validation_1": val1, "validation_2": val2, "test": test_smiles}


def row_counts_by_smiles(smiles: list[str]) -> Counter[str]:
    return Counter(str(text) for text in smiles)


def full_row_count(split_smiles: set[str], counts: Counter[str]) -> int:
    return sum(int(counts[text]) for text in split_smiles)


def shrink_split(
    split_smiles: dict[str, set[str]],
    split: str,
    cap: int,
    counts: Counter[str],
    seed: int,
) -> dict[str, Any]:
    moved: list[str] = []
    while full_row_count(split_smiles[split], counts) > cap:
        items = [(move_key(seed, split, text, counts), text) for text in split_smiles[split]]
        chosen = sorted(items)[0][1]
        split_smiles[split].remove(chosen)
        split_smiles["train"].add(chosen)
        moved.append(chosen)
    return moved_report(moved, counts)


def move_key(seed: int, split: str, smiles: str, counts: Counter[str]) -> tuple[int, str, str]:
    return (-int(counts[smiles]), stable_move_order(seed, split, smiles), smiles)


def moved_report(moved: list[str], counts: Counter[str]) -> dict[str, Any]:
    return {
        "moved_smiles": len(moved),
        "moved_full_metadata_rows": sum(int(counts[text]) for text in moved),
        "smiles": sorted(moved),
    }


def shrink_split_smiles(
    split_smiles: dict[str, set[str]],
    counts: Counter[str],
    seed: int,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    out = {split: set(values) for split, values in split_smiles.items()}
    report = {
        split: shrink_split(out, split, HELDOUT_ROW_CAPS[split], counts, seed)
        for split in HELDOUT_SPLITS
    }
    return out, report


def count_full_splits(base: pa.Table, smiles: list[str], split_smiles: dict[str, set[str]]) -> dict[str, Any]:
    split_indices = record_split_indices(smiles, split_smiles)
    return {split: split_stats(take_rows(base, split_indices[split])) for split in SPLITS}


def count_smiles_splits(split_smiles: dict[str, set[str]], labels: dict[str, int]) -> dict[str, Any]:
    return {split: split_stats(smiles_table(split_smiles[split], labels)) for split in SPLITS}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_readme(path: Path) -> None:
    path.write_text(readme_text())


def readme_text() -> str:
    return """---
pretty_name: Starling Oral Bioavailability Cleaned Aligned With Assay Tool
tags:
- chemistry
- molecular-property-prediction
- tabular
configs:
- config_name: full_metadata
  default: true
  data_files:
  - split: train
    path: data/full_metadata/train/*.parquet
  - split: validation_1
    path: data/full_metadata/validation_1/*.parquet
  - split: validation_2
    path: data/full_metadata/validation_2/*.parquet
  - split: test
    path: data/full_metadata/test/*.parquet
- config_name: smiles_only
  data_files:
  - split: train
    path: data/smiles_only/train/*.parquet
  - split: validation_1
    path: data/smiles_only/validation_1/*.parquet
  - split: validation_2
    path: data/smiles_only/validation_2/*.parquet
  - split: test
    path: data/smiles_only/test/*.parquet
---

# Starling Oral Bioavailability Cleaned Aligned With Assay Tool

`full_metadata` contains cleaned single-record oral-bioavailability rows with
`row_index` and binary label `Y`, where `Y = 1` means
`oral_bioavailability_value >= 20.0`.

`smiles_only` contains exact-SMILES deduplicated rows with only `smiles` and
`Y`. Duplicate SMILES labels are computed from the median raw
`oral_bioavailability_value` across all records for that SMILES.

## Split alignment note

These splits are derived from the condition-key v3_v2 assay-transfer pair
split. The split unit for this record dataset is exact SMILES, so
`full_metadata` and `smiles_only` use the same molecule assignment. To keep
record-level evaluation sizes manageable, some molecules that were present in
the original pair-split held-out molecule pools are intentionally moved back to
train. The row cap policy is recorded in `metadata/split_manifest.json`.
"""


def build(args: argparse.Namespace) -> dict[str, Any]:
    base = base_with_labels(args.base_parquet)
    values = base.column("oral_bioavailability_value").to_pylist()
    smiles = [str(value) for value in base.column("smiles").to_pylist()]
    val_pairs = read_pair_split(args.source_splits_dir, "validation")
    test_pairs = read_pair_split(args.source_splits_dir, "test")
    validate_pairs(val_pairs, test_pairs, values)
    val_indices = eval_record_indices(val_pairs)
    test_indices = eval_record_indices(test_pairs)
    labels = median_labels_by_smiles(smiles, values)
    val_smiles = {smiles[idx] for idx in val_indices}
    test_smiles = {smiles[idx] for idx in test_indices}
    val1_smiles, _val2_smiles = split_validation_smiles(val_smiles - test_smiles, labels, args.seed)
    initial_splits = smiles_split_sets(set(smiles), val_smiles, test_smiles, val1_smiles)
    final_splits, moved = shrink_split_smiles(initial_splits, row_counts_by_smiles(smiles), args.seed)
    return write_outputs(args, base, smiles, labels, initial_splits, final_splits, moved)


def validate_pairs(val_pairs: pa.Table, test_pairs: pa.Table, values: list[Any]) -> None:
    for split, table in (("validation", val_pairs), ("test", test_pairs)):
        verify_pair_ids(table, split)
        verify_source_values(table, values, split)


def write_outputs(
    args: argparse.Namespace,
    base: pa.Table,
    smiles: list[str],
    labels: dict[str, int],
    initial_splits: dict[str, set[str]],
    final_splits: dict[str, set[str]],
    moved: dict[str, Any],
) -> dict[str, Any]:
    prepare_output(args.output_dir, args.overwrite)
    record_splits = record_split_indices(smiles, final_splits)
    initial_full_counts = count_full_splits(base, smiles, initial_splits)
    initial_smiles_counts = count_smiles_splits(initial_splits, labels)
    full_counts = write_full_splits(base, record_splits, args.output_dir)
    smiles_counts = write_smiles_splits(final_splits, labels, args.output_dir)
    manifest = manifest_payload(args, full_counts, smiles_counts)
    add_split_metadata(manifest, initial_full_counts, initial_smiles_counts, moved, final_splits)
    write_json(args.output_dir / "metadata" / "split_manifest.json", manifest)
    write_json(args.output_dir / "metadata" / "dedupe_manifest.json", dedupe_payload(smiles, labels))
    write_readme(args.output_dir / "README.md")
    return manifest


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def manifest_payload(args: argparse.Namespace, full_counts: dict[str, Any], smiles_counts: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "base_parquet": str(args.base_parquet),
        "source_splits_dir": str(args.source_splits_dir),
        "label_policy": "Y = int(oral_bioavailability_value >= 20.0)",
        "split_key": "exact raw smiles string",
        "validation_split_seed": int(args.seed),
        "full_metadata_counts": full_counts,
        "smiles_only_counts": smiles_counts,
    }


def add_split_metadata(
    manifest: dict[str, Any],
    initial_full_counts: dict[str, Any],
    initial_smiles_counts: dict[str, Any],
    moved: dict[str, Any],
    final_splits: dict[str, set[str]],
) -> None:
    manifest["row_cap_policy"] = {"heldout_full_metadata_row_caps": HELDOUT_ROW_CAPS}
    manifest["pre_cap_full_metadata_counts"] = initial_full_counts
    manifest["pre_cap_smiles_only_counts"] = initial_smiles_counts
    manifest["moved_to_train"] = moved
    manifest["exact_smiles_overlap_report"] = overlap_report(final_splits)
    manifest["canonical_smiles_overlap_report"] = canonical_overlap_report(final_splits)


def overlap_report(split_smiles: dict[str, set[str]]) -> dict[str, Any]:
    pairs = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            pairs[f"{left}__{right}"] = len(split_smiles[left] & split_smiles[right])
    return {"policy": "exact raw smiles string", "pairwise_overlap_counts": pairs}


def canonical_overlap_report(split_smiles: dict[str, set[str]]) -> dict[str, Any]:
    try:
        canonical = canonical_split_smiles(split_smiles)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    report = overlap_report(canonical)
    report["policy"] = "rdkit canonical isomeric smiles"
    return {"available": True, **report}


def canonical_split_smiles(split_smiles: dict[str, set[str]]) -> dict[str, set[str]]:
    from rdkit import Chem

    out = {}
    for split, values in split_smiles.items():
        out[split] = {canonical_smiles(Chem, text) for text in values}
    return out


def canonical_smiles(chem: Any, smiles: str) -> str:
    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return str(chem.MolToSmiles(mol, isomericSmiles=True))


def dedupe_payload(smiles: list[str], labels: dict[str, int]) -> dict[str, Any]:
    rows_by_smiles = Counter(smiles)
    return {
        "schema_version": "oral_bioavailability_smiles_dedupe_v1",
        "created_at_utc": utc_now(),
        "dedupe_key": "exact smiles string",
        "label_policy": "median oral_bioavailability_value per smiles, then >= 20.0",
        "input_rows": len(smiles),
        "unique_smiles": len(labels),
        "duplicate_smiles_groups": sum(1 for count in rows_by_smiles.values() if count > 1),
    }


def upload(output_dir: Path, repo_id: str, private: bool) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    return api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, repo_type="dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-parquet", type=Path, default=Path("datasets/base/Oral_bioavailability_cleaned_v3/train.parquet"))
    parser.add_argument("--source-splits-dir", type=Path, default=Path("datasets/pairs_split_full/oral_bioavailability_condition_key_shared_eval_full_v3_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/starling_eval/condition_key_v3_record_splits_hf"))
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default="jiosephlee/starling-oral-bioavailability-cleaned-aligned-with-assay-tool")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(args)
    if args.upload:
        manifest["uploaded_commit"] = upload(args.output_dir, args.repo_id, args.private)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
