#!/usr/bin/env python3
"""Create shared directed full eval splits across compact pair universes."""

from __future__ import annotations

import argparse
import heapq
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_transfer import (  # noqa: E402
    EVAL_SPLITS,
    SPLITS,
    largest_remainder_allocation,
    parquet_files_from_input,
    stable_priority,
    utc_now,
    write_json,
)
from materialize_full_pairs_from_splits import metadata_struct, molecule_struct_type  # noqa: E402


DEFAULT_BASE_INPUT = "datasets/base/Oral_bioavailability_cleaned_v2_condition_key"
DEFAULT_CONDITION_KEY_PAIRS = Path("datasets/pairs_compact/oral_bioavailability_pairs_condition_key_v1")
DEFAULT_SAME_SPECIES_PAIRS = Path("datasets/pairs_compact/oral_bioavailability_pairs_same_species_v2")
DEFAULT_NO_CONSTRAINTS_PAIRS = Path("datasets/pairs_compact/oral_bioavailability_pairs_no_constraints_v1")
DEFAULT_OUTPUT_ROOT = Path("datasets/pairs_split_full")
DEFAULT_METADATA_COLUMNS = [
    "support_text",
    "molecule_name",
    "bioavailability_report_type",
    "species_or_population",
    "dose",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
    "extra_details",
]
DEFAULT_SHARED_COMPATIBILITY_COLUMN = "species_or_population_normalized"
SCHEMA_VERSION = "generic_transfer_pair_splits_directional_full_v1"
SPLIT_VERSION = "oral_bioavailability_shared_eval_directed_v1"
LABEL_TEXT = {0: "not_transfer", 1: "transfer"}
PRESENCE_TEXT = {0: "null<>null", 1: "not_null<>null", 2: "not_null<>not_null"}
EVAL_SUBSETS = ("no_overlap", "a_seen_only", "both_seen")
TRAIN_DIRECTION_MODES = ("bidirectional", "unidirectional")


@dataclass(frozen=True)
class MoleculeIndex:
    record_to_molecule: np.ndarray
    record_canonical_smiles: list[str]
    molecule_canonical_smiles: list[str]
    stats: dict[str, Any]

    def molecule_id(self, record_index: int) -> int:
        return int(self.record_to_molecule[int(record_index)])

    def pair_molecules(self, pair: tuple[int, int]) -> tuple[int, int]:
        return self.molecule_id(pair[0]), self.molecule_id(pair[1])


def canonical_smiles(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError("missing SMILES")
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"invalid SMILES: {text}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def records_path(input_dir: Path) -> Path:
    records_dir = input_dir / "records"
    if records_dir.exists():
        return records_dir
    return input_dir / "records.parquet"


def compact_schema(metadata_columns: list[str]) -> pa.Schema:
    fields = [
        pa.field("row_index_a", pa.uint32()),
        pa.field("row_index_b", pa.uint32()),
        pa.field("transfer_label", pa.int8()),
        pa.field("value_difference", pa.float32()),
        pa.field("weighted_tanimoto", pa.float32()),
        pa.field("similarity_bucket", pa.int8()),
    ]
    for column in metadata_columns:
        fields.append(pa.field(f"{column}_presence_pair", pa.int8()))
    return pa.schema(fields)


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class ProgressLogger:
    def __init__(self, phase: str, total: int | None, interval_seconds: float) -> None:
        self.phase = phase
        self.total = total if total and total > 0 else None
        self.interval_seconds = interval_seconds
        self.start = time.monotonic()
        self.last = 0.0
        self.update(0, force=True)

    def update(self, current: int, *, force: bool = False, extra: str = "") -> None:
        if self.interval_seconds <= 0 and not force:
            return
        now = time.monotonic()
        if not force and now - self.last < self.interval_seconds:
            return
        elapsed = max(now - self.start, 1e-9)
        rate = current / elapsed
        if self.total:
            pct = 100.0 * current / self.total
            remaining = max(self.total - current, 0)
            eta = remaining / rate if rate > 0 else None
            rows = f"{current:,}/{self.total:,} ({pct:.2f}%)"
        else:
            eta = None
            rows = f"{current:,}"
        suffix = f" {extra}" if extra else ""
        print(
            f"[{utc_now()}] {self.phase}: rows={rows} "
            f"rate={rate:,.0f}/s elapsed={format_duration(elapsed)} eta={format_duration(eta)}{suffix}",
            file=sys.stderr,
            flush=True,
        )
        self.last = now

    def finish(self, current: int, *, extra: str = "") -> None:
        self.update(current, force=True, extra=extra)


def parquet_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no Parquet files found under {path}")
    return files


def parquet_row_count(path: Path) -> int:
    return sum(pq.ParquetFile(file_path).metadata.num_rows for file_path in parquet_files(path))


def pair_dir_records(pair_dir: Path) -> Path:
    return records_path(pair_dir)


def pair_key(left: int, right: int) -> tuple[int, int]:
    left = int(left)
    right = int(right)
    return (left, right) if left < right else (right, left)


def molecule_pair_key(left: int, right: int) -> tuple[int, int]:
    left = int(left)
    right = int(right)
    return (left, right) if left < right else (right, left)


def direction_endpoints(pair: tuple[int, int], direction: str) -> tuple[int, int]:
    left, right = pair
    if direction == "a_to_b":
        return left, right
    if direction == "b_to_a":
        return right, left
    raise ValueError(f"invalid direction: {direction}")


def selected_direction(seed: int, split: str, subset: str, pair: tuple[int, int]) -> str:
    return "a_to_b" if stable_priority(seed, split, subset, pair[0], pair[1]) % 2 == 0 else "b_to_a"


def split_output_dirs(output_root: Path, train_direction_mode: str, name_suffix: str = "") -> dict[str, Path]:
    suffix = (
        "shared_eval_unidirectional_full"
        if train_direction_mode == "unidirectional"
        else "shared_eval_full"
    )
    suffix = f"{suffix}{name_suffix}"
    return {
        "condition_key": output_root / f"oral_bioavailability_condition_key_{suffix}",
        "same_species_v2": output_root / f"oral_bioavailability_same_species_v2_{suffix}",
        "no_constraints": output_root / f"oral_bioavailability_no_constraints_{suffix}",
    }


def pair_universes(args: argparse.Namespace) -> dict[str, Path]:
    universes = {
        "condition_key": args.condition_key_pairs,
        "same_species_v2": args.same_species_pairs,
        "no_constraints": args.no_constraints_pairs,
    }
    selected = set(getattr(args, "universes", ()) or universes)
    return {key: value for key, value in universes.items() if key in selected}


def load_train_support_graph(args: argparse.Namespace, molecule_index: MoleculeIndex) -> dict[int, Counter[int]]:
    input_path = pair_dir_records(args.condition_key_pairs)
    total_rows = parquet_row_count(input_path)
    dataset = ds.dataset(input_path, format="parquet")
    graph: dict[int, Counter[int]] = {}
    progress = ProgressLogger("load condition-key train support graph", total_rows, args.progress_every_seconds)
    processed = 0
    for batch in dataset.to_batches(columns=["row_index_a", "row_index_b"], batch_size=args.batch_size):
        left = batch.column("row_index_a").to_pylist()
        right = batch.column("row_index_b").to_pylist()
        for a_raw, b_raw in zip(left, right, strict=True):
            a = molecule_index.molecule_id(int(a_raw))
            b = molecule_index.molecule_id(int(b_raw))
            if a == b:
                graph.setdefault(a, Counter())[a] += 1
            else:
                graph.setdefault(a, Counter())[b] += 1
                graph.setdefault(b, Counter())[a] += 1
        processed += batch.num_rows
        progress.update(processed, extra=f"molecules={len(graph):,}")
    progress.finish(processed, extra=f"molecules={len(graph):,}")
    return graph


def load_base_table(args: argparse.Namespace) -> pa.Table:
    columns = sorted(set([args.smiles_column, args.value_column, *args.metadata_columns]))
    tables = [pq.read_table(path, columns=columns) for path in parquet_files_from_input(args.base_input)]
    if not tables:
        raise RuntimeError("no base Parquet files loaded")
    table = pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]
    if table[args.value_column].null_count:
        raise ValueError(f"{args.value_column} has null values")
    return table


def build_molecule_index(base_table: pa.Table, smiles_column: str) -> MoleculeIndex:
    smiles_values = base_table[smiles_column].combine_chunks().to_pylist()
    molecule_by_smiles: dict[str, int] = {}
    record_to_molecule: list[int] = []
    record_canonical: list[str] = []
    invalid_examples: list[dict[str, Any]] = []
    for row_index, value in enumerate(smiles_values):
        try:
            canonical = canonical_smiles(value)
        except ValueError as exc:
            invalid_examples.append({"row_index": row_index, "smiles": value, "error": str(exc)})
            continue
        molecule_id = molecule_by_smiles.setdefault(canonical, len(molecule_by_smiles))
        record_to_molecule.append(molecule_id)
        record_canonical.append(canonical)
    if invalid_examples:
        preview = "; ".join(
            f"{item['row_index']}={item['smiles']!r}" for item in invalid_examples[:5]
        )
        raise ValueError(f"base input contains invalid SMILES for split molecule identity: {preview}")
    counts = Counter(record_to_molecule)
    duplicate_groups = [count for count in counts.values() if count > 1]
    molecule_canonical = [None] * len(molecule_by_smiles)
    for smiles, molecule_id in molecule_by_smiles.items():
        molecule_canonical[molecule_id] = smiles
    return MoleculeIndex(
        record_to_molecule=np.asarray(record_to_molecule, dtype=np.uint32),
        record_canonical_smiles=record_canonical,
        molecule_canonical_smiles=[str(value) for value in molecule_canonical],
        stats={
            "identity": "rdkit_canonical_isomeric_smiles",
            "record_rows": len(record_to_molecule),
            "unique_molecules": len(molecule_by_smiles),
            "duplicate_molecules": len(duplicate_groups),
            "duplicate_records": int(sum(duplicate_groups)),
            "max_records_per_molecule": int(max(duplicate_groups) if duplicate_groups else 1),
        },
    )


def output_schema(metadata_columns: list[str]) -> pa.Schema:
    molecule_type = molecule_struct_type(metadata_columns)
    return pa.schema(
        [
            ("schema_version", pa.string()),
            ("source_schema_version", pa.string()),
            ("pair_id", pa.string()),
            ("source_pair_id", pa.string()),
            ("direction", pa.string()),
            ("record_id_a", pa.string()),
            ("record_id_b", pa.string()),
            ("row_index_a", pa.uint32()),
            ("row_index_b", pa.uint32()),
            ("group_id", pa.string()),
            ("molecule_a", molecule_type),
            ("molecule_b", molecule_type),
            ("source_oral_bioavailability_value", pa.float64()),
            ("transfer_label", pa.string()),
            ("value_difference", pa.float32()),
            ("T_transfer", pa.float32()),
            ("T_not_transfer", pa.float32()),
            ("weighted_tanimoto", pa.float32()),
            ("similarity_bucket", pa.int8()),
            ("split", pa.string()),
            ("split_version", pa.string()),
            ("eval_subset", pa.string()),
        ]
    )


def constant_array(value: str | None, length: int, type_: pa.DataType = pa.string()) -> pa.Array:
    if value is None:
        return pa.nulls(length, type=type_)
    return pa.array([value] * length, type=type_)


def labels_array(labels: pa.Array) -> pa.Array:
    is_transfer = pc.equal(labels, pa.scalar(1, type=pa.int8()))
    return pc.if_else(
        is_transfer,
        pa.scalar("transfer", type=pa.string()),
        pa.scalar("not_transfer", type=pa.string()),
    )


def as_pylist_array(values: Iterable[Any], type_: pa.DataType) -> pa.Array:
    return pa.array(list(values), type=type_)


def full_table_from_compact(
    table: pa.Table,
    *,
    base_table: pa.Table,
    canonical_smiles_array: pa.Array,
    split: str,
    direction: str,
    eval_subset: str | None,
    metadata_columns: list[str],
    smiles_column: str,
    value_column: str,
    source_schema_version: str,
    transfer_threshold: float,
    not_transfer_threshold: float,
) -> pa.Table:
    length = table.num_rows
    forward = direction == "a_to_b"
    left = table["row_index_a"].combine_chunks().cast(pa.uint32())
    right = table["row_index_b"].combine_chunks().cast(pa.uint32())
    source = left if forward else right
    query = right if forward else left
    source_text = pc.cast(source, pa.string())
    query_text = pc.cast(query, pa.string())
    source_pair_id = pc.binary_join_element_wise(pc.cast(left, pa.string()), pc.cast(right, pa.string()), ":")
    pair_id = pc.binary_join_element_wise(source_pair_id, constant_array(direction, length), ":")
    base_values = base_table[value_column].combine_chunks().cast(pa.float64())

    return pa.Table.from_arrays(
        [
            constant_array(SCHEMA_VERSION, length),
            constant_array(source_schema_version, length),
            pair_id,
            source_pair_id,
            constant_array(direction, length),
            source_text,
            query_text,
            source,
            query,
            constant_array(None, length),
            molecule_struct_from_indices(
                base_table=base_table,
                canonical_smiles_array=canonical_smiles_array,
                indices=source,
                metadata_columns=metadata_columns,
                smiles_column=smiles_column,
            ),
            molecule_struct_from_indices(
                base_table=base_table,
                canonical_smiles_array=canonical_smiles_array,
                indices=query,
                metadata_columns=metadata_columns,
                smiles_column=smiles_column,
            ),
            pc.take(base_values, source),
            labels_array(table["transfer_label"].combine_chunks().cast(pa.int8())),
            table["value_difference"].combine_chunks().cast(pa.float32()),
            pa.array([transfer_threshold] * length, type=pa.float32()),
            pa.array([not_transfer_threshold] * length, type=pa.float32()),
            table["weighted_tanimoto"].combine_chunks().cast(pa.float32()),
            table["similarity_bucket"].combine_chunks().cast(pa.int8()),
            constant_array(split, length),
            constant_array(SPLIT_VERSION, length),
            constant_array(eval_subset, length),
        ],
        schema=output_schema(metadata_columns),
    )


def molecule_struct_from_indices(
    *,
    base_table: pa.Table,
    canonical_smiles_array: pa.Array,
    indices: pa.Array,
    metadata_columns: list[str],
    smiles_column: str,
) -> pa.StructArray:
    record_ids = pc.cast(indices, pa.string())
    return pa.StructArray.from_arrays(
        [
            record_ids,
            indices,
            pc.take(canonical_smiles_array, indices),
            metadata_struct(base_table, indices, metadata_columns),
        ],
        names=["record_id", "row_index", "canonical_smiles", "metadata"],
    )


class RollingParquetWriter:
    def __init__(self, split_dir: Path, schema: pa.Schema, *, file_row_limit: int, compression: str) -> None:
        self.split_dir = split_dir
        self.schema = schema
        self.file_row_limit = file_row_limit
        self.compression = compression
        self.part_id = 0
        self.rows_in_part = 0
        self.writer: pq.ParquetWriter | None = None
        self.split_dir.mkdir(parents=True, exist_ok=True)

    def write_table(self, table: pa.Table) -> None:
        offset = 0
        while offset < table.num_rows:
            if self.writer is None:
                self.writer = pq.ParquetWriter(
                    self.split_dir / f"part-{self.part_id:05d}.parquet",
                    self.schema,
                    compression=self.compression,
                    use_dictionary=False,
                    write_statistics=True,
                )
                self.rows_in_part = 0
            remaining = table.num_rows - offset
            space = max(self.file_row_limit - self.rows_in_part, 1)
            take = min(remaining, space)
            self.writer.write_table(table.slice(offset, take))
            offset += take
            self.rows_in_part += take
            if self.rows_in_part >= self.file_row_limit:
                self.close_part()

    def close_part(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self.part_id += 1
            self.rows_in_part = 0

    def close(self) -> None:
        self.close_part()


def prepare_outputs(output_dirs: dict[str, Path], overwrite: bool, splits: Iterable[str]) -> None:
    for output_dir in output_dirs.values():
        present = [output_dir / split for split in splits] + [output_dir / "metadata.json"]
        if output_dir.exists() and not overwrite and any(path.exists() for path in present):
            raise FileExistsError(f"{output_dir} already contains output; pass --overwrite")
        output_dir.parent.mkdir(parents=True, exist_ok=True)


def make_staged_output_dirs(output_dirs: dict[str, Path]) -> dict[str, Path]:
    staged: dict[str, Path] = {}
    for universe, output_dir in output_dirs.items():
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staged[universe] = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent)
        )
    return staged


def cleanup_staged_output_dirs(staged_dirs: Iterable[Path]) -> None:
    for staged_dir in staged_dirs:
        if staged_dir.exists():
            shutil.rmtree(staged_dir)


def install_staged_output_dirs(staged_dirs: dict[str, Path], final_dirs: dict[str, Path], overwrite: bool) -> None:
    for universe, staged_dir in staged_dirs.items():
        final_dir = final_dirs[universe]
        if final_dir.exists():
            if not overwrite:
                raise FileExistsError(f"{final_dir} exists; pass --overwrite")
            backup = final_dir.with_name(f".{final_dir.name}.replaced.{int(time.time())}")
            final_dir.rename(backup)
            try:
                staged_dir.rename(final_dir)
            except Exception:
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                backup.rename(final_dir)
                raise
            shutil.rmtree(backup)
        else:
            staged_dir.rename(final_dir)


def compact_rows_to_dicts(batch: pa.RecordBatch) -> list[dict[str, Any]]:
    return pa.Table.from_batches([batch]).to_pylist()


def row_priority(seed: int, split: str, subset: str, row: dict[str, Any]) -> int:
    return stable_priority(seed, split, subset, row["row_index_a"], row["row_index_b"], row["transfer_label"])


def stratum_for_row(row: dict[str, Any], metadata_columns: list[str]) -> tuple[Any, ...]:
    return (
        int(row["transfer_label"]),
        int(row["similarity_bucket"]),
        *(int(row[f"{column}_presence_pair"]) for column in metadata_columns),
    )


def stratum_to_string(stratum: tuple[Any, ...], metadata_columns: list[str]) -> str:
    label, bucket, *presence = stratum
    parts = [f"transfer_label={LABEL_TEXT[int(label)]}", f"similarity_bucket={bucket}"]
    for column, code in zip(metadata_columns, presence, strict=True):
        parts.append(f"{column}={PRESENCE_TEXT[int(code)]}")
    return "|".join(parts)


def serializable_counter(counter: Counter[Any] | dict[Any, int], metadata_columns: list[str]) -> dict[str, int]:
    return {
        stratum_to_string(key, metadata_columns) if isinstance(key, tuple) else str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: repr(item[0]))
        if int(value) > 0
    }


def collect_shared_stratum_counts(args: argparse.Namespace) -> Counter[tuple[Any, ...]]:
    compatibility_values = load_shared_compatibility_values(args)
    input_path = pair_dir_records(args.condition_key_pairs)
    total_rows = parquet_row_count(input_path)
    dataset = ds.dataset(input_path, format="parquet")
    columns = [field.name for field in compact_schema(args.pair_metadata_columns)]
    counts: Counter[tuple[Any, ...]] = Counter()
    progress = ProgressLogger("collect shared eval stratum counts", total_rows, args.progress_every_seconds)
    processed = 0
    for batch in dataset.to_batches(columns=columns, batch_size=args.batch_size):
        rows = compact_rows_to_dicts(batch)
        for row in rows:
            if not shared_eval_compatible(row, compatibility_values):
                continue
            counts[stratum_for_row(row, args.pair_metadata_columns)] += 1
        processed += batch.num_rows
        progress.update(processed, extra=f"strata={len(counts):,}")
    progress.finish(processed, extra=f"strata={len(counts):,}")
    return counts


def candidate_pool_capacity(args: argparse.Namespace, quota: int) -> int:
    return max(quota, quota * args.candidate_pool_multiplier)


def collect_eval_pools(
    args: argparse.Namespace,
) -> tuple[
    dict[str, dict[str, dict[tuple[Any, ...], list[dict[str, Any]]]]],
    Counter[tuple[Any, ...]],
    dict[tuple[Any, ...], int],
]:
    compatibility_values = load_shared_compatibility_values(args)
    source_strata = collect_shared_stratum_counts(args)
    allocation = largest_remainder_allocation(args.eval_directions_per_subset, source_strata)
    capacities = {
        stratum: candidate_pool_capacity(args, quota)
        for stratum, quota in allocation.items()
        if quota > 0
    }
    heaps: dict[str, dict[tuple[Any, ...], list[tuple[int, int, int, dict[str, Any]]]]] = {
        split: {stratum: [] for stratum in capacities}
        for split in EVAL_SPLITS
    }
    input_path = pair_dir_records(args.condition_key_pairs)
    total_rows = parquet_row_count(input_path)
    dataset = ds.dataset(input_path, format="parquet")
    columns = [field.name for field in compact_schema(args.pair_metadata_columns)]
    progress = ProgressLogger("collect shared eval stratified candidate pools", total_rows, args.progress_every_seconds)
    processed = 0
    skipped = Counter()
    for batch in dataset.to_batches(columns=columns, batch_size=args.batch_size):
        rows = compact_rows_to_dicts(batch)
        for row in rows:
            if not shared_eval_compatible(row, compatibility_values):
                skipped["shared_eval_incompatible"] += 1
                continue
            stratum = stratum_for_row(row, args.pair_metadata_columns)
            if stratum not in capacities:
                skipped["unallocated_stratum"] += 1
                continue
            for split in EVAL_SPLITS:
                priority = stable_priority(
                    args.seed,
                    split,
                    "eval_pool",
                    repr(stratum),
                    row["row_index_a"],
                    row["row_index_b"],
                )
                entry = (-priority, int(row["row_index_a"]), int(row["row_index_b"]), row)
                heap = heaps[split][stratum]
                capacity = capacities[stratum]
                if len(heap) < capacity:
                    heapq.heappush(heap, entry)
                elif priority < -heap[0][0]:
                    heapq.heapreplace(heap, entry)
        processed += batch.num_rows
        progress.update(
            processed,
            extra=f"pooled={sum(len(heap) for split_heaps in heaps.values() for heap in split_heaps.values()):,}",
        )
    progress.finish(
        processed,
        extra=(
            f"pooled={sum(len(heap) for split_heaps in heaps.values() for heap in split_heaps.values()):,} "
            f"skipped={sum(skipped.values()):,}"
        ),
    )

    pools: dict[str, dict[str, dict[tuple[Any, ...], list[dict[str, Any]]]]] = {}
    for split, split_heaps in heaps.items():
        base_rows_by_stratum = {
            stratum: [entry[3] for entry in heap]
            for stratum, heap in split_heaps.items()
        }
        pools[split] = {}
        for subset in EVAL_SUBSETS:
            pools[split][subset] = {
                stratum: list(rows)
                for stratum, rows in base_rows_by_stratum.items()
            }
    return pools, source_strata, allocation


def load_shared_compatibility_values(args: argparse.Namespace) -> list[Any]:
    tables = [
        pq.read_table(path, columns=[args.shared_eval_compatibility_column])
        for path in parquet_files_from_input(args.base_input)
    ]
    if not tables:
        raise RuntimeError("no base Parquet files loaded")
    table = pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]
    return table[args.shared_eval_compatibility_column].to_pylist()


def shared_eval_compatible(row: dict[str, Any], compatibility_values: list[Any]) -> bool:
    left = int(row["row_index_a"])
    right = int(row["row_index_b"])
    if left >= len(compatibility_values) or right >= len(compatibility_values):
        return False
    left_value = compatibility_values[left]
    right_value = compatibility_values[right]
    if left_value is None or right_value is None:
        return False
    return str(left_value).strip() == str(right_value).strip()


def select_eval_rows(args: argparse.Namespace, molecule_index: MoleculeIndex) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools, source_strata, allocation = collect_eval_pools(args)
    support_graph = load_train_support_graph(args, molecule_index)
    support_counts = {molecule: sum(neighbors.values()) for molecule, neighbors in support_graph.items()}
    removed_train_pairs: set[tuple[int, int]] = set()
    removed_molecule_pairs: Counter[tuple[int, int]] = Counter()
    molecule_degree = eval_candidate_degrees(pools, molecule_index)
    preferred_no_train = {
        molecule
        for molecule, _count in molecule_degree.most_common(args.preferred_no_train_molecules)
    }
    order_eval_pools(
        pools,
        molecule_index=molecule_index,
        molecule_degree=molecule_degree,
        preferred_no_train=preferred_no_train,
        seed=args.seed,
    )
    selected: list[dict[str, Any]] = []
    selected_pairs: set[tuple[int, int]] = set()
    no_train_molecules: set[int] = set()
    train_required_molecules: set[int] = set()
    eval_molecules: set[int] = set()
    eval_molecules_by_split: dict[str, set[int]] = {split: set() for split in EVAL_SPLITS}
    stats: dict[str, Any] = {
        "target_directions_per_eval_split_subset": args.eval_directions_per_subset,
        "molecule_identity": molecule_index.stats,
        "selection_policy": {
            "stratification_source": (
                "condition-key shared-compatible compact pair universe before validation/test molecule removal"
            ),
            "stratification_fields": ["transfer_label", "similarity_bucket", *args.pair_metadata_columns],
            "stratum_allocation": "largest_remainder_proportional_by_stratum",
            "underfilled_quota_policy": "leave_underfilled_no_backfill",
            "disjoint_eval_molecules": bool(args.disjoint_eval_molecules),
        },
        "source_strata": serializable_counter(source_strata, args.pair_metadata_columns),
        "allocation_per_eval_split_subset": serializable_counter(Counter(allocation), args.pair_metadata_columns),
        "selected_counts": {split: Counter() for split in EVAL_SPLITS},
        "pool_sizes": {
            split: {
                subset: int(sum(len(rows) for rows in stratum_rows.values()))
                for subset, stratum_rows in subset_rows.items()
            }
            for split, subset_rows in pools.items()
        },
        "pool_strata": {
            split: {
                subset: int(sum(1 for rows in stratum_rows.values() if rows))
                for subset, stratum_rows in subset_rows.items()
            }
            for split, subset_rows in pools.items()
        },
        "preferred_no_train_molecules": len(preferred_no_train),
        "skipped": Counter(),
        "selected_strata": {
            split: {subset: Counter() for subset in EVAL_SUBSETS}
            for split in EVAL_SPLITS
        },
        "unfilled_allocation": {
            split: {subset: Counter(allocation) for subset in EVAL_SUBSETS}
            for split in EVAL_SPLITS
        },
    }

    def pair_removed(pair: tuple[int, int]) -> bool:
        return pair in removed_train_pairs

    def removed_count_between(left: int, right: int) -> int:
        return int(removed_molecule_pairs[molecule_pair_key(left, right)])

    def support_between(left: int, right: int) -> int:
        return max(0, int(support_graph.get(left, {}).get(right, 0)) - removed_count_between(left, right))

    def support_after_pair_removal(molecule: int, pair: tuple[int, int]) -> int:
        value = support_counts.get(molecule, 0)
        pair_molecules = molecule_index.pair_molecules(pair)
        if molecule in pair_molecules and not pair_removed(pair):
            other = pair_molecules[1] if pair_molecules[0] == molecule else pair_molecules[0]
            if other not in no_train_molecules:
                value -= 1
        return value

    def no_train_decrements(molecules: Iterable[int], *, extra_removed_pair: tuple[int, int] | None = None) -> Counter[int]:
        decrements: Counter[int] = Counter()
        additions = {int(molecule) for molecule in molecules}
        extra_pair_molecules = molecule_index.pair_molecules(extra_removed_pair) if extra_removed_pair else None
        for molecule in additions:
            if molecule in no_train_molecules:
                continue
            for neighbor, count in support_graph.get(molecule, {}).items():
                decrement = int(count) - removed_count_between(molecule, neighbor)
                if extra_pair_molecules and molecule in extra_pair_molecules and neighbor in extra_pair_molecules:
                    decrement = max(0, decrement - 1)
                if decrement <= 0:
                    continue
                if neighbor in no_train_molecules or neighbor in additions:
                    continue
                decrements[neighbor] += decrement
        return decrements

    def can_mark_no_train(molecules: Iterable[int], *, extra_removed_pair: tuple[int, int] | None = None) -> bool:
        decrements = no_train_decrements(molecules, extra_removed_pair=extra_removed_pair)
        for molecule, decrement in decrements.items():
            if molecule in train_required_molecules and support_counts.get(molecule, 0) - decrement <= 0:
                return False
        return True

    def remove_train_pair(pair: tuple[int, int]) -> None:
        if pair_removed(pair):
            return
        removed_train_pairs.add(pair)
        left, right = molecule_index.pair_molecules(pair)
        removed_molecule_pairs[molecule_pair_key(left, right)] += 1
        if left not in no_train_molecules:
            support_counts[left] = support_counts.get(left, 0) - 1
        if right != left and right not in no_train_molecules:
            support_counts[right] = support_counts.get(right, 0) - 1

    def mark_no_train(molecules: Iterable[int], *, extra_removed_pair: tuple[int, int] | None = None) -> None:
        decrements = no_train_decrements(molecules, extra_removed_pair=extra_removed_pair)
        for molecule, decrement in decrements.items():
            support_counts[molecule] = support_counts.get(molecule, 0) - decrement
        for molecule in molecules:
            no_train_molecules.add(int(molecule))
            support_counts[int(molecule)] = 0

    selection_order = [
        (split, subset)
        for subset in ("no_overlap", "a_seen_only", "both_seen")
        for split in EVAL_SPLITS
    ]
    stats["selection_policy"]["selection_order"] = [
        {"split": split, "subset": subset} for split, subset in selection_order
    ]

    for split, subset in selection_order:
            remaining = Counter(allocation)
            exhausted_unfilled: Counter[tuple[Any, ...]] = Counter()
            pointers = {stratum: 0 for stratum in pools[split][subset]}
            step = 0
            while remaining:
                stratum = choose_stratum(
                    remaining,
                    allocation,
                    pools[split][subset],
                    pointers,
                    args.seed,
                    split,
                    subset,
                    step,
                )
                if stratum is None:
                    stats["skipped"]["no_selectable_remaining_stratum"] += 1
                    break
                rows = pools[split][subset].get(stratum) or []
                selected_from_stratum = False
                while pointers[stratum] < len(rows):
                    row = rows[pointers[stratum]]
                    pointers[stratum] += 1
                    pair = pair_key(row["row_index_a"], row["row_index_b"])
                    if pair in selected_pairs:
                        stats["skipped"]["already_selected_pair"] += 1
                        continue
                    left_record, right_record = pair
                    left, right = molecule_index.pair_molecules(pair)
                    pair_molecule_set = {left, right}
                    if args.disjoint_eval_molecules:
                        other_eval_molecules = eval_molecules_by_split["test" if split == "validation" else "validation"]
                        if pair_molecule_set & other_eval_molecules:
                            stats["skipped"]["cross_eval_split_molecule_conflict"] += 1
                            continue
                    if subset == "no_overlap":
                        if left in train_required_molecules or right in train_required_molecules:
                            stats["skipped"]["no_overlap_train_required_conflict"] += 1
                            continue
                        if not can_mark_no_train((left, right), extra_removed_pair=pair):
                            stats["skipped"]["no_overlap_train_support_conflict"] += 1
                            continue
                        direction = selected_direction(args.seed, split, subset, pair)
                        remove_train_pair(pair)
                        mark_no_train((left, right), extra_removed_pair=pair)
                        eval_molecules.update((left, right))
                        eval_molecules_by_split[split].update((left, right))
                    elif subset == "a_seen_only":
                        direction = selected_direction(args.seed, split, subset, pair)
                        source_record, target_record = direction_endpoints(pair, direction)
                        source = molecule_index.molecule_id(source_record)
                        target = molecule_index.molecule_id(target_record)
                        if source == target:
                            stats["skipped"]["a_seen_only_same_molecule_conflict"] += 1
                            continue
                        if source in no_train_molecules or target in train_required_molecules:
                            stats["skipped"]["a_seen_only_seen_status_conflict"] += 1
                            continue
                        if support_counts.get(source, 0) - support_between(source, target) <= 0:
                            stats["skipped"]["a_seen_only_source_without_train_support"] += 1
                            continue
                        if not can_mark_no_train((target,)):
                            stats["skipped"]["a_seen_only_target_train_support_conflict"] += 1
                            continue
                        remove_train_pair(pair)
                        train_required_molecules.add(source)
                        mark_no_train((target,))
                        eval_molecules.update((source, target))
                        eval_molecules_by_split[split].update((source, target))
                    else:
                        if left in no_train_molecules or right in no_train_molecules:
                            stats["skipped"]["both_seen_no_train_molecule_conflict"] += 1
                            continue
                        if support_after_pair_removal(left, pair) <= 0 or support_after_pair_removal(right, pair) <= 0:
                            stats["skipped"]["both_seen_without_train_support"] += 1
                            continue
                        direction = selected_direction(args.seed, split, subset, pair)
                        remove_train_pair(pair)
                        train_required_molecules.update((left, right))
                        eval_molecules.update((left, right))
                        eval_molecules_by_split[split].update((left, right))

                    selected_pairs.add(pair)
                    out = dict(row)
                    out["split"] = split
                    out["eval_subset"] = subset
                    out["direction"] = direction
                    out["molecule_id_a"] = int(molecule_index.molecule_id(left_record))
                    out["molecule_id_b"] = int(molecule_index.molecule_id(right_record))
                    selected.append(out)
                    stats["selected_counts"][split][subset] += 1
                    stats["selected_strata"][split][subset][stratum] += 1
                    remaining[stratum] -= 1
                    if remaining[stratum] <= 0:
                        del remaining[stratum]
                    selected_from_stratum = True
                    break
                if not selected_from_stratum:
                    stats["skipped"]["stratum_exhausted"] += 1
                    exhausted_unfilled[stratum] += remaining[stratum]
                    del remaining[stratum]
                step += 1

            unfilled = Counter({stratum: count for stratum, count in remaining.items() if count > 0})
            unfilled.update(exhausted_unfilled)
            stats["unfilled_allocation"][split][subset] = unfilled
            if stats["selected_counts"][split][subset] < args.eval_directions_per_subset:
                print(
                    f"[{utc_now()}] WARNING: underfilled {split}/{subset}: "
                    f"selected={stats['selected_counts'][split][subset]:,} "
                    f"target={args.eval_directions_per_subset:,} "
                    f"unfilled={sum(unfilled.values()):,}",
                    file=sys.stderr,
                    flush=True,
                )

    stats["selected_counts"] = {
        split: dict(sorted(counter.items())) for split, counter in stats["selected_counts"].items()
    }
    stats["selected_strata"] = {
        split: {
            subset: serializable_counter(counter, args.pair_metadata_columns)
            for subset, counter in subset_counters.items()
        }
        for split, subset_counters in stats["selected_strata"].items()
    }
    stats["unfilled_allocation"] = {
        split: {
            subset: serializable_counter(counter, args.pair_metadata_columns)
            for subset, counter in subset_counters.items()
        }
        for split, subset_counters in stats["unfilled_allocation"].items()
    }
    stats["selected_pairs"] = len(selected_pairs)
    stats["selected_directions"] = len(selected)
    stats["no_train_molecules"] = len(no_train_molecules)
    stats["train_required_molecules"] = len(train_required_molecules)
    stats["eval_molecules"] = len(eval_molecules)
    stats["eval_molecules_by_split"] = {
        split: len(molecules) for split, molecules in sorted(eval_molecules_by_split.items())
    }
    stats["skipped"] = dict(sorted(stats["skipped"].items()))
    return selected, stats


def choose_stratum(
    remaining: Counter[tuple[Any, ...]],
    allocation: dict[tuple[Any, ...], int],
    rows_by_stratum: dict[tuple[Any, ...], list[dict[str, Any]]],
    pointers: dict[tuple[Any, ...], int],
    seed: int,
    split: str,
    subset: str,
    step: int,
) -> tuple[Any, ...] | None:
    best: tuple[float, int, int, int] | None = None
    best_stratum: tuple[Any, ...] | None = None
    for stratum, need in remaining.items():
        if need <= 0:
            continue
        selectable = len(rows_by_stratum.get(stratum) or []) - pointers.get(stratum, 0)
        if selectable <= 0:
            continue
        quota = max(1, allocation.get(stratum, 1))
        score = (
            need / quota,
            need,
            selectable,
            -stable_priority(seed, split, subset, "stratum", step, repr(stratum)),
        )
        if best is None or score > best:
            best = score
            best_stratum = stratum
    return best_stratum


def eval_candidate_degrees(
    pools: dict[str, dict[str, dict[tuple[Any, ...], list[dict[str, Any]]]]],
    molecule_index: MoleculeIndex,
) -> Counter[int]:
    degree: Counter[int] = Counter()
    seen_pairs: set[tuple[int, int]] = set()
    for subset_rows in pools.values():
        for stratum_rows in subset_rows.values():
            for rows in stratum_rows.values():
                for row in rows:
                    pair = pair_key(row["row_index_a"], row["row_index_b"])
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    left, right = molecule_index.pair_molecules(pair)
                    if left == right:
                        degree[left] += 1
                    else:
                        degree.update((left, right))
    return degree


def order_eval_pools(
    pools: dict[str, dict[str, dict[tuple[Any, ...], list[dict[str, Any]]]]],
    *,
    molecule_index: MoleculeIndex,
    molecule_degree: Counter[int],
    preferred_no_train: set[int],
    seed: int,
) -> None:
    def degree_sum(row: dict[str, Any]) -> int:
        left, right = molecule_index.pair_molecules(pair_key(row["row_index_a"], row["row_index_b"]))
        return molecule_degree[left] + molecule_degree[right]

    def preferred_count(row: dict[str, Any]) -> int:
        left, right = molecule_index.pair_molecules(pair_key(row["row_index_a"], row["row_index_b"]))
        return int(left in preferred_no_train) + int(right in preferred_no_train)

    def a_seen_target_preferred(row: dict[str, Any], split: str) -> int:
        pair = pair_key(row["row_index_a"], row["row_index_b"])
        source_record, target_record = direction_endpoints(pair, selected_direction(seed, split, "a_seen_only", pair))
        source = molecule_index.molecule_id(source_record)
        target = molecule_index.molecule_id(target_record)
        return int(target in preferred_no_train) - int(source in preferred_no_train)

    for split, subset_rows in pools.items():
        for subset, stratum_rows in subset_rows.items():
            if subset == "no_overlap":
                for rows in stratum_rows.values():
                    rows.sort(
                        key=lambda row: (
                            -preferred_count(row),
                            -degree_sum(row),
                            row_priority(seed, split, subset, row),
                        )
                    )
            elif subset == "a_seen_only":
                for rows in stratum_rows.values():
                    rows.sort(
                        key=lambda row: (
                            -a_seen_target_preferred(row, split),
                            -degree_sum(row),
                            row_priority(seed, split, subset, row),
                        )
                    )
            else:
                for rows in stratum_rows.values():
                    rows.sort(
                        key=lambda row: (
                            preferred_count(row),
                            -degree_sum(row),
                            row_priority(seed, split, subset, row),
                        )
                    )


def selected_maps(
    selected_rows: list[dict[str, Any]],
    molecule_index: MoleculeIndex,
) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], set[tuple[int, int]], set[int]]:
    by_pair: dict[tuple[int, int], list[dict[str, Any]]] = {}
    selected_pairs: set[tuple[int, int]] = set()
    no_train_molecules: set[int] = set()
    for row in selected_rows:
        pair = pair_key(row["row_index_a"], row["row_index_b"])
        by_pair.setdefault(pair, []).append(row)
        selected_pairs.add(pair)
        source_record, target_record = direction_endpoints(pair, row["direction"])
        source = molecule_index.molecule_id(source_record)
        target = molecule_index.molecule_id(target_record)
        if row["eval_subset"] == "no_overlap":
            no_train_molecules.update((source, target))
        elif row["eval_subset"] == "a_seen_only":
            no_train_molecules.add(target)
    return by_pair, selected_pairs, no_train_molecules


def write_universe(
    *,
    universe: str,
    pair_dir: Path,
    output_dir: Path,
    public_output_dir: Path | None = None,
    selected_rows: list[dict[str, Any]],
    base_table: pa.Table,
    molecule_index: MoleculeIndex,
    args: argparse.Namespace,
) -> dict[str, Any]:
    schema = output_schema(args.metadata_columns)
    canonical_smiles_array = pa.array(molecule_index.record_canonical_smiles, type=pa.large_string())
    writers = {
        split: RollingParquetWriter(
            output_dir / split,
            schema,
            file_row_limit=args.parquet_file_row_limit,
            compression=args.parquet_compression,
        )
        for split in SPLITS
    }
    selected_by_pair, selected_pairs, no_train_molecules = selected_maps(selected_rows, molecule_index)
    selected_pair_rows_found: Counter[tuple[int, int]] = Counter()
    input_path = pair_dir_records(pair_dir)
    total_rows = parquet_row_count(input_path)
    dataset = ds.dataset(input_path, format="parquet")
    columns = [field.name for field in compact_schema(args.pair_metadata_columns)]
    source_meta_path = pair_dir / "metadata.json"
    source_meta = json.loads(source_meta_path.read_text()) if source_meta_path.exists() else {}
    thresholds = source_meta.get("thresholds") or {}
    transfer_threshold = float(thresholds.get("transfer", args.transfer_threshold))
    not_transfer_threshold = float(thresholds.get("not_transfer", args.not_transfer_threshold))
    source_schema_version = str(source_meta.get("schema_version") or "generic_transfer_pairs_compact_parquet_v1")
    train_molecules: set[int] = set()
    rows_by_split: Counter[str] = Counter()
    eval_subset_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    direction_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    label_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    thrown_out: Counter[str] = Counter()
    train_directions = (
        ("a_to_b",)
        if args.train_direction_mode == "unidirectional"
        else ("a_to_b", "b_to_a")
    )

    progress = ProgressLogger(f"write {universe} shared full splits", total_rows, args.progress_every_seconds)
    processed = 0
    no_train_set = set(no_train_molecules)
    selected_set = set(selected_pairs)
    no_train_array = (
        np.fromiter(no_train_set, dtype=np.uint32)
        if no_train_set
        else np.array([], dtype=np.uint32)
    )
    selected_key_array = (
        np.fromiter(
            ((int(left) << 32) | int(right) for left, right in selected_set),
            dtype=np.uint64,
        )
        if selected_set
        else np.array([], dtype=np.uint64)
    )

    try:
        for batch in dataset.to_batches(columns=columns, batch_size=args.batch_size):
            table = pa.Table.from_batches([batch])
            eval_rows_by_direction: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
            left_array = batch.column("row_index_a").to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
            right_array = batch.column("row_index_b").to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
            left_molecule_array = molecule_index.record_to_molecule[left_array]
            right_molecule_array = molecule_index.record_to_molecule[right_array]
            pair_left = np.minimum(left_array, right_array).astype(np.uint64, copy=False)
            pair_right = np.maximum(left_array, right_array).astype(np.uint64, copy=False)
            pair_keys = (pair_left << np.uint64(32)) | pair_right
            selected_mask = (
                np.isin(pair_keys, selected_key_array, assume_unique=False)
                if selected_key_array.size
                else np.zeros(batch.num_rows, dtype=bool)
            )
            if no_train_array.size:
                touches_no_train = np.isin(left_molecule_array, no_train_array, assume_unique=False) | np.isin(
                    right_molecule_array,
                    no_train_array,
                    assume_unique=False,
                )
            else:
                touches_no_train = np.zeros(batch.num_rows, dtype=bool)
            no_train_policy_mask = ~selected_mask & touches_no_train
            train_mask = ~(selected_mask | touches_no_train)
            thrown_out["no_train_molecule_policy"] += int(no_train_policy_mask.sum())

            selected_positions = np.flatnonzero(selected_mask)
            if selected_positions.size:
                selected_table = table.take(pa.array(selected_positions, type=pa.int64()))
                selected_rows_in_batch = selected_table.to_pylist()
            else:
                selected_rows_in_batch = []

            for row in selected_rows_in_batch:
                pair = pair_key(row["row_index_a"], row["row_index_b"])
                if pair in selected_by_pair:
                    selected_pair_rows_found[pair] += 1
                    for selected in selected_by_pair[pair]:
                        eval_row = dict(row)
                        eval_rows_by_direction.setdefault(
                            (selected["split"], selected["eval_subset"], selected["direction"]),
                            [],
                        ).append(eval_row)

            if bool(train_mask.any()) and "train" in args.splits:
                train_table = table.filter(pa.array(train_mask))
                for direction in train_directions:
                    full = full_table_from_compact(
                        train_table,
                        base_table=base_table,
                        canonical_smiles_array=canonical_smiles_array,
                        split="train",
                        direction=direction,
                        eval_subset=None,
                        metadata_columns=args.metadata_columns,
                        smiles_column=args.smiles_column,
                        value_column=args.value_column,
                        source_schema_version=source_schema_version,
                        transfer_threshold=transfer_threshold,
                        not_transfer_threshold=not_transfer_threshold,
                    )
                    writers["train"].write_table(full)
                    rows_by_split["train"] += full.num_rows
                    eval_subset_counts["train"]["none"] += full.num_rows
                    direction_counts["train"][direction] += full.num_rows
                    label_counts["train"].update(full["transfer_label"].to_pylist())
                train_molecules.update(int(value) for value in left_molecule_array[train_mask])
                train_molecules.update(int(value) for value in right_molecule_array[train_mask])

            for (split, subset, direction), eval_rows in eval_rows_by_direction.items():
                if split not in args.splits:
                    continue
                eval_table = pa.Table.from_pylist(eval_rows, schema=compact_schema(args.pair_metadata_columns))
                full = full_table_from_compact(
                    eval_table,
                    base_table=base_table,
                    canonical_smiles_array=canonical_smiles_array,
                    split=split,
                    direction=direction,
                    eval_subset=subset,
                    metadata_columns=args.metadata_columns,
                    smiles_column=args.smiles_column,
                    value_column=args.value_column,
                    source_schema_version=source_schema_version,
                    transfer_threshold=transfer_threshold,
                    not_transfer_threshold=not_transfer_threshold,
                )
                writers[split].write_table(full)
                rows_by_split[split] += full.num_rows
                eval_subset_counts[split][subset] += full.num_rows
                direction_counts[split][direction] += full.num_rows
                label_counts[split].update(full["transfer_label"].to_pylist())

            processed += batch.num_rows
            progress.update(
                processed,
                extra=(
                    f"train={rows_by_split['train']:,} "
                    f"validation={rows_by_split['validation']:,} "
                    f"test={rows_by_split['test']:,}"
                ),
            )
    finally:
        for writer in writers.values():
            writer.close()
    progress.finish(
        processed,
        extra=(
            f"train={rows_by_split['train']:,} "
            f"validation={rows_by_split['validation']:,} "
            f"test={rows_by_split['test']:,}"
        ),
    )

    missing_pairs = sorted(pair for pair in selected_pairs if selected_pair_rows_found[pair] == 0)
    validation_errors: list[str] = []
    if missing_pairs:
        validation_errors.append(f"missing {len(missing_pairs)} selected eval pairs in {universe}")
    for row in selected_rows:
        pair = pair_key(row["row_index_a"], row["row_index_b"])
        source_record, target_record = direction_endpoints(pair, row["direction"])
        source = molecule_index.molecule_id(source_record)
        target = molecule_index.molecule_id(target_record)
        subset = row["eval_subset"]
        source_seen = source in train_molecules
        target_seen = target in train_molecules
        if subset == "no_overlap" and (source_seen or target_seen):
            validation_errors.append(f"no_overlap seen molecule in {universe}: {source}->{target}")
        elif subset == "a_seen_only" and (not source_seen or target_seen):
            validation_errors.append(f"a_seen_only status failure in {universe}: {source}->{target}")
        elif subset == "both_seen" and (not source_seen or not target_seen):
            validation_errors.append(f"both_seen status failure in {universe}: {source}->{target}")
        if len(validation_errors) >= 100:
            break

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "base_input": args.base_input,
        "input_pair_dir": str(pair_dir),
        "output_dir": str(public_output_dir or output_dir),
        "split_version": SPLIT_VERSION,
        "train_direction_mode": args.train_direction_mode,
        "train_directions": list(train_directions),
        "universe": universe,
        "splits": list(args.splits),
        "metadata_columns": args.metadata_columns,
        "rows_by_split": dict(sorted(rows_by_split.items())),
        "eval_subset_counts": {
            split: dict(sorted(eval_subset_counts[split].items())) for split in SPLITS
        },
        "direction_counts": {
            split: dict(sorted(direction_counts[split].items())) for split in SPLITS
        },
        "transfer_label_counts": {
            split: dict(sorted(label_counts[split].items())) for split in SPLITS
        },
        "train_molecules": len(train_molecules),
        "molecule_identity": molecule_index.stats,
        "selected_eval_pairs_found": int(sum(1 for pair in selected_pairs if selected_pair_rows_found[pair] > 0)),
        "selected_eval_pairs_expected": len(selected_pairs),
        "thrown_out_pairs": dict(sorted(thrown_out.items())),
        "validation": {
            "n_errors": len(validation_errors),
            "errors": validation_errors,
            "seen_status_policy": {
                "no_overlap": "source and target RDKit-canonical molecule groups absent from train",
                "a_seen_only": "source RDKit-canonical molecule group present in train and target group absent from train",
                "both_seen": "source and target RDKit-canonical molecule groups present in train",
            },
        },
        "source_pair_metadata": {
            "schema_version": source_meta.get("schema_version"),
            "pairs_written": source_meta.get("pairs_written"),
            "candidate_pairs_seen": source_meta.get("candidate_pairs_seen"),
            "same_metadata_columns": source_meta.get("same_metadata_columns"),
        },
        "parquet": {
            "compression": args.parquet_compression,
            "file_row_limit": args.parquet_file_row_limit,
            "batch_size": args.batch_size,
            "arrow_cpu_count": args.arrow_cpu_count,
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def mode_metadata_stem(train_direction_mode: str) -> str:
    if train_direction_mode == "unidirectional":
        return "oral_bioavailability_shared_eval_unidirectional"
    return "oral_bioavailability_shared_eval"


def write_eval_selection(
    selected_rows: list[dict[str, Any]],
    output_root: Path,
    stats: dict[str, Any],
    train_direction_mode: str,
) -> None:
    payload = {
        "schema_version": f"{SPLIT_VERSION}_selection",
        "created_at_utc": utc_now(),
        "selection_stats": stats,
        "rows": [
            {
                "row_index_a": int(row["row_index_a"]),
                "row_index_b": int(row["row_index_b"]),
                "molecule_id_a": int(row["molecule_id_a"]),
                "molecule_id_b": int(row["molecule_id_b"]),
                "split": row["split"],
                "eval_subset": row["eval_subset"],
                "direction": row["direction"],
                "transfer_label": int(row["transfer_label"]),
                "value_difference": float(row["value_difference"]),
                "weighted_tanimoto": float(row["weighted_tanimoto"]),
                "similarity_bucket": int(row["similarity_bucket"]),
                "stratum": stratum_to_string(stratum_for_row(row, stats["selection_policy"]["stratification_fields"][2:]), stats["selection_policy"]["stratification_fields"][2:]),
            }
            for row in selected_rows
        ],
    }
    write_json(output_root / f"{mode_metadata_stem(train_direction_mode)}_selection.json", payload)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.arrow_cpu_count:
        pa.set_cpu_count(args.arrow_cpu_count)
        pa.set_io_thread_count(args.arrow_cpu_count)
    universes = pair_universes(args)
    outputs = {
        key: value
        for key, value in split_output_dirs(args.output_root, args.train_direction_mode, args.output_name_suffix).items()
        if key in universes
    }
    prepare_outputs(outputs, args.overwrite, args.splits)
    base_table = load_base_table(args)
    molecule_index = build_molecule_index(base_table, args.smiles_column)
    selected_rows, selection_stats = select_eval_rows(args, molecule_index)

    universe_metadata: dict[str, Any] = {}
    staged_outputs = make_staged_output_dirs(outputs)
    installed = False
    try:
        for universe, pair_dir in universes.items():
            universe_metadata[universe] = write_universe(
                universe=universe,
                pair_dir=pair_dir,
                output_dir=staged_outputs[universe],
                public_output_dir=outputs[universe],
                selected_rows=selected_rows,
                base_table=base_table,
                molecule_index=molecule_index,
                args=args,
            )
        install_staged_output_dirs(staged_outputs, outputs, args.overwrite)
        installed = True
    finally:
        if not installed:
            cleanup_staged_output_dirs(staged_outputs.values())

    write_eval_selection(selected_rows, args.output_root, selection_stats, args.train_direction_mode)

    metadata = {
        "schema_version": SPLIT_VERSION,
        "created_at_utc": utc_now(),
        "base_input": args.base_input,
        "condition_key_pair_dir": str(args.condition_key_pairs),
        "same_species_pair_dir": str(args.same_species_pairs),
        "no_constraints_pair_dir": str(args.no_constraints_pairs),
        "train_direction_mode": args.train_direction_mode,
        "shared_eval_compatibility_column": args.shared_eval_compatibility_column,
        "molecule_identity": molecule_index.stats,
        "shared_eval_compatibility_policy": (
            "validation/test candidates are selected from condition-key pairs only when both rows have "
            "identical non-null values in this column, guaranteeing presence in same_species_v2; "
            "no_constraints compatibility is automatic"
        ),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "selection": selection_stats,
        "universe_rows_by_split": {
            key: value["rows_by_split"] for key, value in universe_metadata.items()
        },
        "universe_validation": {
            key: value["validation"] for key, value in universe_metadata.items()
        },
    }
    write_json(args.output_root / f"{mode_metadata_stem(args.train_direction_mode)}_metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", default=DEFAULT_BASE_INPUT)
    parser.add_argument("--condition-key-pairs", type=Path, default=DEFAULT_CONDITION_KEY_PAIRS)
    parser.add_argument("--same-species-pairs", type=Path, default=DEFAULT_SAME_SPECIES_PAIRS)
    parser.add_argument("--no-constraints-pairs", type=Path, default=DEFAULT_NO_CONSTRAINTS_PAIRS)
    parser.add_argument(
        "--universes",
        nargs="+",
        choices=("condition_key", "same_species_v2", "no_constraints"),
        default=["condition_key", "same_species_v2", "no_constraints"],
        help="Pair universes to materialize. Selection is still based on condition_key.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-name-suffix", default="")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument(
        "--train-direction-mode",
        choices=TRAIN_DIRECTION_MODES,
        default="bidirectional",
        help="Whether train writes both pair directions or one deterministic direction per source_pair_id.",
    )
    parser.add_argument("--metadata-columns", nargs="+", default=DEFAULT_METADATA_COLUMNS)
    parser.add_argument("--shared-eval-compatibility-column", default=DEFAULT_SHARED_COMPATIBILITY_COLUMN)
    parser.add_argument(
        "--pair-metadata-columns",
        nargs="+",
        default=[
            "bioavailability_report_type",
            "species_or_population",
            "dose",
            "oral_exposure_mode",
            "qualifying_conditions",
            "comparator",
            "extra_details",
        ],
    )
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--value-column", default="oral_bioavailability_value")
    parser.add_argument("--eval-directions-per-subset", type=int, default=10_000)
    parser.add_argument("--candidate-pool-multiplier", type=int, default=25)
    parser.add_argument("--preferred-no-train-molecules", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--parquet-file-row-limit", type=int, default=10_000_000)
    parser.add_argument("--progress-every-seconds", type=float, default=300.0)
    parser.add_argument("--arrow-cpu-count", type=int, default=None)
    parser.add_argument(
        "--disjoint-eval-molecules",
        action="store_true",
        help="Require validation and test eval pairs to use disjoint RDKit-canonical molecule groups.",
    )
    parser.add_argument("--transfer-threshold", type=float, default=10.0)
    parser.add_argument("--not-transfer-threshold", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.eval_directions_per_subset < 1:
        parser.error("--eval-directions-per-subset must be positive")
    if args.candidate_pool_multiplier < 1:
        parser.error("--candidate-pool-multiplier must be positive")
    if args.preferred_no_train_molecules < 1:
        parser.error("--preferred-no-train-molecules must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.parquet_file_row_limit < 1:
        parser.error("--parquet-file-row-limit must be positive")
    if args.progress_every_seconds < 0:
        parser.error("--progress-every-seconds cannot be negative")
    if args.arrow_cpu_count is not None and args.arrow_cpu_count < 1:
        parser.error("--arrow-cpu-count must be positive")
    return args


def main() -> None:
    metadata = build(parse_args())
    print(
        json.dumps(
            {
                "outputs": metadata["outputs"],
                "selection": metadata["selection"],
                "universe_rows_by_split": metadata["universe_rows_by_split"],
                "universe_validation": metadata["universe_validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
