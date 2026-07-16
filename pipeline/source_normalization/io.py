"""Pinned input validation and deterministic artifact I/O."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from pipeline.source_normalization.structures import collapse_mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("version") != "source_normalization_v1":
        raise ValueError(f"unsupported config version: {config.get('version')!r}")
    return config


def resolve_path(data_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else data_root / path


def artifact_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def parquet_schema(path: Path) -> dict[str, str]:
    schema = pq.ParquetFile(path).schema_arrow
    return {field.name: str(field.type) for field in schema}


def csv_header_and_rows(path: Path) -> tuple[list[str], int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return header, rows


def _assert_metadata(pin: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    for field in ("sha256", "size", "mtime_ns"):
        if actual[field] != pin[field]:
            raise ValueError(f"artifact {field} mismatch for {actual['path']}")


def verify_artifact(pin: Mapping[str, Any], data_root: Path) -> dict[str, Any]:
    path = resolve_path(data_root, pin["path"])
    actual = artifact_metadata(path)
    _assert_metadata(pin, actual)
    if pin["format"] == "csv":
        header, rows = csv_header_and_rows(path)
        if header != pin["expected_columns"]:
            raise ValueError(f"CSV schema mismatch for {path}")
    else:
        rows = pq.ParquetFile(path).metadata.num_rows
        if parquet_schema(path) != pin["schema"]:
            raise ValueError(f"Parquet schema mismatch for {path}")
    if rows != pin["rows"]:
        raise ValueError(f"row-count mismatch for {path}: {rows} != {pin['rows']}")
    return {**actual, "rows": rows, "schema": pin.get("schema", pin.get("expected_columns"))}


def read_source(pin: Mapping[str, Any], data_root: Path) -> pd.DataFrame:
    path = resolve_path(data_root, pin["path"])
    if pin["format"] == "csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    return pq.read_table(path).to_pandas()


def required_identifiers(config: Mapping[str, Any], data_root: Path) -> set[str]:
    identifiers: set[str] = set()
    for spec in config["sources"].values():
        if spec["structure_mode"] != "mapped":
            continue
        path = resolve_path(data_root, spec["path"])
        frame = pd.read_csv(path, usecols=["global_identifier"], dtype=str, keep_default_na=False)
        identifiers.update(value for value in frame["global_identifier"] if value)
    return identifiers


def _mapping_rows(table: pa.Table) -> list[tuple[str, str]]:
    identifiers = table["global_identifier"].to_pylist()
    smiles = table["smiles"].to_pylist()
    return list(zip(identifiers, smiles, strict=True))


def load_mapping(path: Path, identifiers: set[str]) -> tuple[dict[str, str], dict[str, int]]:
    table = pq.read_table(path, columns=["global_identifier", "smiles"])
    unique = pc.count_distinct(table["global_identifier"]).as_py()
    duplicate_rows = len(table) - unique
    if duplicate_rows:
        collapse_mapping(_mapping_rows(table))
    mask = pc.is_in(table["global_identifier"], value_set=pa.array(sorted(identifiers)))
    selected = table.filter(mask)
    mapping, selected_duplicates = collapse_mapping(_mapping_rows(selected))
    stats = {
        "mapping_rows": len(table),
        "unique_identifiers": unique,
        "identical_duplicate_rows": duplicate_rows,
        "selected_duplicate_rows": selected_duplicates,
        "requested_identifiers": len(identifiers),
        "resolved_identifiers": len(mapping),
    }
    return mapping, stats


def load_tdc_exclusions(path: Path) -> tuple[set[str], dict[str, Any]]:
    table = pq.read_table(path)
    raw = {value for value in table["raw_smiles"].to_pylist() if value}
    canonical = {value for value in table["canonical_smiles"].to_pylist() if value}
    splits = sorted(set(table["split"].to_pylist()))
    stats = {"rows": len(table), "raw_unique": len(raw), "canonical_unique": len(canonical), "splits": splits}
    return raw | canonical, stats


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    os.replace(temporary, path)


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)

