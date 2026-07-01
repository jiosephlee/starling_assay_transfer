"""Record-native data loading for Starling KNN evaluation."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CANONICAL_QUERY_SPLITS = ("validation_1", "validation_2", "test")
SOURCE_SPLIT = "train"
FULL_METADATA_CONFIG = "full_metadata"
RECORD_SCHEMA_VERSION = "condition_key_v3_record_splits_hf_full_metadata_v1"
REQUIRED_RECORD_COLUMNS = ("row_index", "smiles", "Y", "oral_bioavailability_value")
MISSING_FIELDS = (
    "support_text",
    "molecule_name",
    "bioavailability_report_type",
    "species_or_population",
    "dose",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
    "extra_details",
)
MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "unknown", "not reported"}


@dataclass(frozen=True)
class RecordKnnDataset:
    dataset_dir: Path
    config_name: str
    split: str
    sources: pd.DataFrame
    queries: pd.DataFrame
    manifest_hash: str
    source_key_hash: str
    query_key_hash: str


def read_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open() as fh:
        return yaml.safe_load(fh) or {}


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    out = Path(path)
    if out.is_absolute():
        return out
    return Path(root or Path.cwd()) / out


def normalize_record_split(split: str) -> str:
    text = str(split)
    aliases = {f"val{chr(95)}1": "validation_1", f"val{chr(95)}2": "validation_2"}
    aliases["valid"] = "validation_1"
    normalized = aliases.get(text, text)
    allowed = (*CANONICAL_QUERY_SPLITS, SOURCE_SPLIT)
    if normalized not in allowed:
        raise ValueError(f"unknown record KNN split {split!r}; expected one of {allowed}")
    return normalized


def normalize_record_splits(splits: list[str] | tuple[str, ...] | str) -> list[str]:
    if isinstance(splits, str):
        values = [splits]
    else:
        values = list(splits)
    return [normalize_record_split(value) for value in values]


def record_split_path(dataset_dir: str | Path, split: str, config_name: str = FULL_METADATA_CONFIG) -> Path:
    root = Path(dataset_dir)
    canonical = normalize_record_split(split)
    candidates = (
        root / "data" / config_name / canonical / "part-00000.parquet",
        root / "data" / config_name / canonical / f"{canonical}.parquet",
        root / canonical / "part-00000.parquet",
        root / f"{canonical}.parquet",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no {config_name} record split {canonical!r} under {root}")


def dataset_manifest_path(dataset_dir: str | Path) -> Path:
    root = Path(dataset_dir)
    for path in (root / "metadata" / "split_manifest.json", root / "manifest.json"):
        if path.exists():
            return path
    raise FileNotFoundError(f"no dataset manifest under {root}")


def canonicalize_smiles(values: list[Any] | pd.Series) -> list[str | None]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.warning")
    out: list[str | None] = []
    for value in values:
        text = "" if value is None else str(value).strip()
        mol = Chem.MolFromSmiles(text) if text else None
        out.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None)
    return out


def load_record_dataset(
    dataset_dir: str | Path,
    split: str,
    *,
    config_name: str = FULL_METADATA_CONFIG,
    max_queries: int = 0,
    check_disjoint: bool = True,
) -> RecordKnnDataset:
    root = Path(dataset_dir)
    canonical = normalize_record_split(split)
    manifest_path = dataset_manifest_path(root)
    sources = load_record_sources(root, config_name=config_name)
    queries = load_record_queries(root, canonical, config_name=config_name, max_queries=max_queries)
    if check_disjoint:
        assert_source_query_disjoint(sources, queries, canonical)
    return RecordKnnDataset(
        dataset_dir=root,
        config_name=config_name,
        split=canonical,
        sources=sources,
        queries=queries,
        manifest_hash=sha256_file(manifest_path),
        source_key_hash=record_key_hash(sources, ["record_key", "canonical_smiles", "label"]),
        query_key_hash=record_key_hash(queries, ["record_key", "canonical_smiles", "label"]),
    )


def load_record_sources(dataset_dir: str | Path, config_name: str = FULL_METADATA_CONFIG) -> pd.DataFrame:
    rows = pd.read_parquet(record_split_path(dataset_dir, SOURCE_SPLIT, config_name))
    return normalize_record_frame(rows, role="source")


def load_record_queries(
    dataset_dir: str | Path,
    split: str,
    *,
    config_name: str = FULL_METADATA_CONFIG,
    max_queries: int = 0,
) -> pd.DataFrame:
    rows = pd.read_parquet(record_split_path(dataset_dir, split, config_name))
    rows = rows.head(int(max_queries)).copy() if max_queries and max_queries > 0 else rows
    return normalize_record_frame(rows, role="query")


def normalize_record_frame(rows: pd.DataFrame, *, role: str) -> pd.DataFrame:
    validate_record_schema(rows, role)
    out = annotate_record_rows(rows).reset_index(drop=True)
    out["record_key"] = out["row_index"].astype(str)
    out["canonical_smiles"] = canonicalize_smiles(out["smiles"])
    out["label"] = out["Y"].astype(np.int8)
    out[f"{role}_position"] = np.arange(len(out), dtype=np.int64)
    out[f"{role}_row_index"] = out["row_index"].astype(np.int64)
    return out


def validate_record_schema(rows: pd.DataFrame, role: str) -> None:
    missing = [col for col in REQUIRED_RECORD_COLUMNS if col not in rows.columns]
    if missing:
        raise ValueError(f"record KNN {role} rows missing required columns: {missing}")
    if rows["row_index"].isna().any():
        raise ValueError(f"record KNN {role} rows contain null row_index")
    if rows["smiles"].isna().any():
        raise ValueError(f"record KNN {role} rows contain null smiles")
    bad_labels = ~rows["Y"].isin([0, 1, False, True])
    if bad_labels.any():
        raise ValueError(f"record KNN {role} rows contain non-binary Y values")


def annotate_record_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["ob_bin"] = np.where(out["oral_bioavailability_value"] < 20.0, "ob_lt20", "ob_ge20")
    masks = [missing_mask(row) for _, row in out.iterrows()]
    out["metadata_missing_mask"] = masks
    out["missing_count"] = [mask.count("1") for mask in masks]
    out["missing_count_bucket"] = [missing_count_bucket(v) for v in out["missing_count"]]
    return out


def missing_mask(row: pd.Series) -> str:
    return "".join("1" if is_missing(row.get(field)) else "0" for field in MISSING_FIELDS)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in MISSING_TOKENS


def missing_count_bucket(value: int) -> str:
    if value <= 2:
        return "missing_0_2"
    if value <= 5:
        return "missing_3_5"
    return "missing_6_9"


def assert_source_query_disjoint(sources: pd.DataFrame, queries: pd.DataFrame, split: str) -> None:
    source_keys = set(sources["record_key"].astype(str))
    query_keys = set(queries["record_key"].astype(str))
    record_overlap = source_keys & query_keys
    if record_overlap:
        raise ValueError(f"{split} overlaps train by record_key; first={sorted(record_overlap)[:5]}")
    source_smiles = set(sources["canonical_smiles"].dropna().astype(str))
    query_smiles = set(queries["canonical_smiles"].dropna().astype(str))
    smiles_overlap = source_smiles & query_smiles
    if smiles_overlap:
        raise ValueError(f"{split} overlaps train by canonical_smiles; n={len(smiles_overlap)}")


def validate_smiles_only_alignment(dataset_dir: str | Path) -> None:
    root = Path(dataset_dir)
    for split in (SOURCE_SPLIT, *CANONICAL_QUERY_SPLITS):
        full = load_record_queries(root, split) if split != SOURCE_SPLIT else load_record_sources(root)
        smiles = pd.read_parquet(record_split_path(root, split, "smiles_only"))
        validate_membership_alignment(full, normalize_record_frame(smiles, role="query"), split)


def validate_membership_alignment(full: pd.DataFrame, smiles: pd.DataFrame, split: str) -> None:
    full_keys = set(full["record_key"].astype(str))
    smiles_keys = set(smiles["record_key"].astype(str))
    if full_keys != smiles_keys:
        raise ValueError(f"{split} full_metadata/smiles_only row membership mismatch")


def record_key_hash(rows: pd.DataFrame, columns: list[str]) -> str:
    h = hashlib.sha256()
    for row in rows[columns].itertuples(index=False, name=None):
        h.update("\t".join(str(v) for v in row).encode())
        h.update(b"\n")
    return h.hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
