#!/usr/bin/env python3
"""Shared helpers for generic molecular transfer dataset scripts."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq


NULL_TOKEN = "__NULL__"
SPLITS = ("train", "validation", "test")
EVAL_SPLITS = ("validation", "test")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_json(row: dict[str, Any]) -> str:
    return json.dumps(row, separators=(",", ":"), sort_keys=False)


def read_jsonl_gz(path: Path, max_rows: int | None = None) -> Iterator[dict[str, Any]]:
    count = 0
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if max_rows is not None and count >= max_rows:
                return


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def json_default(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Counter):
        return dict(sorted(value.items()))
    return str(value)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def compact_float(value: Any) -> float | int | None:
    out = finite_float(value)
    if out is None:
        return None
    return int(out) if out.is_integer() else out


def stable_priority(seed: int, *parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return seed ^ int.from_bytes(digest, "big")


def stable_hash_text(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()


def normalize_stratum_value(value: Any) -> str:
    if value is None:
        return NULL_TOKEN
    text = str(value).strip()
    return text if text else NULL_TOKEN


def compute_quantile_thresholds(values: list[float], n_buckets: int) -> list[float]:
    if n_buckets < 2:
        raise ValueError("n_buckets must be >= 2")
    values = sorted(values)
    if not values:
        raise ValueError("cannot compute quantiles from empty values")
    thresholds: list[float] = []
    for index in range(1, n_buckets):
        pos = (len(values) - 1) * (index / n_buckets)
        lower = math.floor(pos)
        upper = math.ceil(pos)
        if lower == upper:
            thresholds.append(values[lower])
        else:
            frac = pos - lower
            thresholds.append(values[lower] * (1.0 - frac) + values[upper] * frac)
    return thresholds


def bucket_for_value(value: float, thresholds: list[float]) -> int:
    bucket = 0
    for threshold in thresholds:
        if value > threshold:
            bucket += 1
        else:
            break
    return bucket


def largest_remainder_allocation(total: int, counts: Counter[Any]) -> dict[Any, int]:
    if total <= 0 or not counts:
        return {}
    available = sum(counts.values())
    if available <= 0:
        return {}
    target = min(total, available)
    raw = {key: target * (count / available) for key, count in counts.items()}
    allocation = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = target - sum(allocation.values())
    order = sorted(
        ((raw[key] - allocation[key], counts[key], repr(key), key) for key in counts),
        key=lambda item: (-item[0], -item[1], item[2]),
    )
    for _rem, _count, _repr_key, key in order:
        if remaining <= 0:
            break
        allocation[key] += 1
        remaining -= 1
    return {key: value for key, value in allocation.items() if value > 0}


def prepare_output_dir(output_dir: Path, expected_files: Iterable[str], overwrite: bool) -> None:
    expected = [output_dir / name for name in expected_files]
    present = [path for path in expected if path.exists()]
    if present and not overwrite:
        formatted = "\n".join(str(path) for path in present)
        raise FileExistsError(f"output files exist; pass --overwrite:\n{formatted}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_dir.exists():
        for path in expected:
            if path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


@contextmanager
def atomic_output_dir(output_dir: Path) -> Iterator[Path]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as tmp:
        yield Path(tmp)


def parquet_files_from_input(input_ref: str, *, repo_type: str = "dataset") -> list[Path]:
    path = Path(input_ref)
    if path.exists():
        if path.is_dir():
            files = sorted(path.glob("*.parquet")) + sorted((path / "data").glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"no parquet files found under {path}")
            return files
        return [path]

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(input_ref) if repo_type == "dataset" else api.repo_info(input_ref)
    candidates = [
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.endswith(".parquet")
    ]
    if not candidates:
        raise FileNotFoundError(f"no parquet files found in HF repo {input_ref}")
    return [Path(hf_hub_download(input_ref, name, repo_type=repo_type)) for name in sorted(candidates)]


def iter_parquet_rows(
    input_ref: str,
    *,
    columns: list[str] | None = None,
    batch_size: int = 8192,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    count = 0
    for path in parquet_files_from_input(input_ref):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                yield row
                count += 1
                if max_rows is not None and count >= max_rows:
                    return


def write_parquet_pylist(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    schema: pa.Schema | None = None,
    compression: str = "zstd",
) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression=compression)


class FingerprintCache:
    def __init__(self, radius: int = 2, nbits: int = 2048) -> None:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator

        RDLogger.DisableLog("rdApp.warning")
        inv = rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership=True)
        feat_inv = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        self.chem = Chem
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=nbits,
            atomInvariantsGenerator=inv,
        )
        self.feature_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=nbits,
            atomInvariantsGenerator=feat_inv,
        )
        self.cache: dict[str, tuple[Any, Any] | None] = {}

    def get(self, smiles: str) -> tuple[Any, Any] | None:
        if smiles in self.cache:
            return self.cache[smiles]
        mol = self.chem.MolFromSmiles(smiles)
        if mol is None:
            self.cache[smiles] = None
            return None
        self.cache[smiles] = (
            self.morgan_gen.GetFingerprint(mol),
            self.feature_gen.GetFingerprint(mol),
        )
        return self.cache[smiles]

    def similarity(self, left_fp: tuple[Any, Any], right_fp: tuple[Any, Any]) -> float:
        from rdkit import DataStructs

        morgan = DataStructs.TanimotoSimilarity(left_fp[0], right_fp[0])
        feature = DataStructs.TanimotoSimilarity(left_fp[1], right_fp[1])
        return 0.8 * float(morgan) + 0.2 * float(feature)


def upload_folder_to_hf(
    *,
    folder_path: Path,
    repo_id: str,
    private: bool,
    path_in_repo: str | None,
    commit_message: str,
) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    return api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )


def replace_dir_from_tmp(tmp: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in tmp.iterdir():
        os.replace(src, output_dir / src.name)
