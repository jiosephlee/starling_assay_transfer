"""RDKit retrieval and model-independent candidate caches for record KNN."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .knn_data import RECORD_SCHEMA_VERSION, RecordKnnDataset, sha256_json

FINGERPRINT_PARAMS = {
    "morgan_radius": 2,
    "morgan_fp_size": 2048,
    "feature_radius": 2,
    "feature_fp_size": 2048,
}
SIMILARITY_WEIGHTS = {"morgan": 0.8, "feature": 0.2}
EXCLUSION_POLICY = ["same_record_key", "same_canonical_smiles"]
CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RetrievalConfig:
    top_fraction: float = 0.10
    top_n: int | None = None
    max_queries: int = 0


@dataclass(frozen=True)
class CandidateSet:
    positions: np.ndarray
    similarities: np.ndarray
    metadata: dict[str, Any]
    data_path: Path | None = None


def rdkit_version() -> str:
    import rdkit

    return str(rdkit.__version__)


def fingerprints(smiles: list[Any] | pd.Series) -> list[Any]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.warning")
    inv = rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership=True)
    feat = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=inv)
    feature = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=feat)
    return [_fingerprint_one(value, morgan, feature, Chem) for value in smiles]


def _fingerprint_one(value: Any, morgan: Any, feature: Any, chem: Any) -> Any:
    mol = chem.MolFromSmiles("" if value is None else str(value))
    if mol is None:
        return None
    return morgan.GetFingerprint(mol), feature.GetFingerprint(mol)


def weighted_similarity(left: Any, right: Any) -> float:
    from rdkit import DataStructs

    if left is None or right is None:
        return -1.0
    morgan = float(DataStructs.TanimotoSimilarity(left[0], right[0]))
    feature = float(DataStructs.TanimotoSimilarity(left[1], right[1]))
    return SIMILARITY_WEIGHTS["morgan"] * morgan + SIMILARITY_WEIGHTS["feature"] * feature


def candidate_count(n_sources: int, config: RetrievalConfig) -> int:
    if config.top_n is not None and config.top_n > 0:
        return min(int(config.top_n), int(n_sources))
    count = int(np.ceil(int(n_sources) * float(config.top_fraction)))
    return min(max(1, count), int(n_sources))


def ranked_candidates(
    query_fp: Any,
    source_fps: list[Any],
    source_rows: pd.DataFrame,
    query: Any,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray([weighted_similarity(query_fp, fp) for fp in source_fps], dtype=np.float32)
    scores[exclusion_mask(source_rows, query)] = -1.0
    limit = min(max(1, int(limit)), len(scores))
    chosen = np.argpartition(scores, -limit)[-limit:]
    chosen = chosen[np.argsort(scores[chosen])[::-1]]
    return chosen.astype(np.int64), scores[chosen].astype(np.float32)


def exclusion_mask(source_rows: pd.DataFrame, query: Any) -> np.ndarray:
    source_key = source_rows["record_key"].astype(str).to_numpy()
    same_record = source_key == str(query.record_key)
    source_smiles = source_rows["canonical_smiles"].astype(object).to_numpy()
    same_smiles = source_smiles == getattr(query, "canonical_smiles", None)
    return same_record | same_smiles


def build_or_load_candidate_cache(
    dataset: RecordKnnDataset,
    cache_dir: str | Path,
    config: RetrievalConfig,
    *,
    require_existing: bool = False,
) -> CandidateSet:
    metadata = cache_metadata(dataset, config)
    data_path, meta_path = candidate_cache_paths(cache_dir, metadata)
    if data_path.exists() and meta_path.exists():
        return load_candidate_cache(data_path, meta_path, metadata)
    if require_existing:
        raise FileNotFoundError(f"missing record-KNN candidate cache {data_path}")
    candidates = build_candidates(dataset, config)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_path, positions=candidates.positions, similarities=candidates.similarities)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return CandidateSet(candidates.positions, candidates.similarities, metadata, data_path)


def build_candidates(dataset: RecordKnnDataset, config: RetrievalConfig) -> CandidateSet:
    n_candidates = candidate_count(len(dataset.sources), config)
    source_fps = fingerprints(dataset.sources["smiles"])
    query_fps = fingerprints(dataset.queries["smiles"])
    positions = np.empty((len(dataset.queries), n_candidates), dtype=np.uint32)
    similarities = np.empty((len(dataset.queries), n_candidates), dtype=np.float32)
    for idx, (query, fp) in enumerate(zip(dataset.queries.itertuples(index=False), query_fps, strict=True)):
        pos, sims = ranked_candidates(fp, source_fps, dataset.sources, query, n_candidates)
        positions[idx, :] = pos.astype(np.uint32)
        similarities[idx, :] = sims.astype(np.float32)
    return CandidateSet(positions, similarities, cache_metadata(dataset, config), None)


def cache_metadata(dataset: RecordKnnDataset, config: RetrievalConfig) -> dict[str, Any]:
    n_candidates = candidate_count(len(dataset.sources), config)
    top_policy = {"top_fraction": float(config.top_fraction), "top_n": _top_n_value(config)}
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "dataset_dir": str(dataset.dataset_dir),
        "dataset_config": dataset.config_name,
        "split": dataset.split,
        "source_split": "train",
        "n_sources": int(len(dataset.sources)),
        "n_queries": int(len(dataset.queries)),
        "n_candidates": int(n_candidates),
        "source_key_hash": dataset.source_key_hash,
        "query_key_hash": dataset.query_key_hash,
        "dataset_manifest_hash": dataset.manifest_hash,
        "rdkit_version": rdkit_version(),
        "fingerprint_params": FINGERPRINT_PARAMS,
        "similarity_weights": SIMILARITY_WEIGHTS,
        "exclusion_policy": EXCLUSION_POLICY,
        "top_policy": top_policy,
        "max_queries": int(config.max_queries),
    }


def _top_n_value(config: RetrievalConfig) -> int | None:
    if config.top_n is not None and config.top_n > 0:
        return int(config.top_n)
    return None


def candidate_cache_paths(cache_dir: str | Path, metadata: dict[str, Any]) -> tuple[Path, Path]:
    digest = sha256_json(metadata)
    root = Path(cache_dir)
    return root / f"candidates_{digest}.npz", root / f"candidates_{digest}.json"


def load_candidate_cache(data_path: Path, meta_path: Path, expected: dict[str, Any]) -> CandidateSet:
    metadata = json.loads(meta_path.read_text())
    validate_cache_metadata(metadata, expected)
    data = np.load(data_path)
    positions = data["positions"].astype(np.uint32)
    similarities = data["similarities"].astype(np.float32)
    validate_candidate_arrays(positions, similarities, metadata)
    return CandidateSet(positions, similarities, metadata, data_path)


def validate_cache_metadata(metadata: dict[str, Any], expected: dict[str, Any]) -> None:
    if metadata != expected:
        diff = [key for key in sorted(set(metadata) | set(expected)) if metadata.get(key) != expected.get(key)]
        raise RuntimeError(f"record-KNN cache metadata mismatch: {diff[:8]}")
    forbidden = "embedding_manifest_hash"
    if forbidden in metadata:
        raise RuntimeError(f"retrieval cache metadata must not include {forbidden}")


def validate_candidate_arrays(positions: np.ndarray, similarities: np.ndarray, metadata: dict[str, Any]) -> None:
    shape = (int(metadata["n_queries"]), int(metadata["n_candidates"]))
    if positions.shape != shape or similarities.shape != shape:
        raise RuntimeError(f"candidate cache shape mismatch: {positions.shape}/{similarities.shape} != {shape}")
    if positions.size and int(positions.max()) >= int(metadata["n_sources"]):
        raise RuntimeError("candidate cache contains source index outside train source pool")
