"""Scorer implementations for record KNN candidate aggregation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .knn_data import RecordKnnDataset, sha256_file


@dataclass(frozen=True)
class CandidateBatch:
    dataset: RecordKnnDataset
    query_position: int
    candidate_positions: np.ndarray
    retrieval_scores: np.ndarray


class KnnScorer(Protocol):
    name: str
    requires_source_value: bool

    def validate(self, dataset: RecordKnnDataset, context: dict[str, Any]) -> None:
        ...

    def prepare(self, split: str) -> None:
        ...

    def score_candidates(self, batch: CandidateBatch) -> np.ndarray:
        ...

    def cache_identity(self) -> dict[str, Any]:
        ...


@dataclass
class TanimotoScorer:
    name: str = "tanimoto_knn"
    requires_source_value: bool = False

    def validate(self, dataset: RecordKnnDataset, context: dict[str, Any]) -> None:
        return None

    def prepare(self, split: str) -> None:
        return None

    def score_candidates(self, batch: CandidateBatch) -> np.ndarray:
        return np.asarray(batch.retrieval_scores, dtype=np.float32)

    def cache_identity(self) -> dict[str, Any]:
        return {"name": self.name}


@dataclass
class MLPTransferScorer:
    model: Any
    source_value_scale: float
    batch_size: int = 4096
    device: str | None = None
    config_path: str | Path | None = None
    checkpoint: str | Path | None = None
    embedding_manifest_path: str | Path | None = None
    base_parquet: str | Path | None = None
    name: str = "starling_rerank_knn"
    requires_source_value: bool = True

    def validate(self, dataset: RecordKnnDataset, context: dict[str, Any]) -> None:
        core = unwrap_model(self.model)
        self.requires_source_value = bool(getattr(core, "use_source_value", False))
        self._validate_embeddings(core, dataset)
        if self.requires_source_value and "oral_bioavailability_value" not in dataset.sources:
            raise ValueError("MLP record KNN requires source oral_bioavailability_value")
        if self.source_value_scale <= 0:
            raise ValueError("source_value_scale must be positive for MLP record KNN")

    def prepare(self, split: str) -> None:
        core = unwrap_model(self.model)
        core.eval()

    def score_candidates(self, batch: CandidateBatch) -> np.ndarray:
        import torch

        core = unwrap_model(self.model)
        device = self._device(core)
        positions = batch.candidate_positions.astype(np.int64)
        scores = np.empty(len(positions), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(positions), int(self.batch_size)):
                end = min(start + int(self.batch_size), len(positions))
                scores[start:end] = self._score_slice(core, batch, positions[start:end], device)
        return scores

    def cache_identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "use_source_value": bool(getattr(unwrap_model(self.model), "use_source_value", False)),
            "source_value_scale": float(self.source_value_scale),
            "checkpoint": self._path_identity(self.checkpoint),
            "config": self._path_identity(self.config_path),
            "embedding_manifest_hash": self._file_hash(self.embedding_manifest_path),
            "base_parquet": self._path_identity(self.base_parquet),
        }

    def _validate_embeddings(self, core: Any, dataset: RecordKnnDataset) -> None:
        if not hasattr(core, "mol_emb"):
            raise ValueError("MLP scorer model must expose mol_emb")
        n_emb = int(core.mol_emb.shape[0])
        max_source = int(dataset.sources["row_index"].max())
        max_query = int(dataset.queries["row_index"].max())
        if max(max_source, max_query) >= n_emb:
            raise RuntimeError(f"record-KNN row_index exceeds model embedding rows: max={max(max_source, max_query)}")

    def _score_slice(self, core: Any, batch: CandidateBatch, positions: np.ndarray, device: Any) -> np.ndarray:
        import torch

        rows = batch.dataset.sources.iloc[positions]
        a_idx = torch.from_numpy(rows["row_index"].to_numpy(dtype=np.int64)).to(device)
        qrow = int(batch.dataset.queries.iloc[int(batch.query_position)]["row_index"])
        b_idx = torch.full((len(positions),), qrow, dtype=torch.long, device=device)
        kwargs = self._source_value_kwargs(core, rows, device)
        logits = core(a_idx=a_idx, b_idx=b_idx, **kwargs)["logits"]
        return torch.sigmoid(logits).detach().float().cpu().numpy()

    def _source_value_kwargs(self, core: Any, rows: Any, device: Any) -> dict[str, Any]:
        if not bool(getattr(core, "use_source_value", False)):
            return {}
        import torch

        values = rows["oral_bioavailability_value"].to_numpy(dtype=np.float32)
        scaled = values / np.float32(self.source_value_scale)
        return {"source_value": torch.from_numpy(scaled).to(device)}

    def _device(self, core: Any) -> Any:
        if self.device:
            return self.device
        return next(core.parameters()).device

    def _file_hash(self, path: str | Path | None) -> str | None:
        return sha256_file(path) if path and Path(path).exists() else None

    def _path_identity(self, path: str | Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        target = Path(path)
        if target.is_dir():
            return {"path": str(target), "digest": _directory_digest(target)}
        if target.exists():
            return {"path": str(target), "sha256": sha256_file(target)}
        return {"path": str(target), "missing": True}


def unwrap_model(model: Any) -> Any:
    core = model
    while hasattr(core, "module"):
        core = core.module
    if hasattr(core, "_orig_mod"):
        core = core._orig_mod
    return core


def _directory_digest(path: Path) -> str:
    h = hashlib.sha256()
    for child in sorted(path.iterdir()):
        if child.name in {"model.safetensors", "pytorch_model.bin", "config.json"} and child.is_file():
            h.update(child.name.encode())
            h.update(sha256_file(child).encode())
    return h.hexdigest()


def scorer_identity_hash(scorer: KnnScorer) -> str:
    payload = json.dumps(scorer.cache_identity(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
