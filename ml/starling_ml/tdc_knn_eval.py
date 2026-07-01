"""TDC Bioavailability_Ma downstream KNN evaluation for live MLP runs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config
from .metrics import simple_transfer_metrics


@dataclass(frozen=True)
class TdcEvalResult:
    macro_f1: float
    accuracy: float
    transfer_precision: float
    transfer_recall: float
    n_queries: int
    elapsed_seconds: float


def tdc_eval_result_dict(result: TdcEvalResult) -> dict[str, float | int]:
    return {
        "macro_f1": result.macro_f1,
        "accuracy": result.accuracy,
        "transfer_precision": result.transfer_precision,
        "transfer_recall": result.transfer_recall,
        "n_queries": result.n_queries,
        "elapsed_seconds": result.elapsed_seconds,
    }


class SmilesCanonicalizer:
    def __init__(self) -> None:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.warning")
        self.chem = Chem
        self.cache: dict[str, str | None] = {}

    def canonical(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text in self.cache:
            return self.cache[text]
        mol = self.chem.MolFromSmiles(text)
        if mol is None:
            self.cache[text] = None
            return None
        self.cache[text] = self.chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return self.cache[text]


def load_tdc_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    canonicalizer = SmilesCanonicalizer()
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            raw = row.get("drug") or row.get("Drug")
            can = canonicalizer.canonical(raw)
            if can is None:
                continue
            rows.append({"smiles": str(raw).strip(), "canonical_smiles": can, "Y": int(row["Y"])})
    return rows


def _fingerprints(smiles: list[str]):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.warning")
    inv = rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership=True)
    feat_inv = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=inv)
    feature_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=feat_inv)
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(None)
        else:
            out.append((morgan_gen.GetFingerprint(mol), feature_gen.GetFingerprint(mol)))
    return out


def _weighted_similarity(left_fp, right_fp) -> float:
    from rdkit import DataStructs

    return 0.8 * float(DataStructs.TanimotoSimilarity(left_fp[0], right_fp[0])) + 0.2 * float(
        DataStructs.TanimotoSimilarity(left_fp[1], right_fp[1])
    )


def load_base(cfg: Config) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(cfg.paths.base_parquet, columns=["smiles", "oral_bioavailability_value"])
    smiles = [str(value).strip() for value in table["smiles"].to_pylist()]
    values = table["oral_bioavailability_value"].to_numpy(zero_copy_only=False).astype(np.float32)
    return {
        "smiles": smiles,
        "values": values,
        "bioavailable": values >= np.float32(20.0),
    }


def _tdc_cache_stem(tdc_path: str | Path) -> str:
    path = Path(tdc_path)
    return f"{path.parent.name}_{path.stem}"


def _legacy_tdc_cache_stem(tdc_path: str | Path) -> str:
    return Path(tdc_path).stem


def candidate_cache_path(cache_dir: str | Path, tdc_path: str | Path, base_parquet: str, top_fraction: float) -> Path:
    safe = f"{_tdc_cache_stem(tdc_path)}_{Path(base_parquet).stem}_top{top_fraction:.3f}".replace(".", "p")
    return Path(cache_dir) / f"{safe}_candidates.npz"


def _legacy_candidate_cache_path(
    cache_dir: str | Path, tdc_path: str | Path, base_parquet: str, top_fraction: float
) -> Path:
    safe = f"{_legacy_tdc_cache_stem(tdc_path)}_{Path(base_parquet).stem}_top{top_fraction:.3f}".replace(".", "p")
    return Path(cache_dir) / f"{safe}_candidates.npz"


def build_or_load_candidates(
    *,
    cfg: Config,
    tdc_path: str | Path,
    tdc_rows: list[dict[str, Any]],
    cache_dir: str | Path,
    top_fraction: float,
) -> np.ndarray:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = candidate_cache_path(cache_dir, tdc_path, cfg.paths.base_parquet, top_fraction)
    if path.exists():
        return np.load(path)["candidates"].astype(np.uint32)
    legacy_path = _legacy_candidate_cache_path(cache_dir, tdc_path, cfg.paths.base_parquet, top_fraction)
    if legacy_path.exists():
        data = np.load(legacy_path)["candidates"].astype(np.uint32)
        if data.shape[0] == len(tdc_rows):
            np.savez_compressed(path, candidates=data)
            return data

    base = load_base(cfg)
    base_fps = _fingerprints(base["smiles"])
    query_fps = _fingerprints([row["canonical_smiles"] for row in tdc_rows])
    n_base = len(base_fps)
    k = max(1, int(np.ceil(n_base * top_fraction)))
    candidates = np.empty((len(tdc_rows), k), dtype=np.uint32)
    for qi, qfp in enumerate(query_fps):
        if qfp is None:
            candidates[qi, :] = np.arange(k, dtype=np.uint32)
            continue
        scores = np.asarray(
            [(-1.0 if bfp is None else _weighted_similarity(qfp, bfp)) for bfp in base_fps],
            dtype=np.float32,
        )
        chosen = np.argpartition(scores, -k)[-k:]
        chosen = chosen[np.argsort(scores[chosen])[::-1]]
        candidates[qi, :] = chosen.astype(np.uint32)
    np.savez_compressed(path, candidates=candidates)
    return candidates


def query_embedding_cache_path(cache_dir: str | Path, tdc_path: str | Path, embeddings_dir: str) -> Path:
    safe = f"{_tdc_cache_stem(tdc_path)}_{Path(embeddings_dir).name}".replace(".", "p")
    return Path(cache_dir) / f"{safe}_query_embeddings.npz"


def _legacy_query_embedding_cache_path(cache_dir: str | Path, tdc_path: str | Path, embeddings_dir: str) -> Path:
    safe = f"{_legacy_tdc_cache_stem(tdc_path)}_{Path(embeddings_dir).name}".replace(".", "p")
    return Path(cache_dir) / f"{safe}_query_embeddings.npz"


def build_or_load_query_embeddings(
    *,
    cfg: Config,
    tdc_path: str | Path,
    tdc_rows: list[dict[str, Any]],
    cache_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = query_embedding_cache_path(cache_dir, tdc_path, cfg.paths.embeddings_dir)
    if path.exists():
        data = np.load(path)
        return data["mol_emb"], data["meta_emb"], data["meta_present"]
    legacy_path = _legacy_query_embedding_cache_path(cache_dir, tdc_path, cfg.paths.embeddings_dir)
    if legacy_path.exists():
        data = np.load(legacy_path)
        if data["mol_emb"].shape[0] == len(tdc_rows):
            mol_emb, meta_emb, meta_present = data["mol_emb"], data["meta_emb"], data["meta_present"]
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, mol_emb=mol_emb, meta_emb=meta_emb, meta_present=meta_present)
            return mol_emb, meta_emb, meta_present

    from .precompute_embeddings import compute_metadata, compute_molformer

    columns: dict[str, list[Any]] = {"smiles": [row["canonical_smiles"] for row in tdc_rows]}
    fixed_metadata = {
        "molecule_name": None,
        "species_or_population": "human",
        "dose": None,
        "oral_exposure_mode": "oral",
        "qualifying_conditions": None,
        "comparator": None,
        "extra_details": "TDC Bioavailability_Ma oral bioavailability",
    }
    for field in cfg.embedding.metadata_fields:
        columns[field] = [fixed_metadata.get(field) for _row in tdc_rows]
    mol_emb = compute_molformer(columns["smiles"], cfg)
    meta_emb, meta_present = compute_metadata(columns, cfg)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mol_emb=mol_emb, meta_emb=meta_emb, meta_present=meta_present)
    return mol_emb, meta_emb, meta_present


def _model_core(model):
    return model.module if hasattr(model, "module") else model


def evaluate_tdc_knn(
    *,
    cfg: Config,
    model,
    tdc_path: str | Path,
    cache_dir: str | Path,
    top_fraction: float = 0.25,
    k: int = 10,
    batch_size: int = 65536,
) -> TdcEvalResult:
    start_time = time.perf_counter()
    rows = load_tdc_jsonl(tdc_path)
    if not rows:
        raise RuntimeError(f"no valid TDC rows found in {tdc_path}")
    base = load_base(cfg)
    candidates = build_or_load_candidates(
        cfg=cfg,
        tdc_path=tdc_path,
        tdc_rows=rows,
        cache_dir=cache_dir,
        top_fraction=top_fraction,
    )
    q_mol, q_meta, q_present = build_or_load_query_embeddings(
        cfg=cfg,
        tdc_path=tdc_path,
        tdc_rows=rows,
        cache_dir=cache_dir,
    )

    core = _model_core(model)
    device = next(core.parameters()).device
    old_mol, old_meta, old_present = core.mol_emb, core.meta_emb, core.meta_present
    base_count = old_mol.shape[0]
    try:
        core.mol_emb = torch.cat([old_mol, torch.from_numpy(q_mol).to(device=device, dtype=old_mol.dtype)], dim=0)
        core.meta_emb = torch.cat([old_meta, torch.from_numpy(q_meta).to(device=device, dtype=old_meta.dtype)], dim=0)
        core.meta_present = torch.cat(
            [old_present, torch.from_numpy(q_present).to(device=device, dtype=old_present.dtype)],
            dim=0,
        )
        labels = np.asarray([row["Y"] for row in rows], dtype=np.int8)
        predictions: list[int] = []
        source_values = base["values"].astype(np.float32) / np.float32(cfg.model.source_value_scale)
        core.eval()
        with torch.no_grad():
            for qi in range(len(rows)):
                cand = candidates[qi]
                scores = np.empty(len(cand), dtype=np.float32)
                query_idx = np.full(len(cand), base_count + qi, dtype=np.int64)
                for start in range(0, len(cand), batch_size):
                    end = min(start + batch_size, len(cand))
                    a = torch.from_numpy(cand[start:end].astype(np.int64)).to(device)
                    b = torch.from_numpy(query_idx[start:end]).to(device)
                    kwargs = {}
                    if core.use_source_value:
                        kwargs["source_value"] = torch.from_numpy(source_values[cand[start:end]]).to(device)
                    logits = core(a_idx=a, b_idx=b, **kwargs)["logits"]
                    scores[start:end] = torch.sigmoid(logits).detach().float().cpu().numpy()
                top = cand[np.argsort(scores)[-k:]]
                vote = float(np.mean(base["bioavailable"][top]))
                predictions.append(1 if vote >= 0.5 else 0)
    finally:
        core.mol_emb, core.meta_emb, core.meta_present = old_mol, old_meta, old_present

    pred_logits = np.where(np.asarray(predictions, dtype=np.int8) == 1, 10.0, -10.0).astype(np.float32)
    metrics = simple_transfer_metrics(pred_logits, labels.astype(np.float32))
    return TdcEvalResult(
        macro_f1=float(metrics["macro_f1"]),
        accuracy=float(metrics["accuracy"]),
        transfer_precision=float(metrics["transfer_precision"]),
        transfer_recall=float(metrics["transfer_recall"]),
        n_queries=int(len(rows)),
        elapsed_seconds=float(time.perf_counter() - start_time),
    )
