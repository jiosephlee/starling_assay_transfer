"""Offline frozen-encoder cache for the V6 100M MLP ablation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np


MOLFORMER_MODEL = "ibm-research/MoLFormer-XL-both-10pct"
MOLFORMER_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"
PUBMEDBERT_MODEL = "NeuML/pubmedbert-base-embeddings"
PUBMEDBERT_REVISION = "b79526d6ef3645e0df4530322e266f24c829f5ef"
CONCEPTS = ("Fa", "Fg", "Fh", "oral_bioavailability", "oral_exposure")
CACHE_VERSIONS = ("v6_molformer_pubmedbert_100m_cache_v1",
                  "v6_5_molformer_pubmedbert_100m_cache_v2")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    columns = ["record_index", "molecule_index", "canonical_smiles", "assay_paragraph",
               "assay_concept"]
    rows = pq.read_table(path, columns=columns).to_pylist()
    rows.sort(key=lambda row: row["record_index"])
    if [row["record_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("record_index must be contiguous and sorted")
    return rows


def _identity_records(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    columns = ["record_index", "record_id", "molecule_index", "canonical_smiles",
               "assay_paragraph", "assay_concept"]
    rows = pq.read_table(path, columns=columns).to_pylist()
    rows.sort(key=lambda row: row["record_index"])
    if [row["record_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("record_index must be contiguous and sorted")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("record_id must be unique")
    return rows


def _molecules(rows: list[dict]) -> list[str]:
    values: dict[int, str] = {}
    for row in rows:
        index, smiles = row["molecule_index"], row["canonical_smiles"]
        if index in values and values[index] != smiles:
            raise ValueError(f"molecule index {index} maps to multiple SMILES")
        values[index] = smiles
    if sorted(values) != list(range(len(values))):
        raise ValueError("molecule_index must be contiguous")
    return [values[index] for index in range(len(values))]


def _token_batches(lengths: list[int], budget: int, maximum: int) -> list[list[int]]:
    batches, current, padded = [], [], 0
    for index in sorted(range(len(lengths)), key=lambda item: (lengths[item], item)):
        next_padded = max(padded, lengths[index])
        if current and (len(current) == maximum or next_padded * (len(current) + 1) > budget):
            batches.append(current)
            current, padded = [], 0
        current.append(index)
        padded = max(padded, lengths[index])
    if current:
        batches.append(current)
    return batches


def _molformer(smiles: list[str], device: str, token_budget: int) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from .precompute_embeddings import _ensure_transformers_compat, _patch_model_compat

    _ensure_transformers_compat()
    tokenizer = AutoTokenizer.from_pretrained(MOLFORMER_MODEL, revision=MOLFORMER_REVISION,
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(MOLFORMER_MODEL, revision=MOLFORMER_REVISION,
                                      trust_remote_code=True, deterministic_eval=True)
    _patch_model_compat(model)
    model.config.deterministic_eval = True
    model.to(device).eval()
    lengths = [len(tokenizer(value, truncation=False)["input_ids"]) for value in smiles]
    output = np.empty((len(smiles), 768), dtype=np.float16)
    with torch.inference_mode():
        for indices in _token_batches(lengths, token_budget, 64):
            encoded = tokenizer([smiles[i] for i in indices], padding=True, truncation=False,
                                return_tensors="pt").to(device)
            pooled = model(**encoded).pooler_output.float().cpu().numpy().astype(np.float16)
            output[np.asarray(indices)] = pooled
    return output, np.asarray(lengths, dtype=np.uint16)


def _mean_pool(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


def _pubmedbert(paragraphs: list[str], device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(PUBMEDBERT_MODEL, revision=PUBMEDBERT_REVISION)
    model = AutoModel.from_pretrained(PUBMEDBERT_MODEL, revision=PUBMEDBERT_REVISION)
    model.to(device).eval()
    lengths = [len(tokenizer(value, truncation=False)["input_ids"]) for value in paragraphs]
    if max(lengths, default=0) > 256:
        raise ValueError("an assay paragraph exceeds the frozen 256-token contract")
    output = np.empty((len(paragraphs), 768), dtype=np.float16)
    with torch.inference_mode():
        for start in range(0, len(paragraphs), batch_size):
            batch = paragraphs[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=False, return_tensors="pt").to(device)
            pooled = _mean_pool(model(**encoded).last_hidden_state.float(), encoded["attention_mask"])
            output[start:start + len(batch)] = pooled.cpu().numpy().astype(np.float16)
    return output, np.asarray(lengths, dtype=np.uint16)


def _write_arrays(root: Path, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    hashes = {}
    for name, values in arrays.items():
        path = root / name
        np.save(path, values, allow_pickle=False)
        hashes[name] = file_sha256(path)
    return hashes


def build_embedding_cache(records_path: Path, output: Path, device: str = "cuda",
                          molformer_fn: Callable = _molformer,
                          pubmedbert_fn: Callable = _pubmedbert) -> dict:
    """Build cache atomically; encoder callables are injectable for CPU unit tests."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {output}")
    rows, parent = _records(records_path), output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=parent))
    molecules, paragraphs = _molecules(rows), [row["assay_paragraph"] for row in rows]
    molecule, mol_lengths = molformer_fn(molecules, device, 8192)
    assay, assay_lengths = pubmedbert_fn(paragraphs, device, 64)
    concept = np.asarray([CONCEPTS.index(row["assay_concept"]) for row in rows], dtype=np.uint8)
    record_molecule = np.asarray([row["molecule_index"] for row in rows], dtype=np.uint32)
    arrays = {"molecule_embeddings.npy": molecule, "assay_embeddings.npy": assay,
              "record_molecule_index.npy": record_molecule, "record_assay_concept.npy": concept,
              "molecule_token_lengths.npy": mol_lengths, "assay_token_lengths.npy": assay_lengths}
    hashes = _write_arrays(temporary, arrays)
    manifest = _manifest(records_path, rows, molecule, assay, hashes)
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temporary, output)
    return manifest


def _reuse_indices(old_rows: list[dict], new_rows: list[dict]) -> np.ndarray:
    old_by_id = {row["record_id"]: row for row in old_rows}
    fields = ("molecule_index", "canonical_smiles", "assay_paragraph", "assay_concept")
    indices = []
    for row in new_rows:
        parent = old_by_id.get(row["record_id"])
        if parent is None:
            raise ValueError(f"new record cannot be reused: {row['record_id']}")
        if any(row[field] != parent[field] for field in fields):
            raise ValueError(f"record encoder inputs changed: {row['record_id']}")
        indices.append(parent["record_index"])
    return np.asarray(indices, dtype=np.int64)


def _reuse_arrays(parent: Path, indices: np.ndarray,
                  new_rows: list[dict]) -> dict[str, np.ndarray]:
    molecule_names = ("molecule_embeddings.npy", "molecule_token_lengths.npy")
    arrays = {name: np.array(np.load(parent / name, mmap_mode="r"), copy=True)
              for name in molecule_names}
    arrays["assay_embeddings.npy"] = np.array(
        np.load(parent / "assay_embeddings.npy", mmap_mode="r")[indices], copy=True)
    arrays["assay_token_lengths.npy"] = np.array(
        np.load(parent / "assay_token_lengths.npy", mmap_mode="r")[indices], copy=True)
    arrays["record_molecule_index.npy"] = np.asarray(
        [row["molecule_index"] for row in new_rows], dtype=np.uint32)
    arrays["record_assay_concept.npy"] = np.asarray(
        [CONCEPTS.index(row["assay_concept"]) for row in new_rows], dtype=np.uint8)
    return arrays


def reindex_embedding_cache(parent: Path, parent_records: Path, new_records: Path,
                            output: Path) -> dict:
    """Create an immutable subset/reindex cache without running either encoder."""
    parent_manifest = validate_embedding_cache(parent, parent_records)
    old_rows, new_rows = _identity_records(parent_records), _identity_records(new_records)
    if _molecules(old_rows) != _molecules(new_rows):
        raise ValueError("molecule universe changed; a full embedding rebuild is required")
    indices = _reuse_indices(old_rows, new_rows)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    arrays = _reuse_arrays(parent, indices, new_rows)
    hashes = _write_arrays(temporary, arrays)
    manifest = _manifest(new_records, new_rows, arrays["molecule_embeddings.npy"],
                         arrays["assay_embeddings.npy"], hashes)
    manifest["version"] = "v6_5_molformer_pubmedbert_100m_cache_v2"
    manifest["reuse"] = {"parent_cache": str(parent),
                         "parent_manifest_sha256": file_sha256(parent / "manifest.json"),
                         "parent_records_sha256": parent_manifest["records_sha256"],
                         "reused_records": len(new_rows),
                         "removed_records": len(old_rows) - len(new_rows)}
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temporary, output)
    return manifest


def _manifest(records_path: Path, rows: list[dict], molecule: np.ndarray,
              assay: np.ndarray, hashes: dict[str, str]) -> dict:
    import torch
    import transformers

    return {"version": "v6_molformer_pubmedbert_100m_cache_v1",
            "records_path": str(records_path), "records_sha256": file_sha256(records_path),
            "records": len(rows), "molecules": len(molecule), "embedding_dim": 768,
            "molecule_shape": list(molecule.shape), "assay_shape": list(assay.shape),
            "dtype": "float16", "concepts": list(CONCEPTS), "query_values_present": False,
            "molformer": {"model": MOLFORMER_MODEL, "revision": MOLFORMER_REVISION,
                          "pooling": "pooler_output", "truncation": False},
            "pubmedbert": {"model": PUBMEDBERT_MODEL, "revision": PUBMEDBERT_REVISION,
                           "pooling": "attention_mask_mean", "max_length": 256},
            "torch": torch.__version__, "transformers": transformers.__version__, "sha256": hashes}


def validate_embedding_cache(root: Path, records_path: Path | None = None) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("version") not in CACHE_VERSIONS:
        raise ValueError("unsupported embedding cache version")
    expected = (("molformer", "model", MOLFORMER_MODEL),
                ("molformer", "revision", MOLFORMER_REVISION),
                ("pubmedbert", "model", PUBMEDBERT_MODEL),
                ("pubmedbert", "revision", PUBMEDBERT_REVISION))
    if any(manifest[section][key] != value for section, key, value in expected):
        raise ValueError("embedding cache encoder contract mismatch")
    if records_path and manifest["records_sha256"] != file_sha256(records_path):
        raise ValueError("embedding cache records hash is stale")
    for name, expected in manifest["sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"embedding cache hash mismatch: {name}")
    molecule = np.load(root / "molecule_embeddings.npy", mmap_mode="r")
    assay = np.load(root / "assay_embeddings.npy", mmap_mode="r")
    if list(molecule.shape) != manifest["molecule_shape"] or list(assay.shape) != manifest["assay_shape"]:
        raise ValueError("embedding cache shape mismatch")
    return manifest
