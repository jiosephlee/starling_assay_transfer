"""Validated embedding and pair-cache access for V6 100M training."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from numpy.lib.format import open_memmap

from .v6_embedding_cache import CONCEPTS, file_sha256, validate_embedding_cache


PAIR_NAMES = ("query_record_index", "retrieval_record_index", "retrieval_value",
              "target_z", "target_a", "group_index", "member_index")
PAIR_DTYPES = {"query_record_index": np.uint32, "retrieval_record_index": np.uint32,
               "retrieval_value": np.float32, "target_z": np.float32,
               "target_a": np.float32, "group_index": np.uint32,
               "member_index": np.uint16}
SPLITS = ("train", "validation", "test", "validation_ranking", "test_ranking")


def _cache_split(source: Path, output: Path) -> dict:
    parquet, rows, arrays = pq.ParquetFile(source), pq.ParquetFile(source).metadata.num_rows, {}
    output.mkdir(parents=True)
    for name in PAIR_NAMES:
        arrays[name] = open_memmap(output / f"{name}.npy", mode="w+",
                                   dtype=PAIR_DTYPES[name], shape=(rows,))
    offset = 0
    for batch in parquet.iter_batches(batch_size=131072, columns=list(PAIR_NAMES)):
        size = len(batch)
        for name in PAIR_NAMES:
            arrays[name][offset:offset + size] = np.asarray(batch.column(name))
        offset += size
    arrays.clear()
    if "ranking_query_id" in parquet.schema_arrow.names:
        values = pq.read_table(source, columns=["ranking_query_id"]).column(0).to_pylist()
        width = max(map(len, values), default=1)
        np.save(output / "ranking_query_id.npy", np.asarray(values, dtype=f"U{width}"))
    return {"rows": rows, "source_sha256": file_sha256(source)}


def build_pair_cache(dataset: Path, output: Path) -> dict:
    """Build an immutable, pair-only V6.5 cache bound to all source Parquets."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing pair cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    split_info = {name: _cache_split(dataset / f"{name}.parquet", temporary / name)
                  for name in SPLITS}
    files = sorted(temporary.rglob("*.npy"))
    manifest = {"version": "v6_5_mlp_100m_pair_cache_v2", "dataset": str(dataset),
                "dataset_manifest_sha256": file_sha256(dataset / "manifest.json"),
                "dataset_records_sha256": file_sha256(dataset / "records.parquet"),
                "splits": split_info,
                "sha256": {str(path.relative_to(temporary)): file_sha256(path)
                            for path in files}}
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temporary, output)
    return manifest


def validate_pair_cache(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != "v6_5_mlp_100m_pair_cache_v2":
        raise ValueError("unsupported V6.5 pair cache version")
    for name, expected in manifest["sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"pair cache hash mismatch: {name}")
    return manifest


class V6CachedPairs:
    def __init__(self, embeddings: Path, pairs: Path, device: torch.device):
        self.embedding_manifest = validate_embedding_cache(embeddings)
        self.pair_manifest = validate_pair_cache(pairs)
        if self.pair_manifest["dataset_records_sha256"] != self.embedding_manifest["records_sha256"]:
            raise ValueError("pair and embedding caches refer to different record tables")
        self.device, self.pairs, self.splits = device, pairs, {}
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.molecule = self._tensor(embeddings / "molecule_embeddings.npy", dtype)
        self.assay = self._tensor(embeddings / "assay_embeddings.npy", dtype)
        self.record_molecule = self._tensor(
            embeddings / "record_molecule_index.npy", torch.long)
        self.record_concept = np.load(embeddings / "record_assay_concept.npy", mmap_mode="r")

    def _tensor(self, path: Path, dtype: torch.dtype) -> torch.Tensor:
        values = np.load(path, mmap_mode="r")
        writable = np.array(values, copy=True)
        return torch.from_numpy(writable).to(device=self.device, dtype=dtype)

    def split(self, name: str) -> dict[str, np.ndarray]:
        if name not in self.splits:
            self.splits[name] = {key: np.load(self.pairs / name / f"{key}.npy", mmap_mode="r")
                                 for key in PAIR_NAMES}
        return self.splits[name]

    def features(self, split: dict[str, np.ndarray], indices: np.ndarray):
        query = self._indices(split["query_record_index"], indices)
        retrieval = self._indices(split["retrieval_record_index"], indices)
        value = torch.as_tensor(np.asarray(split["retrieval_value"][indices]),
                                device=self.device, dtype=torch.float32)
        qm = self.molecule[self.record_molecule[query]]
        rm = self.molecule[self.record_molecule[retrieval]]
        return qm, rm, self.assay[query], self.assay[retrieval], value

    def targets(self, split: dict[str, np.ndarray], indices: np.ndarray) -> torch.Tensor:
        values = np.asarray(split["target_a"][indices])
        return torch.as_tensor(values, device=self.device, dtype=torch.float32)

    def concepts(self, split: dict[str, np.ndarray]) -> np.ndarray:
        query = np.asarray(split["query_record_index"])
        return np.asarray(self.record_concept[query])

    def ranking_subset(self, split_name: str, maximum: int = 100,
                       seed: int = 42) -> np.ndarray:
        split = self.split(split_name)
        return ranking_subset_indices(np.asarray(split["group_index"]), self.concepts(split),
                                      maximum, seed)

    def _indices(self, values: np.ndarray, indices: np.ndarray) -> torch.Tensor:
        selected = np.asarray(values[indices]).astype(np.int64, copy=False)
        return torch.as_tensor(selected, device=self.device)


def group_indices(groups: np.ndarray, group_size: int) -> np.ndarray:
    offsets = np.arange(group_size, dtype=np.int64)[None, :]
    return (groups[:, None].astype(np.int64) * group_size + offsets).reshape(-1)


def build_group_schedule(pair_cache: Path, output: Path, steps: int = 5000,
                         batch_groups: int = 128, group_size: int = 40,
                         seed: int = 4878, sampling_mode: str = "with_replacement") -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite schedule: {output}")
    pair_manifest = validate_pair_cache(pair_cache)
    target = np.load(pair_cache / "train" / "target_a.npy", mmap_mode="r")
    if len(target) % group_size:
        raise ValueError("training rows do not form complete groups")
    rng, groups = np.random.default_rng(seed), len(target) // group_size
    schedule = _schedule_values(rng, groups, steps, batch_groups, sampling_mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, schedule, allow_pickle=False)
    manifest = {"version": "v6_5_mlp_100m_group_schedule_v2", "seed": seed,
                "steps": steps, "batch_groups": batch_groups, "group_size": group_size,
                "training_groups": groups, "schedule_sha256": file_sha256(output),
                "pair_manifest_sha256": file_sha256(pair_cache / "manifest.json"),
                "train_source_sha256": pair_manifest["splits"]["train"]["source_sha256"],
                "sampling_mode": sampling_mode, "retained_groups": int(schedule.size),
                "dropped_groups": groups - int(schedule.size)}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _schedule_values(rng: np.random.Generator, groups: int, steps: int,
                     batch_groups: int, mode: str) -> np.ndarray:
    size = steps * batch_groups
    if mode == "with_replacement":
        return rng.integers(0, groups, size=(steps, batch_groups), dtype=np.uint32)
    if mode != "one_pass":
        raise ValueError(f"unsupported schedule sampling mode: {mode}")
    if size > groups:
        raise ValueError("one-pass schedule requests more groups than the pair cache contains")
    values = rng.permutation(groups)[:size].astype(np.uint32, copy=False)
    return values.reshape(steps, batch_groups)


def validate_group_schedule(path: Path, expected_steps: int = 5000,
                            pair_cache: Path | None = None) -> tuple[np.ndarray, dict]:
    manifest = json.loads(path.with_suffix(".json").read_text())
    if file_sha256(path) != manifest["schedule_sha256"]:
        raise ValueError("group schedule hash mismatch")
    schedule = np.load(path, mmap_mode="r")
    expected = (manifest["steps"], manifest["batch_groups"])
    if schedule.shape != expected or len(schedule) < expected_steps:
        raise ValueError("group schedule shape is incompatible")
    if pair_cache and manifest["pair_manifest_sha256"] != file_sha256(pair_cache / "manifest.json"):
        raise ValueError("group schedule belongs to a different pair cache")
    return schedule, manifest


def _ranking_groups(group_ids: np.ndarray, concepts: np.ndarray) -> dict[int, int]:
    groups = {}
    for group in np.unique(group_ids):
        indices = np.flatnonzero(group_ids == group)
        if len(indices) != 20:
            raise ValueError(f"ranking group {group} has {len(indices)} rows")
        codes = np.unique(concepts[indices])
        if len(codes) != 1:
            raise ValueError(f"ranking group {group} crosses assay concepts")
        groups[int(group)] = int(codes[0])
    return groups


def ranking_subset_indices(group_ids: np.ndarray, concepts: np.ndarray,
                           maximum: int = 100, seed: int = 42) -> np.ndarray:
    groups, rng = _ranking_groups(group_ids, concepts), random.Random(seed)
    queues = {}
    for code in dict.fromkeys(groups.values()):
        values = sorted(group for group, group_code in groups.items() if group_code == code)
        rng.shuffle(values)
        queues[code] = values
    order = sorted(queues, key=lambda code: (len(queues[code]), CONCEPTS[code]))
    chosen = []
    while len(chosen) < min(maximum, len(groups)):
        before = len(chosen)
        for code in order:
            if queues[code] and len(chosen) < maximum:
                chosen.append(queues[code].pop(0))
        if len(chosen) == before:
            break
    chosen_set = set(chosen)
    return np.asarray([index for index, group in enumerate(group_ids)
                       if int(group) in chosen_set], dtype=np.int64)
