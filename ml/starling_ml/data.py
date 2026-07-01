"""Compact pair index + dataset.

The split shards store, per pair, two ``uint32`` molecule row indices and an ``int8``
label (plus columns we don't need for training). We flatten each split once into three
numpy memmaps ``a (uint32)``, ``b (uint32)``, ``label (int8)`` so the training loop never
parses parquet and the DataLoader ships only small integer tensors — the actual
embeddings are gathered inside the model from GPU-resident buffers.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from torch.utils.data import Dataset

_A_FILE = "a.u32"
_B_FILE = "b.u32"
_L_FILE = "label.i8"
_SOURCE_VALUE_FILE = "source_value.f32"
_EVAL_SUBSET_FILE = "eval_subset.u8"
_SIMILARITY_BUCKET_FILE = "similarity_bucket.u8"
_META_FILE = "meta.json"
DEFAULT_EVAL_SUBSETS = ("no_overlap", "a_seen_only", "both_seen")
DEFAULT_SIMILARITY_BUCKET_NAMES = (
    "tanimoto_0_0p2",
    "tanimoto_0p2_0p4",
    "tanimoto_0p4_0p6",
    "tanimoto_0p6_0p8",
    "tanimoto_0p8_1",
)
_SIMILARITY_CUTS = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
_MISSING_SIMILARITY_BUCKET = np.uint8(255)


def labels_to_int8(col) -> np.ndarray:
    """Normalize a ``transfer_label`` column to int8 {0,1}.

    Handles both formats: full splits use string ``"transfer"``/``"not_transfer"``; the older
    compact splits use int8 0/1 (passed through unchanged).
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
        return pc.equal(col, "transfer").cast(pa.int8()).to_numpy(zero_copy_only=False)
    return col.to_numpy(zero_copy_only=False).astype(np.int8)


def _split_files(splits_dir: str, split: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(splits_dir, split, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet shards for split {split!r} under {splits_dir}")
    return files


def _load_scaled_source_values(base_parquet: str, scale: float) -> np.ndarray:
    import pyarrow.parquet as pq

    if not base_parquet:
        raise ValueError("base_parquet is required when source-value memmaps are enabled")
    if scale == 0:
        raise ValueError("source_value_scale must be nonzero")
    table = pq.read_table(base_parquet, columns=["oral_bioavailability_value"])
    col = table.column("oral_bioavailability_value")
    if col.null_count:
        raise ValueError(f"oral_bioavailability_value has {col.null_count} nulls in {base_parquet}")
    values = col.to_numpy(zero_copy_only=False).astype(np.float32) / np.float32(scale)
    if not np.isfinite(values).all():
        raise ValueError(f"oral_bioavailability_value contains non-finite values in {base_parquet}")
    return values


def fixed_tanimoto_bucket(values) -> np.ndarray:
    """Map Tanimoto scores to fixed bins: [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]."""

    scores = np.asarray(values, dtype=np.float32)
    out = np.full(scores.shape, _MISSING_SIMILARITY_BUCKET, dtype=np.uint8)
    valid = np.isfinite(scores)
    if valid.any():
        clipped = np.clip(scores[valid], 0.0, 1.0)
        out[valid] = np.searchsorted(_SIMILARITY_CUTS, clipped, side="right").astype(np.uint8)
    return out


def build_split_memmap(
    splits_dir: str,
    memmap_dir: str,
    split: str,
    rebuild: bool = False,
    *,
    base_parquet: str | None = None,
    use_source_value: bool = False,
    source_value_scale: float = 100.0,
    store_eval_subset: bool = False,
    eval_subset_names: tuple[str, ...] = DEFAULT_EVAL_SUBSETS,
    store_similarity_bucket: bool = False,
    similarity_bucket_names: tuple[str, ...] = DEFAULT_SIMILARITY_BUCKET_NAMES,
) -> dict:
    """Materialize (a, b, label) memmaps for one split. Idempotent unless ``rebuild``."""
    import pyarrow.parquet as pq

    out_dir = os.path.join(memmap_dir, split)
    meta_path = os.path.join(out_dir, _META_FILE)
    files = _split_files(splits_dir, split)
    signature = [[os.path.basename(f), os.path.getsize(f)] for f in files]
    if use_source_value:
        signature.append(
            [
                "source_value",
                os.path.basename(base_parquet or ""),
                os.path.getsize(base_parquet or ""),
                float(source_value_scale),
            ]
        )
    if store_eval_subset:
        signature.append(["eval_subset", *eval_subset_names])
    if store_similarity_bucket:
        signature.append(["fixed_tanimoto_similarity_bucket", *similarity_bucket_names])

    if os.path.exists(meta_path) and not rebuild:
        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
        except json.JSONDecodeError:
            meta = {}
        if meta.get("signature") == signature:
            return meta  # already built and source unchanged

    os.makedirs(out_dir, exist_ok=True)
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f"[memmap:{split}] {len(files)} shards, {total} pairs -> {out_dir}")

    a = np.memmap(os.path.join(out_dir, _A_FILE), dtype=np.uint32, mode="w+", shape=(total,))
    b = np.memmap(os.path.join(out_dir, _B_FILE), dtype=np.uint32, mode="w+", shape=(total,))
    lab = np.memmap(os.path.join(out_dir, _L_FILE), dtype=np.int8, mode="w+", shape=(total,))
    source_values = _load_scaled_source_values(base_parquet or "", source_value_scale) if use_source_value else None
    source_value = (
        np.memmap(os.path.join(out_dir, _SOURCE_VALUE_FILE), dtype=np.float32, mode="w+", shape=(total,))
        if use_source_value
        else None
    )
    eval_subset = (
        np.memmap(os.path.join(out_dir, _EVAL_SUBSET_FILE), dtype=np.uint8, mode="w+", shape=(total,))
        if store_eval_subset
        else None
    )
    similarity_bucket = (
        np.memmap(os.path.join(out_dir, _SIMILARITY_BUCKET_FILE), dtype=np.uint8, mode="w+", shape=(total,))
        if store_similarity_bucket
        else None
    )
    subset_to_code = {name: code for code, name in enumerate(eval_subset_names)}
    subset_counts = {name: 0 for name in eval_subset_names}
    similarity_bucket_counts = {name: 0 for name in similarity_bucket_names}

    offset = 0
    cols = ["row_index_a", "row_index_b", "transfer_label"]
    if store_eval_subset:
        cols.append("eval_subset")
    if store_similarity_bucket:
        cols.append("weighted_tanimoto")
    for fi, f in enumerate(files):
        table = pq.read_table(f, columns=cols)
        m = table.num_rows
        row_a = table.column("row_index_a").to_numpy(zero_copy_only=False)
        a[offset : offset + m] = row_a
        b[offset : offset + m] = table.column("row_index_b").to_numpy(zero_copy_only=False)
        lab[offset : offset + m] = labels_to_int8(table.column("transfer_label"))
        if source_value is not None and source_values is not None:
            source_value[offset : offset + m] = source_values[row_a]
        if eval_subset is not None:
            values = table.column("eval_subset").to_pylist()
            encoded = np.empty(m, dtype=np.uint8)
            for i, value in enumerate(values):
                if value not in subset_to_code:
                    raise ValueError(
                        f"unexpected eval_subset value {value!r} in {f}; "
                        f"expected one of {list(eval_subset_names)!r}"
                    )
                encoded[i] = subset_to_code[value]
                subset_counts[value] += 1
            eval_subset[offset : offset + m] = encoded
        if similarity_bucket is not None:
            encoded = fixed_tanimoto_bucket(
                table.column("weighted_tanimoto").to_numpy(zero_copy_only=False)
            )
            for code, name in enumerate(similarity_bucket_names):
                similarity_bucket_counts[name] += int((encoded == code).sum())
            similarity_bucket[offset : offset + m] = encoded
        offset += m
        if fi % 10 == 0 or fi == len(files) - 1:
            print(f"[memmap:{split}] {offset}/{total}")
    a.flush(); b.flush(); lab.flush()
    if source_value is not None:
        source_value.flush()
    if eval_subset is not None:
        eval_subset.flush()
    if similarity_bucket is not None:
        similarity_bucket.flush()
    assert offset == total, (offset, total)

    meta = {
        "split": split,
        "count": int(total),
        "signature": signature,
        "use_source_value": bool(use_source_value),
        "source_value_scale": float(source_value_scale),
        "has_eval_subset": bool(store_eval_subset),
        "eval_subset_names": list(eval_subset_names) if store_eval_subset else [],
        "eval_subset_counts": subset_counts if store_eval_subset else {},
        "has_similarity_bucket": bool(store_similarity_bucket),
        "similarity_bucket_names": list(similarity_bucket_names) if store_similarity_bucket else [],
        "similarity_bucket_counts": similarity_bucket_counts if store_similarity_bucket else {},
    }
    tmp_meta_path = f"{meta_path}.tmp.{os.getpid()}"
    with open(tmp_meta_path, "w") as fh:
        json.dump(meta, fh)
    os.replace(tmp_meta_path, meta_path)
    return meta


class PairDataset(Dataset):
    """Map-style dataset over the compact (a, b, label) memmaps for one split."""

    def __init__(self, memmap_dir: str, split: str):
        out_dir = os.path.join(memmap_dir, split)
        with open(os.path.join(out_dir, _META_FILE)) as fh:
            meta = json.load(fh)
        self.count = meta["count"]
        self.use_source_value = bool(meta.get("use_source_value", False))
        self.has_eval_subset = bool(meta.get("has_eval_subset", False))
        self.has_similarity_bucket = bool(meta.get("has_similarity_bucket", False))
        self.eval_subset_names = list(meta.get("eval_subset_names") or [])
        self.eval_subset_counts = dict(meta.get("eval_subset_counts") or {})
        self.similarity_bucket_names = list(meta.get("similarity_bucket_names") or [])
        self.similarity_bucket_counts = dict(meta.get("similarity_bucket_counts") or {})
        self.a = np.memmap(os.path.join(out_dir, _A_FILE), dtype=np.uint32, mode="r", shape=(self.count,))
        self.b = np.memmap(os.path.join(out_dir, _B_FILE), dtype=np.uint32, mode="r", shape=(self.count,))
        self.lab = np.memmap(os.path.join(out_dir, _L_FILE), dtype=np.int8, mode="r", shape=(self.count,))
        self.source_value = (
            np.memmap(os.path.join(out_dir, _SOURCE_VALUE_FILE), dtype=np.float32, mode="r", shape=(self.count,))
            if self.use_source_value
            else None
        )
        self.eval_subset = (
            np.memmap(os.path.join(out_dir, _EVAL_SUBSET_FILE), dtype=np.uint8, mode="r", shape=(self.count,))
            if self.has_eval_subset
            else None
        )
        self.similarity_bucket = (
            np.memmap(
                os.path.join(out_dir, _SIMILARITY_BUCKET_FILE),
                dtype=np.uint8,
                mode="r",
                shape=(self.count,),
            )
            if self.has_similarity_bucket
            else None
        )

    def __len__(self) -> int:
        return self.count

    def eval_subset_indices(self, name: str) -> np.ndarray:
        if self.eval_subset is None:
            raise ValueError("eval_subset was not stored for this dataset")
        if name not in self.eval_subset_names:
            raise KeyError(f"unknown eval_subset {name!r}; expected {self.eval_subset_names!r}")
        code = self.eval_subset_names.index(name)
        return np.flatnonzero(np.asarray(self.eval_subset) == code)

    def similarity_bucket_indices(self, name: str) -> np.ndarray:
        if self.similarity_bucket is None:
            raise ValueError("similarity_bucket was not stored for this dataset")
        if name not in self.similarity_bucket_names:
            raise KeyError(f"unknown similarity bucket {name!r}; expected {self.similarity_bucket_names!r}")
        code = self.similarity_bucket_names.index(name)
        return np.flatnonzero(np.asarray(self.similarity_bucket) == code)

    def __getitem__(self, idx: int):
        if self.source_value is not None:
            return int(self.a[idx]), int(self.b[idx]), int(self.lab[idx]), float(self.source_value[idx])
        return int(self.a[idx]), int(self.b[idx]), int(self.lab[idx])

    def __getitems__(self, indices: list[int]):
        # Batched fetch (PyTorch DataLoader uses this when present) — vectorized gather
        # avoids per-index Python overhead at billion-pair scale.
        idx = np.asarray(indices, dtype=np.int64)
        a = self.a[idx].astype(np.int64)
        b = self.b[idx].astype(np.int64)
        lab = self.lab[idx].astype(np.float32)
        if self.source_value is not None:
            source_value = self.source_value[idx].astype(np.float32)
            return list(zip(a.tolist(), b.tolist(), lab.tolist(), source_value.tolist()))
        return list(zip(a.tolist(), b.tolist(), lab.tolist()))


def collate_pairs(batch):
    import torch

    arr = np.asarray(batch, dtype=np.float64)  # columns: a, b, label[, source_value]
    a = torch.from_numpy(arr[:, 0].astype(np.int64))
    b = torch.from_numpy(arr[:, 1].astype(np.int64))
    labels = torch.from_numpy(arr[:, 2].astype(np.float32))
    out = {"a_idx": a, "b_idx": b, "labels": labels}
    if arr.shape[1] > 3:
        out["source_value"] = torch.from_numpy(arr[:, 3].astype(np.float32))
    return out
