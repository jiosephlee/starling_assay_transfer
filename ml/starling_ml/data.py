"""v2 SMILES-addressed example index + dataset.

The v2 materialized artifact (``docs/assay_transfer_design_v2.md`` 16.2) is one row per
directed transfer example: ``retrieved_smiles`` (x_A), ``query_smiles`` (x_B),
``retrieved_value`` (y_A), inline ``retrieved_<metadata>`` (Z_A), the continuous primary
target ``continuous_target`` (D_expected), and a nullable ``binary_label``.

Molecules are addressed by structure: precompute assigns each unique SMILES a structure
row and each unique retrieved-metadata tuple a metadata row (see
``precompute_embeddings.py``). This module flattens one split into memmaps of those
indices plus the targets so the training loop ships only small integer/float tensors and
the model gathers the frozen embeddings on-device:

    a.u32       retrieval structure row  (retrieved_smiles -> structure table)
    b.u32       query structure row      (query_smiles     -> structure table)
    meta_a.u32  retrieval metadata row   (Z_A tuple        -> metadata table)
    yA.f32      retrieval value y_A       (native; the model divides by source_value_scale)
    target.f32  continuous target D_expected (always present)
    label.i8    binary label 0/1 (0 where absent; masked by label_present)
    present.u8  1 where the hard binary label exists, else 0
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from torch.utils.data import Dataset

_A_FILE = "a.u32"
_B_FILE = "b.u32"
_META_A_FILE = "meta_a.u32"
_YA_FILE = "yA.f32"
_TARGET_FILE = "target.f32"
_LABEL_FILE = "label.i8"
_PRESENT_FILE = "present.u8"
_META_FILE = "meta.json"

# Retrieval-side metadata fields (Z_A) fed to the metadata encoder, as ``retrieved_*``
# columns of the materialized artifact. Keep in sync with precompute_embeddings.py.
DEFAULT_METADATA_FIELDS = (
    "retrieved_species_exact",
    "retrieved_assay_system",
    "retrieved_assay_system_family",
    "retrieved_administration_route",
    "retrieved_anatomical_site",
    "retrieved_formulation_class",
    "retrieved_qualifying_conditions",
)

_LABEL_TO_INT = {"transfer": 1, "not_transfer": 0}


def metadata_key(row: dict, fields: tuple[str, ...]) -> str:
    """Stable hash of one row's retrieval metadata tuple (the Z_A identity)."""
    parts = [f"{f}={row.get(f)}" for f in fields]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def _binary_to_int_present(value) -> tuple[int, int]:
    """(int8 label, present flag). Accepts 1/0/None or 'transfer'/'not_transfer'/None."""
    if value is None:
        return 0, 0
    if isinstance(value, str):
        v = _LABEL_TO_INT.get(value)
        return (v, 1) if v is not None else (0, 0)
    return int(value), 1


def build_split_memmap(
    dataset_parquet: str,
    memmap_dir: str,
    split: str,
    smiles_to_row: dict[str, int],
    meta_to_row: dict[str, int],
    *,
    metadata_fields: tuple[str, ...] = DEFAULT_METADATA_FIELDS,
    rebuild: bool = False,
) -> dict:
    """Materialize the v2 index/target memmaps for one split. Idempotent unless ``rebuild``."""
    import pyarrow.parquet as pq

    out_dir = os.path.join(memmap_dir, split)
    meta_path = os.path.join(out_dir, _META_FILE)
    signature = [os.path.basename(dataset_parquet), os.path.getsize(dataset_parquet), split,
                 list(metadata_fields)]
    if os.path.exists(meta_path) and not rebuild:
        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
            if meta.get("signature") == signature:
                return meta
        except json.JSONDecodeError:
            pass

    rows = [r for r in pq.read_table(dataset_parquet).to_pylist() if r.get("split") == split]
    n = len(rows)
    os.makedirs(out_dir, exist_ok=True)
    a = np.memmap(os.path.join(out_dir, _A_FILE), dtype=np.uint32, mode="w+", shape=(n,))
    b = np.memmap(os.path.join(out_dir, _B_FILE), dtype=np.uint32, mode="w+", shape=(n,))
    meta_a = np.memmap(os.path.join(out_dir, _META_A_FILE), dtype=np.uint32, mode="w+", shape=(n,))
    yA = np.memmap(os.path.join(out_dir, _YA_FILE), dtype=np.float32, mode="w+", shape=(n,))
    target = np.memmap(os.path.join(out_dir, _TARGET_FILE), dtype=np.float32, mode="w+", shape=(n,))
    label = np.memmap(os.path.join(out_dir, _LABEL_FILE), dtype=np.int8, mode="w+", shape=(n,))
    present = np.memmap(os.path.join(out_dir, _PRESENT_FILE), dtype=np.uint8, mode="w+", shape=(n,))

    n_hard = 0
    for i, r in enumerate(rows):
        a[i] = smiles_to_row[r["retrieved_smiles"]]
        b[i] = smiles_to_row[r["query_smiles"]]
        meta_a[i] = meta_to_row[metadata_key(r, metadata_fields)]
        yA[i] = float(r["retrieved_value"])
        target[i] = float(r["continuous_target"])
        lab, pres = _binary_to_int_present(r.get("binary_label"))
        label[i] = lab
        present[i] = pres
        n_hard += pres
    for arr in (a, b, meta_a, yA, target, label, present):
        arr.flush()

    meta = {
        "split": split,
        "count": int(n),
        "hard_binary_count": int(n_hard),
        "signature": signature,
        "metadata_fields": list(metadata_fields),
    }
    tmp = f"{meta_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(meta, fh)
    os.replace(tmp, meta_path)
    return meta


class PairDataset(Dataset):
    """Map-style dataset over the v2 index/target memmaps for one split."""

    def __init__(self, memmap_dir: str, split: str):
        out_dir = os.path.join(memmap_dir, split)
        with open(os.path.join(out_dir, _META_FILE)) as fh:
            meta = json.load(fh)
        self.count = meta["count"]
        self.hard_binary_count = meta.get("hard_binary_count", 0)
        c = (self.count,)
        self.a = np.memmap(os.path.join(out_dir, _A_FILE), dtype=np.uint32, mode="r", shape=c)
        self.b = np.memmap(os.path.join(out_dir, _B_FILE), dtype=np.uint32, mode="r", shape=c)
        self.meta_a = np.memmap(os.path.join(out_dir, _META_A_FILE), dtype=np.uint32, mode="r", shape=c)
        self.yA = np.memmap(os.path.join(out_dir, _YA_FILE), dtype=np.float32, mode="r", shape=c)
        self.target = np.memmap(os.path.join(out_dir, _TARGET_FILE), dtype=np.float32, mode="r", shape=c)
        self.label = np.memmap(os.path.join(out_dir, _LABEL_FILE), dtype=np.int8, mode="r", shape=c)
        self.present = np.memmap(os.path.join(out_dir, _PRESENT_FILE), dtype=np.uint8, mode="r", shape=c)

    def __len__(self) -> int:
        return self.count

    def _label_or_nan(self, i: int) -> float:
        return float(self.label[i]) if self.present[i] else float("nan")

    def __getitem__(self, idx: int):
        return (
            int(self.a[idx]), int(self.b[idx]), int(self.meta_a[idx]),
            float(self.yA[idx]), float(self.target[idx]), self._label_or_nan(idx),
        )

    def __getitems__(self, indices: list[int]):
        idx = np.asarray(indices, dtype=np.int64)
        labels = np.where(self.present[idx] == 1, self.label[idx].astype(np.float32), np.float32("nan"))
        return list(
            zip(
                self.a[idx].astype(np.int64).tolist(),
                self.b[idx].astype(np.int64).tolist(),
                self.meta_a[idx].astype(np.int64).tolist(),
                self.yA[idx].astype(np.float32).tolist(),
                self.target[idx].astype(np.float32).tolist(),
                labels.tolist(),
            )
        )


def collate_pairs(batch):
    import torch

    arr = np.asarray(batch, dtype=np.float64)  # a, b, meta_a, yA, target, label(nan)
    return {
        "a_idx": torch.from_numpy(arr[:, 0].astype(np.int64)),
        "b_idx": torch.from_numpy(arr[:, 1].astype(np.int64)),
        "meta_a_idx": torch.from_numpy(arr[:, 2].astype(np.int64)),
        "source_value": torch.from_numpy(arr[:, 3].astype(np.float32)),
        "distance": torch.from_numpy(arr[:, 4].astype(np.float32)),
        "labels": torch.from_numpy(arr[:, 5].astype(np.float32)),  # NaN where absent
    }
