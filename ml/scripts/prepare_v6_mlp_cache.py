#!/usr/bin/env python3
"""Create deterministic fixed features and streaming NumPy caches for V6 MLP training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numpy.lib.format import open_memmap
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.feature_extraction.text import HashingVectorizer


PAIR_COLUMNS = ("query_record_index", "retrieval_record_index", "retrieval_value",
                "retrieval_is_approximate", "target_z", "target_a", "group_index")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _molecule_features(records: list[dict], size: int) -> np.ndarray:
    count = max(row["molecule_index"] for row in records) + 1
    matrix, seen = np.zeros((count, size), dtype=np.uint8), set()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=size)
    for row in records:
        index = row["molecule_index"]
        if index in seen:
            continue
        molecule = Chem.MolFromSmiles(row["canonical_smiles"])
        if molecule is None:
            raise ValueError(f"invalid canonical SMILES at molecule index {index}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), matrix[index])
        seen.add(index)
    return matrix


def _assay_features(records: list[dict], size: int, batch_size: int = 8192) -> np.ndarray:
    vectorizer = HashingVectorizer(n_features=size, analyzer="char_wb", ngram_range=(3, 5),
                                   lowercase=True, alternate_sign=True, norm="l2", dtype=np.float32)
    matrix = np.empty((len(records), size), dtype=np.float16)
    for start in range(0, len(records), batch_size):
        paragraphs = [row["assay_paragraph"] for row in records[start:start + batch_size]]
        matrix[start:start + len(paragraphs)] = vectorizer.transform(paragraphs).toarray()
    return matrix


def _write_array(path: Path, values: np.ndarray) -> None:
    target = open_memmap(path, mode="w+", dtype=values.dtype, shape=values.shape)
    target[:] = values
    del target


def _cache_pairs(source: Path, output: Path) -> dict[str, list[int]]:
    parquet, arrays = pq.ParquetFile(source), {}
    rows = parquet.metadata.num_rows
    types = {"query_record_index": np.uint32, "retrieval_record_index": np.uint32,
             "retrieval_value": np.float16, "retrieval_is_approximate": np.uint8,
             "target_z": np.float32, "target_a": np.float32, "group_index": np.uint32}
    for name in PAIR_COLUMNS:
        arrays[name] = open_memmap(output / f"{name}.npy", mode="w+", dtype=types[name], shape=(rows,))
    offset = 0
    for batch in parquet.iter_batches(batch_size=131072, columns=list(PAIR_COLUMNS)):
        size = len(batch)
        for name in PAIR_COLUMNS:
            arrays[name][offset:offset + size] = np.asarray(batch.column(name))
        offset += size
    shapes = {name: list(array.shape) for name, array in arrays.items()}
    arrays.clear()
    return shapes


def _prepare_features(output: Path, records: list[dict], mol_dim: int, assay_dim: int):
    paths = [output / name for name in ("molecule_features.npy", "assay_features.npy",
                                        "record_molecule_index.npy")]
    if all(path.exists() for path in paths):
        molecule, assay, record_molecule = (np.load(path, mmap_mode="r") for path in paths)
        if len(assay) != len(records) or len(record_molecule) != len(records):
            raise RuntimeError("existing feature cache has a different record count")
        return molecule, assay
    molecule, assay = _molecule_features(records, mol_dim), _assay_features(records, assay_dim)
    record_molecule = np.asarray([row["molecule_index"] for row in records], dtype=np.uint32)
    _write_array(paths[0], molecule)
    _write_array(paths[1], assay)
    _write_array(paths[2], record_molecule)
    return molecule, assay


def prepare(dataset: Path, output: Path, mol_dim: int, assay_dim: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    records = pq.read_table(dataset / "records.parquet").to_pylist()
    molecule, assay = _prepare_features(output, records, mol_dim, assay_dim)
    pair_shapes = {}
    for split in ("train", "validation", "test", "validation_ranking", "test_ranking"):
        split_dir = output / split
        split_dir.mkdir(exist_ok=True)
        pair_shapes[split] = _cache_pairs(dataset / f"{split}.parquet", split_dir)
    files = sorted(path for path in output.rglob("*.npy"))
    manifest = {"dataset": str(dataset), "dataset_records_sha256": _hash(dataset / "records.parquet"),
                "mol_dim": mol_dim, "assay_dim": assay_dim,
                "records": len(records), "molecules": len(molecule), "pairs": pair_shapes,
                "sha256": {str(path.relative_to(output)): _hash(path) for path in files}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/parquet/assay_transfer_raw_pair_v6_mlp"))
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/v6_mlp/cache"))
    parser.add_argument("--mol-dim", type=int, default=1024)
    parser.add_argument("--assay-dim", type=int, default=512)
    args = parser.parse_args()
    result = prepare(args.dataset, args.output, args.mol_dim, args.assay_dim)
    print(json.dumps({key: result[key] for key in ("records", "molecules", "pairs")}, indent=2))


if __name__ == "__main__":
    main()
