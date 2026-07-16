#!/usr/bin/env python3
"""Split stage: composed base records -> molecule-first split assignment.

Molecules are assigned to ``train`` / ``validation`` / ``test`` *before* pairs are built
(``docs/assay_transfer_design.md`` section 12.1), over the **composed union** of every
base table in the build (``dataset_system_design.md`` sections 5.2, 7.1). Because
assignment is keyed on the canonical molecule structure, a molecule lands in one split
across every source and endpoint, so retrieval-side records can never leak a held-out
query molecule.

Assignment is a stable hash of ``seed:canonical_smiles`` mapped through the split ratios,
so it is deterministic and independent of row order or which sources are composed. Output:
``<output_dir>/molecule_splits.parquet`` (``canonical_smiles`` -> ``split``) +
``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SPLITS = ("train", "validation", "test")
MOLECULE_CANONICALIZATION_VERSION = "rdkit_isomeric_v1"


def canonical_smiles(smiles: str) -> Optional[str]:
    """RDKit canonical isomeric SMILES, or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def assign_split(smiles: str, seed: int, ratios: tuple[float, float, float]) -> str:
    """Deterministic split for one canonical molecule via a stable hash of seed:smiles."""
    digest = hashlib.sha1(f"{seed}:{smiles}".encode()).hexdigest()
    frac = int(digest[:12], 16) / float(1 << 48)  # in [0, 1)
    train, val, _ = ratios
    if frac < train:
        return "train"
    if frac < train + val:
        return "validation"
    return "test"


def _base_file(path: Path) -> Path:
    if not path.is_dir():
        return path
    records = path / "records.parquet"
    return records if records.exists() else path / "base.parquet"


def _collect_molecules(base_paths: list[Path]) -> tuple[dict[str, str], Counter]:
    """Map raw smiles -> canonical smiles across all base tables; count parse failures."""
    canon: dict[str, str] = {}
    stats: Counter = Counter()
    for path in base_paths:
        schema = pq.read_schema(path)
        canonical_input = "canonical_smiles" in schema.names
        if canonical_input:
            stats["canonical_input_tables"] += 1
        column = "canonical_smiles" if canonical_input else "smiles"
        table = pq.read_table(path, columns=[column])
        for raw in table.column(column).to_pylist():
            if raw is None or not str(raw).strip():
                stats["missing_smiles"] += 1
                continue
            raw = str(raw).strip()
            if raw in canon:
                continue
            c = raw if canonical_input else canonical_smiles(raw)
            if c is None:
                stats["uncanonicalizable"] += 1
                # Fall back to the raw string so the molecule is still splittable.
                canon[raw] = raw
            else:
                canon[raw] = c
    stats["raw_smiles"] = len(canon)
    return canon, stats


def build(args: argparse.Namespace) -> dict[str, Any]:
    ratios = (args.train_frac, args.val_frac, args.test_frac)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")
    base_paths = [_base_file(Path(p)) for p in args.base]
    for p in base_paths:
        if not p.exists():
            raise FileNotFoundError(f"base parquet not found: {p}")

    raw_to_canon, stats = _collect_molecules(base_paths)
    # One split per canonical molecule (joint across raw spellings and sources).
    canon_split: dict[str, str] = {}
    for canon in set(raw_to_canon.values()):
        canon_split[canon] = assign_split(canon, args.seed, ratios)

    raw = sorted(raw_to_canon)
    table = pa.table(
        {
            "smiles": pa.array(raw, type=pa.string()),
            "canonical_smiles": pa.array([raw_to_canon[r] for r in raw], type=pa.string()),
            "split": pa.array([canon_split[raw_to_canon[r]] for r in raw], type=pa.string()),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output_dir / "molecule_splits.parquet", compression="zstd")

    split_counts = Counter(canon_split.values())
    manifest = {
        "stage": "split",
        "base_inputs": [str(p) for p in base_paths],
        "molecule_canonicalization_version": (
            "canonical_base_passthrough_v1" if stats.get("canonical_input_tables") == len(base_paths)
            else MOLECULE_CANONICALIZATION_VERSION
        ),
        "split_seed": args.seed,
        "split_ratios": {"train": ratios[0], "validation": ratios[1], "test": ratios[2]},
        "unique_canonical_molecules": len(canon_split),
        "unique_raw_smiles": len(raw_to_canon),
        "molecules_per_split": {s: split_counts.get(s, 0) for s in SPLITS},
        "uncanonicalizable_smiles": stats.get("uncanonicalizable", 0),
        "missing_smiles_rows": stats.get("missing_smiles", 0),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, nargs="+", required=True, help="Base dirs or parquets to compose.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
