"""Tests for the molecule-first split stage (requires RDKit)."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from pipeline.stages import split as split_stage

    _HAS_RDKIT = True
except Exception:  # pragma: no cover - env without rdkit
    _HAS_RDKIT = False


def _Args(**kw):
    ns = type("Args", (), {})()
    ns.__dict__.update(kw)
    return ns


@unittest.skipUnless(_HAS_RDKIT, "rdkit not available in this interpreter")
class SplitStageTest(unittest.TestCase):
    def test_assignment_is_deterministic_and_partitions(self) -> None:
        ratios = (0.8, 0.1, 0.1)
        a1 = split_stage.assign_split("CCO", 17, ratios)
        a2 = split_stage.assign_split("CCO", 17, ratios)
        self.assertEqual(a1, a2)
        self.assertIn(a1, split_stage.SPLITS)
        # different seed can move the molecule
        seeds = {split_stage.assign_split("CCO", s, ratios) for s in range(50)}
        self.assertGreater(len(seeds), 1)

    def test_same_molecule_one_split_across_sources(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            # Two "sources" sharing a molecule (ethanol) with different raw spellings.
            pq.write_table(pa.table({"smiles": ["OCC", "c1ccccc1"]}), d / "b1.parquet")
            pq.write_table(pa.table({"smiles": ["CCO", "CCCCO"]}), d / "b2.parquet")
            out = d / "splits"
            manifest = split_stage.build(
                _Args(
                    base=[d / "b1.parquet", d / "b2.parquet"],
                    output_dir=out,
                    seed=17,
                    train_frac=0.8,
                    val_frac=0.1,
                    test_frac=0.1,
                )
            )
            table = pq.read_table(out / "molecule_splits.parquet").to_pylist()
            by_raw = {r["smiles"]: r for r in table}
            # OCC and CCO both canonicalize to ethanol -> identical canonical + split.
            self.assertEqual(by_raw["OCC"]["canonical_smiles"], by_raw["CCO"]["canonical_smiles"])
            self.assertEqual(by_raw["OCC"]["split"], by_raw["CCO"]["split"])
            self.assertEqual(manifest["unique_canonical_molecules"], 3)  # ethanol, benzene, butanol

    def test_ratios_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            split_stage.build(
                _Args(base=[Path("x")], output_dir=Path("y"), seed=1, train_frac=0.5, val_frac=0.1, test_frac=0.1)
            )


if __name__ == "__main__":
    unittest.main()
