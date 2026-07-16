"""Tests for the v2 pairs stage: record votes, strict majority, continuous target, firewall."""

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

from pipeline.normalize.conditions import CONDITION_COLUMNS  # noqa: E402

try:
    from pipeline.stages import pairs  # imports FingerprintCache (rdkit)

    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    _HAS_RDKIT = False


def _Args(**kw):
    ns = type("Args", (), {})()
    ns.__dict__.update(kw)
    return ns


def _row(smiles, value, endpoint="q2.fraction_absorbed.percent", metric="bounded_percentage",
         species="human", ext="e0"):
    r = {
        "smiles": smiles,
        "source_id": "q2",
        "canonical_endpoint_id": endpoint,
        "canonical_endpoint_key": endpoint,
        "metric_type": metric,
        "property_value": float(value),
        "property_value_native": float(value),
        "species_exact": species,
        "pmid": "1",
        "extraction_id": ext,
    }
    for c in CONDITION_COLUMNS:
        r[c] = None
    return r


def _write_base(path, rows):
    cols = list(rows[0].keys())
    pq.write_table(pa.table({c: pa.array([r[c] for r in rows]) for c in cols}), path)


def _write_split(path, molecules, split="train"):
    pq.write_table(
        pa.table({"smiles": molecules, "canonical_smiles": molecules, "split": [split] * len(molecules)}),
        path,
    )


@unittest.skipUnless(_HAS_RDKIT, "rdkit not available")
class PairsV2Test(unittest.TestCase):
    def _run(self, rows, molecules, splits, profile="same_endpoint"):
        tmp = Path(self._tmp.name)
        base = tmp / "base.parquet"
        split_dir = tmp / "splits"
        split_dir.mkdir(exist_ok=True)
        _write_base(base, rows)
        sm = pa.table({"smiles": molecules, "canonical_smiles": molecules, "split": splits})
        pq.write_table(sm, split_dir / "molecule_splits.parquet")
        out = tmp / f"pairs_{profile}"
        manifest = pairs.build(
            _Args(base=[base], split_dir=split_dir, profile=profile, output_dir=out,
                  split_version="v2", max_queries=0)
        )
        table = pq.read_table(out / "pairs.parquet").to_pylist()
        return manifest, table

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_votes_majority_and_continuous(self) -> None:
        # CC=ethane, CCO=ethanol, CCCO=propanol, c1ccccc1=benzene; use valid SMILES so rdkit fps work.
        rows = [
            _row("CCO", 50, ext="a"),                # retrieved m1=50
            _row("CCCO", 52, ext="b1"),              # query m2 records: 52,51,90
            _row("CCCO", 51, ext="b2"),
            _row("CCCO", 90, ext="b3"),
            _row("c1ccccc1", 70, ext="c"),           # query m3 single: 70 (ambiguous vs 50)
            # a different endpoint for m1 -> must never pair with the % endpoint (firewall)
            _row("CCO", 2.0, endpoint="q4.metabolic_half_life", metric="half_life", ext="d"),
        ]
        mols = ["CCO", "CCCO", "c1ccccc1"]
        _, table = self._run(rows, mols, ["train"] * 3)
        by = {(r["retrieved_smiles"], r["query_smiles"]): r for r in table}

        # firewall: no candidate crosses endpoints.
        self.assertTrue(all(r["canonical_endpoint_id"] == "q2.fraction_absorbed.percent" for r in table))

        # m1(50) -> m2 evidence [52,51,90]: votes t,t,nt -> N=3 n_t=2 -> majority transfer(1).
        c = by[("CCO", "CCCO")]
        self.assertEqual(c["n_records"], 3)
        self.assertEqual(c["n_transfer"], 2)
        self.assertEqual(c["n_nontransfer"], 1)
        self.assertEqual(c["n_ambiguous"], 0)
        self.assertEqual(c["binary_label"], 1)
        self.assertEqual(c["majority_side"], "transfer")
        self.assertAlmostEqual(c["continuous_target"], (2 + 1 + 40) / 3, places=6)  # mean of |50-y|

        # m1(50) -> m3 single 70: d=20 ambiguous -> null binary, but continuous target kept.
        c2 = by[("CCO", "c1ccccc1")]
        self.assertEqual(c2["n_records"], 1)
        self.assertEqual(c2["n_ambiguous"], 1)
        self.assertIsNone(c2["binary_label"])
        self.assertAlmostEqual(c2["continuous_target"], 20.0)
        self.assertIn(c2["tanimoto_bucket"], ("low", "high"))
        self.assertEqual(c2["assay_concept"], "Fa")

    def test_within_split_no_leakage(self) -> None:
        rows = [_row("CCO", 50, ext="a"), _row("CCCO", 55, ext="b")]
        with self.assertRaises(RuntimeError):  # m1 train, m2 test -> no in-split partner
            self._run(rows, ["CCO", "CCCO"], ["train", "test"])


if __name__ == "__main__":
    unittest.main()
