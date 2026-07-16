"""Tests for v3 record voting, majority labels, and endpoint/split firewalls."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.stages import pairs

RELEASE = Path("configs/assay_transfer/v3/release.yaml")


def _args(**values):
    obj = type("Args", (), {})()
    obj.__dict__.update(values)
    return obj


def _row(smiles, value, child, endpoint="oral_bioavailability.absolute.percent"):
    return {"child_id": child, "source_id": "starling", "canonical_smiles": smiles,
            "canonical_endpoint_key": endpoint, "endpoint_family": "oral_bioavailability",
            "endpoint_subtype": "absolute", "unit_basis": "percent", "scalar_value": float(value),
            "comparison_value": float(value), "assay_concept": "oral_bioavailability",
            "metric_type": "bounded_percentage", "transfer_max": 10.0,
            "not_transfer_min": 30.0, "threshold_display": "10/30"}


class PairsV3Test(unittest.TestCase):
    def _run(self, rows, splits):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pq.write_table(pa.Table.from_pylist(rows), root / "records.parquet")
            split_dir = root / "splits"
            split_dir.mkdir()
            mols = sorted({row["canonical_smiles"] for row in rows})
            pq.write_table(pa.table({"canonical_smiles": mols, "split": [splits[m] for m in mols]}),
                           split_dir / "molecule_splits.parquet")
            out = root / "pairs"
            report = pairs.build(_args(base=[root / "records.parquet"], split_dir=split_dir,
                                       release=str(RELEASE), output_dir=out,
                                       query_caps={"oral_bioavailability": 0}, max_queries=None))
            return report, pq.read_table(out / "pairs.parquet").to_pylist()

    def test_strict_majority_and_no_continuous_target(self):
        rows = [_row("CCO", 50, "a"), _row("CCCO", 52, "b1"),
                _row("CCCO", 51, "b2"), _row("CCCO", 90, "b3"),
                _row("c1ccccc1", 70, "c")]
        report, output = self._run(rows, {r["canonical_smiles"]: "train" for r in rows})
        candidate = next(r for r in output if r["retrieval_record_id"] == "a" and r["query_smiles"] == "CCCO")
        self.assertEqual((candidate["n_transfer"], candidate["n_nontransfer"]), (2, 1))
        self.assertEqual(candidate["binary_label"], 1)
        ambiguous = next(r for r in output if r["retrieval_record_id"] == "a" and r["query_smiles"] == "c1ccccc1")
        self.assertIsNone(ambiguous["binary_label"])
        self.assertNotIn("continuous_target", ambiguous)
        self.assertGreater(report["binary_label_counts"]["null"], 0)

    def test_split_firewall(self):
        rows = [_row("CCO", 50, "a"), _row("CCCO", 55, "b")]
        with self.assertRaises(RuntimeError):
            self._run(rows, {"CCO": "train", "CCCO": "test"})


if __name__ == "__main__":
    unittest.main()
