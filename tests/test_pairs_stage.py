"""Tests for the pairs stage: endpoint firewall, labeling, setting exclusion."""

from __future__ import annotations

import json
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
from pipeline.stages import pairs  # noqa: E402


def _Args(**kw):
    ns = type("Args", (), {})()
    ns.__dict__.update(kw)
    return ns


def _base_row(smiles, endpoint, metric, value, species="human", **conditions):
    row = {
        "smiles": smiles,
        "source_id": "q2",
        "canonical_endpoint_id": endpoint,
        "canonical_endpoint_key": endpoint,
        "metric_type": metric,
        "property_value": float(value),
        "property_value_native": float(value),
        "species_exact": species,
    }
    for c in CONDITION_COLUMNS:
        row[c] = conditions.get(c)
    return row


def _write_base(path, rows):
    cols = list(rows[0].keys())
    pq.write_table(pa.table({c: pa.array([r[c] for r in rows]) for c in cols}), path)


def _write_split(path, molecules, split="train"):
    pq.write_table(
        pa.table({"smiles": molecules, "canonical_smiles": molecules, "split": [split] * len(molecules)}),
        path,
    )


class PairsStageTest(unittest.TestCase):
    def _run(self, rows, molecules, profile, tmp):
        base = Path(tmp) / "base.parquet"
        split_dir = Path(tmp) / "splits"
        split_dir.mkdir()
        _write_base(base, rows)
        _write_split(split_dir / "molecule_splits.parquet", molecules)
        out = Path(tmp) / "pairs"
        manifest = pairs.build(
            _Args(base=[base], split_dir=split_dir, profile=profile, output_dir=out, seed=1, max_queries=64)
        )
        table = pq.read_table(out / "pairs.parquet").to_pylist()
        return manifest, table

    def test_firewall_and_labels_same_endpoint(self) -> None:
        rows = [
            _base_row("m1", "q2.fraction_absorbed.percent", "bounded_percentage", 50),
            _base_row("m2", "q2.fraction_absorbed.percent", "bounded_percentage", 55),  # close -> transfer
            _base_row("m3", "q2.fraction_absorbed.percent", "bounded_percentage", 90),  # far -> not_transfer
            _base_row("m1", "q4.metabolic_half_life", "half_life", 2.0),  # different endpoint
        ]
        with tempfile.TemporaryDirectory() as tmp:
            manifest, table = self._run(rows, ["m1", "m2", "m3"], "same_endpoint", tmp)
            # No pair crosses an endpoint.
            self.assertTrue(all(p["canonical_endpoint_id"] == "q2.fraction_absorbed.percent" for p in table))
            # m1(50) -> m2(55) transfer ; m1(50) -> m3(90) not_transfer.
            by_pair = {(p["retrieved_smiles"], p["query_smiles"]): p["transfer_label"] for p in table}
            self.assertEqual(by_pair[("m1", "m2")], "transfer")
            self.assertEqual(by_pair[("m1", "m3")], "not_transfer")
            # self-pairs excluded.
            self.assertNotIn(("m1", "m1"), by_pair)

    def test_same_species_excludes_null_species(self) -> None:
        rows = [
            _base_row("m1", "q2.fraction_absorbed.percent", "bounded_percentage", 50, species="human"),
            _base_row("m2", "q2.fraction_absorbed.percent", "bounded_percentage", 52, species=None),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):  # m2 excluded -> no cross-molecule query
                self._run(rows, ["m1", "m2"], "same_species_same_endpoint", tmp)

    def test_within_split_no_leakage(self) -> None:
        rows = [
            _base_row("m1", "q2.fraction_absorbed.percent", "bounded_percentage", 50),
            _base_row("m2", "q2.fraction_absorbed.percent", "bounded_percentage", 55),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.parquet"
            split_dir = Path(tmp) / "splits"
            split_dir.mkdir()
            _write_base(base, rows)
            # m1 train, m2 test -> they must never pair.
            pq.write_table(
                pa.table({"smiles": ["m1", "m2"], "canonical_smiles": ["m1", "m2"], "split": ["train", "test"]}),
                split_dir / "molecule_splits.parquet",
            )
            with self.assertRaises(RuntimeError):
                pairs.build(
                    _Args(base=[base], split_dir=split_dir, profile="same_endpoint",
                          output_dir=Path(tmp) / "pairs", seed=1, max_queries=64)
                )


if __name__ == "__main__":
    unittest.main()
