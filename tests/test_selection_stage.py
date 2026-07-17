"""Tests for v3 all-split hard-label selection and degree spreading."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.stages import selection


def _args(**values):
    obj = type("Args", (), {})()
    obj.__dict__.update(values)
    return obj


def _candidate(cid, split, query, retrieval, label, concept="Fa", bucket="low"):
    return {"candidate_id": cid, "split": split, "assay_concept": concept,
            "tanimoto_bucket": bucket, "query_smiles": query,
            "retrieved_smiles": retrieval, "retrieval_record_id": f"child-{cid}",
            "binary_label": label}


class SelectionV3Test(unittest.TestCase):
    def _run(self, rows, quota=20):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        candidates = root / "candidates"
        candidates.mkdir()
        pq.write_table(pa.Table.from_pylist(rows), candidates / "pairs.parquet")
        report = selection.build(_args(candidates=[candidates], output_dir=root / "selected",
                                       train_quota=quota, val_quota=quota, test_quota=quota,
                                       release=None))
        return root, report

    def test_null_labels_excluded_from_every_split(self):
        rows = [_candidate("t1", "train", "q1", "r1", 1),
                _candidate("t2", "train", "q2", "r2", None),
                _candidate("v1", "validation", "q3", "r3", 0),
                _candidate("v2", "validation", "q4", "r4", None)]
        root, report = self._run(rows)
        self.assertEqual(report["splits"]["train"]["eligible_candidates"], 1)
        self.assertEqual(report["splits"]["validation"]["eligible_candidates"], 1)
        selected = pq.read_table(root / "selected/selected/train/selected.parquet").to_pylist()
        self.assertTrue(all(row["binary_label"] in (0, 1) for row in selected))

    def test_soft_evidence_retains_null_only_for_training(self):
        rows = [_candidate("t1", "train", "q1", "r1", 1),
                _candidate("t2", "train", "q2", "r2", None),
                _candidate("v1", "validation", "q3", "r3", None)]
        train = selection._eligible_rows(rows[:2], "train", soft_evidence=True)
        validation = selection._eligible_rows(rows[2:], "validation", soft_evidence=True)
        self.assertEqual([row["candidate_id"] for row in train], ["t1", "t2"])
        self.assertEqual(validation, [])

    def test_frozen_heldout_reuses_exact_candidate_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = root / "selected" / "validation"
            frozen.mkdir(parents=True)
            rows = [_candidate("v2", "validation", "q2", "r2", 0),
                    _candidate("v1", "validation", "q1", "r1", 1)]
            for row in rows:
                row["canonical_endpoint_key"] = "endpoint"
            pq.write_table(pa.Table.from_pylist(list(reversed(rows))), frozen / "selected.parquet")
            selected = selection._frozen_heldout(rows, root, "validation")
            self.assertEqual([row["candidate_id"] for row in selected], ["v1", "v2"])

    def test_frozen_heldout_rejects_missing_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = root / "selected" / "test"
            frozen.mkdir(parents=True)
            row = _candidate("missing", "test", "q", "r", 0)
            row["canonical_endpoint_key"] = "endpoint"
            pq.write_table(pa.Table.from_pylist([row]), frozen / "selected.parquet")
            with self.assertRaises(ValueError):
                selection._frozen_heldout([], root, "test")

    def test_round_robin_spreads_query_degree(self):
        rows = [_candidate(f"h{i}", "train", "hot", f"r{i}", 1) for i in range(20)]
        rows += [_candidate(f"o{i}", "train", f"q{i}", f"r{i}", 1) for i in range(5)]
        root, _ = self._run(rows, quota=6)
        selected = pq.read_table(root / "selected/selected/train/selected.parquet").to_pylist()
        degrees = Counter(row["query_smiles"] for row in selected)
        self.assertLessEqual(degrees["hot"], 2)
        self.assertGreaterEqual(len(degrees), 5)

    def test_per_stratum_targets_can_take_all_sparse_rows(self):
        rows = [_candidate(f"a{i}", "train", f"q{i}", f"r{i}", 1,
                           concept="Fa", bucket="high") for i in range(3)]
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        candidates = root / "candidates"
        candidates.mkdir()
        pq.write_table(pa.Table.from_pylist(rows), candidates / "pairs.parquet")
        targets = {"train": {"Fa|high": None}}
        report = selection.build(_args(candidates=[candidates], output_dir=root / "selected",
                                       train_quota=3, val_quota=0, test_quota=0,
                                       release=None, stratum_targets=targets))
        self.assertEqual(report["splits"]["train"]["selected"], 3)
        self.assertEqual(report["splits"]["train"]["underfilled_strata"], {})


if __name__ == "__main__":
    unittest.main()
