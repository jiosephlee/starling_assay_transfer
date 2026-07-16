"""Tests for the v2 selection stage: continuous-vs-hard-binary pools, strata, disjointness."""

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

from pipeline.stages import selection  # noqa: E402


def _Args(**kw):
    ns = type("Args", (), {})()
    ns.__dict__.update(kw)
    return ns


def _cand(cid, split, concept, bucket, q, r, binary, cont=1.0):
    return {
        "candidate_id": cid,
        "split": split,
        "assay_concept": concept,
        "tanimoto_bucket": bucket,
        "canonical_endpoint_id": "q2.fraction_absorbed.percent",
        "query_smiles": q,
        "retrieved_smiles": r,
        "retrieval_record_id": f"rec_{r}",
        "binary_label": binary,          # 1 / 0 / None
        "continuous_target": cont,
    }


class SelectionStageTest(unittest.TestCase):
    def _write(self, rows, path):
        pq.write_table(pa.Table.from_pylist(rows), path)

    def test_pools_and_disjointness(self) -> None:
        rows = [
            # train: 2 hard-binary + 2 null-binary (all have continuous targets)
            _cand("t1", "train", "Fa", "low", "mA", "mR1", 1),
            _cand("t2", "train", "Fa", "low", "mB", "mR2", 0),
            _cand("t3", "train", "Fa", "high", "mC", "mR3", None),
            _cand("t4", "train", "Fh", "low", "mD", "mR4", None),
            # validation: 2 hard-binary + 1 null (null must be excluded)
            _cand("v1", "validation", "Fa", "low", "mV1", "mVR1", 1),
            _cand("v2", "validation", "Fh", "low", "mV2", "mVR2", 0),
            _cand("v3", "validation", "Fa", "low", "mV3", "mVR3", None),
        ]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cand_dir = d / "cand"
            cand_dir.mkdir()
            self._write(rows, cand_dir / "pairs.parquet")
            out = d / "sel"
            report = selection.build(
                _Args(candidates=[cand_dir], output_dir=out, train_quota=10, val_quota=10, test_quota=10)
            )
            # train pool = all 4 (continuous), incl the 2 null-binary.
            self.assertEqual(report["splits"]["train"]["eligible_candidates"], 4)
            self.assertEqual(report["splits"]["train"]["selected"], 4)
            # validation pool = only the 2 hard-binary (null excluded).
            self.assertEqual(report["splits"]["validation"]["eligible_candidates"], 2)
            val = pq.read_table(out / "selected" / "validation" / "selected.parquet").to_pylist()
            self.assertTrue(all(r["binary_label"] is not None for r in val))
            # train contains null-binary rows (continuous-only training rows).
            train = pq.read_table(out / "selected" / "train" / "selected.parquet").to_pylist()
            self.assertTrue(any(r["binary_label"] is None for r in train))
            # per-stratum accounting present.
            self.assertIn("Fa|low", report["splits"]["train"]["per_stratum_selected"])

    def test_degree_control_spreads_across_query_molecules(self) -> None:
        # One stratum, one high-degree query molecule vs several others; round-robin should
        # not let the single molecule dominate a tiny quota.
        rows = [_cand(f"h{i}", "train", "Fa", "low", "hot", f"r{i}", 1) for i in range(20)]
        rows += [_cand(f"o{i}", "train", "Fa", "low", f"q{i}", f"r{i}", 1) for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cand_dir = d / "cand"
            cand_dir.mkdir()
            self._write(rows, cand_dir / "pairs.parquet")
            report = selection.build(
                _Args(candidates=[cand_dir], output_dir=d / "sel", train_quota=6, val_quota=0, test_quota=0)
            )
            train = pq.read_table(d / "sel" / "selected" / "train" / "selected.parquet").to_pylist()
            from collections import Counter
            per_q = Counter(r["query_smiles"] for r in train)
            # 'hot' must not take all 6 slots — round-robin gives the 5 other molecules a turn.
            self.assertLessEqual(per_q["hot"], 2)
            self.assertGreaterEqual(len(per_q), 5)


if __name__ == "__main__":
    unittest.main()
