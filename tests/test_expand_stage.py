"""Tests for the v2 expand stage: frozen ledger + nested-prefix containment."""

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

from pipeline.stages import expand  # noqa: E402


def _Args(**kw):
    ns = type("Args", (), {})()
    ns.__dict__.update(kw)
    return ns


def _cand(i, concept="Fa", bucket="low", split="train"):
    return {
        "candidate_id": f"c{i:04d}",
        "split": split,
        "assay_concept": concept,
        "tanimoto_bucket": bucket,
        "query_smiles": f"q{i % 10}",
        "retrieved_smiles": f"r{i}",
        "retrieval_record_id": f"rec{i}",
        "continuous_target": 1.0,
        "binary_label": (1 if i % 2 == 0 else None),
    }


class ExpandStageTest(unittest.TestCase):
    def test_nested_prefix_containment(self) -> None:
        # 60 train candidates across 2 strata (Fa|low, Fh|low).
        rows = [_cand(i, concept=("Fa" if i < 30 else "Fh")) for i in range(60)]
        rows += [_cand(i, split="validation") for i in range(100, 105)]  # ignored by expand
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cand = d / "cand"
            cand.mkdir()
            pq.write_table(pa.Table.from_pylist(rows), cand / "pairs.parquet")
            out = d / "expand"
            manifest = expand.build(_Args(candidates=[cand], output_dir=out, prefixes=[10, 20, 40]))

            self.assertTrue(manifest["nested_containment_holds"])
            self.assertEqual(manifest["train_eligible_candidates"], 60)  # validation excluded

            prefixes = json.loads((out / "prefixes.json").read_text())
            ids10 = set(prefixes["10"]["candidate_ids"])
            ids20 = set(prefixes["20"]["candidate_ids"])
            ids40 = set(prefixes["40"]["candidate_ids"])
            self.assertTrue(ids10 <= ids20 <= ids40)   # strict nesting
            self.assertLess(len(ids10), len(ids40))

            # ledger covers every train candidate exactly once.
            ledger = pq.read_table(out / "ledger.parquet").to_pylist()
            self.assertEqual(len({r["candidate_id"] for r in ledger}), 60)


if __name__ == "__main__":
    unittest.main()
