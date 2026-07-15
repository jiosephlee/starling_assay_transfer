"""Tests for the materialize and render_hf stages."""

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

from pipeline.stages import materialize, render_hf  # noqa: E402


def _Args(**kw):
    ns = type("Args", (), {})()
    ns.__dict__.update(kw)
    return ns


def _write_base(path):
    pq.write_table(
        pa.table(
            {
                "smiles": ["mA"],
                "source_id": ["q4"],
                "canonical_endpoint_id": ["q4.metabolic_half_life"],
                "canonical_endpoint_key": ["q4.metabolic_half_life"],
                "metric_type": ["half_life"],
                "property_value": [2.0],
                "property_value_native": [120.0],
                "species_exact": ["human"],
                "assay_system": ["human liver microsomes"],
            }
        ),
        path,
    )


def _write_pairs(path):
    pq.write_table(
        pa.table(
            {
                "split": ["train", "validation"],
                "canonical_endpoint_id": ["q4.metabolic_half_life"] * 2,
                "k_profile": ["same_endpoint"] * 2,
                "setting_key": ["", ""],
                "retrieved_row_index": [0, 0],
                "retrieved_smiles": ["mA", "mA"],
                "retrieved_value": [2.0, 2.0],
                "retrieved_value_native": [120.0, 120.0],
                "query_smiles": ["mB", "mC"],
                "query_value_mean": [2.1, 3.0],
                "query_n": [1, 1],
                "query_std": [0.0, 0.0],
                "metric_type": ["half_life", "half_life"],
                "transfer_label": ["transfer", "not_transfer"],
            }
        ),
        path,
    )


class MaterializeRenderTest(unittest.TestCase):
    def test_materialize_inlines_retrieved_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_base(d / "base.parquet")
            _write_pairs(d / "pairs.parquet")
            out = d / "mat"
            manifest = materialize.build(_Args(pairs=d / "pairs.parquet", base=[d / "base.parquet"], output_dir=out))
            self.assertEqual(manifest["rows"], 2)
            recs = pq.read_table(out / "dataset.parquet").to_pylist()
            # Z_A inlined from base, prefixed retrieved_*; not the pair's own value/index.
            self.assertEqual(recs[0]["retrieved_assay_system"], "human liver microsomes")
            self.assertEqual(recs[0]["retrieved_species_exact"], "human")
            self.assertNotIn("retrieved_property_value", recs[0])  # already on pair

    def test_render_hf_splits_and_renders(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_base(d / "base.parquet")
            _write_pairs(d / "pairs.parquet")
            mat = d / "mat"
            materialize.build(_Args(pairs=d / "pairs.parquet", base=[d / "base.parquet"], output_dir=mat))
            hf = d / "hf"
            info = render_hf.build(
                _Args(dataset=mat / "dataset.parquet", template=render_hf.DEFAULT_TEMPLATE, output_dir=hf)
            )
            self.assertEqual(info["rows_per_split"], {"train": 1, "validation": 1})
            train = pq.read_table(hf / "train" / "data.parquet").to_pylist()[0]
            self.assertIn("Query molecule structure", train["prompt"])
            self.assertIn("human liver microsomes", train["prompt"])
            self.assertEqual(train["completion"], " transfer")
            # query aggregates must not leak into the prompt (labels/diagnostics only).
            self.assertNotIn("query_value_mean", train["prompt"])


if __name__ == "__main__":
    unittest.main()
