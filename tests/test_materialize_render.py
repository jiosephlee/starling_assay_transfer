"""Tests for the v2 materialize and render_hf stages (continuous primary + nullable binary)."""

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


def _selected():
    # Two candidates: one hard-binary (train), one null-binary (train), one hard (validation).
    base = dict(
        canonical_endpoint_id="q4.metabolic_half_life",
        k_profile="same_endpoint",
        setting_key="",
        metric_type="half_life",
        retrieval_record_id="e0",
        retrieved_row_index=0,
        retrieved_smiles="mA",
        retrieved_source_id="q4",
        retrieved_value=2.0,
        retrieved_value_native=120.0,
        n_records=3,
        n_transfer=2,
        n_nontransfer=0,
        n_ambiguous=1,
        transfer_fraction=0.66,
        nontransfer_fraction=0.0,
        ambiguous_fraction=0.33,
        majority_side="transfer",
        majority_margin=0.16,
        dist_std=0.1,
        dist_median=0.2,
        dist_max=0.3,
        tanimoto=0.2,
    )
    return [
        {**base, "candidate_id": "c1", "split": "train", "assay_concept": "Fh",
         "tanimoto_bucket": "low", "query_smiles": "mB", "binary_label": 1, "continuous_target": 0.2},
        {**base, "candidate_id": "c2", "split": "train", "assay_concept": "Fh",
         "tanimoto_bucket": "low", "query_smiles": "mC", "binary_label": None, "continuous_target": 0.5},
        {**base, "candidate_id": "c3", "split": "validation", "assay_concept": "Fh",
         "tanimoto_bucket": "low", "query_smiles": "mD", "binary_label": 0, "continuous_target": 0.9},
    ]


class MaterializeRenderV2Test(unittest.TestCase):
    def _materialize(self, d: Path):
        _write_base(d / "base.parquet")
        sel_dir = d / "sel"
        sel_dir.mkdir()
        pq.write_table(pa.Table.from_pylist(_selected()), sel_dir / "selected.parquet")
        out = d / "mat"
        report = materialize.build(_Args(pairs=[sel_dir], base=[d / "base.parquet"], output_dir=out))
        return report, out

    def test_materialize_inlines_metadata_and_keeps_null_binary(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            report, out = self._materialize(Path(d))
            self.assertEqual(report["rows"], 3)
            self.assertEqual(report["rows_by_split"], {"train": 2, "validation": 1})
            recs = pq.read_table(out / "dataset.parquet").to_pylist()
            self.assertEqual(recs[0]["retrieved_assay_system"], "human liver microsomes")
            # continuous target present on all; binary nullable.
            self.assertTrue(all(r["continuous_target"] is not None for r in recs))
            self.assertTrue(any(r["binary_label"] is None for r in recs))

    def test_materialize_requires_v2_fields(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_base(d / "base.parquet")
            sel = d / "sel"
            sel.mkdir()
            pq.write_table(pa.table({"query_smiles": ["x"], "retrieved_row_index": [0]}),
                           sel / "selected.parquet")
            with self.assertRaises(ValueError):
                materialize.build(_Args(pairs=[sel], base=[d / "base.parquet"], output_dir=d / "mat"))

    def test_render_hf_continuous_primary_nullable_binary(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _, mat = self._materialize(d)
            hf = d / "hf"
            info = render_hf.build(_Args(dataset=mat / "dataset.parquet",
                                         template=render_hf.DEFAULT_TEMPLATE, output_dir=hf))
            self.assertEqual(info["primary_target"], "continuous_target")
            self.assertEqual(info["rows_per_split"], {"train": 2, "validation": 1})
            train = pq.read_table(hf / "train" / "data.parquet").to_pylist()
            # continuous target always present; the null-binary row has no completion.
            self.assertTrue(all(r["continuous_target"] is not None for r in train))
            null_row = next(r for r in train if r["binary_label"] is None)
            self.assertIsNone(null_row["completion"])
            hard_row = next(r for r in train if r["binary_label"] == "transfer")
            self.assertEqual(hard_row["completion"], " transfer")
            self.assertIn("human liver microsomes", hard_row["prompt"])


if __name__ == "__main__":
    unittest.main()
