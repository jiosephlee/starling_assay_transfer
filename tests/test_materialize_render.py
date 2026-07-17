"""Tests for v3 hard-label materialization and three-column MCQA rendering."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.stages import materialize, render_hf


def _args(**values):
    obj = type("Args", (), {})()
    obj.__dict__.update(values)
    return obj


def _retrieval():
    row = {"child_id": "child-a", "parent_provenance_id": "parent-a", "record_id": "record-a",
           "input_sha256": "abc", "scalar_is_approximate": False,
           "source_smiles": "OCC", "canonical_smiles": "CCO",
           "canonical_endpoint_key": "q4.half_life.minute", "measurement_label": "fa x fg"}
    for name in render_hf.CONTEXT_NAMES:
        row[f"context_{name}"] = None
    row["context_study_or_assay_system"] = "human liver microsomes"
    return row


def _query_record():
    row = {"child_id": "child-q", "parent_provenance_id": "parent-q", "record_id": "record-q",
           "input_sha256": "def", "scalar_is_approximate": False,
           "source_smiles": "OCCC", "canonical_smiles": "CCCO",
           "canonical_endpoint_key": "q4.half_life.minute"}
    row.update({f"context_{name}": None for name in render_hf.CONTEXT_NAMES})
    return row


def _candidate(cid, split, label, query):
    return {"candidate_id": cid, "split": split, "canonical_endpoint_key": "q4.half_life.minute",
            "endpoint_family": "hepatic_metabolism", "endpoint_subtype": "half_life",
            "unit_basis": "minute", "assay_concept": "Fh", "k_profile": "same_endpoint",
            "setting_key": "q4.half_life.minute", "metric_type": "positive_scalar",
            "retrieval_record_id": "child-a", "retrieved_smiles": "CCO", "retrieved_source_id": "q4",
            "retrieved_value": 120.0, "query_smiles": query, "transfer_max": 0.30103,
            "not_transfer_min": 0.69897, "threshold_display": "within 2-fold / at least 5-fold apart",
            "n_records": 2, "n_transfer": 2 if label else 0, "n_nontransfer": 0 if label else 2,
            "n_ambiguous": 0, "transfer_fraction": float(label),
            "nontransfer_fraction": float(1 - label), "ambiguous_fraction": 0.0,
            "binary_label": label, "majority_margin": 0.5, "tanimoto": 0.2,
            "tanimoto_bucket": "low"}


class MaterializeRenderV3Test(unittest.TestCase):
    def _build(self, root: Path):
        base = root / "base"
        base.mkdir()
        pq.write_table(pa.Table.from_pylist([_retrieval(), _query_record()]), base / "records.parquet")
        selected = root / "selected"
        selected.mkdir()
        pq.write_table(pa.Table.from_pylist([_candidate("c1", "train", 1, "CCCO"),
                                             _candidate("c2", "validation", 0, "CCCCO")]),
                       selected / "selected.parquet")
        materialized = root / "materialized"
        materialize.build(_args(pairs=[selected], base=[base], output_dir=materialized))
        return materialized

    def test_rejects_null_binary_label(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base"
            base.mkdir()
            pq.write_table(pa.Table.from_pylist([_retrieval()]), base / "records.parquet")
            selected = root / "selected"
            selected.mkdir()
            row = _candidate("bad", "train", 1, "CCC")
            row["binary_label"] = None
            pq.write_table(pa.Table.from_pylist([row]), selected / "selected.parquet")
            with self.assertRaises(ValueError):
                materialize.build(_args(pairs=[selected], base=[base], output_dir=root / "out"))

    def test_soft_v4_renders_distribution_and_c_for_tied_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base"
            base.mkdir()
            pq.write_table(pa.Table.from_pylist([_retrieval(), _query_record()]), base / "records.parquet")
            selected = root / "selected"
            selected.mkdir()
            row = _candidate("soft", "train", 1, "CCCO")
            row.update({"binary_label": None, "n_records": 2, "n_transfer": 1,
                        "n_nontransfer": 1, "n_ambiguous": 0, "transfer_fraction": 0.5,
                        "nontransfer_fraction": 0.5, "ambiguous_fraction": 0.0,
                        "majority_margin": None})
            pq.write_table(pa.Table.from_pylist([row]), selected / "selected.parquet")
            materialized = root / "materialized"
            materialize.build(_args(pairs=[selected], base=[base], output_dir=materialized,
                                    allow_null_train=True))
            hf = root / "hf"
            info = render_hf.build(_args(dataset=materialized / "dataset.parquet",
                                         template_dir=Path("templates/assay_transfer_v4"),
                                         output_dir=hf, soft_evidence=True,
                                         target_policy_version="empirical_vote_distribution_v4"))
            output = pq.read_table(hf / "train/data.parquet").to_pylist()[0]
            self.assertEqual(output["completion"], "C")
            self.assertEqual(output["target_distribution"],
                             {"transfer": 0.5, "nontransfer": 0.5, "ambiguous": 0.0})
            self.assertEqual(info["top_level_features"],
                             ["prompt", "completion", "target_distribution", "metadata"])
            self.assertIn("(C) ambiguous evidence", output["prompt"])

    def test_soft_v4_rejects_null_heldout_label(self):
        row = _candidate("bad-val", "validation", 1, "CCCO")
        row["binary_label"] = None
        with self.assertRaises(ValueError):
            materialize._validate([row], allow_null_train=True)

    def test_hf_schema_and_mcqa_leakage_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            materialized = self._build(root)
            hf = root / "hf"
            info = render_hf.build(_args(dataset=materialized / "dataset.parquet",
                                         template_dir=Path("templates/assay_transfer_v3"), output_dir=hf))
            table = pq.read_table(hf / "train/data.parquet")
            self.assertEqual(table.column_names, ["prompt", "completion", "metadata"])
            row = table.to_pylist()[0]
            self.assertEqual(row["completion"], "A")
            self.assertIn("human liver microsomes", row["prompt"])
            self.assertNotIn("n_transfer", row["prompt"])
            self.assertNotIn("continuous_target", table.schema.names)
            self.assertEqual(info["completion_map"], {"1": "A", "0": "B"})
            self.assertEqual(row["metadata"]["retrieved_measurement_label"], "fa x fg")

    def test_renderer_accepts_release_schema_version(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            materialized = self._build(root)
            info = render_hf.build(_args(dataset=materialized / "dataset.parquet",
                                         template_dir=Path("templates/assay_transfer_v3"),
                                         output_dir=root / "hf",
                                         schema_version="assay_transfer_binary_v4"))
            table = pq.read_table(root / "hf/train/data.parquet")
            self.assertEqual(info["schema_version"], "assay_transfer_binary_v4")
            self.assertEqual(table["metadata"][0].as_py()["schema_version"],
                             "assay_transfer_binary_v4")

    def test_intern_variant_wraps_only_smiles_and_versions_template(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            materialized = self._build(root)
            hf = root / "hf-intern"
            render_hf.build(_args(dataset=materialized / "dataset.parquet",
                                  template_dir=Path("templates/assay_transfer_v3_intern"),
                                  template_variant="intern", output_dir=hf))
            row = pq.read_table(hf / "train/data.parquet").to_pylist()[0]
            self.assertIn("<SMILES>OCC</SMILES>", row["prompt"])
            self.assertIn("<SMILES>OCCC</SMILES>", row["prompt"])
            self.assertEqual(row["completion"], "(A)")
            self.assertEqual(row["metadata"]["template_id"], "Fh_intern_mcqa_v3")


if __name__ == "__main__":
    unittest.main()
