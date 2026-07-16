"""Policy, concept-routing, expansion, and function-size tests for v3."""

from __future__ import annotations

import ast
import tarfile
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.stages import expand_v3
from pipeline.v3_policy import V3Policies

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "configs/assay_transfer/v3/release.yaml"
V4_RELEASE = ROOT / "configs/assay_transfer/v4/release.yaml"


class V3PolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = V3Policies(RELEASE)

    def test_q1_family_creates_two_concepts(self):
        common = {"source_id": "q1"}
        self.assertEqual(self.policy.concept_for({**common, "endpoint_family": "oral_exposure"}),
                         "oral_exposure")
        self.assertEqual(self.policy.concept_for({**common, "endpoint_family": "oral_bioavailability"}),
                         "oral_bioavailability")

    def test_exact_ratio_rule_precedes_oral_exposure_wildcard(self):
        row = {"endpoint_family": "oral_exposure", "endpoint_subtype": "oral_iv_ratio",
               "unit_basis": "dimensionless_ratio"}
        self.assertEqual(self.policy.metric_for(row).name, "dimensionless_ratio")

    def test_metric_domains(self):
        bounded = self.policy.metric_for({"endpoint_family": "oral_bioavailability",
                                          "endpoint_subtype": "absolute", "unit_basis": "percent"})
        scalar = self.policy.metric_for({"endpoint_family": "oral_exposure",
                                         "endpoint_subtype": "cmax", "unit_basis": "ng_ml"})
        self.assertIsNone(bounded.transform_value(101.0))
        self.assertIsNone(scalar.transform_value(0.0))
        self.assertIsNotNone(scalar.transform_value(0.001))

    def test_sparse_strata_policy_is_explicit(self):
        sparse = self.policy.sampling["sparse_strata"]
        self.assertEqual(sparse["train_all_available"], ["Fa|high", "Fg|high", "Fh|high"])
        self.assertEqual(sparse["heldout_matched_min"], ["Fa|high", "Fg|high"])

    def test_v4_uses_base_v2_and_retains_all_enumerated_train_strata(self):
        policy = V3Policies(V4_RELEASE)
        sparse = policy.sampling["sparse_strata"]
        self.assertEqual(policy.release["base_schema_version"], "canonical_endpoints_v2")
        self.assertEqual(policy.sampling["quotas"], {"train": 0, "validation": 2000, "test": 2000})
        self.assertEqual(len(sparse["train_all_available"]), 10)
        self.assertEqual(sparse["heldout_matched_min"], [])
        self.assertNotIn("heldout_backfill_within_concept", sparse)
        self.assertEqual(policy.sampling["split"], {"train": 0.60, "validation": 0.20, "test": 0.20})


class ExpansionBundleTest(unittest.TestCase):
    def test_bundle_contains_only_train_records_and_policy_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            eligible, splits, selection = root / "eligible", root / "splits", root / "selection"
            eligible.mkdir(); splits.mkdir(); (selection / "selected/train").mkdir(parents=True)
            rows = [{"child_id": "a", "canonical_smiles": "CCO"},
                    {"child_id": "b", "canonical_smiles": "CCCO"}]
            pq.write_table(pa.Table.from_pylist(rows), eligible / "records.parquet")
            pq.write_table(pa.table({"canonical_smiles": ["CCO", "CCCO"],
                                     "split": ["train", "test"]}), splits / "molecule_splits.parquet")
            pq.write_table(pa.table({"candidate_id": ["c1"]}),
                           selection / "selected/train/selected.parquet")
            report = expand_v3.build(Namespace(eligible=eligible, split_dir=splits,
                                                selection_dir=selection, release=str(RELEASE),
                                                template_dir=ROOT / "templates/assay_transfer_v3",
                                                output_dir=root / "out", query_caps={"Fa": 16}))
            self.assertEqual(report["eligible_train_records"], 1)
            with tarfile.open(report["archive"], "r:gz") as archive:
                names = set(archive.getnames())
            self.assertIn("release.yaml", names)
            self.assertIn("eligible_train_records.parquet", names)


class V3FunctionLengthTest(unittest.TestCase):
    def test_v3_functions_are_at_most_sixty_lines(self):
        paths = [ROOT / "pipeline/v3_policy.py"] + list((ROOT / "pipeline/stages").glob("*.py"))
        touched = {"compose_v3.py", "split.py", "pairs.py", "selection.py", "materialize.py",
                   "render_hf.py", "expand_v3.py", "run.py"}
        violations = []
        for path in paths:
            if path.name not in touched and path.name != "v3_policy.py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    length = (node.end_lineno or node.lineno) - node.lineno + 1
                    if length > 60:
                        violations.append(f"{path.name}:{node.name}:{length}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
