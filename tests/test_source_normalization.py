from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.source_normalization.artifacts import combination_fields  # noqa: E402
from pipeline.source_normalization.measurement import (  # noqa: E402
    approved_category,
    parse_measurement,
)
from pipeline.source_normalization.normalize import (  # noqa: E402
    annotate_duplicates,
    normalize_row,
    primary_reason,
    stable_record_id,
)
from pipeline.source_normalization.structures import (  # noqa: E402
    canonicalize_smiles,
    collapse_mapping,
    compare_exported_smiles,
    tdc_match,
)
from pipeline.source_normalization.io import (  # noqa: E402
    load_config, load_identifier_reconciliations, load_tdc_exclusions,
)
from pipeline.source_normalization.fg import split_explicit_fg_measurements  # noqa: E402
from pipeline.source_normalization.endpoint_keys import assign_canonical_endpoint  # noqa: E402
from pipeline.source_normalization.text import normalize_lexical, normalize_unit  # noqa: E402


class StructureNormalizationTest(unittest.TestCase):
    def test_identical_mapping_duplicates_collapse(self) -> None:
        mapping, duplicates = collapse_mapping([("id", "CCO"), ("id", "CCO")])
        self.assertEqual(mapping, {"id": "CCO"})
        self.assertEqual(duplicates, 1)

    def test_conflicting_mapping_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting"):
            collapse_mapping([("id", "CCO"), ("id", "CCN")])

    def test_canonicalization_and_wildcard_rejection(self) -> None:
        self.assertEqual(canonicalize_smiles("C(C)O"), ("CCO", None))
        self.assertEqual(canonicalize_smiles("*CC")[1], "wildcard_structure")
        self.assertEqual(canonicalize_smiles("not smiles")[1], "invalid_structure")

    def test_raw_and_canonical_tdc_matches(self) -> None:
        exclusions = {"C(C)O", "CCN"}
        self.assertTrue(tdc_match("C(C)O", "CCO", exclusions))
        self.assertTrue(tdc_match("NCC", "CCN", exclusions))
        self.assertFalse(tdc_match("CCC", "CCC", exclusions))

    def test_v2_excludes_only_heldout_tdc_splits(self) -> None:
        config = load_config(REPO_ROOT / "configs/source_normalization_v2.yaml")
        spec = config["references"]["tdc_exclusion"]
        path = REPO_ROOT / spec["path"]
        exclusions, stats = load_tdc_exclusions(path, spec["included_splits"])
        self.assertEqual(config["artifact_schema_version"], "canonical_endpoints_v2")
        self.assertEqual(stats["source_rows"], 640)
        self.assertEqual(stats["rows"], 192)
        self.assertEqual(stats["splits"], ["test", "valid"])
        self.assertEqual(len(exclusions), 192)

    def test_v3_identifier_reconciliation_is_explicit_and_excludes_ambiguous_rows(self) -> None:
        config = load_config(REPO_ROOT / "configs/source_normalization_v3.yaml")
        spec = config["references"]["identifier_reconciliation"]
        mappings, stats = load_identifier_reconciliations(REPO_ROOT / spec["path"])
        self.assertEqual(stats["rows"], 37)
        self.assertEqual(stats["methods"], {"same_pmid_unique_name": 9, "source_unique_name": 28})
        self.assertNotIn(("q3", 4990), mappings)
        self.assertNotIn(("q3", 18843), mappings)

    def test_exported_mapping_comparison_statuses(self) -> None:
        self.assertEqual(compare_exported_smiles("CCO", "CCO"), "exact_match")
        self.assertEqual(compare_exported_smiles("C(C)O", "CCO"), "canonical_match")
        self.assertEqual(compare_exported_smiles("CCN", "CCO"), "mismatch")
        self.assertEqual(compare_exported_smiles("", "CCO"), "missing")


class MeasurementNormalizationTest(unittest.TestCase):
    def test_text_and_unit_normalization(self) -> None:
        self.assertEqual(normalize_lexical("  Intrinsic_Clearance – Mean "), "intrinsic clearance - mean")
        self.assertEqual(normalize_unit(" µL · min⁻¹ / mg "), "ul*min-1/mg")

    def test_scalar_scientific_and_approximate(self) -> None:
        parsed = parse_measurement("approximately 3.6 × 10⁻⁶ cm/s")
        self.assertAlmostEqual(parsed.scalar_value or 0, 3.6e-6)
        self.assertTrue(parsed.scalar_is_approximate)

    def test_interval_bound_and_plus_minus(self) -> None:
        interval = parse_measurement("16.7–29.4")
        self.assertEqual((interval.interval_lower, interval.interval_upper), (16.7, 29.4))
        self.assertEqual(parse_measurement("0.99 or higher").lower_bound, 0.99)
        central = parse_measurement("0.14 ± 0.02")
        self.assertEqual(central.scalar_value, 0.14)
        self.assertTrue(central.scalar_is_approximate)

    def test_versioned_categorical_allowlist(self) -> None:
        allowed = ["substrate", "not_substrate"]
        self.assertEqual(approved_category("not_substrate", allowed), "not substrate")
        self.assertIsNone(approved_category("inhibited", allowed))

    def test_explicit_unitless_fg_fraction_becomes_percent(self) -> None:
        emissions, rejected = split_explicit_fg_measurements("Fg = 0.14") or ([], [])
        self.assertFalse(rejected)
        self.assertEqual(len(emissions), 1)
        self.assertEqual(emissions[0].unit_normalized, "percent")
        self.assertEqual(emissions[0].scalar_value, 14.0)

    def test_condition_specific_fg_values_are_separate_children(self) -> None:
        text = "Fg 3.4% ± 1.8% (competent) vs 4.5% ± 1.6% (deficient)"
        emissions, rejected = split_explicit_fg_measurements(text) or ([], [])
        self.assertFalse(rejected)
        self.assertEqual([item.scalar_value for item in emissions], [3.4, 4.5])
        self.assertEqual([item.local_measurement_context for item in emissions], ["competent", "deficient"])

    def test_fa_times_fg_is_retained_with_its_own_measurement_label(self) -> None:
        parsed = split_explicit_fg_measurements("Fa×Fg = 0.22; Fg = 0.14")
        emissions, rejected = parsed or ([], [])
        self.assertFalse(rejected)
        self.assertEqual([item.scalar_value for item in emissions], [22.0, 14.0])
        self.assertEqual([item.measurement_label for item in emissions], ["fa x fg", "fg"])

    def test_fa_times_fg_maps_to_the_broad_fg_concept(self) -> None:
        emissions, _ = split_explicit_fg_measurements("Fa×Fg = 0.22") or ([], [])
        assignment, reason = assign_canonical_endpoint("q3", {"endpoint_alias_raw": "intestinal availability"}, emissions[0])
        self.assertIsNone(reason)
        self.assertEqual(assignment.canonical_endpoint_key if assignment else None, "q3.gut_wall_escape.fg.percent")

    def test_curated_support_text_reextraction_is_marked(self) -> None:
        emissions, rejected = split_explicit_fg_measurements("0.75", "F_G = 0.75") or ([], [])
        self.assertFalse(rejected)
        self.assertEqual(emissions[0].scalar_value, 75.0)
        self.assertIn("support text re-extraction", emissions[0].local_measurement_context or "")


def _base_row(columns: list[str]) -> dict[str, object]:
    return {column: "" for column in columns}


def _context(source_id: str, spec: dict, mapping: dict[str, str] | None = None) -> dict:
    config_path = REPO_ROOT / "configs/source_normalization_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {
        "source_id": source_id,
        "spec": spec,
        "mapping": mapping or {},
        "tdc": set(),
        "allowlists": config["categorical_allowlists"],
        "input_hash": "a" * 64,
    }


class FiveSchemaFixtureIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config_path = REPO_ROOT / "configs/source_normalization_v1.yaml"
        cls.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    def _mapped_fixture(self, source_id: str, value: str) -> tuple[dict, list[str]]:
        spec = self.config["sources"][source_id]
        row = _base_row(spec["expected_columns"])
        row.update({"global_identifier": "id", "smiles": "C(C)O", spec["endpoint_column"]: "endpoint"})
        row[spec["measurement_column"]] = value
        if source_id == "q3":
            row["substrate_status"] = "substrate"
        return normalize_row(row, _context(source_id, spec, {"id": "CCO"}), 1)

    def test_q1_q2_q3_q4_schemas_accept(self) -> None:
        for source_id, value in {"q1": "5", "q2": "1-2", "q3": "", "q4": ">= 3"}.items():
            with self.subTest(source_id=source_id):
                record, reasons = self._mapped_fixture(source_id, value)
                self.assertEqual(reasons, [])
                self.assertEqual(record["canonical_smiles"], "CCO")
                self.assertTrue(set(self.config["sources"][source_id]["expected_columns"]) <= set(record))

    def test_starling_schema_accepts(self) -> None:
        spec = self.config["sources"]["starling"]
        row = _base_row(list(spec["schema"]))
        row.update({"smiles": "CCO", "oral_bioavailability_value": 88.0})
        record, reasons = normalize_row(row, _context("starling", spec), 1)
        self.assertEqual(reasons, [])
        self.assertEqual(record["endpoint_alias_normalized"], "oral bioavailability")
        self.assertTrue(set(spec["schema"]) <= set(record))
        self.assertEqual(record["species_or_population_normalized"], "")
        self.assertIn("species_or_population_mechanical_normalized", record)

    def test_q_sources_never_fall_back_to_exported_smiles(self) -> None:
        spec = self.config["sources"]["q1"]
        row = _base_row(spec["expected_columns"])
        row.update({"global_identifier": "missing", "smiles": "CCO", "exposure_measure": "AUC", "parameter_value": "5"})
        record, reasons = normalize_row(row, _context("q1", spec), 1)
        self.assertIn("unresolved_smiles_mapping", reasons)
        self.assertIsNone(record["canonical_smiles"])

    def test_tdc_canonical_match_is_rejected(self) -> None:
        spec = self.config["sources"]["starling"]
        row = _base_row(list(spec["schema"]))
        row.update({"smiles": "C(C)O", "oral_bioavailability_value": 80.0})
        context = _context("starling", spec)
        context["tdc"] = {"CCO"}
        _, reasons = normalize_row(row, context, 1)
        self.assertIn("tdc_molecule_match", reasons)


class StableEvidenceTest(unittest.TestCase):
    def test_record_ids_are_stable_and_row_specific(self) -> None:
        first = stable_record_id("q1", "hash", 1)
        self.assertEqual(first, stable_record_id("q1", "hash", 1))
        self.assertNotEqual(first, stable_record_id("q1", "hash", 2))

    def test_duplicate_annotations_preserve_rows(self) -> None:
        spec = {"normalized_fields": []}
        fields = combination_fields(spec)
        template = {field: None for field in fields}
        records = [{**template, "canonical_smiles": "CCO"} for _ in range(2)]
        annotate_duplicates(records, fields)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["duplicate_group_size"], 2)
        self.assertEqual(records[0]["duplicate_group_id"], records[1]["duplicate_group_id"])

    def test_primary_reason_precedence(self) -> None:
        reasons = ["missing_measurement", "tdc_molecule_match", "missing_global_identifier"]
        self.assertEqual(primary_reason(reasons), "missing_global_identifier")


class NewFunctionLengthTest(unittest.TestCase):
    def test_new_functions_are_at_most_sixty_lines(self) -> None:
        roots = [
            REPO_ROOT / "pipeline/source_normalization",
            REPO_ROOT / "scripts/normalize_sources.py",
            REPO_ROOT / "scripts/publish_cleaned_canonical_bases.py",
        ]
        violations = []
        paths = [path for root in roots for path in ([root] if root.is_file() else root.glob("*.py"))]
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    length = (node.end_lineno or node.lineno) - node.lineno + 1
                    if length > 60:
                        violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} {node.name} ({length})")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
