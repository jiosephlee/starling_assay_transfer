from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pyarrow as pa
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.source_normalization.hf_export import (  # noqa: E402
    COMMON_COLUMNS,
    clean_table,
    omitted_columns,
    public_columns,
    replaced_columns,
    source_columns,
)


def _config() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs/source_normalization_v1.yaml").read_text())


def _record(spec: dict) -> dict:
    row = {column: "Raw Value" for column in source_columns(spec)}
    row.update({
        "canonical_smiles": "CCO",
        "canonical_endpoint_key": "q1.example.endpoint",
        "scalar_value": 7.5,
        "unit_normalized": "ng*h/ml",
        "species_exact": "rat",
        "source_id": "q1",
        "source_name": "oral_bioavailability",
        "record_id": "record", "source_row_number": 1, "input_sha256": "hash",
        "parent_provenance_id": "parent", "child_id": "child",
        "scalar_is_approximate": False, "variation_value": None, "variation_type": None,
        "accompanying_interval_lower": None, "accompanying_interval_upper": None,
        "endpoint_family": "family", "endpoint_subtype": "subtype", "unit_basis": "basis",
        "direction": None, "target": None, "kinetic_parameter": None, "auc_window": None,
        "defining_timepoint": None,
    })
    for column in spec["normalized_fields"]:
        suffix = "species_or_population_mechanical_normalized" if column == "species_or_population" else f"{column}_normalized"
        row[suffix] = f"normalized {column}"
    return row


class CleanedExportTest(unittest.TestCase):
    def test_q1_replaces_all_configured_normalized_fields(self) -> None:
        spec = _config()["sources"]["q1"]
        clean = clean_table(pa.Table.from_pylist([_record(spec)]), spec, "q1").to_pylist()[0]
        replacements = replaced_columns(spec)
        self.assertEqual(clean["smiles"], "CCO")
        self.assertEqual(clean["exposure_measure"], "q1.example.endpoint")
        self.assertEqual(clean["parameter_value"], 7.5)
        self.assertEqual(clean["parameter_units"], "ng*h/ml")
        self.assertEqual(clean["statistic_type"], "normalized statistic_type")
        self.assertEqual(clean["species_exact"], "rat")
        self.assertEqual(set(replacements), {"smiles", *spec["normalized_fields"], "parameter_value"})
        self.assertNotIn("condition_key", clean)
        self.assertNotIn("extraction_id", clean)
        self.assertNotIn("global_identifier", clean)

    def test_starling_uses_override_and_keeps_legacy_key(self) -> None:
        spec = _config()["sources"]["starling"]
        raw = _record(spec)
        raw["condition_key_repro"] = "legacy key"
        clean = clean_table(pa.Table.from_pylist([raw]), spec).to_pylist()[0]
        self.assertEqual(clean["species_or_population"], "normalized species_or_population")
        self.assertEqual(clean["oral_bioavailability_value"], 7.5)
        self.assertEqual(clean["smiles"], "CCO")
        self.assertEqual(clean["condition_key_repro"], "legacy key")
        self.assertEqual(clean["unit_normalized"], "ng*h/ml")

    def test_public_schema_has_original_fields_and_common_fields_once(self) -> None:
        spec = _config()["sources"]["q3"]
        columns = public_columns(spec)
        self.assertEqual(len(columns), len(set(columns)))
        self.assertTrue(set(COMMON_COLUMNS) <= set(columns))
        self.assertEqual(columns[:3], ["smiles", "gut_wall_process", "measured_value"])
        self.assertEqual(columns[-3:], ["extra_details", "support_text", "paragraph_idx"])

    def test_starling_order_without_endpoint_or_unit_source_columns(self) -> None:
        spec = _config()["sources"]["starling"]
        columns = public_columns(spec, "starling")
        self.assertEqual(columns[:2], ["smiles", "oral_bioavailability_value"])
        self.assertEqual(columns[-2:], ["extra_details", "support_text"])

    def test_all_null_columns_are_omitted_and_reported(self) -> None:
        spec = _config()["sources"]["q3"]
        clean = clean_table(pa.Table.from_pylist([_record(spec)]), spec, "q3")
        omissions = omitted_columns(clean, spec, "q3")
        reasons = {(item["column"], item["reason"]) for item in omissions}
        self.assertNotIn("direction", clean.column_names)
        self.assertIn(("direction", "all_null_in_dataset"), reasons)
        self.assertIn(("extraction_id", "internal_identifier"), reasons)
        self.assertFalse(any(clean[name].null_count == clean.num_rows for name in clean.column_names))


if __name__ == "__main__":
    unittest.main()
