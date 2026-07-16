from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.source_normalization.endpoint_keys import assign_canonical_endpoint  # noqa: E402
from pipeline.source_normalization.scalar import split_scalar_measurements  # noqa: E402


def _one(value: object, alias: str, unit: str | None = None):
    emissions, rejections = split_scalar_measurements(value, alias, unit)
    if rejections or len(emissions) != 1:
        raise AssertionError((emissions, rejections))
    return emissions[0]


class ScalarSplittingTest(unittest.TestCase):
    def test_five_source_schema_styles(self) -> None:
        q1 = _one("5", "AUC0_inf", "ng*h/mL")
        q2, rejected = split_scalar_measurements(
            "Papp A→B 7.22 ± 0.60 nm/s; Papp B→A 28.73 ± 1.20 nm/s",
            "caco2_mdck_pampa_permeability",
        )
        q3, q3_rejected = split_scalar_measurements("Km 53 µM; Vmax 390 pmol/min/mg", "intestinal_metabolism")
        q4 = _one("Mean 0.28; low 0.057; high 0.378", "intrinsic_clearance", "µL/min/mg")
        starling = _one(88.0, "oral_bioavailability", "percent")
        self.assertEqual((q1.scalar_value, len(q2), len(q3), q4.scalar_value, starling.scalar_value), (5, 2, 2, 0.28, 88))
        self.assertEqual(rejected + q3_rejected, [])

    def test_directional_typed_and_named_ratio_split(self) -> None:
        emissions, rejected = split_scalar_measurements(
            "Papp A→B 7.22 ± 0.60 nm/s, Papp B→A 28.73 ± 1.20 nm/s; efflux ratio 4.2",
            "bidirectional_permeability",
        )
        self.assertEqual([item.scalar_value for item in emissions], [7.22, 28.73, 4.2])
        self.assertEqual([item.measurement_label for item in emissions], ["papp", "papp", "efflux ratio"])
        self.assertEqual(rejected, [])

    def test_repeated_concentration_and_timepoint_series(self) -> None:
        concentration, rejected = split_scalar_measurements("1.2 at 5 µM, 2.3 at 10 µM", "solubility", "mg/mL")
        time, time_rejected = split_scalar_measurements("5 min: 40%; 10 min: 70%", "dissolution", "%")
        self.assertEqual([item.measurement_concentration for item in concentration], ["5 μm", "10 μm"])
        self.assertEqual([item.measurement_timepoint for item in time], ["5 min", "10 min"])
        self.assertEqual(rejected + time_rejected, [])

    def test_partial_parent_retains_unambiguous_child(self) -> None:
        emissions, rejected = split_scalar_measurements("Km 5 µM; Vmax high", "intestinal_metabolism")
        self.assertEqual([item.scalar_value for item in emissions], [5.0])
        self.assertEqual([item.rejection_reason for item in rejected], ["qualitative_or_unparseable_measurement"])

    def test_scalar_acceptance_and_audit_metadata(self) -> None:
        approximate = _one("approximately 3.6 × 10⁻⁶ cm/s", "Papp")
        central = _one("0.14 ± 0.02", "Km", "µM")
        interval = _one("5 (95% CI 4-6)", "bioavailability", "%")
        self.assertTrue(approximate.scalar_is_approximate)
        self.assertTrue(math.isclose(approximate.scalar_value, 3.6e-6))
        self.assertEqual((central.scalar_value, central.variation_value), (0.14, 0.02))
        self.assertEqual((interval.accompanying_interval_lower, interval.accompanying_interval_upper), (4, 6))

    def test_rejected_measurement_classes(self) -> None:
        cases = {
            "high": "qualitative_or_unparseable_measurement",
            ">= 3": "bound_only_measurement",
            "16.7-29.4": "range_or_interval_only",
            "1, 2, 3": "ambiguous_multiple_measurements",
            "1.4-fold increase": "comparison_only_fold_change",
        }
        for value, reason in cases.items():
            with self.subTest(value=value):
                emissions, rejected = split_scalar_measurements(value, "permeability", "fold")
                self.assertEqual(emissions, [])
                self.assertEqual(rejected[0].rejection_reason, reason)


class EndpointAssignmentTest(unittest.TestCase):
    def test_direction_and_unit_define_permeability_key(self) -> None:
        record = {"endpoint_alias_raw": "caco2_mdck_pampa_permeability"}
        forward, _ = assign_canonical_endpoint("q2", record, _one("Papp A→B 7.2 nm/s", "x"))
        reverse, _ = assign_canonical_endpoint("q2", record, _one("Papp B→A 7.2 nm/s", "x"))
        other_unit, _ = assign_canonical_endpoint("q2", record, _one("Papp A→B 7.2 cm/s", "x"))
        self.assertNotEqual(forward.canonical_endpoint_key, reverse.canonical_endpoint_key)
        self.assertNotEqual(forward.canonical_endpoint_key, other_unit.canonical_endpoint_key)

    def test_condition_metadata_does_not_define_key(self) -> None:
        first = {"endpoint_alias_raw": "caco2_mdck_pampa_permeability", "assay_system": "Caco-2", "condition_medium": "HBSS"}
        second = {"endpoint_alias_raw": "caco2_mdck_pampa_permeability", "assay_system": "MDCK", "condition_medium": "DMEM"}
        emission = _one("Papp A→B 7.2 nm/s", "x")
        left, _ = assign_canonical_endpoint("q2", first, emission)
        right, _ = assign_canonical_endpoint("q2", second, emission)
        self.assertEqual(left.canonical_endpoint_key, right.canonical_endpoint_key)

    def test_target_and_kinetic_parameter_define_key(self) -> None:
        cyp = {"endpoint_alias_raw": "cyp_metabolism", "enzyme_or_pathway": "CYP3A4"}
        other = {"endpoint_alias_raw": "cyp_metabolism", "enzyme_or_pathway": "CYP2D6"}
        km, _ = assign_canonical_endpoint("q4", cyp, _one("Km 2.5 µM", "x"))
        vmax, _ = assign_canonical_endpoint("q4", cyp, _one("Vmax 2.5 pmol/min/mg", "x"))
        other_km, _ = assign_canonical_endpoint("q4", other, _one("Km 2.5 µM", "x"))
        self.assertEqual(len({km.canonical_endpoint_key, vmax.canonical_endpoint_key, other_km.canonical_endpoint_key}), 3)

    def test_only_explicit_absolute_bioavailability_shares(self) -> None:
        q1_record = {"endpoint_alias_raw": "bioavailability", "comparator_exposure": "IV reference"}
        starling_record = {"bioavailability_report_type": "absolute"}
        q1, _ = assign_canonical_endpoint("q1", q1_record, _one("88", "bioavailability", "%"))
        starling, _ = assign_canonical_endpoint("starling", starling_record, _one(88, "oral_bioavailability", "percent"))
        relative, _ = assign_canonical_endpoint(
            "q1", {"endpoint_alias_raw": "relative_bioavailability"}, _one("1.2", "relative_bioavailability", "ratio")
        )
        unspecified, reason = assign_canonical_endpoint(
            "starling", {"bioavailability_report_type": "unspecified"}, _one(88, "oral_bioavailability", "percent")
        )
        self.assertEqual(q1.canonical_endpoint_key, starling.canonical_endpoint_key)
        self.assertNotEqual(q1.canonical_endpoint_key, relative.canonical_endpoint_key)
        self.assertIsNone(unspecified)
        self.assertEqual(reason, "missing_absolute_or_relative_identity")

    def test_auc_window_defines_key_and_missing_window_rejects(self) -> None:
        record = {"endpoint_alias_raw": "AUC0_24"}
        assigned, _ = assign_canonical_endpoint("q1", record, _one("5", "AUC0_24", "ng*h/mL"))
        missing, reason = assign_canonical_endpoint(
            "q1", {"endpoint_alias_raw": "AUC"}, _one("5", "AUC", "ng*h/mL")
        )
        self.assertIn("zero_to_24h", assigned.canonical_endpoint_key)
        self.assertIsNone(missing)
        self.assertEqual(reason, "missing_auc_window_or_unit")


class SemanticSourceNameTest(unittest.TestCase):
    def test_semantic_names_are_unique_and_stable(self) -> None:
        config = yaml.safe_load((REPO_ROOT / "configs/source_normalization_v1.yaml").read_text())
        names = {source: spec["semantic_name"] for source, spec in config["sources"].items()}
        self.assertEqual(names, {
            "q1": "oral_bioavailability", "q2": "intestinal_absorption",
            "q3": "gut_wall", "q4": "hepatic", "starling": "starling_oba",
        })
        self.assertEqual(len(set(names.values())), len(names))


if __name__ == "__main__":
    unittest.main()
