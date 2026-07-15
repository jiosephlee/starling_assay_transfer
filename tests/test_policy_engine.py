"""Tests for the policy-driven endpoint engine: policy, value_canon, endpoints."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.policy import load_condition_key_policy, load_metric_policy  # noqa: E402
from pipeline.endpoints import load_endpoint_resolver  # noqa: E402
from pipeline.normalize import value_canon  # noqa: E402


class MetricPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_metric_policy()

    def test_bounded_percentage_deadband(self) -> None:
        m = self.policy.for_metric("bounded_percentage")
        self.assertEqual(m.transform_value(42.0), 42.0)  # identity
        self.assertEqual(m.label(50.0, 55.0), "transfer")  # 5 <= 10
        self.assertEqual(m.label(50.0, 90.0), "not_transfer")  # 40 >= 30
        self.assertIsNone(m.label(50.0, 70.0))  # 20 in deadband

    def test_bounded_fraction_deadband(self) -> None:
        m = self.policy.for_metric("bounded_fraction")
        self.assertEqual(m.label(0.50, 0.55), "transfer")
        self.assertEqual(m.label(0.10, 0.60), "not_transfer")
        self.assertIsNone(m.label(0.10, 0.30))  # 0.20 in deadband

    def test_log_metric_uses_transformed_scale(self) -> None:
        m = self.policy.for_metric("half_life")
        # 2-fold difference -> log10(2) ~ 0.301 == transfer_max (inclusive).
        a, b = m.transform_value(10.0), m.transform_value(20.0)
        self.assertAlmostEqual(abs(a - b), math.log10(2), places=6)
        self.assertEqual(m.label(a, b), "transfer")
        # 10-fold -> log10(10) = 1.0 >= not_transfer_min (0.699).
        c = m.transform_value(100.0)
        self.assertEqual(m.label(a, c), "not_transfer")

    def test_unknown_metric_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.policy.for_metric("does_not_exist")


class ConditionKeyPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ck = load_condition_key_policy()

    def test_same_endpoint_join_fields(self) -> None:
        self.assertEqual(self.ck.join_fields("same_endpoint"), ["canonical_endpoint_key"])

    def test_same_species_requires_species(self) -> None:
        self.assertIn("species_exact", self.ck.join_fields("same_species_same_endpoint"))
        self.assertIn("species_exact", self.ck.required_non_null("same_species_same_endpoint"))

    def test_most_specific_needs_schema(self) -> None:
        fields = self.ck.join_fields("most_specific", "q4_half_life_v1")
        self.assertIn("canonical_endpoint_key", fields)
        self.assertIn("species_exact", fields)
        with self.assertRaises(ValueError):
            self.ck.join_fields("most_specific")


class ValueCanonTest(unittest.TestCase):
    def test_percent(self) -> None:
        self.assertEqual(value_canon.percent("55", "%"), 55.0)
        self.assertEqual(value_canon.percent("0.7", "fraction"), 70.0)
        self.assertEqual(value_canon.percent("0.95", None), 95.0)  # bare 0-1 -> percent
        self.assertEqual(value_canon.percent("80", None), 80.0)  # bare 0-100 stays
        self.assertIsNone(value_canon.percent("5", "h^-1"))  # rate unit rejected
        self.assertIsNone(value_canon.percent("150", "%"))  # out of range

    def test_fraction(self) -> None:
        self.assertEqual(value_canon.fraction("50", "%"), 0.5)
        self.assertAlmostEqual(value_canon.fraction("0.3", "unitless"), 0.3)
        self.assertEqual(value_canon.fraction("70", None), 0.7)  # bare >1 -> /100
        self.assertIsNone(value_canon.fraction("5", "µM"))

    def test_time_minutes(self) -> None:
        self.assertEqual(value_canon.time_minutes("30", "min"), 30.0)
        self.assertEqual(value_canon.time_minutes("2", "h"), 120.0)
        self.assertIsNone(value_canon.time_minutes("50", "%"))  # non-time unit
        self.assertIsNone(value_canon.time_minutes("-5", "min"))  # non-positive


class EndpointResolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = load_endpoint_resolver()

    def test_fraction_absorbed_assigns_bounded_percentage(self) -> None:
        row = {"endpoint_category": "fraction_absorbed", "reported_value": "82", "reported_units": "%"}
        a, reason = self.resolver.assign("q2", row)
        self.assertIsNone(reason)
        self.assertEqual(a.canonical_endpoint_id, "q2.fraction_absorbed.percent")
        self.assertEqual(a.metric_type, "bounded_percentage")
        self.assertEqual(a.transformed_value, 82.0)

    def test_half_life_canonicalizes_to_log_minutes(self) -> None:
        row = {"metric_type": "metabolic_half_life", "reported_value": "2", "reported_units": "h"}
        a, reason = self.resolver.assign("q4", row)
        self.assertIsNone(reason)
        self.assertEqual(a.native_value, 120.0)
        self.assertAlmostEqual(a.transformed_value, math.log10(120.0), places=6)

    def test_extraction_ratio_percent_to_fraction(self) -> None:
        row = {"metric_type": "extraction_ratio", "reported_value": "40", "reported_units": "%"}
        a, reason = self.resolver.assign("q4", row)
        self.assertIsNone(reason)
        self.assertEqual(a.canonical_endpoint_id, "q4.extraction_ratio")
        self.assertAlmostEqual(a.native_value, 0.4)

    def test_disabled_endpoint_quarantines(self) -> None:
        row = {"metric_type": "intrinsic_clearance", "reported_value": "12", "reported_units": "mL/min/kg"}
        a, reason = self.resolver.assign("q4", row)
        self.assertIsNone(a)
        self.assertEqual(reason, "unmapped_or_disabled_endpoint")

    def test_bad_unit_quarantines_enabled_endpoint(self) -> None:
        row = {"metric_type": "metabolic_half_life", "reported_value": "5", "reported_units": "µM"}
        a, reason = self.resolver.assign("q4", row)
        self.assertIsNone(a)
        self.assertEqual(reason, "unresolved_value_or_unit")


if __name__ == "__main__":
    unittest.main()
