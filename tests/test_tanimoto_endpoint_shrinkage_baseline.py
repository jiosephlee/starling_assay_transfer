"""Tests for support-gated endpoint-specific weighted-Tanimoto thresholding."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

from pipeline import tanimoto_endpoint_shrinkage_baseline as shrinkage


class EndpointShrinkageBaselineTest(unittest.TestCase):
    def test_eligible_thresholds_exclude_endpoints_below_train_support(self):
        thresholds = {"well_supported": 0.2, "sparse": 0.8}
        selected = shrinkage.eligible_thresholds(thresholds, {"well_supported": 100, "sparse": 24}, 25)
        self.assertEqual(selected, {"well_supported": 0.2})

    def test_support_tie_break_prefers_the_larger_minimum(self):
        rows = [
            {"min_train_rows": 25, "macro_f1": 0.6, "accuracy": 0.7, "transfer_precision": 0.5},
            {"min_train_rows": 50, "macro_f1": 0.6, "accuracy": 0.7, "transfer_precision": 0.5},
        ]
        self.assertEqual(shrinkage.choose_support(rows)["min_train_rows"], 50)

    def test_all_functions_are_at_most_sixty_lines(self):
        path = Path(shrinkage.__file__)
        violations = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = (node.end_lineno or node.lineno) - node.lineno + 1
                if length > 60:
                    violations.append((node.name, length))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
