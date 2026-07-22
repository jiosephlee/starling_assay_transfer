"""Tests for endpoint-specific weighted-Tanimoto thresholding."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

from pipeline import tanimoto_endpoint_baseline as endpoint_baseline


class EndpointTanimotoBaselineTest(unittest.TestCase):
    def test_endpoint_sweep_evaluates_each_cutoff_and_selects_per_endpoint(self):
        train = _data([0.1, 0.9, 0.2, 0.8], [0, 1, 1, 0], ["a", "a", "b", "b"])
        sweep, thresholds, selected = endpoint_baseline.endpoint_threshold_sweep(train, count=3)
        self.assertEqual(len(sweep), 6)
        self.assertEqual(set(thresholds), {"a", "b"})
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(row["threshold"] in {0.0, 0.5, 1.0} for row in selected))

    def test_predictions_use_global_fallback_only_for_unseen_endpoints(self):
        data = _data([0.7, 0.7], [1, 0], ["seen", "unseen"])
        predicted, used_fallback = endpoint_baseline.predictions(data, {"seen": 0.6}, 0.8)
        self.assertEqual(predicted.tolist(), [True, False])
        self.assertEqual(used_fallback.tolist(), [False, True])

    def test_all_functions_are_at_most_sixty_lines(self):
        path = Path(endpoint_baseline.__file__)
        violations = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = (node.end_lineno or node.lineno) - node.lineno + 1
                if length > 60:
                    violations.append((node.name, length))
        self.assertEqual(violations, [])


def _data(scores: list[float], labels: list[int], endpoints: list[str]) -> dict[str, np.ndarray]:
    size = len(scores)
    return {
        "score": np.asarray(scores, dtype=float), "label": np.asarray(labels, dtype=np.int8),
        "concept": np.full(size, "Fa", dtype=object), "bucket": np.full(size, "high", dtype=object),
        "endpoint": np.asarray(endpoints, dtype=object),
    }


if __name__ == "__main__":
    unittest.main()
