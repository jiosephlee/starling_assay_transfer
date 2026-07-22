"""Tests for the V3 weighted-Tanimoto baseline."""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline import tanimoto_baseline as baseline


class TanimotoBaselineTest(unittest.TestCase):
    def test_grid_has_exactly_one_hundred_inclusive_thresholds(self):
        grid = baseline.threshold_grid()
        self.assertEqual(len(grid), 100)
        self.assertEqual(grid[0], 0.0)
        self.assertEqual(grid[-1], 1.0)

    def test_sweep_uses_inclusive_threshold(self):
        data = _data([0.0, 0.5, 1.0], [0, 1, 1])
        rows = baseline.sweep_thresholds(data, count=3)
        middle = rows[1]
        self.assertEqual(middle["threshold"], 0.5)
        self.assertEqual(middle["predicted_transfer"], 2)
        self.assertEqual(middle["macro_f1"], 1.0)

    def test_metrics_match_hand_calculated_confusion_matrix(self):
        metrics = baseline.binary_metrics(np.array([1, 1, 0, 0]),
                                          np.array([1, 0, 1, 0]))
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["transfer_precision"], 0.5)
        self.assertEqual(metrics["transfer_recall"], 0.5)

    def test_tie_break_prefers_accuracy_then_precision_then_lower_threshold(self):
        rows = [
            {"threshold": 0.4, "macro_f1": 0.7, "accuracy": 0.8,
             "transfer_precision": 0.6},
            {"threshold": 0.3, "macro_f1": 0.7, "accuracy": 0.8,
             "transfer_precision": 0.6},
            {"threshold": 0.2, "macro_f1": 0.7, "accuracy": 0.7,
             "transfer_precision": 0.9},
        ]
        self.assertEqual(baseline.choose_threshold(rows)["threshold"], 0.3)

    def test_configurable_output_stem_and_report_heading(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            paths = baseline.output_paths(output_dir, "assay_transfer_v4_fg_v3_tanimoto")
            evaluation = [{
                "split": "validation", "slice_type": "overall", "slice_value": "all",
                "threshold": 0.5, "n": 1, "actual_transfer": 1,
                "predicted_transfer": 1, "macro_f1": 1.0, "accuracy": 1.0,
                "transfer_precision": 1.0, "transfer_recall": 1.0,
            }]
            baseline.write_markdown(
                paths["markdown"], {"threshold": 0.5, "macro_f1": 1.0, "accuracy": 1.0},
                evaluation, "Assay Transfer V4-on-V3 Weighted-Tanimoto Baseline",
                paths["baseline"].name,
            )
            report = paths["markdown"].read_text()
        self.assertEqual(paths["sweep"].name, "assay_transfer_v4_fg_v3_tanimoto_threshold_sweep.tsv")
        self.assertIn("# Assay Transfer V4-on-V3 Weighted-Tanimoto Baseline", report)
        self.assertIn("`assay_transfer_v4_fg_v3_tanimoto_baseline.tsv`", report)

    def test_all_functions_are_at_most_sixty_lines(self):
        path = Path(baseline.__file__)
        violations = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = (node.end_lineno or node.lineno) - node.lineno + 1
                if length > 60:
                    violations.append((node.name, length))
        self.assertEqual(violations, [])


def _data(scores: list[float], labels: list[int]) -> dict[str, np.ndarray]:
    size = len(scores)
    return {
        "score": np.asarray(scores, dtype=float),
        "label": np.asarray(labels, dtype=np.int8),
        "concept": np.full(size, "Fa", dtype=object),
        "bucket": np.full(size, "high", dtype=object),
    }


if __name__ == "__main__":
    unittest.main()
