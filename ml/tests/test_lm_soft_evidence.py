"""Tests for the V4 LM soft-target objective."""

from __future__ import annotations

import unittest

import torch

from starling_ml.lm_soft_evidence import (
    argmax_predictions,
    soft_evidence_loss,
    soft_evidence_metrics,
)


class LmSoftEvidenceTest(unittest.TestCase):
    def test_soft_loss_matches_manual_cross_entropy(self):
        scores = torch.tensor([[0.4, -0.2, 0.1]])
        targets = torch.tensor([[0.6, 0.1, 0.3]])
        expected = -(targets * scores.log_softmax(dim=-1)).sum()
        self.assertTrue(torch.allclose(soft_evidence_loss(scores, targets), expected))

    def test_prediction_uses_argmax(self):
        probabilities = torch.tensor([[0.45, 0.40, 0.15], [0.20, 0.20, 0.60]])
        scores = probabilities.log()
        self.assertEqual(argmax_predictions(scores).tolist(), [0, 2])

    def test_invalid_target_is_rejected(self):
        with self.assertRaises(ValueError):
            soft_evidence_loss(torch.zeros(1, 3), torch.tensor([[0.6, 0.6, -0.2]]))

    def test_metrics_count_c_as_incorrect(self):
        probabilities = torch.tensor([[0.20, 0.20, 0.60], [0.70, 0.20, 0.10]])
        metrics = soft_evidence_metrics(probabilities.log(), probabilities,
                                        torch.tensor([1, 1]))
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["c_prediction_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
