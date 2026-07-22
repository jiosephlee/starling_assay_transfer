"""Tests for variance-temperature binary V5 targets."""

from __future__ import annotations

import math
import unittest

from pipeline.variance_soft import (
    argmax_completion,
    completion,
    completion_is_tie_anchor,
    target_distribution,
    target_temperature,
)


def _row(transfer: int, nontransfer: int, ambiguous: int) -> dict[str, int]:
    return {"n_records": transfer + nontransfer + ambiguous,
            "n_transfer": transfer, "n_nontransfer": nontransfer,
            "n_ambiguous": ambiguous}


class VarianceSoftTargetTest(unittest.TestCase):
    def test_exact_decisive_and_mixed_targets(self):
        for counts in ((9, 1, 0), (6, 4, 0), (6, 4, 10)):
            row = _row(*counts)
            temperature = math.sqrt(sum(counts)) * (1 + counts[2] / sum(counts))
            expected = 1 / (1 + math.exp(-(counts[0] - counts[1]) / temperature))
            self.assertAlmostEqual(target_temperature(row), temperature)
            self.assertAlmostEqual(target_distribution(row)["transfer"], expected)

    def test_ties_and_all_deadband_are_anchored_to_a(self):
        for row in (_row(3, 3, 2), _row(0, 0, 7)):
            self.assertEqual(target_distribution(row),
                             {"transfer": 0.5, "nontransfer": 0.5})
            self.assertEqual(completion(row), "A")
            self.assertTrue(completion_is_tie_anchor(row))

    def test_more_consistent_evidence_increases_confidence(self):
        small = target_distribution(_row(1, 0, 0))["transfer"]
        large = target_distribution(_row(9, 0, 0))["transfer"]
        self.assertGreater(large, small)

    def test_deadbands_move_target_toward_half(self):
        no_deadband = target_distribution(_row(6, 2, 0))["transfer"]
        deadband_heavy = target_distribution(_row(6, 2, 20))["transfer"]
        self.assertLess(deadband_heavy, no_deadband)
        self.assertGreater(deadband_heavy, 0.5)

    def test_swapping_counts_swaps_probabilities(self):
        forward = target_distribution(_row(7, 2, 3))
        reverse = target_distribution(_row(2, 7, 3))
        self.assertAlmostEqual(forward["transfer"], reverse["nontransfer"])
        self.assertAlmostEqual(forward["nontransfer"], reverse["transfer"])

    def test_exact_logit_tie_predicts_a(self):
        self.assertEqual(argmax_completion([0.5, 0.5]), "A")
        self.assertEqual(argmax_completion([0.4, 0.6]), "B")


if __name__ == "__main__":
    unittest.main()
