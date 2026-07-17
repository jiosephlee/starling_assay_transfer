"""Tests for canonical V4 target construction and decision rules."""

from __future__ import annotations

import unittest

from pipeline.soft_evidence import argmax_completion, modal_completion, target_distribution


class SoftEvidenceTest(unittest.TestCase):
    def test_target_distribution_uses_all_three_vote_counts(self):
        row = {"n_records": 10, "n_transfer": 6, "n_nontransfer": 1, "n_ambiguous": 3}
        self.assertEqual(target_distribution(row),
                         {"transfer": 0.6, "nontransfer": 0.1, "ambiguous": 0.3})
        self.assertEqual(modal_completion(row), "A")

    def test_modal_tie_falls_back_to_c(self):
        row = {"n_records": 10, "n_transfer": 4, "n_nontransfer": 4, "n_ambiguous": 2}
        self.assertEqual(modal_completion(row), "C")

    def test_prediction_uses_argmax(self):
        self.assertEqual(argmax_completion([0.45, 0.35, 0.20]), "A")
        self.assertEqual(argmax_completion([0.45, 0.45, 0.10]), "C")

    def test_invalid_counts_are_rejected(self):
        row = {"n_records": 3, "n_transfer": 1, "n_nontransfer": 1, "n_ambiguous": 0}
        with self.assertRaises(ValueError):
            target_distribution(row)

    def test_stale_stored_fraction_is_rejected(self):
        row = {"n_records": 2, "n_transfer": 2, "n_nontransfer": 0, "n_ambiguous": 0,
               "transfer_fraction": 0.5}
        with self.assertRaises(ValueError):
            target_distribution(row)


if __name__ == "__main__":
    unittest.main()
