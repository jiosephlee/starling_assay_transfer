import unittest

import torch

from starling_ml.intern_v6 import (PackedGroup, assign_optimizer_batches, group_bfd,
                                  listnet_loss, ranking_metrics)


class TestInternV6(unittest.TestCase):
    def test_listnet_has_gradients_and_handles_ties(self):
        scores = torch.tensor([[1., 0.], [0., 1.], [.5, .5], [.5, .5]], requires_grad=True)
        loss = listnet_loss(scores, torch.tensor([2., 0., 1., 1.]), torch.tensor([0, 0, 1, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_groups_are_never_split_and_tail_is_dropped(self):
        groups = [PackedGroup(str(i), tuple(), 1000) for i in range(5)]
        chunks = group_bfd(groups, 4096)
        self.assertEqual(sum(len(chunk) for chunk in chunks), 5)
        self.assertEqual(len(assign_optimizer_batches(chunks, 2)), 2)

    def test_ranking_metrics_are_query_macro(self):
        metrics = ranking_metrics([3, 2, 1, 0, 0, 1], [3, 2, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1])
        self.assertAlmostEqual(metrics["top1_regret"], 0.5)
        self.assertAlmostEqual(metrics["best_in_top10_regret"], 0.0)
