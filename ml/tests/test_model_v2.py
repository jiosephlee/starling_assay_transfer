"""Synthetic forward/loss test for the v2 asymmetric dual-head model (no MoLFormer needed)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

_ML = Path(__file__).resolve().parents[1]
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

try:
    import numpy as np
    import torch

    from starling_ml.config import LossConfig, ModelConfig
    from starling_ml.model import TransferPairModel

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def _model(aux_weight=0.0):
    mcfg = ModelConfig(mol_hidden=16, mol_out=8, meta_field_proj=4, d_model=16, d_ff=32, n_blocks=1)
    lcfg = LossConfig(aux_binary_weight=aux_weight)
    n_smiles, n_meta, n_fields, text_dim = 10, 5, 7, 384
    mol_emb = np.random.randn(n_smiles, 768).astype("float32")
    meta_emb = np.random.randn(n_meta, n_fields, text_dim).astype("float32")
    meta_present = np.ones((n_meta, n_fields), dtype="float32")
    return TransferPairModel(mcfg, lcfg, mol_emb, meta_emb, meta_present)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class ModelV2Test(unittest.TestCase):
    def test_asymmetric_dual_head_shapes_and_loss(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        model = _model(aux_weight=0.5)
        B = 4
        a_idx = torch.tensor([0, 1, 2, 3])
        b_idx = torch.tensor([4, 5, 6, 7])
        meta_a_idx = torch.tensor([0, 1, 2, 3])
        source_value = torch.tensor([50.0, 20.0, 80.0, 10.0])
        distance = torch.tensor([2.0, 5.0, 0.5, 20.0])
        # nullable binary labels: two present, two missing (NaN).
        labels = torch.tensor([1.0, float("nan"), 0.0, float("nan")])

        out = model(a_idx, b_idx, meta_a_idx, source_value, distance=distance, labels=labels)
        self.assertEqual(out["distance"].shape, (B,))
        self.assertEqual(out["logits"].shape, (B,))
        self.assertTrue(torch.all(out["distance"] >= 0))  # softplus -> non-negative distance
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()  # gradients flow through both heads

    def test_continuous_only_ignores_labels(self) -> None:
        model = _model(aux_weight=0.0)  # binary aux disabled
        a = torch.tensor([0, 1]); b = torch.tensor([2, 3]); ma = torch.tensor([0, 1])
        sv = torch.tensor([50.0, 60.0]); dist = torch.tensor([1.0, 2.0])
        # all labels missing -> continuous-only training still produces a finite loss.
        out = model(a, b, ma, sv, distance=dist, labels=torch.tensor([float("nan"), float("nan")]))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_query_branch_is_structure_only(self) -> None:
        # The query encoder must not touch the metadata tables (inference contract 3).
        model = _model()
        b = torch.tensor([0, 1, 2])
        z = model._encode_query(b)
        self.assertEqual(z.shape, (3, model.mol_mlp[-1].out_features))  # only mol_out dims


if __name__ == "__main__":
    unittest.main()
