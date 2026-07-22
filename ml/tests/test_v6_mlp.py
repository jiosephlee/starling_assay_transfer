import torch

from starling_ml.v6_mlp import (V6ContrastiveRetriever, V6DirectPredictor,
                                direct_listnet_loss, graded_list_loss, soft_ab_loss)


def _inputs(size=8):
    return (torch.randn(size, 1024), torch.randn(size, 1024),
            torch.randn(size, 512), torch.randn(size, 512),
            torch.randn(size), torch.zeros(size))


def test_direct_predictor_and_soft_loss_have_gradients():
    model = V6DirectPredictor(feature_dim=16, hidden_dim=32)
    logits = model(*_inputs())
    loss = soft_ab_loss(logits, torch.linspace(0.1, 0.9, 8))
    loss = loss + 0.1 * direct_listnet_loss(logits, torch.randn(8), 4)
    loss.backward()
    assert logits.shape == (8, 2)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_contrastive_retriever_and_tied_list_are_finite():
    model = V6ContrastiveRetriever(feature_dim=16, embedding_dim=8)
    scores = model(*_inputs()).reshape(2, 4)
    loss = graded_list_loss(scores, torch.ones_like(scores))
    loss.backward()
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    assert all(parameter.grad is not None for parameter in model.parameters())
