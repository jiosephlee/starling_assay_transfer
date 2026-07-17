"""Logit-level objective and prediction helpers for LM soft evidence V4."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_shapes(choice_scores: torch.Tensor, targets: torch.Tensor) -> None:
    if choice_scores.shape != targets.shape or choice_scores.shape[-1] != 3:
        raise ValueError(
            f"choice scores and targets must have matching [..., 3] shapes: "
            f"{choice_scores.shape} != {targets.shape}"
        )


def soft_evidence_loss(
    choice_scores: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Cross-entropy over A/B/C choice scores without raw-N weighting."""
    _validate_shapes(choice_scores, targets)
    if not torch.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("soft-evidence targets must be finite and nonnegative")
    if not torch.allclose(targets.sum(dim=-1), torch.ones_like(targets[..., 0]), atol=1e-6):
        raise ValueError("soft-evidence targets must sum to one")
    losses = -(targets * F.log_softmax(choice_scores, dim=-1)).sum(dim=-1)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"unsupported reduction: {reduction}")


def strict_majority_predictions(choice_scores: torch.Tensor) -> torch.Tensor:
    """Return 0=A, 1=B, 2=C using probability > 0.5, never argmax."""
    if choice_scores.shape[-1] != 3:
        raise ValueError(f"choice scores must end in dimension 3: {choice_scores.shape}")
    probabilities = choice_scores.softmax(dim=-1)
    output = torch.full(probabilities.shape[:-1], 2, dtype=torch.long, device=choice_scores.device)
    output = torch.where(probabilities[..., 0] > 0.5, 0, output)
    return torch.where(probabilities[..., 1] > 0.5, 1, output)


def _class_f1(predictions: torch.Tensor, expected: torch.Tensor, class_id: int) -> float:
    predicted_class = predictions == class_id
    expected_class = expected == class_id
    tp = int((predicted_class & expected_class).sum())
    fp = int((predicted_class & ~expected_class).sum())
    fn = int((~predicted_class & expected_class).sum())
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def soft_evidence_metrics(
    choice_scores: torch.Tensor, targets: torch.Tensor, binary_labels: torch.Tensor
) -> dict[str, float]:
    """Evaluate soft fit and frozen A/B labels, counting every C prediction as wrong."""
    _validate_shapes(choice_scores, targets)
    probabilities = choice_scores.softmax(dim=-1)
    predictions = strict_majority_predictions(choice_scores)
    expected = torch.where(binary_labels == 1, 0, 1)
    if not torch.isin(binary_labels, torch.tensor([0, 1], device=binary_labels.device)).all():
        raise ValueError("binary evaluation labels must be 0 or 1")
    nll = soft_evidence_loss(choice_scores, targets)
    brier = ((probabilities - targets) ** 2).sum(dim=-1).mean()
    accuracy = (predictions == expected).float().mean()
    macro_f1 = (_class_f1(predictions, expected, 0) + _class_f1(predictions, expected, 1)) / 2
    return {"soft_nll": float(nll), "brier": float(brier), "accuracy": float(accuracy),
            "macro_f1": macro_f1, "c_prediction_rate": float((predictions == 2).float().mean())}
