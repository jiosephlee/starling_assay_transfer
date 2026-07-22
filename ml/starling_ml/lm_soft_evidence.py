"""Logit-level objective and metrics for V4/V5 LM soft evidence."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_shapes(choice_scores: torch.Tensor, targets: torch.Tensor) -> None:
    valid_choices = choice_scores.ndim > 0 and choice_scores.shape[-1] in (2, 3)
    if choice_scores.shape != targets.shape or not valid_choices:
        raise ValueError(
            f"choice scores and targets must have matching [..., 2|3] shapes: "
            f"{choice_scores.shape} != {targets.shape}"
        )


def soft_evidence_loss(
    choice_scores: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Cross-entropy over A/B or A/B/C choice scores without raw-N weighting."""
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


def argmax_predictions(choice_scores: torch.Tensor) -> torch.Tensor:
    """Return the largest choice index, using A for an exact score tie."""
    if choice_scores.shape[-1] not in (2, 3):
        raise ValueError(f"choice scores must end in dimension 2 or 3: {choice_scores.shape}")
    return choice_scores.argmax(dim=-1)


def _binary_reliability(probabilities: torch.Tensor, targets: torch.Tensor,
                        bins: int = 10) -> float:
    predicted_a = probabilities[:, 0]
    target_a = targets[:, 0]
    indices = torch.clamp((predicted_a * bins).long(), max=bins - 1)
    reliability = predicted_a.new_tensor(0.0)
    for index in range(bins):
        selected = indices == index
        if selected.any():
            gap = predicted_a[selected].mean() - target_a[selected].mean()
            reliability += selected.float().mean() * gap.square()
    return float(reliability)


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
    """Evaluate soft fit and frozen A/B labels for V4 or binary V5."""
    _validate_shapes(choice_scores, targets)
    probabilities = choice_scores.softmax(dim=-1)
    predictions = argmax_predictions(choice_scores)
    expected = torch.where(binary_labels == 1, 0, 1)
    if not torch.isin(binary_labels, torch.tensor([0, 1], device=binary_labels.device)).all():
        raise ValueError("binary evaluation labels must be 0 or 1")
    nll = soft_evidence_loss(choice_scores, targets)
    brier = ((probabilities - targets) ** 2).sum(dim=-1).mean()
    accuracy = (predictions == expected).float().mean()
    macro_f1 = (_class_f1(predictions, expected, 0) + _class_f1(predictions, expected, 1)) / 2
    metrics = {"soft_nll": float(nll), "brier": float(brier),
               "accuracy": float(accuracy), "macro_f1": macro_f1}
    if choice_scores.shape[-1] == 2:
        metrics["reliability"] = _binary_reliability(probabilities, targets)
    else:
        metrics["c_prediction_rate"] = float((predictions == 2).float().mean())
    return metrics
