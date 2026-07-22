"""Shared fixed-feature encoders and V6 direct/contrastive MLP objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
                                    nn.SiLU(), nn.Dropout(dropout),
                                    nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class V6Backbone(nn.Module):
    def __init__(self, mol_dim: int = 1024, assay_dim: int = 512, width: int = 256,
                 output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.molecule = Encoder(mol_dim, width, output_dim, dropout)
        self.assay = Encoder(assay_dim, width, output_dim, dropout)

    def encode(self, molecule: torch.Tensor, assay: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.molecule(molecule), self.assay(assay)


class V6DirectPredictor(nn.Module):
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.backbone = V6Backbone(output_dim=feature_dim, dropout=dropout)
        fused = feature_dim * 8 + 2
        self.head = nn.Sequential(nn.LayerNorm(fused), nn.Linear(fused, hidden_dim), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2),
                                  nn.SiLU(), nn.Linear(hidden_dim // 2, 2))

    def forward(self, qm, rm, qa, ra, value, approximate) -> torch.Tensor:
        hq, aq = self.backbone.encode(qm, qa)
        hr, ar = self.backbone.encode(rm, ra)
        pieces = [hq, hr, (hq - hr).abs(), hq * hr, aq, ar, (aq - ar).abs(), aq * ar,
                  value[:, None], approximate[:, None]]
        return self.head(torch.cat(pieces, dim=-1))


class V6ContrastiveRetriever(nn.Module):
    def __init__(self, feature_dim: int = 128, embedding_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.backbone = V6Backbone(output_dim=feature_dim, dropout=dropout)
        self.query_head = Encoder(feature_dim * 2, feature_dim * 2, embedding_dim, dropout)
        self.retrieval_head = Encoder(feature_dim * 2 + 2, feature_dim * 2, embedding_dim, dropout)

    def forward(self, qm, rm, qa, ra, value, approximate) -> torch.Tensor:
        hq, aq = self.backbone.encode(qm, qa)
        hr, ar = self.backbone.encode(rm, ra)
        query = F.normalize(self.query_head(torch.cat([hq, aq], dim=-1)), dim=-1)
        retrieval = F.normalize(self.retrieval_head(
            torch.cat([hr, ar, value[:, None], approximate[:, None]], dim=-1)), dim=-1)
        return (query * retrieval).sum(dim=-1)


def soft_ab_loss(logits: torch.Tensor, target_a: torch.Tensor) -> torch.Tensor:
    targets = torch.stack([target_a, 1.0 - target_a], dim=-1)
    return -(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def graded_list_loss(scores: torch.Tensor, target_z: torch.Tensor,
                     target_temperature: float = 1.0, model_temperature: float = 1.0,
                     minimum_spread: float = 1e-6) -> torch.Tensor:
    spread = target_z.max(dim=-1).values - target_z.min(dim=-1).values
    valid = spread > minimum_spread
    if not bool(valid.any()):
        return scores.sum() * 0.0
    targets = F.softmax(target_z[valid].float() / target_temperature, dim=-1)
    predictions = F.log_softmax(scores[valid].float() / model_temperature, dim=-1)
    return -(targets * predictions).sum(dim=-1).mean()


def direct_listnet_loss(logits: torch.Tensor, target_z: torch.Tensor, group_size: int) -> torch.Tensor:
    margins = (logits[:, 0] - logits[:, 1]).reshape(-1, group_size)
    return graded_list_loss(margins, target_z.reshape(-1, group_size))
