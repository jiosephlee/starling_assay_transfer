"""Eight-block, approximately 100M-parameter V6 fusion MLP variants."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FusionSpec:
    input_dim: int
    ffn_dim: int
    parameters: int


FUSION_SPECS = {
    "concat": FusionSpec(2049, 3872, 99_990_020),
    "difference": FusionSpec(3073, 3824, 99_860_228),
    "difference_product": FusionSpec(4097, 3792, 100_123_908),
}


class Adapter(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 1024), nn.SiLU(),
                                    nn.Dropout(dropout), nn.Linear(1024, 512),
                                    nn.LayerNorm(512))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class SwiGLUBlock(nn.Module):
    def __init__(self, width: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.input = nn.Linear(width, ffn_dim * 2)
        self.output = nn.Linear(ffn_dim, width)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((width,), 1e-4))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gate, content = self.input(self.norm(values)).chunk(2, dim=-1)
        update = self.output(F.silu(gate) * content)
        return values + self.layer_scale * self.dropout(update)


class V6FusionMLP100M(nn.Module):
    def __init__(self, fusion_mode: str, dropout: float = 0.1, blocks: int = 8):
        super().__init__()
        if fusion_mode not in FUSION_SPECS:
            raise ValueError(f"unknown fusion mode: {fusion_mode}")
        if blocks != 8:
            raise ValueError("the frozen V6 100M contract requires exactly eight blocks")
        self.fusion_mode, spec = fusion_mode, FUSION_SPECS[fusion_mode]
        self.molecule, self.assay = Adapter(dropout), Adapter(dropout)
        self.fusion = nn.Sequential(nn.LayerNorm(spec.input_dim), nn.Linear(spec.input_dim, 1024))
        self.blocks = nn.ModuleList([SwiGLUBlock(1024, spec.ffn_dim, dropout)
                                     for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 2))
        self._assert_parameter_count(spec.parameters)

    def _pieces(self, qm, rm, qa, ra, value) -> list[torch.Tensor]:
        pieces = [qm, rm]
        if self.fusion_mode != "concat":
            pieces.append((qm - rm).abs())
        if self.fusion_mode == "difference_product":
            pieces.append(qm * rm)
        pieces.extend([qa, ra])
        if self.fusion_mode != "concat":
            pieces.append((qa - ra).abs())
        if self.fusion_mode == "difference_product":
            pieces.append(qa * ra)
        return [*pieces, value[:, None]]

    def _assert_parameter_count(self, expected: int) -> None:
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != expected:
            raise RuntimeError(f"parameter contract mismatch: expected {expected}, found {actual}")

    def forward(self, qm, rm, qa, ra, value) -> torch.Tensor:
        qm, rm = self.molecule(qm), self.molecule(rm)
        qa, ra = self.assay(qa), self.assay(ra)
        hidden = self.fusion(torch.cat(self._pieces(qm, rm, qa, ra, value), dim=-1))
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)


def soft_ab_loss(logits: torch.Tensor, target_a: torch.Tensor) -> torch.Tensor:
    targets = torch.stack((target_a, 1.0 - target_a), dim=-1)
    return -(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()
