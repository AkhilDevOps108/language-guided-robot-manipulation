from __future__ import annotations

import torch
from torch import nn


class LanguageEncoder(nn.Module):
    """Learned task instruction embedding (not a pretrained language model)."""

    def __init__(self, vocab_size: int = 128, embed_dim: int = 32, output_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(embed_dim, output_dim), nn.ReLU())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(tokens)
        mask = (tokens != 0).unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = summed / denom
        return self.proj(pooled)

