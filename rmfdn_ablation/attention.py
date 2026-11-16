"""Attention blocks shared by the fusion module."""

import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    """Single-token self-attention with a residual LayerNorm (pre-fusion)."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(tokens, tokens, tokens)
        return self.norm(tokens + attended)
