import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionBlock(nn.Module):
    """Single-token self-attention with a residual LayerNorm (pre-fusion)."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(tokens, tokens, tokens)
        return self.norm(tokens + attended)

class CrossModalAttention(nn.Module):
    """Cross-modal alignment attention block."""

    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)

    def forward(self, x1, x2, x2_mask=None):
        alignment_scores = torch.bmm(self.W(x1), x2.transpose(-2, -1))

        if x2_mask is not None:
            mask = x2_mask.unsqueeze(1)
            alignment_scores = alignment_scores.masked_fill(mask == 0, float('-inf'))

        score_dist = F.softmax(alignment_scores, dim=-1)
        return torch.bmm(score_dist, x2)

class CrossModalTransformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cross_modal_attention = CrossModalAttention(dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x1, x2, x2_mask):
        attended = self.cross_modal_attention(x1, x2, x2_mask)
        out = self.layer_norm(x1 + attended)
        return out
