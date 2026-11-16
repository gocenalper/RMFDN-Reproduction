"""Per-modality encoders for R-MFDN.

Each modality is a self-contained ``nn.Module`` with its own ``forward`` that
turns a raw batch tensor into one ``(B, embed_dim)`` clip feature:

* :class:`VisualEncoder` -- ImageNet ResNet-18 over frames, then a small
  transformer over each temporal *group* of frames; group means are averaged.
* :class:`AudioEncoder` -- AudioSet-pretrained AST, fine-tuned end to end.
* :class:`TextEncoder` -- BERT-base with mask-weighted mean pooling.

``RMFDN.py`` assembles these into the full :class:`~rmfdn_ablation.RMFDN.RMFDN`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Visual frames arrive from the DataLoader as uint8 [0,255] (smaller
# worker->main IPC); /255 + ImageNet-normalise happens here on the GPU.
_VIS_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_VIS_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class VisualEncoder(nn.Module):
    """ImageNet ResNet-18 frames -> temporal transformer (E_v) -> clip feature."""

    def __init__(self, config: dict, embed_dim: int, num_heads: int):
        super().__init__()
        import timm

        # Per-frame ResNet-18 embeddings feed E_v directly, fine-tuned end to
        # end (paper setup); the ResNet width must therefore match embed_dim.
        self.backbone = timm.create_model("resnet18", pretrained=True, num_classes=0)
        assert self.backbone.num_features == embed_dim, \
            "visual encoder requires ResNet width == embed_dim"
        self.num_frame_groups = config["frame_groups"]
        self.frames_per_group = config["frames_per_clip"] // self.num_frame_groups
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.frames_per_group, embed_dim))
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4, batch_first=True),
            num_layers=2)

    def encode(self, frames: torch.Tensor):
        """Return group tokens and their clip-level mean."""
        # frames: (B, T, 3, H, W) uint8 [0,255]; /255 + ImageNet-normalise on the GPU.
        frames = frames.float().div_(255)
        frames = (frames - _VIS_MEAN.to(frames.device)) / _VIS_STD.to(frames.device)
        batch_size = frames.shape[0]
        frame_features = self.backbone(frames.flatten(0, 1))       # (B*T, embed_dim)
        grouped = frame_features.view(
            batch_size * self.num_frame_groups, self.frames_per_group, -1
        )                                                          # (B*G, T/G, d)
        group_features = self.temporal_transformer(grouped + self.pos_embedding).mean(1)
        sequence = group_features.view(batch_size, self.num_frame_groups, -1)
        return sequence, sequence.mean(1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encode(frames)[1]


class AudioEncoder(nn.Module):
    """AudioSet-pretrained AST -> projected token sequence / clip feature."""

    def __init__(self, config: dict, embed_dim: int, _num_heads: int):
        super().__init__()
        from transformers import ASTModel

        self.ast_model = ASTModel.from_pretrained(config["ast_checkpoint"])
        # AST's long attention sequence is memory-heavy; recompute activations
        # during backward so the multimodal cells fit the target GPUs.
        self.ast_model.gradient_checkpointing_enable()
        self.projection = nn.Linear(self.ast_model.config.hidden_size, embed_dim)

    def encode(self, mel: torch.Tensor):
        """Return projected AST tokens and AST's pooled clip feature."""
        outputs = self.ast_model(input_values=mel)
        return self.projection(outputs.last_hidden_state), self.projection(outputs.pooler_output)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encode(mel)[1]


class TextEncoder(nn.Module):
    """BERT with mask-weighted mean pooling -> clip feature."""

    def __init__(self, config: dict, embed_dim: int):
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(config["bert_model"])
        self.projection = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Return projected token states and their mask-weighted mean."""
        hidden_states = self.bert(
            input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        sequence = self.projection(hidden_states)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (sequence * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        return sequence, pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encode(input_ids, attention_mask)[1]
