"""Per-modality encoders for R-MFDN.

Each modality is a self-contained ``nn.Module`` with its own ``forward`` that
turns a raw batch tensor into one ``(B, embed_dim)`` clip feature:

* :class:`VisualEncoder` -- ImageNet ResNet-18 over frames, then a small
  transformer over each temporal *group* of frames; group means are averaged.
* :class:`AudioEncoder` -- AudioSet-pretrained AST, fine-tuned end to end.
* :class:`MelCNN` -- compact alternative: small 2D CNN on the same fbank,
  trained from scratch (``config["audio_encoder"] == "melcnn"``).
* :class:`TextEncoder` -- BERT-base with mask-weighted mean pooling.
* :class:`EmbedBagText` -- compact alternative: learned embedding bag over the
  same WordPiece ids, trained from scratch (``config["text_encoder"] == "embedbag"``).
* :class:`MiniTransformerText` -- the embedding bag plus a 2-layer transformer,
  i.e. word order without pretraining (``config["text_encoder"] == "minitf"``).
* ``config["text_encoder"] == "bertmini"`` / ``"bertsmall"`` / ``"bertmedium"`` --
  :class:`TextEncoder` over the pretrained BERT-mini / -small / -medium
  checkpoint at ``config["bert_mini_model"]`` / ``config["bert_small_model"]`` /
  ``config["bert_medium_model"]``.

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
        # no_temporal_transformer=True drops E_v entirely: group features become
        # the plain mean of the per-frame backbone embeddings (pure perceiver).
        if config.get("no_temporal_transformer"):
            self.pos_embedding = None
            self.temporal_transformer = None
        else:
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
        group_features = (grouped.mean(1) if self.temporal_transformer is None
                          else self.temporal_transformer(grouped + self.pos_embedding).mean(1))
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


class MelCNN(nn.Module):
    """Small 2D CNN over the AST fbank -> clip feature (compact audio stream).

    Consumes exactly what :class:`AudioEncoder` consumes -- the ``(B, 1024, 128)``
    AudioSet-normalised kaldi fbank from ``ASTFeatureExtractor`` -- so the data
    pipeline is untouched. Five stride-2 conv blocks (32-64-128-256-256), global
    average pooling over the remaining 32x4 map, linear projection to
    ``embed_dim``. About 1.1M parameters against AST's 86M; no audio
    pretraining. Like the 16-frame mean-pool on the visual side, the pooling
    is order-free: the fusion only ever saw one clip-level audio token anyway.
    """

    CHANNELS = (32, 64, 128, 256, 256)

    def __init__(self, config: dict, embed_dim: int, _num_heads: int):
        super().__init__()
        blocks, previous = [], 1
        for width in self.CHANNELS:
            blocks += [nn.Conv2d(previous, width, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(width), nn.GELU()]
            previous = width
        self.blocks = nn.Sequential(*blocks)
        self.projection = nn.Linear(previous, embed_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, T, F) -> (B, 1, T, F) -> (B, C, T/32, F/32) -> (B, embed_dim)
        features = self.blocks(mel.unsqueeze(1))
        return self.projection(features.mean((2, 3)))


class TextEncoder(nn.Module):
    """BERT with mask-weighted mean pooling -> clip feature.

    ``checkpoint`` defaults to ``config["bert_model"]`` (BERT-base, the paper
    encoder); the compact ladder passes a smaller BERT-family checkpoint that
    shares the same WordPiece vocabulary, so the tokenizer is unchanged.
    """

    def __init__(self, config: dict, embed_dim: int, checkpoint: str | None = None):
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(checkpoint or config["bert_model"])
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


class EmbedBagText(nn.Module):
    """Learned embedding bag over the BERT WordPiece ids (compact text stream).

    Consumes exactly what :class:`TextEncoder` consumes -- ``input_ids`` and
    ``attention_mask`` from the BERT tokenizer -- so the collate is untouched.
    A 64-d embedding table (padding id 0 stays zero), an attention-mask
    weighted mean over tokens, then a two-layer MLP to ``embed_dim``. About
    2.1M parameters against BERT-base's 110M; no language pretraining and no
    word order: the only bit the text stream owns (llm_textgen) is expected to
    be lexical and length-level.
    """

    VOCAB_SIZE = 30522
    TOKEN_DIM = 64
    HIDDEN_DIM = 256

    def __init__(self, config: dict, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.TOKEN_DIM, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(self.TOKEN_DIM, self.HIDDEN_DIM), nn.GELU(),
            nn.Linear(self.HIDDEN_DIM, embed_dim))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Return projected token states and their mask-weighted mean."""
        sequence = self.projection(self.embedding(input_ids))
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (sequence * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        return sequence, pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encode(input_ids, attention_mask)[1]


class MiniTransformerText(nn.Module):
    """Small from-scratch transformer over the BERT WordPiece ids (compact text stream).

    Same inputs as :class:`TextEncoder` and :class:`EmbedBagText`. The 64-d
    token table of the embedding bag, projected to a 128-d model width, plus a
    learned position table, then two pre-norm transformer encoder layers
    (4 heads, FFN 256) with the attention mask as key padding; mask-weighted
    mean over tokens, linear to ``embed_dim``. About 2.3M parameters: the
    embedding bag plus word order, so a gain over :class:`EmbedBagText` on the
    llm_textgen bit isolates order/context from language pretraining.
    """

    VOCAB_SIZE = 30522
    TOKEN_DIM = 64
    MODEL_DIM = 128
    HEADS = 4
    FFN_DIM = 256
    LAYERS = 2

    def __init__(self, config: dict, embed_dim: int):
        super().__init__()
        max_tokens = config.get("max_text_tokens", 50)
        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.TOKEN_DIM, padding_idx=0)
        self.input_projection = nn.Linear(self.TOKEN_DIM, self.MODEL_DIM)
        self.position = nn.Parameter(torch.zeros(1, max_tokens, self.MODEL_DIM))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            self.MODEL_DIM, self.HEADS, dim_feedforward=self.FFN_DIM,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, self.LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.MODEL_DIM)
        self.projection = nn.Linear(self.MODEL_DIM, embed_dim)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Return projected token states and their mask-weighted mean."""
        tokens = self.input_projection(self.embedding(input_ids))
        tokens = tokens + self.position[:, :tokens.shape[1]]
        states = self.encoder(tokens, src_key_padding_mask=attention_mask == 0)
        sequence = self.projection(self.norm(states))
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (sequence * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        return sequence, pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encode(input_ids, attention_mask)[1]
