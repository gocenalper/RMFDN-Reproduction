"""R-MFDN model: per-modality encoders, progressive fusion, two output heads.

Architecture (mirroring the paper's end-to-end setup):

* **Visual** -- ImageNet-pretrained ResNet-18 over individual frames, then a
  small transformer over each temporal *group* of frames; group means are
  averaged into one clip feature.
* **Audio** -- AudioSet-pretrained AST, fine-tuned end to end; or, with
  ``audio_encoder="melcnn"``, the compact :class:`~rmfdn_ablation.encoders.MelCNN`.
* **Text** -- BERT-base, mask-weighted mean pooling over token states; or,
  with ``text_encoder="embedbag"`` / ``"minitf"``, the compact
  :class:`~rmfdn_ablation.encoders.EmbedBagText` /
  :class:`~rmfdn_ablation.encoders.MiniTransformerText`; with
  ``"bertmini"`` / ``"bertsmall"`` / ``"bertmedium"``, the same encoder over
  the pretrained BERT-mini / BERT-small / BERT-medium checkpoint.
* **Fusion** (paper Eq. 1-3, "progressive") -- with all three modalities the
  text token cross-attends into audio, the result cross-attends into visual.
  With two modalities a single cross-attention is used (text, else audio, is
  the query). A single modality passes through self-attention + FFN only.
* **Heads** -- two *parallel* linear heads on the fused feature:

      binary_head     Linear(d, 2)            -> real / fake decision
      technique_head  Linear(d, n_techniques) -> independent per-technique
                                                 (multi-label) logits

  Every matrix cell uses the same seven-output taxonomy by default; with
  ``mlc_loss="type11"`` the technique head instead emits an 11-way
  forgery-type softmax. The binary decision is not derived from the
  technique outputs; the heads are siblings.
* **Contrastive projections** -- one 2-layer MLP per modality feeding the
  InfoNCE losses (Eq. 4-5) and the live shared-encoder reference path (Eq. 6).

When a reference feature is supplied, the classification input becomes
``fused + reference_alpha * reference_fused`` (Eq. 6).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from .attention import SelfAttentionBlock
from .config import TECHNIQUE_INDICES, NUM_FORGERY_TYPES
from .encoders import (AudioEncoder, EmbedBagText, MelCNN, MiniTransformerText,
                       TextEncoder, VisualEncoder)


class RMFDN(nn.Module):
    """Modality-configurable R-MFDN (any non-empty subset of {v, a, t})."""

    def __init__(self, config: dict, modalities: Sequence[str]):
        super().__init__()
        embed_dim, num_heads = config["embed_dim"], config["attention_heads"]
        self.config = config
        self.modalities = list(modalities)

        # ----- per-modality encoders, each its own nn.Module (own forward) ----
        self.encoders = nn.ModuleDict()
        if "v" in self.modalities:
            self.encoders["v"] = VisualEncoder(config, embed_dim, num_heads)
        if "a" in self.modalities:
            # audio_encoder="melcnn" swaps AST for the compact from-scratch CNN;
            # absent/"ast" keeps the paper encoder (every earlier run).
            audio_encoder = {"ast": AudioEncoder, "melcnn": MelCNN}[
                config.get("audio_encoder", "ast")]
            self.encoders["a"] = audio_encoder(config, embed_dim, num_heads)
        if "t" in self.modalities:
            # text_encoder="embedbag"/"minitf" swaps BERT for a compact
            # from-scratch encoder, "bertmini"/"bertsmall"/"bertmedium" for the
            # pretrained BERT-mini/-small/-medium at config["bert_mini_model"/
            # "bert_small_model"/"bert_medium_model"]; absent/"bert" keeps the
            # paper encoder (every earlier run).
            kind = config.get("text_encoder", "bert")
            pretrained_small = {"bertmini": "bert_mini_model",
                                "bertsmall": "bert_small_model",
                                "bertmedium": "bert_medium_model"}
            if kind in pretrained_small:
                self.encoders["t"] = TextEncoder(
                    config, embed_dim, checkpoint=config[pretrained_small[kind]])
            else:
                text_encoder = {"bert": TextEncoder, "embedbag": EmbedBagText,
                                "minitf": MiniTransformerText}[kind]
                self.encoders["t"] = text_encoder(config, embed_dim)

        # ----- fusion ---------------------------------------------------------
        # The visual encoder output gets no additional modality self-attention;
        # audio/text use this block only in multimodal fusion.
        if len(self.modalities) >= 2:
            self.modality_self_attention = nn.ModuleDict(
                {m: SelfAttentionBlock(embed_dim, num_heads)
                 for m in self.modalities if m != "v"})

        if len(self.modalities) == 3:
            # Progressive fusion (Eq. 1-3): text -> audio, then -> visual.
            self.text_audio_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.text_audio_norm = nn.LayerNorm(embed_dim)
            self.fusion_ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim))
            self.audiotext_visual_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.fusion_norm = nn.LayerNorm(embed_dim)

        elif len(self.modalities) == 2:
            # Single cross-attention; text (else audio) acts as the query.
            self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.attention_norm = nn.LayerNorm(embed_dim)
            self.fusion_ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim))
            self.output_norm = nn.LayerNorm(embed_dim)
            
        # ----- output heads (parallel, both read the fused feature) ----------
        # ``no_binary_head`` drops the real/fake head entirely (Phase-1: pure
        # multi-label model; no detection branch is built or evaluated).
        if not config.get("no_binary_head"):
            self.binary_head = nn.Linear(embed_dim, 2)
        # 11-way forgery-TYPE softmax head when mlc_loss="type11" (mutually
        # exclusive types), else the standard independent 7-bit technique head.
        n_technique_out = (NUM_FORGERY_TYPES if config.get("mlc_loss") == "type11"
                           else len(TECHNIQUE_INDICES))
        self.technique_head = nn.Linear(embed_dim, n_technique_out)
        # Training-only auxiliary head (oracle-PoE + aux): plain-BCE supervision
        # that restores the trunk's full technique gradient without touching the
        # measured head's objective. Never read at evaluation.
        if config.get("aux_plain_head"):
            self.aux_technique_head = nn.Linear(embed_dim, n_technique_out)

        # ----- contrastive projections for the InfoNCE losses ----------------
        # raw_contrastive=True drops the projection heads entirely: Eq. 4/5 then
        # run on the very features the fusion module consumes (the paper's
        # literal notation), so the alignment gradient reshapes the encoders
        # instead of being absorbed in a side branch.
        # Set by freeze_encoders(); keeps frozen encoders out of train mode so
        # their batch-norm running statistics cannot drift.
        self.frozen_encoders = False

        contrastive_dim = config["contrastive_dim"]
        if config.get("raw_contrastive"):
            self.contrastive_projections = None
        else:
            self.contrastive_projections = nn.ModuleDict({
                m: nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                 nn.Linear(embed_dim, contrastive_dim))
                for m in self.modalities})

    def train(self, mode: bool = True):
        """Standard train/eval switch, except that frozen encoders stay in eval."""
        super().train(mode)
        if getattr(self, "frozen_encoders", False):
            for encoder in self.encoders.values():
                encoder.eval()
        return self

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #

    def encode_modalities(self, batch: dict) -> Dict[str, torch.Tensor]:
        """Per-modality clip features, each ``(B, embed_dim)``."""
        features: Dict[str, torch.Tensor] = {}
        if "v" in self.modalities:
            features["v"] = self.encoders["v"](batch["frames"])
        if "a" in self.modalities:
            features["a"] = self.encoders["a"](batch["mel"])
        if "t" in self.modalities:
            features["t"] = self.encoders["t"](batch["input_ids"], batch["attention_mask"])
        return features

    def fuse(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Progressive fusion into a single ``(B, embed_dim)`` clip feature."""
        if len(self.modalities) == 1:
            return features[self.modalities[0]]

        tokens = {m: (features[m].unsqueeze(1) if m == "v"
                      else self.modality_self_attention[m](features[m].unsqueeze(1)))
                  for m in self.modalities}

        if len(self.modalities) == 3:
            attended_audio, _ = self.text_audio_attention(tokens["t"], tokens["a"], tokens["a"])
            text_audio = self.text_audio_norm(tokens["t"] + attended_audio)
            query = self.fusion_ffn(text_audio)
            attended_visual, _ = self.audiotext_visual_attention(query, tokens["v"], tokens["v"])
            return self.fusion_norm(query + attended_visual).squeeze(1)

        if len(self.modalities) == 2:
            query_modality = "t" if "t" in self.modalities else "a"
            key_modality = [m for m in self.modalities if m != query_modality][0]
            attended, _ = self.cross_attention(
                tokens[query_modality], tokens[key_modality], tokens[key_modality])
            fused = self.attention_norm(tokens[query_modality] + attended)
            return self.output_norm(fused + self.fusion_ffn(fused)).squeeze(1)

        raise RuntimeError(f"unsupported modality configuration: {self.modalities}")

    def encode(self, batch: dict) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Convenience: per-modality features plus the fused clip feature
        (the contract used by the live shared reference path)."""
        features = self.encode_modalities(batch)
        return features, self.fuse(features)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, batch: dict,
                reference_fused: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Returns a dict with the fused feature, the per-modality contrastive
        projections (``contrastive_<m>``), and the two heads' logits. When a
        reference feature is given, the heads read the identity-anchored
        feature ``fused + reference_alpha * reference_fused`` (Eq. 6).
        """
        features = self.encode_modalities(batch)
        fused = self.fuse(features)
        outputs: Dict[str, torch.Tensor] = {"fused": fused}
        for modality in self.modalities:
            outputs[f"contrastive_{modality}"] = (
                features[modality] if self.contrastive_projections is None
                else self.contrastive_projections[modality](features[modality]))

        anchored = fused if reference_fused is None \
            else fused + self.config["reference_alpha"] * reference_fused
        if hasattr(self, "binary_head"):
            outputs["binary_logits"] = self.binary_head(anchored)
        outputs["technique_logits"] = self.technique_head(anchored)
        if hasattr(self, "aux_technique_head"):
            outputs["aux_technique_logits"] = self.aux_technique_head(anchored)
        return outputs
