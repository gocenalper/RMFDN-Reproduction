from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from .attention import CrossModalTransformer
from .config import TECHNIQUE_INDICES
from .encoders import AudioEncoder, TextEncoder, VisualEncoder


class MulGRmfdn(nn.Module):
    """Modified R-MFDN according to the MulG paper.

    Only supports the default BCE 7-bit technique head and projected
    contrastive features; the ``type11`` / ``raw_contrastive`` config keys
    are ignored here.
    """

    def __init__(self, config: dict, modalities: Sequence[str]):
        super().__init__()
        embed_dim, num_heads = config["embed_dim"], config["attention_heads"]
        self.config = config
        self.modalities = list(modalities)
        self.encoders = nn.ModuleDict()
        if "v" in self.modalities:
            self.encoders["v"] = VisualEncoder(config, embed_dim, num_heads)
        if "a" in self.modalities:
            self.encoders["a"] = AudioEncoder(config, embed_dim, num_heads)
        if "t" in self.modalities:
            self.encoders["t"] = TextEncoder(config, embed_dim)

        # ----- fusion ---------------------------------------------------------
        if len(self.modalities) >= 2:
            self.cross_transformers = nn.ModuleDict({
                target: nn.ModuleDict({
                    source: CrossModalTransformer(embed_dim)
                        for source in self.modalities if source != target
                })
                for target in self.modalities
            })
            num_sources = len(self.modalities) - 1
            self.grus = nn.ModuleDict({
                m: nn.GRU(input_size=num_sources * embed_dim, hidden_size=embed_dim,
                          batch_first=True)
                for m in self.modalities
            })
            self.fusion_fc = nn.Linear(len(self.modalities) * embed_dim, embed_dim)
        
        # ----- output heads (parallel, both read the fused feature) ----------
        self.binary_head = nn.Linear(embed_dim, 2)
        self.technique_head = nn.Linear(embed_dim, len(TECHNIQUE_INDICES))

        # ----- contrastive projections for the InfoNCE losses ----------------
        contrastive_dim = config["contrastive_dim"]
        self.contrastive_projections = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                             nn.Linear(embed_dim, contrastive_dim))
            for m in self.modalities})

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #

    def encode_modalities(self, batch: dict):
        """Return per-modality token sequences and canonical pooled features."""
        sequences: Dict[str, torch.Tensor] = {}
        pooled: Dict[str, torch.Tensor] = {}

        if "v" in self.modalities:
            sequences["v"], pooled["v"] = self.encoders["v"].encode(batch["frames"])

        if "a" in self.modalities:
            sequences["a"], pooled["a"] = self.encoders["a"].encode(batch["mel"])

        if "t" in self.modalities:
            sequences["t"], pooled["t"] = self.encoders["t"].encode(
                batch["input_ids"], batch["attention_mask"])

        return sequences, pooled

    def fuse(self, features, batch):
        branch_inputs = {}
        for target in self.modalities:
            ct_outs = []
            for source in self.modalities:
                if source == target:
                    continue
                x2_mask = batch["attention_mask"] if source == "t" else None
                ct_outs.append(self.cross_transformers[target][source](features[target], features[source], x2_mask))
            branch_inputs[target] = torch.cat(ct_outs, dim=-1)
            
        hiddens = []
        for m in self.modalities:
            gru_out, _ = self.grus[m](branch_inputs[m])      # (B, T_m, d)
            if m == "t":
                lengths  = batch["attention_mask"].sum(1)    # (B,) real tokens
                last_idx = (lengths - 1).clamp(min=0)        # last real token index
                h_m = gru_out[torch.arange(gru_out.size(0), device=gru_out.device), last_idx]   # (B, d)
            else:
                h_m = gru_out[:, -1]                         # last step
            hiddens.append(h_m)
        
        fused = self.fusion_fc(torch.cat(hiddens, dim=-1))  # (B, d)
        return fused

    def encode(self, batch):
        seqs, pooled = self.encode_modalities(batch)
        fused = (pooled[self.modalities[0]] if len(self.modalities) == 1
                 else self.fuse(seqs, batch))
        return pooled, fused
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
        features, fused = self.encode(batch)
        outputs: Dict[str, torch.Tensor] = {"fused": fused}
        for modality in self.modalities:
            outputs[f"contrastive_{modality}"] = self.contrastive_projections[modality](
                features[modality])

        anchored = fused if reference_fused is None \
            else fused + self.config["reference_alpha"] * reference_fused
        outputs["binary_logits"] = self.binary_head(anchored)
        outputs["technique_logits"] = self.technique_head(anchored)
        return outputs
