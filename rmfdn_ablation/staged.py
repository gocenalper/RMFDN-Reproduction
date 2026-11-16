"""Staged fusion: hold the features fixed, vary only the fusion mechanism.

A jointly-trained cross-attention cell cannot say how much of its advantage
came from the interaction, because its encoders moved too: better
per-modality features and the fusion module are confounded in a single
number. Isolating fusion needs two cells that differ in the fusion mechanism
and in nothing else -- same encoders, same weights, same frozen state:

* ``m0`` (:class:`StagedFusion`) concatenates the expert features and reads
  out. There is no path from one modality to another.
* ``m1`` (:class:`~rmfdn_ablation.RMFDN.RMFDN`) run with ``freeze_experts``
  keeps the paper's progressive cross-attention on top of the same frozen
  encoders.

Their difference is what cross-attention contributes at fixed features. The
difference between the frozen ``m1`` and the jointly-trained ``m1`` is what
training the encoders through the fusion adds on top.

Freezing is strict: parameters get ``requires_grad=False`` and the encoders
are pinned to eval mode, so the ResNet's batch-norm running statistics stay
exactly as the expert left them.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

from .RMFDN import RMFDN


def load_expert_encoders(model, checkpoints: Dict[str, str]) -> None:
    """Copy ``encoders.<m>.*`` tensors from each modality's unimodal run."""
    for modality in model.modalities:
        path = checkpoints.get(modality)
        if not path:
            continue
        state = torch.load(path, map_location="cpu")
        state = state.get("model", state)
        prefix = f"encoders.{modality}."
        weights = {key[len(prefix):]: value for key, value in state.items()
                   if key.startswith(prefix)}
        if not weights:
            raise ValueError(f"no '{prefix}*' tensors in {path}")
        model.encoders[modality].load_state_dict(weights)
        print(f"[staged] loaded '{modality}' expert encoder "
              f"({len(weights)} tensors) from {path}", flush=True)


def freeze_encoders(model) -> int:
    """Freeze every encoder; returns the remaining trainable parameter count."""
    for encoder in model.encoders.values():
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        encoder.eval()
    model.frozen_encoders = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class StagedFusion(RMFDN):
    """RMFDN with cross-attention replaced by a concatenation readout.

    This is the no-interaction floor of the ladder, not the fusion test
    itself: it is the cell that the frozen cross-attention cell is measured
    against. ``config["expert_checkpoints"]`` supplies the encoders; it is
    optional, so test evaluation -- which loads the run's own state dict
    immediately afterwards -- can construct the model without them.
    """

    def __init__(self, config: dict, modalities: Sequence[str]):
        super().__init__(config, modalities)
        embed_dim = config["embed_dim"]

        # Delete the cross-attention machinery outright: this model must have
        # no path by which one modality can look at another.
        for attribute in ("modality_self_attention", "cross_attention",
                          "attention_norm", "output_norm", "fusion_ffn",
                          "text_audio_attention", "text_audio_norm",
                          "audiotext_visual_attention", "fusion_norm"):
            if hasattr(self, attribute):
                delattr(self, attribute)

        width = embed_dim * len(self.modalities)
        self.readout = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))

        load_expert_encoders(self, config.get("expert_checkpoints") or {})
        # ``staged_trainable`` turns the frozen concat cell into its trained
        # twin (the fusion ablation's fm_c rung): same architecture, encoders
        # random-initialised and trained jointly through the readout.
        if not config.get("staged_trainable"):
            freeze_encoders(self)

    def fuse(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate the per-modality features and read out. No interaction."""
        if len(self.modalities) == 1:
            return features[self.modalities[0]]
        stacked = torch.cat([features[m] for m in self.modalities], dim=-1)
        return self.readout(stacked)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
