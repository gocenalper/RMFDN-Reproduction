"""Shared-encoder reference path for paper Eq. 5-6."""

from __future__ import annotations

from typing import Dict

import torch


def encode_reference_batch(model, batch: dict) -> Dict[str, torch.Tensor]:
    """Encode the paired pristine references with the model's live encoders."""
    reference_batch = {
        "frames": batch["reference_frames"],
        "mel": batch["reference_mel"],
        "input_ids": batch["reference_input_ids"],
        "attention_mask": batch["reference_attention_mask"],
    }
    features, fused = model.encode(reference_batch)
    encoded = {"fused": fused}
    for modality in model.modalities:
        encoded[f"contrastive_{modality}"] = model.contrastive_projections[modality](
            features[modality])
    return encoded
