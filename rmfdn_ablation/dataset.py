"""PyTorch dataset and batch collation for multimodal clips.

The dataset is *modality-aware*: it loads only the inputs the current
configuration consumes (a text-only run never decodes a single video frame),
which is what makes the lighter ablation cells cheap. Inactive modalities are
replaced by 1-element placeholder tensors so every sample has the same keys
and default stacking works.

Batch keys produced by the collate function:

    frames            (B, T, 3, H, W)  -- placeholder when "v" inactive
    mel               (B, 1024, 128) AST input -- placeholder when "a" inactive
    input_ids,
    attention_mask    BERT-tokenised transcripts (empty string when "t" inactive)
    is_fake           (B,) long, binary target
    technique_targets (B, 7) float, multi-hot ground truth
    identity          list[str], speaker tags for the identity-aware loss
"""

from __future__ import annotations

import random
import zlib
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .data_catalog import ClipRecord, clip_basename
from .media_io import load_ast_input, load_frame_tensor, load_transcript


class MultimodalClipDataset(Dataset):
    """Clips with on-the-fly media loading for the active modalities only."""

    def __init__(self, clips: Sequence[ClipRecord], modalities: Sequence[str], config: dict,
                 reference_index: Optional[Dict[str, List[str]]] = None,
                 reference_seed: Optional[int] = None):
        self.clips = list(clips)
        self.modalities = list(modalities)
        self.config = config
        self.reference_index = reference_index
        self.reference_seed = reference_seed

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict:
        clip = self.clips[index]
        sample = {
            "is_fake": clip.is_fake,
            "technique_targets": torch.tensor(clip.technique_targets, dtype=torch.float32),
            "identity": clip.identity,
        }
        if "v" in self.modalities:
            sample["frames"] = load_frame_tensor(clip.directory, self.config)
        else:
            sample["frames"] = torch.zeros(1)
        sample["mel"] = (
            load_ast_input(clip.directory, clip.basename, self.config)
            if "a" in self.modalities else torch.zeros(1)
        )
        sample["transcript"] = (
            load_transcript(clip.directory, clip.basename)
            if "t" in self.modalities else ""
        )
        if self.reference_index is not None:
            reference_clips = self.reference_index.get(clip.identity)
            if not reference_clips:
                raise KeyError(f"no pristine reference clips for identity {clip.identity!r}")
            if self.reference_seed is None:
                reference_directory = random.choice(reference_clips)
            else:
                identity_seed = zlib.crc32(clip.identity.encode("utf-8"))
                rng = random.Random(self.reference_seed + identity_seed + index * 1_000_003)
                reference_directory = rng.choice(reference_clips)
            reference_basename = clip_basename(reference_directory)
            sample["reference_frames"] = (
                load_frame_tensor(reference_directory, self.config)
                if "v" in self.modalities else torch.zeros(1)
            )
            sample["reference_mel"] = (
                load_ast_input(reference_directory, reference_basename, self.config)
                if "a" in self.modalities else torch.zeros(1)
            )
            sample["reference_transcript"] = (
                load_transcript(reference_directory, reference_basename)
                if "t" in self.modalities else ""
            )
        return sample


def build_collate_fn(tokenizer, config: dict) -> Callable[[List[dict]], dict]:
    """Collate that stacks tensors and BERT-tokenises the transcripts.
    """

    def collate(batch: List[dict]) -> dict:
        collated = {
            "frames": torch.stack([sample["frames"] for sample in batch]),
            "mel": torch.stack([sample["mel"] for sample in batch]),
            "is_fake": torch.tensor([sample["is_fake"] for sample in batch], dtype=torch.long),
            "technique_targets": torch.stack([sample["technique_targets"] for sample in batch]),
            "identity": [sample["identity"] for sample in batch],
        }
        encoded = tokenizer(
            [sample["transcript"] for sample in batch],
            padding="max_length", truncation=True,
            max_length=config["max_text_tokens"], return_tensors="pt",
        )
        collated["input_ids"] = encoded["input_ids"]
        collated["attention_mask"] = encoded["attention_mask"]
        if "reference_frames" in batch[0]:
            collated["reference_frames"] = torch.stack(
                [sample["reference_frames"] for sample in batch])
            collated["reference_mel"] = torch.stack(
                [sample["reference_mel"] for sample in batch])
            reference_encoded = tokenizer(
                [sample["reference_transcript"] for sample in batch],
                padding="max_length", truncation=True,
                max_length=config["max_text_tokens"], return_tensors="pt",
            )
            collated["reference_input_ids"] = reference_encoded["input_ids"]
            collated["reference_attention_mask"] = reference_encoded["attention_mask"]
        return collated

    return collate
