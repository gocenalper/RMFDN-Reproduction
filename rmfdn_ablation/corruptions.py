"""Input corruptions: training-time modality dropout and test-time conditions.

Everything here works on the tensors the dataset produces, so the same
definition of "missing" is used in training (modality dropout) and in testing
(the missing-modality conditions):

* video frames are uint8 ``(T, 3, H, W)`` in [0, 255]; a missing video is
  all-black frames (zeros, before ImageNet normalisation);
* the audio input is the AST log-mel ``(1024, 128)``; a missing audio track
  is the all-zero input, which is also what an unreadable audio file becomes.

Test conditions (report Section 6) are applied to archived inputs with fixed
seeds, so every arm sees exactly the same corrupted clips.
"""

from __future__ import annotations

import zlib
from typing import Dict, List, Optional

import numpy as np
import torch

#: Test-time conditions, in report order. ``audio``/``video`` give the kind of
#: corruption; ``snr_db`` is used by the noise conditions.
CONDITIONS: Dict[str, Dict] = {
    "clean":            {"audio": None,       "video": None},
    "audio_missing":    {"audio": "missing",  "video": None},
    "audio_noise_10db": {"audio": "noise",    "video": None, "snr_db": 10.0},
    "audio_noise_0db":  {"audio": "noise",    "video": None, "snr_db": 0.0},
    "audio_mismatch":   {"audio": "mismatch", "video": None},
    "video_blur":       {"audio": None,       "video": "blur", "sigma": 3.0},
    "video_missing":    {"audio": None,       "video": "missing"},
}

#: Conditions that also occur in training for the dropout arms.
SEEN_IN_DROPOUT_TRAINING = ("clean", "audio_missing", "video_missing")

#: Conditions averaged for the "audio corrupted" summary of the hypothesis.
AUDIO_CORRUPTED = ("audio_missing", "audio_noise_10db", "audio_noise_0db", "audio_mismatch")


def condition(name: str) -> Dict:
    if name not in CONDITIONS:
        raise ValueError(f"unknown condition {name!r}; choose from {list(CONDITIONS)}")
    return CONDITIONS[name]


def clip_seed(directory: str, seed: int) -> int:
    """Deterministic per-clip seed, independent of data order and workers."""
    return (zlib.crc32(directory.encode("utf-8")) + 1_000_003 * seed) % (2 ** 32)


def add_white_noise(waveform: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    """Add white Gaussian noise at ``snr_db`` relative to the clip's own power.

    A silent clip (zero power) gets noise of unit-free power 1e-4 so that the
    condition is still defined.
    """
    generator = torch.Generator().manual_seed(int(seed))
    signal_power = float(waveform.pow(2).mean())
    if signal_power <= 0.0:
        signal_power = 1e-4
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn(waveform.shape, generator=generator) * noise_power ** 0.5
    return waveform + noise.to(waveform.dtype)


def blank_frames(frames: torch.Tensor) -> torch.Tensor:
    """All-black frames with the same shape and dtype."""
    return torch.zeros_like(frames)


def blank_audio(mel: torch.Tensor) -> torch.Tensor:
    """All-zero AST input with the same shape and dtype."""
    return torch.zeros_like(mel)


def blur_frames(frames: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    """Gaussian blur of every frame (uint8 in, uint8 out)."""
    from torchvision.transforms.functional import gaussian_blur

    kernel = int(2 * round(3 * sigma) + 1)
    blurred = gaussian_blur(frames.float(), kernel_size=[kernel, kernel], sigma=[sigma, sigma])
    return blurred.round().clamp(0, 255).to(frames.dtype)


def mismatch_permutation(count: int, seed: int) -> List[int]:
    """A fixed derangement: index i takes its audio from clip ``perm[i] != i``."""
    if count < 2:
        return list(range(count))
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    # shifting a random cycle by one guarantees no fixed point
    perm = np.empty(count, dtype=int)
    perm[order] = np.roll(order, -1)
    return perm.tolist()


def apply_modality_dropout(batch: dict, modalities, probability: float,
                           generator: torch.Generator) -> Dict[str, int]:
    """Training-time modality dropout (report Section 5.2, after ModDrop).

    Each sample independently loses its audio with ``probability`` and its
    video with ``probability``, never both. Labels are left unchanged. Works
    in place on the batch tensors; returns how many samples lost each input.
    """
    counts = {"audio": 0, "video": 0}
    if probability <= 0.0 or not ("a" in modalities and "v" in modalities):
        return counts
    size = batch["is_fake"].shape[0]
    draw = torch.rand(size, generator=generator)
    drop_audio = draw < probability
    drop_video = (draw >= probability) & (draw < 2 * probability)
    if drop_audio.any():
        index = drop_audio.nonzero().squeeze(1).to(batch["mel"].device)
        batch["mel"][index] = 0
        counts["audio"] = int(drop_audio.sum())
    if drop_video.any():
        index = drop_video.nonzero().squeeze(1).to(batch["frames"].device)
        batch["frames"][index] = 0
        counts["video"] = int(drop_video.sum())
    return counts
