"""Canonical frame, pretrained-AST audio, and transcript loading.

Each loader returns a fixed-shape tensor so that batches can be stacked
without padding logic downstream:

* frames      -> uint8 ``(frames_per_clip, 3, image_size, image_size)`` in
                 [0,255] (the model does /255 + ImageNet-normalise on the GPU),
* AST input   -> float32 ``(1024, 128)``,
* transcript  -> plain string (tokenised later, in the collate function).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

#: Lazily-constructed AST feature extractor (pretrained-AST audio path).
_AST_EXTRACTOR = None

#: Set after the first undecodable-audio warning so logs are not flooded.
_AUDIO_WARNING_SHOWN = False


def _ensure_numpy_pickle_compat() -> None:
    """Let NumPy 1.x open object arrays created by NumPy 2.x."""
    try:
        __import__("numpy._core")
        return
    except ImportError:
        import numpy.core as numpy_core

    sys.modules.setdefault("numpy._core", numpy_core)
    for name in ("multiarray", "numeric", "umath", "overrides", "_multiarray_umath"):
        module = getattr(numpy_core, name, None)
        if module is not None:
            sys.modules.setdefault(f"numpy._core.{name}", module)


def load_frame_tensor(clip_directory: str, config: dict) -> torch.Tensor:
    """Load a clip's frames as a uint8 ``(T, 3, H, W)`` tensor in [0,255].

    Frames are stored either as a JPEG archive (``frames.jpz[.npz]``: an
    object array of encoded JPEG buffers, key ``"j"``) or as a raw uint8
    array (``frames_ndarray.npy``).

    NOTE (memory): full-dataset JPEG clips may contain far more frames than
    the model consumes, so temporal subsampling happens *before* JPEG
    decoding. Decoding everything first ballooned DataLoader worker RSS and
    OOM-killed the overnight 90k run; this ordering is load-bearing.
    """
    archive = os.path.join(clip_directory, "frames.jpz.npz")
    if not os.path.exists(archive):
        archive = os.path.join(clip_directory, "frames.jpz")
    if os.path.exists(archive):
        import cv2

        _ensure_numpy_pickle_compat()
        encoded_frames = np.load(archive, allow_pickle=True)["j"]
        target_count = int(config["frames_per_clip"])
        if len(encoded_frames) > target_count:  # subsample BEFORE decode
            keep = np.linspace(0, len(encoded_frames) - 1, target_count).astype(int)
            encoded_frames = encoded_frames[keep]
        frames = np.stack([
            cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
            for buffer in encoded_frames
        ])
    else:
        frames = np.load(os.path.join(clip_directory, "frames_ndarray.npy"))

    tensor = torch.from_numpy(np.ascontiguousarray(frames))
    if tensor.ndim == 4 and tensor.shape[-1] == 3:          # (T,H,W,C) -> (T,C,H,W)
        tensor = tensor.permute(0, 3, 1, 2).contiguous()

    target_count = config["frames_per_clip"]
    if tensor.shape[0] != target_count:                      # uniform subsample / pad
        if tensor.shape[0] > target_count:
            tensor = tensor[torch.linspace(0, tensor.shape[0] - 1, target_count).long()]
        else:
            last = tensor[-1:].repeat(target_count - tensor.shape[0], 1, 1, 1)
            tensor = torch.cat([tensor, last], 0)

    # Return uint8 [0,255]; the model does /255 + ImageNet-normalise on the GPU,
    # keeping the worker->main IPC payload 4x smaller than float32. The rare
    # sources that arrive as float (or need a resize) are rounded back to uint8.
    if tensor.dtype != torch.uint8:
        scaled = tensor.float()
        if scaled.max() <= 1.5:                              # source was [0,1] float
            scaled = scaled * 255.0
        tensor = scaled.round().clamp(0, 255).to(torch.uint8)
    if tensor.shape[-1] != config["image_size"] or tensor.shape[-2] != config["image_size"]:
        tensor = F.interpolate(tensor.float(), size=(config["image_size"],) * 2,
                               mode="bilinear", align_corners=False
                               ).round().clamp(0, 255).to(torch.uint8)
    return tensor


def _decode_audio(path: str):
    """Decode an audio file to a float32 ``(channels, samples)`` tensor.

    libsndfile (via ``soundfile``) reads wav / flac / mp3 identically on every
    platform and needs no system codec. Containers it cannot open (``.m4a``)
    fall back to ``torchaudio.load``, whose backend depends on the installed
    torchaudio version.
    """
    try:
        import soundfile

        samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(samples.T.copy()), sample_rate
    except Exception:
        import torchaudio

        return torchaudio.load(path)


def _warn_undecodable_audio(path: str, error: Exception) -> None:
    """Say once per process that an existing audio file could not be decoded.

    The clip still loads (as silence) so a single broken file cannot abort a
    long run, but a decoder that fails for *every* file -- a missing audio
    backend -- must not pass unnoticed.
    """
    global _AUDIO_WARNING_SHOWN
    if not _AUDIO_WARNING_SHOWN:
        _AUDIO_WARNING_SHOWN = True
        print(f"[media_io] WARNING: could not decode {path} ({error!r}); "
              f"using silence. Further decode failures are not reported.",
              file=sys.stderr, flush=True)


def load_ast_input(clip_directory: str, basename: str, config: dict,
                   noise_snr_db: Optional[float] = None,
                   noise_seed: int = 0) -> torch.Tensor:
    """Load audio as pretrained-AST ``input_values`` ``(1024, 128)``.

    Uses the model's own ``ASTFeatureExtractor`` (kaldi fbank, 128 mel bins,
    AudioSet normalisation). Missing/unreadable audio degrades to a silent
    all-zero tensor of the same shape so one broken file cannot abort a run.

    ``noise_snr_db`` adds white noise to the waveform at that signal-to-noise
    ratio before the filterbank (test-time corruption, seeded by ``noise_seed``).
    """
    global _AST_EXTRACTOR
    import torchaudio

    if _AST_EXTRACTOR is None:
        from transformers import ASTFeatureExtractor
        _AST_EXTRACTOR = ASTFeatureExtractor.from_pretrained(config["ast_checkpoint"])
    max_len = _AST_EXTRACTOR.max_length          # 1024 frames
    num_bins = _AST_EXTRACTOR.num_mel_bins        # 128

    waveform = sample_rate = None
    for extension in (".mp3", ".wav", ".flac", ".m4a"):
        candidate = os.path.join(clip_directory, basename + extension)
        if os.path.exists(candidate):
            try:
                waveform, sample_rate = _decode_audio(candidate)
                break
            except Exception as error:
                _warn_undecodable_audio(candidate, error)
                waveform = None
    if waveform is None:
        return torch.zeros(max_len, num_bins)

    if waveform.shape[0] > 1:                                # stereo -> mono
        waveform = waveform.mean(0, keepdim=True)
    if sample_rate != config["sample_rate"]:
        waveform = torchaudio.functional.resample(waveform, sample_rate, config["sample_rate"])
    waveform = waveform[:, : int(config["max_audio_seconds"] * config["sample_rate"])]
    if noise_snr_db is not None:
        from .corruptions import add_white_noise
        waveform = add_white_noise(waveform, noise_snr_db, noise_seed)
    features = _AST_EXTRACTOR(waveform.squeeze(0).numpy(), sampling_rate=config["sample_rate"],
                              return_tensors="pt")
    audio = features["input_values"].squeeze(0)            # (T, 128)
    # AST's position embeddings are fixed to ``max_length`` frames; short clips
    # come back shorter, so pad/truncate the time axis to exactly max_length.
    if audio.shape[0] < max_len:
        audio = F.pad(audio, (0, 0, 0, max_len - audio.shape[0]))
    return audio[:max_len]                                 # (1024, 128)


def load_transcript(clip_directory: str, basename: str) -> str:
    """Load ``<basename>.txt``; missing transcripts become the empty string."""
    path = os.path.join(clip_directory, basename + ".txt")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read().strip()
