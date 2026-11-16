"""Clip discovery and ground-truth labelling.

The dataset lives on disk as::

    {data_root}/{split}/{forgery_type}/{identity}/.../<clip>/
        frames.jpz.npz | frames.jpz | frames_ndarray.npy   (video frames)
        <clip>.mp3                                          (audio track)
        <clip>.txt                                          (transcript)

A directory is recognised as a clip when it contains a frame archive. The
clip's labels are derived entirely from its ``forgery_type`` directory name
via :data:`~rmfdn_ablation.config.FORGERY_TYPE_TO_TECHNIQUES`:

* ``is_fake``            -- binary target (0 = pristine, 1 = any manipulation),
* ``technique_targets``  -- 7-bit multi-hot vector over the atomic techniques.

This module is intentionally torch-free so the labelling logic can be unit
tested without the deep-learning stack.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

from .config import (BIT_PATTERN_TYPES, FAVC_TYPE_TO_TECHNIQUES,
                     FORGERY_TYPE_TO_TECHNIQUES)

#: IDForge types plus the FakeAVCeleb adapter types; directory names resolve here.
_ALL_TYPES = {**FORGERY_TYPE_TO_TECHNIQUES, **FAVC_TYPE_TO_TECHNIQUES,
              **BIT_PATTERN_TYPES}

#: Filenames that mark a directory as a clip. ``frames.jpz(.npz)`` holds
#: JPEG-encoded frames (full dataset); ``frames_ndarray.npy`` raw uint8
#: frames (subset).
_FRAME_ARCHIVE_PREFIX = "frames.jpz"
_FRAME_RAW_ARRAY = "frames_ndarray.npy"


def clip_basename(clip_directory: str) -> str:
    """Clip stem used for its sidecar files (``<stem>.mp3`` / ``<stem>.txt``)."""
    name = os.path.basename(clip_directory.rstrip("/"))
    return name[:-4] if name.endswith(".mp4") else name


def identity_from_path(path: str) -> str:
    """Extract the speaker identity tag (``idNN``) from any path segment."""
    for segment in path.split(os.sep):
        if segment.startswith("id") and len(segment) >= 4 and segment[2:4].isdigit():
            return segment[:4]
    return "id??"


def _has_frame_archive(filenames: Sequence[str]) -> bool:
    return _FRAME_RAW_ARRAY in filenames or any(
        name.startswith(_FRAME_ARCHIVE_PREFIX) for name in filenames
    )


class ClipRecord:
    """One dataset clip: where it lives and what its ground truth is."""

    __slots__ = ("directory", "basename", "forgery_type", "identity",
                 "is_fake", "technique_targets")

    def __init__(self, directory: str, forgery_type: str):
        self.directory = directory
        self.basename = clip_basename(directory)
        self.forgery_type = forgery_type
        self.identity = identity_from_path(directory)
        self.is_fake = 0 if forgery_type == "pristine" else 1
        self.technique_targets = _ALL_TYPES[forgery_type]


def scan_split_clips(data_root: str, split: str) -> List[ClipRecord]:
    """Walk ``{data_root}/{split}`` and return every clip with its labels.

    A missing split directory yields an empty list (with a warning) rather than
    failing -- this lets a split be trained with the test data unextracted
    (paired with ``skip_test_eval``); the deferred test pass scans it later.
    """
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        print(f"[data_catalog] WARNING: split directory absent -> 0 clips: {split_dir}",
              flush=True)
        return []
    clips: List[ClipRecord] = []
    for forgery_type in sorted(os.listdir(split_dir)):
        type_dir = os.path.join(split_dir, forgery_type)
        if forgery_type not in _ALL_TYPES or not os.path.isdir(type_dir):
            continue
        for directory, _subdirs, filenames in os.walk(type_dir):
            if _has_frame_archive(filenames):
                clips.append(ClipRecord(directory, forgery_type))
    return clips


def scan_reference_clips(reference_root: str) -> Dict[str, List[str]]:
    """Index pristine reference clips by identity for paper Eq. 5-6."""
    index: Dict[str, List[str]] = {}
    for directory, _subdirs, filenames in os.walk(reference_root):
        if _has_frame_archive(filenames):
            index.setdefault(identity_from_path(directory), []).append(directory)
    return {identity: sorted(directories)
            for identity, directories in index.items()}
