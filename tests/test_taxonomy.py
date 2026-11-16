#!/usr/bin/env python3
"""Pure-logic tests: taxonomy consistency and clip filtering.

Torch-free on purpose -- runs anywhere with ``python3 tests/test_taxonomy.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmfdn_ablation.config import (FORGERY_TYPE_TO_TECHNIQUES, MANIPULATION_TECHNIQUES,
                                   MODALITY_CONFIGS, TECHNIQUE_INDICES)
from rmfdn_ablation.data_catalog import ClipRecord, clip_basename, identity_from_path


def test_taxonomy_shapes():
    assert len(MANIPULATION_TECHNIQUES) == 7
    for forgery_type, vector in FORGERY_TYPE_TO_TECHNIQUES.items():
        assert len(vector) == 7, forgery_type
        if forgery_type == "pristine":
            assert sum(vector) == 0
        else:
            assert sum(vector) >= 1, forgery_type


def test_every_matrix_cell_uses_the_complete_taxonomy():
    assert TECHNIQUE_INDICES == tuple(range(7))


def test_modality_config_names_match_their_modalities():
    for name, modalities in MODALITY_CONFIGS:
        assert name == "".join(sorted(m.upper() for m in modalities),) \
            or set(name.lower()) == set(modalities)


def test_path_parsing():
    assert identity_from_path("/data/train/face_tts/id07_xyz/clip001") == "id07"
    assert clip_basename("/data/.../clip001.mp4") == "clip001"
    assert clip_basename("/data/.../clip001") == "clip001"


def test_clip_targets_use_the_complete_taxonomy():
    pristine = ClipRecord("/d/train/pristine/id01/c1", "pristine")
    face_tts = ClipRecord("/d/train/face_tts/id01/c2", "face_tts")

    assert pristine.is_fake == 0 and face_tts.is_fake == 1
    assert face_tts.technique_targets == (1, 1, 1, 0, 0, 0, 0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all taxonomy tests passed")
