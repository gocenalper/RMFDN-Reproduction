#!/usr/bin/env python3
"""Regression tests for evaluation metrics.

Runs without media files or model downloads by using a tiny Dataset plus a
deterministic dummy model. Run: ``python3 tests/test_evaluation.py``.
"""

import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:
    print("skipped: torch not installed")
    sys.exit(0)

from rmfdn_ablation.evaluation import _reference_apmeter_metrics, evaluate


TARGETS = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0],
], dtype=np.float32)

# Every matrix cell scores the complete taxonomy. Columns 0,1,2 vary in this
# fixture; columns 3-6 are single-valued and therefore excluded from mAP/F1.
FULL_LOGITS = torch.tensor([
    [-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0],
    [4.0, -4.0, 4.0, -4.0, -4.0, -4.0, -4.0],
    [4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0],
    [-4.0, 4.0, -4.0, -4.0, -4.0, -4.0, -4.0],
    [-4.0, 4.0, 4.0, -4.0, -4.0, -4.0, -4.0],
])


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(TARGETS)

    def __getitem__(self, index):
        return {
            "frames": torch.zeros(1),
            "mel": torch.zeros(1),
            "transcript": "",
            "is_fake": int(TARGETS[index].any()),
            "technique_targets": torch.tensor(TARGETS[index]),
            "identity": f"id{index:02d}",
        }


class TinyTokenizer:
    def __call__(self, texts, **_kwargs):
        batch_size = len(texts)
        return {
            "input_ids": torch.zeros((batch_size, 1), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 1), dtype=torch.long),
        }


class TinyModel:
    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, batch, reference_fused=None):
        batch_size = batch["is_fake"].shape[0]
        return {
            "binary_logits": torch.tensor([[-4.0, 4.0]]).repeat(batch_size, 1),
            "technique_logits": FULL_LOGITS[:batch_size],
        }


def test_map_scores_only_varying_pins_of_the_full_taxonomy():
    metrics = evaluate(
        TinyModel(),
        TinyDataset(),
        TinyTokenizer(),
        {"batch_size": len(TARGETS), "num_workers": 0, "max_text_tokens": 1},
        device="cpu",
    )

    scores = torch.sigmoid(FULL_LOGITS).numpy()
    scoreable = [i for i in range(TARGETS.shape[1]) if 0 < TARGETS[:, i].sum() < len(TARGETS)]
    expected_map = average_precision_score(
        TARGETS[:, scoreable], scores[:, scoreable], average="macro"
    ) * 100

    assert metrics["nkeep"] == len(scoreable) == 3
    assert "tts" in metrics["per_bit_acc"]
    assert math.isclose(metrics["mAP"], expected_map, rel_tol=1e-9)


def test_metrics_can_be_restricted_to_training_visible_bits():
    metrics = evaluate(
        TinyModel(),
        TinyDataset(),
        TinyTokenizer(),
        {"batch_size": len(TARGETS), "num_workers": 0, "max_text_tokens": 1},
        device="cpu",
        technique_indices=(0, 1),
    )

    scores = torch.sigmoid(FULL_LOGITS).numpy()
    expected_map = average_precision_score(
        TARGETS[:, [0, 1]], scores[:, [0, 1]], average="macro"
    ) * 100

    assert metrics["nkeep"] == 2
    assert metrics["metric_bits"] == ["lip_sync", "face_swap"]
    assert set(metrics["per_bit_acc"]) == {"lip_sync", "face_swap"}
    assert math.isclose(metrics["mAP"], expected_map, rel_tol=1e-9)


def test_reference_apmeter_matches_published_ap_and_f1_formulas():
    labels = np.array([
        [1, 1],
        [0, 1],
        [1, 0],
        [0, 0],
    ])
    logits = np.array([
        [3.0, 1.0],
        [2.0, -1.0],
        [1.0, 2.0],
        [-1.0, -2.0],
    ])

    metrics = _reference_apmeter_metrics(labels, logits, (0, 1))

    expected_map = ((1.0 + 2.0 / 3.0) / 2.0
                    + (1.0 / 2.0 + 2.0 / 3.0) / 2.0) / 2.0 * 100
    expected_of1 = 2.0 * 0.6 * 0.75 / (0.6 + 0.75) * 100
    class_precision = ((2.0 / 3.0) + (1.0 / 2.0)) / 2.0
    class_recall = (1.0 + 0.5) / 2.0
    expected_cf1 = (2.0 * class_precision * class_recall
                    / (class_precision + class_recall) * 100)

    assert math.isclose(metrics["mAP"], expected_map, rel_tol=1e-9)
    assert math.isclose(metrics["OF1"], expected_of1, rel_tol=1e-9)
    assert math.isclose(metrics["CF1"], expected_cf1, rel_tol=1e-9)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all evaluation tests passed")
