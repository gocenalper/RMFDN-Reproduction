"""Compact audio stream: MelCNN replaces AST behind config["audio_encoder"]."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmfdn_ablation import RMFDN as rmfdn_module
from rmfdn_ablation.RMFDN import RMFDN
from rmfdn_ablation.encoders import MelCNN


class _TinyEncoder(nn.Module):
    def __init__(self, *_args):
        super().__init__()
        self.backbone = nn.Linear(2, 2)

    def forward(self, values, *_args):
        return values.float().mean(1)


_CONFIG = {"embed_dim": 512, "attention_heads": 8, "contrastive_dim": 128,
           "reference_alpha": 0.001}


def test_melcnn_output_shape_and_size():
    torch.manual_seed(0)
    encoder = MelCNN(_CONFIG, 512, 8)
    out = encoder(torch.randn(2, 1024, 128))
    assert out.shape == (2, 512)
    n = sum(p.numel() for p in encoder.parameters())
    assert 0.9e6 < n < 1.4e6, n


def test_rmfdn_builds_melcnn_when_asked():
    model = RMFDN(dict(_CONFIG, audio_encoder="melcnn"), ["a"])
    assert isinstance(model.encoders["a"], MelCNN)
    outputs = model({"mel": torch.randn(2, 1024, 128)})
    assert outputs["technique_logits"].shape == (2, 7)
    assert outputs["binary_logits"].shape == (2, 2)
    assert outputs["contrastive_a"].shape == (2, 128)


def test_rmfdn_defaults_to_ast_without_the_key():
    with mock.patch.object(rmfdn_module, "AudioEncoder", _TinyEncoder):
        model = RMFDN(_CONFIG, ["a"])
    assert isinstance(model.encoders["a"], _TinyEncoder)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all melcnn tests passed")
