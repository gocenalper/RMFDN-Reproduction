"""Compact text stream: EmbedBagText replaces BERT behind config["text_encoder"]."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmfdn_ablation import RMFDN as rmfdn_module
from rmfdn_ablation.RMFDN import RMFDN
from rmfdn_ablation.encoders import EmbedBagText


class _TinyEncoder(nn.Module):
    def __init__(self, *_args):
        super().__init__()
        self.backbone = nn.Linear(2, 2)

    def forward(self, values, *_args):
        return values.float().mean(1, keepdim=True).expand(-1, 512)


_CONFIG = {"embed_dim": 512, "attention_heads": 8, "contrastive_dim": 128,
           "reference_alpha": 0.001}


def _batch(padded_from: int = 30):
    ids = torch.randint(1, EmbedBagText.VOCAB_SIZE, (2, 50))
    mask = torch.ones(2, 50, dtype=torch.long)
    ids[1, padded_from:] = 0
    mask[1, padded_from:] = 0
    return ids, mask


def test_embedbag_output_shape_and_size():
    torch.manual_seed(0)
    encoder = EmbedBagText(_CONFIG, 512)
    ids, mask = _batch()
    sequence, pooled = encoder.encode(ids, mask)
    assert sequence.shape == (2, 50, 512)
    assert pooled.shape == (2, 512)
    assert torch.equal(encoder(ids, mask), pooled)
    n = sum(p.numel() for p in encoder.parameters())
    assert 1.9e6 < n < 2.3e6, n


def test_embedbag_pooling_ignores_padding():
    torch.manual_seed(0)
    encoder = EmbedBagText(_CONFIG, 512)
    ids, mask = _batch(padded_from=30)
    pooled = encoder(ids, mask)
    ids_garbage = ids.clone()
    ids_garbage[1, 30:] = torch.randint(1, EmbedBagText.VOCAB_SIZE, (20,))
    assert torch.allclose(encoder(ids_garbage, mask), pooled)


def test_rmfdn_builds_embedbag_when_asked():
    model = RMFDN(dict(_CONFIG, text_encoder="embedbag"), ["t"])
    assert isinstance(model.encoders["t"], EmbedBagText)
    ids, mask = _batch()
    outputs = model({"input_ids": ids, "attention_mask": mask})
    assert outputs["technique_logits"].shape == (2, 7)
    assert outputs["binary_logits"].shape == (2, 2)
    assert outputs["contrastive_t"].shape == (2, 128)


def test_rmfdn_defaults_to_bert_without_the_key():
    with mock.patch.object(rmfdn_module, "TextEncoder", _TinyEncoder):
        model = RMFDN(_CONFIG, ["t"])
    assert isinstance(model.encoders["t"], _TinyEncoder)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all embedbag tests passed")
