"""Compact text stream: MiniTransformerText behind config["text_encoder"] == "minitf"."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmfdn_ablation.RMFDN import RMFDN
from rmfdn_ablation.encoders import MiniTransformerText

_CONFIG = {"embed_dim": 512, "attention_heads": 8, "contrastive_dim": 128,
           "reference_alpha": 0.001, "max_text_tokens": 50}


def _batch(padded_from: int = 30):
    ids = torch.randint(1, MiniTransformerText.VOCAB_SIZE, (2, 50))
    mask = torch.ones(2, 50, dtype=torch.long)
    ids[1, padded_from:] = 0
    mask[1, padded_from:] = 0
    return ids, mask


def test_minitf_output_shape_and_size():
    torch.manual_seed(0)
    encoder = MiniTransformerText(_CONFIG, 512)
    ids, mask = _batch()
    sequence, pooled = encoder.encode(ids, mask)
    assert sequence.shape == (2, 50, 512)
    assert pooled.shape == (2, 512)
    assert not torch.isnan(pooled).any()
    n = sum(p.numel() for p in encoder.parameters())
    assert 2.1e6 < n < 2.6e6, n


def test_minitf_padding_does_not_leak():
    torch.manual_seed(0)
    encoder = MiniTransformerText(_CONFIG, 512).eval()
    ids, mask = _batch(padded_from=30)
    with torch.no_grad():
        pooled = encoder(ids, mask)
        ids_garbage = ids.clone()
        ids_garbage[1, 30:] = torch.randint(1, MiniTransformerText.VOCAB_SIZE, (20,))
        assert torch.allclose(encoder(ids_garbage, mask), pooled, atol=1e-5)


def test_minitf_is_order_sensitive():
    torch.manual_seed(0)
    encoder = MiniTransformerText(_CONFIG, 512).eval()
    ids, mask = _batch(padded_from=50)
    with torch.no_grad():
        a = encoder(ids, mask)
        b = encoder(ids.flip(1), mask)
    assert not torch.allclose(a, b, atol=1e-4)


def test_rmfdn_builds_minitf_when_asked():
    model = RMFDN(dict(_CONFIG, text_encoder="minitf"), ["t"])
    assert isinstance(model.encoders["t"], MiniTransformerText)
    ids, mask = _batch()
    outputs = model({"input_ids": ids, "attention_mask": mask})
    assert outputs["technique_logits"].shape == (2, 7)
    assert outputs["binary_logits"].shape == (2, 2)
    assert outputs["contrastive_t"].shape == (2, 128)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all minitf tests passed")
