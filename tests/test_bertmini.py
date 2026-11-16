"""Compact text stream: pretrained BERT-mini/-small/-medium behind config["text_encoder"]."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmfdn_ablation.RMFDN import RMFDN
from rmfdn_ablation.encoders import TextEncoder

_CONFIG = {"embed_dim": 512, "attention_heads": 8, "contrastive_dim": 128,
           "reference_alpha": 0.001, "bert_model": "/models/bert-base-uncased",
           "bert_mini_model": "/models/bert-mini",
           "bert_small_model": "/models/bert-small",
           "bert_medium_model": "/models/bert-medium"}


class _FakeBert(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.table = nn.Embedding(30522, hidden, padding_idx=0)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.table(input_ids))


def _patched(hidden: int):
    return mock.patch("transformers.BertModel.from_pretrained",
                      side_effect=lambda path: _FakeBert(hidden))


def test_bertmini_loads_the_mini_checkpoint():
    with _patched(256) as loader:
        model = RMFDN(dict(_CONFIG, text_encoder="bertmini"), ["t"])
    loader.assert_called_once_with("/models/bert-mini")
    assert isinstance(model.encoders["t"], TextEncoder)
    assert model.encoders["t"].projection.in_features == 256
    ids = torch.randint(1, 30522, (2, 50))
    mask = torch.ones(2, 50, dtype=torch.long)
    outputs = model({"input_ids": ids, "attention_mask": mask})
    assert outputs["technique_logits"].shape == (2, 7)
    assert outputs["contrastive_t"].shape == (2, 128)


def test_bertsmall_loads_the_small_checkpoint():
    with _patched(512) as loader:
        model = RMFDN(dict(_CONFIG, text_encoder="bertsmall"), ["t"])
    loader.assert_called_once_with("/models/bert-small")
    assert model.encoders["t"].projection.in_features == 512


def test_bertmedium_loads_the_medium_checkpoint():
    with _patched(512) as loader:
        model = RMFDN(dict(_CONFIG, text_encoder="bertmedium"), ["t"])
    loader.assert_called_once_with("/models/bert-medium")
    assert model.encoders["t"].projection.in_features == 512


def test_absent_key_still_loads_bert_base():
    with _patched(768) as loader:
        model = RMFDN(_CONFIG, ["t"])
    loader.assert_called_once_with("/models/bert-base-uncased")
    assert model.encoders["t"].projection.in_features == 768


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all bertmini tests passed")
