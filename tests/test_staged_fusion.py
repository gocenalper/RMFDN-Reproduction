"""Staged fusion (m0): frozen experts, trained readout, no interaction."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmfdn_ablation import RMFDN as rmfdn_module
from rmfdn_ablation.experiment import model_class_for
from rmfdn_ablation.staged import StagedFusion, freeze_encoders, load_expert_encoders
from rmfdn_ablation.trainer import _build_optimizer


class _TinyPooledEncoder(nn.Module):
    def __init__(self, *_args):
        super().__init__()
        self.backbone = nn.Linear(2, 2)

    def forward(self, values, *_args):
        return values.float().mean(1)


_CONFIG = {
    "embed_dim": 2, "attention_heads": 1, "contrastive_dim": 2,
    "reference_alpha": 0.001, "peak_lr": 1e-4, "weight_decay": 0.01,
    "backbone_lr_scale": 0.1,
}
_BATCH = {
    "frames": torch.randn(2, 3, 2), "mel": torch.randn(2, 4, 2),
    "input_ids": torch.randn(2, 5, 2),
    "attention_mask": torch.ones(2, 5, dtype=torch.long),
}


def _tiny(modalities, config=None):
    with mock.patch.object(rmfdn_module, "VisualEncoder", _TinyPooledEncoder), \
         mock.patch.object(rmfdn_module, "AudioEncoder", _TinyPooledEncoder), \
         mock.patch.object(rmfdn_module, "TextEncoder", _TinyPooledEncoder):
        return StagedFusion(config or _CONFIG, modalities)


def test_m0_resolves_to_staged_fusion():
    assert model_class_for("m0") is StagedFusion


def test_staged_fusion_has_no_cross_modal_path():
    for modalities in (["v", "a"], ["a", "t"], ["v", "a", "t"]):
        model = _tiny(modalities)
        for attribute in ("cross_attention", "text_audio_attention",
                          "audiotext_visual_attention", "modality_self_attention",
                          "fusion_ffn"):
            assert not hasattr(model, attribute), (modalities, attribute)
        outputs = model(_BATCH)
        assert outputs["binary_logits"].shape == (2, 2)
        assert outputs["technique_logits"].shape == (2, 7)


def test_encoders_are_frozen_and_stay_in_eval():
    model = _tiny(["v", "a", "t"])
    assert all(not p.requires_grad
               for encoder in model.encoders.values()
               for p in encoder.parameters())
    assert model.trainable_parameter_count() > 0

    model.train()                       # the training loop calls this every epoch
    assert all(not encoder.training for encoder in model.encoders.values())
    assert model.readout.training       # the readout does train


def test_optimizer_skips_frozen_encoders():
    model = _tiny(["v", "a"])
    optimizer = _build_optimizer(model, ["v", "a"], _CONFIG)

    held = {id(p) for group in optimizer.param_groups for p in group["params"]}
    frozen = {id(p) for encoder in model.encoders.values()
              for p in encoder.parameters()}
    assert not (held & frozen)
    assert held == {id(p) for p in model.parameters() if p.requires_grad}


def test_readout_sees_every_modality():
    """A concatenation readout must actually depend on each input stream."""
    model = _tiny(["v", "a", "t"])
    base = model(_BATCH)["technique_logits"]
    for key in ("frames", "mel", "input_ids"):
        perturbed = {**_BATCH, key: _BATCH[key] + 5.0}
        assert not torch.allclose(model(perturbed)["technique_logits"], base), key


def test_expert_checkpoint_loading(tmp_path):
    donor = _tiny(["v"])
    with torch.no_grad():
        for parameter in donor.encoders["v"].parameters():
            parameter.fill_(0.375)
    path = tmp_path / "expert_V.pt"
    torch.save(donor.state_dict(), path)

    model = _tiny(["v", "a"], {**_CONFIG, "expert_checkpoints": {"v": str(path)}})
    assert all(torch.allclose(p, torch.full_like(p, 0.375))
               for p in model.encoders["v"].parameters())
    # the untouched modality keeps its initialisation
    assert not all(torch.allclose(p, torch.full_like(p, 0.375))
                   for p in model.encoders["a"].parameters())


def test_frozen_m1_keeps_its_cross_attention():
    """The isolated fusion test: same frozen encoders, cross-attention intact."""
    with mock.patch.object(rmfdn_module, "VisualEncoder", _TinyPooledEncoder), \
         mock.patch.object(rmfdn_module, "AudioEncoder", _TinyPooledEncoder), \
         mock.patch.object(rmfdn_module, "TextEncoder", _TinyPooledEncoder):
        model = rmfdn_module.RMFDN(_CONFIG, ["v", "a"])

    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable = freeze_encoders(model)

    assert hasattr(model, "cross_attention")        # the mechanism survives
    assert trainable < trainable_before             # the encoders do not
    assert all(not p.requires_grad
               for encoder in model.encoders.values()
               for p in encoder.parameters())
    assert any(p.requires_grad for p in model.cross_attention.parameters())

    model.train()
    assert all(not encoder.training for encoder in model.encoders.values())
    assert model.cross_attention.training


def test_freezing_is_opt_in_for_the_base_model():
    with mock.patch.object(rmfdn_module, "VisualEncoder", _TinyPooledEncoder), \
         mock.patch.object(rmfdn_module, "AudioEncoder", _TinyPooledEncoder), \
         mock.patch.object(rmfdn_module, "TextEncoder", _TinyPooledEncoder):
        model = rmfdn_module.RMFDN(_CONFIG, ["v", "a"])

    model.train()
    assert all(encoder.training for encoder in model.encoders.values())
    assert all(p.requires_grad for p in model.parameters())
