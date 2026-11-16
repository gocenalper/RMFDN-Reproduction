#!/usr/bin/env python3
"""Smoke tests for the torch-dependent pieces that need no downloads/data.

Covers the learning-rate schedule and the InfoNCE losses. Skips gracefully
when torch is not installed. Run: ``python3 tests/test_torch_components.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:
    print("skipped: torch not installed")
    sys.exit(0)

from rmfdn_ablation.config import DEFAULT_CONFIG
from rmfdn_ablation.losses import (cross_modal_consistency_loss,
                                   identity_aware_contrastive_loss, info_nce_loss)
from rmfdn_ablation.trainer import cosine_warmup_lr


def test_schedule_endpoints():
    config = DEFAULT_CONFIG
    assert abs(cosine_warmup_lr(0, config) - config["warmup_start_lr"]) < 1e-12
    assert abs(cosine_warmup_lr(config["warmup_iters"], config) - config["peak_lr"]) < 1e-12
    assert abs(cosine_warmup_lr(config["lr_schedule_iters"], config) - config["min_lr"]) < 1e-9
    mid = cosine_warmup_lr((config["warmup_iters"] + config["lr_schedule_iters"]) // 2, config)
    assert config["min_lr"] < mid < config["peak_lr"]


def test_schedule_horizon_is_independent_of_run_length():
    short_run = dict(DEFAULT_CONFIG, total_iters=12_000, lr_schedule_iters=90_000)
    full_run = dict(DEFAULT_CONFIG, total_iters=90_000, lr_schedule_iters=90_000)

    assert cosine_warmup_lr(6_000, short_run) == cosine_warmup_lr(6_000, full_run)
    assert cosine_warmup_lr(12_000, short_run) > short_run["min_lr"]


def test_info_nce_prefers_the_positive():
    torch.manual_seed(0)
    anchor = torch.randn(1, 128)
    aligned = info_nce_loss(anchor, anchor.clone(), torch.randn(1, 16, 128), 0.1)
    misaligned = info_nce_loss(anchor, -anchor, torch.randn(1, 16, 128), 0.1)
    assert torch.isfinite(aligned) and torch.isfinite(misaligned)
    assert aligned.item() < misaligned.item()  # matching positive -> lower loss


def _fake_outputs(batch_size=6, dim=128):
    torch.manual_seed(0)
    return {"fused": torch.randn(batch_size, 512),
            "contrastive_v": torch.randn(batch_size, dim),
            "contrastive_a": torch.randn(batch_size, dim)}


def test_cross_modal_loss_shapes_and_guards():
    outputs = _fake_outputs()
    is_fake = torch.tensor([0, 1, 0, 1, 1, 0])
    loss = cross_modal_consistency_loss(outputs, is_fake, ["v", "a"], 0.1)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
    # single modality and all-fake batches degrade to zero, not an error
    assert cross_modal_consistency_loss(outputs, is_fake, ["v"], 0.1).item() == 0.0
    assert cross_modal_consistency_loss(outputs, torch.ones(6, dtype=torch.long),
                                        ["v", "a"], 0.1).item() == 0.0


def test_identity_loss_guards_and_value():
    outputs = _fake_outputs()
    is_fake = torch.tensor([0, 1, 0, 1, 1, 0])
    identities = ["id01", "id01", "id02", "id02", "id03", "id09"]
    references = {"contrastive_v": torch.randn(6, 128),
                  "contrastive_a": torch.randn(6, 128)}
    loss = identity_aware_contrastive_loss(
        outputs, identities, is_fake, references, ["v", "a"], 0.1)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
    # no reference branch -> exactly zero
    assert identity_aware_contrastive_loss(outputs, identities, is_fake, None,
                                           ["v", "a"], 0.1).item() == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all torch-component tests passed")
