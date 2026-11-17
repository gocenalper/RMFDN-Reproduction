"""Gate, modality dropout and test-time corruptions (no downloads, no data)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import rmfdn_ablation.RMFDN as rmfdn_module
from rmfdn_ablation.config import DEFAULT_CONFIG, VARIANTS
from rmfdn_ablation.corruptions import (CONDITIONS, add_white_noise, apply_modality_dropout,
                                        blur_frames, condition, mismatch_permutation)
from rmfdn_ablation.experiment import expand_cells


def _va_model(monkeypatch, gate: bool):
    """VA model whose encoders are identity maps, so fuse() can be driven directly."""
    for name in ("VisualEncoder", "AudioEncoder", "TextEncoder"):
        monkeypatch.setattr(rmfdn_module, name, lambda *args, **kwargs: nn.Identity())
    config = dict(DEFAULT_CONFIG, embed_dim=16, attention_heads=2, contrastive_dim=8,
                  fusion_gate=gate)
    torch.manual_seed(0)
    return rmfdn_module.RMFDN(config, ["v", "a"]).eval()


def test_gate_adds_one_linear_layer(monkeypatch):
    plain = _va_model(monkeypatch, gate=False)
    gated = _va_model(monkeypatch, gate=True)
    extra = sum(p.numel() for p in gated.parameters()) - sum(p.numel() for p in plain.parameters())
    assert extra == 2 * 16 * 16 + 16
    assert not hasattr(plain, "fusion_gate")


def test_gate_at_one_half_reproduces_the_residual_sum(monkeypatch):
    plain = _va_model(monkeypatch, gate=False)
    gated = _va_model(monkeypatch, gate=True)
    gated.load_state_dict(plain.state_dict(), strict=False)
    nn.init.zeros_(gated.fusion_gate.weight)
    nn.init.zeros_(gated.fusion_gate.bias)              # sigmoid(0) = 0.5 everywhere
    features = {"v": torch.randn(4, 16), "a": torch.randn(4, 16)}
    with torch.no_grad():
        expected = plain.fuse(features)
        actual = gated.fuse(features)
    assert torch.allclose(actual, expected, atol=1e-4)
    assert torch.allclose(gated.last_gate, torch.full((4,), 0.5))
    assert plain.last_gate is None


def test_gate_value_is_returned_per_clip(monkeypatch):
    gated = _va_model(monkeypatch, gate=True)
    batch = {"frames": torch.randn(3, 16), "mel": torch.randn(3, 16)}
    outputs = gated(batch)
    assert outputs["gate"].shape == (3,)
    assert ((outputs["gate"] > 0) & (outputs["gate"] < 1)).all()


def test_modality_dropout_blanks_audio_xor_video():
    generator = torch.Generator().manual_seed(0)
    batch = {"is_fake": torch.zeros(4000, dtype=torch.long),
             "mel": torch.ones(4000, 3, 2), "frames": torch.full((4000, 2, 3, 2, 2), 7, dtype=torch.uint8)}
    counts = apply_modality_dropout(batch, ["v", "a"], 0.1, generator)
    audio_gone = batch["mel"].flatten(1).abs().sum(1) == 0
    video_gone = batch["frames"].flatten(1).sum(1) == 0
    assert not (audio_gone & video_gone).any()
    assert counts == {"audio": int(audio_gone.sum()), "video": int(video_gone.sum())}
    assert 300 < counts["audio"] < 500 and 300 < counts["video"] < 500


def test_modality_dropout_is_off_for_zero_probability_and_single_modality():
    batch = {"is_fake": torch.zeros(8, dtype=torch.long), "mel": torch.ones(8, 2),
             "frames": torch.ones(8, 2, dtype=torch.uint8)}
    generator = torch.Generator().manual_seed(0)
    assert apply_modality_dropout(batch, ["v", "a"], 0.0, generator) == {"audio": 0, "video": 0}
    assert apply_modality_dropout(batch, ["v"], 0.5, generator) == {"audio": 0, "video": 0}
    assert batch["mel"].sum() == 16


@pytest.mark.parametrize("snr", [10.0, 0.0])
def test_white_noise_hits_the_requested_snr(snr):
    waveform = torch.sin(torch.linspace(0, 2000, 128000)).unsqueeze(0)
    noisy = add_white_noise(waveform, snr, seed=3)
    noise = noisy - waveform
    measured = 10 * torch.log10(waveform.pow(2).mean() / noise.pow(2).mean())
    assert abs(float(measured) - snr) < 0.1
    assert torch.equal(noisy, add_white_noise(waveform, snr, seed=3))


def test_mismatch_is_a_fixed_derangement():
    perm = mismatch_permutation(50, seed=1)
    assert sorted(perm) == list(range(50))
    assert all(source != index for index, source in enumerate(perm))
    assert perm == mismatch_permutation(50, seed=1)


def test_blur_keeps_shape_dtype_and_mean():
    frames = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
    blurred = blur_frames(frames, sigma=3.0)
    assert blurred.shape == frames.shape and blurred.dtype == torch.uint8
    assert abs(blurred.float().mean() - frames.float().mean()) < 8
    assert blurred.float().std() < frames.float().std()


def test_conditions_match_the_report():
    assert list(CONDITIONS) == ["clean", "audio_missing", "audio_noise_10db", "audio_noise_0db",
                                "audio_mismatch", "video_blur", "video_missing"]
    with pytest.raises(ValueError):
        condition("nope")


def test_improvement_matrix_expands_to_four_arms():
    import json
    cells = json.load(open(Path(__file__).resolve().parents[1]
                           / "experiments" / "improvement_matrix.json"))["cells"]
    names = [spec.name for spec in expand_cells(cells)]
    assert names == ["m1_i3_p1_VA", "m1_i3_p1_VA_dropout", "m1_i3_p1_VA_gate",
                     "m1_i3_p1_VA_gate_dropout"]
    assert set(VARIANTS) == {"base", "dropout", "gate", "gate_dropout"}
    with pytest.raises(ValueError):
        expand_cells([{**cells[0], "variants": ["unknown"]}])
