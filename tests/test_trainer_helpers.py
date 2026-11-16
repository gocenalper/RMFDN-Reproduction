"""Characterization tests for the private trainer building blocks."""

from __future__ import annotations

from contextlib import nullcontext
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rmfdn_ablation.trainer as trainer
from rmfdn_ablation.config import DEFAULT_CONFIG
from rmfdn_ablation.reference import encode_reference_batch
from rmfdn_ablation.trainer import (_build_optimizer, _build_train_loader,
                                    _compute_training_loss, _evaluate_test_split,
                                    _move_training_batch,
                                    _restore_checkpoint, _save_periodic_checkpoint,
                                    _seed_everything, _visible_bit_mask,
                                    _CheckpointPaths)


class _TinyVisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.projection = nn.Linear(2, 2)


class _TinySharedEncoderModel(nn.Module):
    """Minimal stand-in for either M1 or M2's shared encoder contract."""

    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({"v": _TinyVisualEncoder()})
        self.fusion = nn.Linear(2, 1)


class _TinySequenceEncoder(nn.Module):
    def __init__(self, *_args):
        super().__init__()
        self.backbone = nn.Linear(2, 2)

    def encode(self, *args):
        sequence = args[0].float()
        return sequence, sequence.mean(1)


class _TinyPooledEncoder(nn.Module):
    def __init__(self, *_args):
        super().__init__()
        self.backbone = nn.Linear(2, 2)

    def forward(self, values, *_args):
        return values.float().mean(1)


class _TinyRunModel(nn.Module):
    def __init__(self, config, modalities):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch, reference_fused=None):
        return {"loss_anchor": self.weight}


class _FakeTrainingScaler:
    def state_dict(self):
        return {"scale": 1.0}

    def load_state_dict(self, state):
        self.state = state

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def _optimizer_config():
    return {"peak_lr": 5e-5, "weight_decay": 0.01, "backbone_lr_scale": 0.2}


def test_seed_everything_preserves_seed_call_order():
    calls = []
    with mock.patch.object(
            trainer.torch, "manual_seed",
            side_effect=lambda seed: calls.append(("torch", seed))), \
         mock.patch.object(
            trainer.np.random, "seed",
            side_effect=lambda seed: calls.append(("numpy", seed))), \
         mock.patch.object(
            trainer.random, "seed",
            side_effect=lambda seed: calls.append(("python", seed))):
        _seed_everything(17)

    assert calls == [("torch", 17), ("numpy", 17), ("python", 17)]


def test_build_optimizer_preserves_visual_parameter_groups():
    model = _TinySharedEncoderModel()

    optimizer = _build_optimizer(model, ["v"], _optimizer_config())

    backbone_ids = {id(parameter) for parameter in model.encoders["v"].backbone.parameters()}
    first_group_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    second_group_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    assert first_group_ids == {id(parameter) for parameter in model.parameters()} - backbone_ids
    assert second_group_ids == backbone_ids
    assert [group["lr_scale"] for group in optimizer.param_groups] == [1.0, 0.2]
    assert optimizer.defaults["lr"] == 5e-5
    assert optimizer.defaults["weight_decay"] == 0.01


def test_m2_uses_the_shared_encoder_contract_and_optimizer_path():
    import rmfdn_ablation.mulG_RMFDN as mulg_module

    class TinyBackbone(nn.Linear):
        num_features = 2

    class TinyBert(nn.Module):
        config = types.SimpleNamespace(hidden_size=2)

        @classmethod
        def from_pretrained(cls, _name):
            return cls()

        def __init__(self):
            super().__init__()

    fake_timm = types.SimpleNamespace(
        create_model=lambda *_args, **_kwargs: TinyBackbone(2, 2))
    fake_transformers = types.SimpleNamespace(BertModel=TinyBert)
    config = {
        "embed_dim": 2,
        "attention_heads": 1,
        "contrastive_dim": 2,
        "frame_groups": 1,
        "frames_per_clip": 1,
        "n_mels": 16,
        "mel_frames": 16,
        "bert_model": "unused",
        "peak_lr": 5e-5,
        "weight_decay": 0.01,
        "backbone_lr_scale": 0.2,
        "reference_alpha": 0.001,
    }

    with mock.patch.dict(sys.modules, {"timm": fake_timm,
                                       "transformers": fake_transformers}), \
         mock.patch.object(mulg_module, "VisualEncoder", _TinySequenceEncoder,
                           create=True), \
         mock.patch.object(mulg_module, "AudioEncoder", _TinySequenceEncoder,
                           create=True), \
         mock.patch.object(mulg_module, "TextEncoder", _TinySequenceEncoder,
                           create=True):
        model = mulg_module.MulGRmfdn(config, ["v", "a", "t"])

    assert set(model.encoders) == {"v", "a", "t"}
    optimizer = _build_optimizer(model, ["v", "a", "t"], config)
    assert len(optimizer.param_groups) == 2

    batch = {
        "frames": torch.randn(2, 3, 2),
        "mel": torch.randn(2, 4, 2),
        "input_ids": torch.ones(2, 5, 2),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
    }
    outputs = model(batch)
    assert outputs["binary_logits"].shape == (2, 2)
    assert outputs["technique_logits"].shape == (2, 7)


def test_m1_forwards_every_declared_modality_combination():
    import rmfdn_ablation.RMFDN as rmfdn_module

    config = {
        "embed_dim": 2,
        "attention_heads": 1,
        "contrastive_dim": 2,
        "reference_alpha": 0.001,
    }
    batch = {
        "frames": torch.randn(2, 3, 2),
        "mel": torch.randn(2, 4, 2),
        "input_ids": torch.randn(2, 5, 2),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
    }
    modality_sets = [
        ["v"], ["a"], ["t"], ["v", "a"], ["v", "t"],
        ["a", "t"], ["v", "a", "t"],
    ]

    with mock.patch.object(rmfdn_module, "VisualEncoder", _TinyPooledEncoder), \
         mock.patch.object(rmfdn_module, "AudioEncoder", _TinyPooledEncoder), \
         mock.patch.object(rmfdn_module, "TextEncoder", _TinyPooledEncoder):
        for modalities in modality_sets:
            model = rmfdn_module.RMFDN(config, modalities)
            outputs = model(batch)
            assert outputs["binary_logits"].shape == (2, 2)
            assert outputs["technique_logits"].shape == (2, 7)
            if len(modalities) == 1:
                assert not hasattr(model, "fusion_ffn")
                assert not hasattr(model, "modality_self_attention")


def test_build_optimizer_uses_one_group_without_visual_modality():
    model = nn.Linear(2, 1)

    optimizer = _build_optimizer(model, ["a"], _optimizer_config())

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr_scale"] == 1.0
    assert {id(parameter) for parameter in optimizer.param_groups[0]["params"]} == {
        id(parameter) for parameter in model.parameters()
    }


def test_build_train_loader_preserves_unbalanced_loader_options():
    config = {
        "bounded_sampler_pristine": None,
        "batch_size": 2,
        "num_workers": 0,
        "seed": 3,
    }

    loader = _build_train_loader(
        "run", [object(), object(), object()], ["v"], tokenizer=None, config=config)

    assert isinstance(loader.sampler, RandomSampler)
    assert loader.batch_size == 2
    assert loader.drop_last is True
    assert loader.num_workers == 0
    assert loader.pin_memory is True
    assert loader.persistent_workers is False


def test_move_training_batch_preserves_all_seven_bit_targets():
    batch = {
        "frames": torch.tensor([1.0]),
        "mel": torch.tensor([2.0]),
        "input_ids": torch.tensor([[3]]),
        "attention_mask": torch.tensor([[1]]),
        "is_fake": torch.tensor([0, 1]),
        "technique_targets": torch.tensor([[0.0, 1.0, 0.0, 1.0],
                                             [1.0, 0.0, 1.0, 0.0]]),
        "identity": ["id01", "id02"],
    }

    technique_targets = _move_training_batch(batch, "cpu")

    for key in ("frames", "mel", "input_ids", "attention_mask", "is_fake"):
        assert batch[key].device.type == "cpu"
    assert torch.equal(technique_targets, batch["technique_targets"])
    assert batch["identity"] == ["id01", "id02"]


def test_compute_training_loss_preserves_weights_and_accumulation():
    auxiliary_calls = []

    def cross_modal(outputs, is_fake, modalities, temperature):
        auxiliary_calls.append(("cross", outputs, is_fake, modalities, temperature))
        return torch.tensor(2.0)

    def identity(outputs, identities, is_fake, references, modalities, temperature):
        auxiliary_calls.append(
            ("identity", outputs, identities, is_fake, references, modalities, temperature))
        return torch.tensor(3.0)

    outputs = {
        "binary_logits": torch.tensor([[2.0, -1.0], [-1.0, 2.0]]),
        "technique_logits": torch.tensor([[0.5, -0.5], [-0.25, 0.25]]),
    }
    batch = {"is_fake": torch.tensor([0, 1]), "identity": ["id01", "id02"]}
    technique_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    references = {"contrastive_v": torch.ones(2, 2)}

    with mock.patch.object(trainer, "cross_modal_consistency_loss", cross_modal), \
         mock.patch.object(trainer, "identity_aware_contrastive_loss", identity):
        loss = _compute_training_loss(
            outputs, batch, technique_targets, ["v", "a"], temperature=0.7,
            cross_modal_beta=0.2, identity_gamma=0.3,
            reference_outputs=references,
            accum_steps=2)

    expected = (
        torch.nn.functional.cross_entropy(outputs["binary_logits"], batch["is_fake"])
        + torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["technique_logits"], technique_targets)
        + 0.2 * 2.0
        + 0.3 * 3.0
    ) / 2
    assert torch.allclose(loss, expected)
    assert [call[0] for call in auxiliary_calls] == ["cross", "identity"]
    assert auxiliary_calls[0][3:] == (["v", "a"], 0.7)
    assert auxiliary_calls[1][2:] == (
        ["id01", "id02"], batch["is_fake"], references, ["v", "a"], 0.7)


def _checkpoint_components():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = _FakeTrainingScaler()
    return model, optimizer, scaler


def test_checkpoint_paths_preserve_existing_artifact_names():
    with tempfile.TemporaryDirectory() as temp_dir:
        paths = _CheckpointPaths.for_run(temp_dir, "m1_i1_p1_V")
        run_dir = Path(temp_dir) / "runs"

        assert paths.checkpoint_path == str(run_dir / "m1_i1_p1_V_ck.pt")
        assert paths.best_path == str(run_dir / "m1_i1_p1_V.pt")
        assert paths.best_meta_path == str(run_dir / "m1_i1_p1_V.best.json")
        assert not hasattr(paths, "best_map_path")
        assert not hasattr(paths, "best_map_meta_path")


def test_validation_run_writes_only_the_canonical_checkpoint():
    class FakeClip:
        is_fake = 0

    config = dict(DEFAULT_CONFIG)
    config.update(total_iters=2, eval_interval=1, batch_size=1, num_workers=0,
                  bounded_sampler_pristine=None)
    batch = {
        "frames": torch.zeros(1),
        "mel": torch.zeros(1),
        "input_ids": torch.zeros(1, 1, dtype=torch.long),
        "attention_mask": torch.ones(1, 1, dtype=torch.long),
        "is_fake": torch.zeros(1, dtype=torch.long),
        "technique_targets": torch.zeros(1, 7),
        "identity": ["id01"],
    }
    validation_metrics = {"ACC": 70.0, "AUC": 80.0, "mAP": 60.0}

    with tempfile.TemporaryDirectory() as temp_dir, \
         mock.patch.object(
            trainer.torch.amp, "GradScaler", return_value=_FakeTrainingScaler()), \
         mock.patch.object(trainer.torch.amp, "autocast",
                           side_effect=lambda *_args, **_kwargs: nullcontext()), \
         mock.patch.object(trainer, "_build_train_loader", return_value=[batch]), \
         mock.patch.object(
            trainer, "_compute_training_loss",
            side_effect=lambda outputs, *args, **kwargs: outputs["loss_anchor"] * 0 + 1.0), \
         mock.patch.object(trainer, "evaluate", return_value=validation_metrics):
        result = trainer.run_single_experiment(
            "run", modalities=["a"], use_reference=False,
            train_clips=[FakeClip()], val_clips=[FakeClip()],
            test_clips=[], reference_index={},
            tokenizer=None, config=config, output_dir=temp_dir, device="cpu",
            model_class=_TinyRunModel, skip_test_eval=True)
        run_dir = Path(temp_dir) / "runs"

        assert result["test_eval"] == "skipped"
        assert result["config"]["early_stop_patience"] == 10
        assert (run_dir / "run.pt").exists()
        assert not (run_dir / "run_final.pt").exists()
        assert not (run_dir / "run_bestmap.pt").exists()
        assert not (run_dir / "run.bestmap.json").exists()


def test_reference_batch_uses_live_shared_encoder_gradients():
    class TinyReferenceModel(nn.Module):
        modalities = ["v"]

        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(2, 2)
            self.contrastive_projections = nn.ModuleDict({"v": nn.Linear(2, 2)})

        def encode(self, batch):
            feature = self.encoder(batch["frames"].float())
            return {"v": feature}, feature

    model = TinyReferenceModel()
    batch = {
        "reference_frames": torch.ones(3, 2),
        "reference_mel": torch.zeros(3, 1),
        "reference_input_ids": torch.zeros(3, 1, dtype=torch.long),
        "reference_attention_mask": torch.ones(3, 1, dtype=torch.long),
    }

    encoded = encode_reference_batch(model, batch)
    (encoded["fused"].sum() + encoded["contrastive_v"].sum()).backward()

    assert model.encoder.weight.grad is not None
    assert model.contrastive_projections["v"].weight.grad is not None


def test_periodic_checkpoint_preserves_schema_and_round_trips():
    with tempfile.TemporaryDirectory() as temp_dir:
        run_dir = Path(temp_dir) / "runs"
        run_dir.mkdir()
        paths = _CheckpointPaths.for_run(temp_dir, "run")
        model, optimizer, scaler = _checkpoint_components()
        best_model_state = {key: value.detach().clone()
                            for key, value in model.state_dict().items()}

        _save_periodic_checkpoint(
            paths, model, optimizer, scaler, iteration=11, best_score=0.75,
            best_model_state=best_model_state, bad_evals=2)

        payload = torch.load(paths.checkpoint_path, map_location="cpu")
        assert set(payload) == {
            "model_state", "optimizer_state", "scaler_state", "iteration",
            "best_score", "best_model_state", "bad_evals",
        }
        assert payload["iteration"] == 12
        assert payload["best_score"] == 0.75
        assert payload["bad_evals"] == 2
        assert not (run_dir / "run_ck.pt.tmp").exists()

        restored = _restore_checkpoint(
            paths, model, optimizer, scaler, device="cpu", run_name="run")
        iteration, best_score, restored_state, bad_evals = restored
        assert iteration == 12
        assert best_score == 0.75
        assert bad_evals == 2
        for key in best_model_state:
            assert torch.equal(restored_state[key], best_model_state[key])


def test_restore_checkpoint_restarts_when_periodic_file_is_corrupt():
    with tempfile.TemporaryDirectory() as temp_dir:
        run_dir = Path(temp_dir) / "runs"
        run_dir.mkdir()
        paths = _CheckpointPaths.for_run(temp_dir, "run")
        Path(paths.checkpoint_path).write_text("not a torch checkpoint")
        model, optimizer, scaler = _checkpoint_components()

        restored = _restore_checkpoint(
            paths, model, optimizer, scaler, device="cpu", run_name="run")

        assert restored == (0, -1.0, None, 0)


def test_restore_checkpoint_prefers_newer_persisted_best():
    with tempfile.TemporaryDirectory() as temp_dir:
        run_dir = Path(temp_dir) / "runs"
        run_dir.mkdir()
        paths = _CheckpointPaths.for_run(temp_dir, "run")
        model, optimizer, scaler = _checkpoint_components()
        periodic_best = {key: value.detach().clone()
                         for key, value in model.state_dict().items()}
        _save_periodic_checkpoint(
            paths, model, optimizer, scaler, iteration=11, best_score=0.5,
            best_model_state=periodic_best, bad_evals=3)
        persisted_best = {key: torch.full_like(value, 7)
                          for key, value in model.state_dict().items()}
        torch.save(persisted_best, paths.best_path)
        Path(paths.best_meta_path).write_text(
            json.dumps({"best_score": 0.9, "iteration": 25}))

        iteration, best_score, restored_state, bad_evals = _restore_checkpoint(
            paths, model, optimizer, scaler, device="cpu", run_name="run")

        assert iteration == 12
        assert best_score == 0.9
        assert bad_evals == 0
        for key in persisted_best:
            assert torch.equal(restored_state[key], persisted_best[key])


def test_evaluate_test_split_enables_live_references():
    calls = []

    class FakeDataset:
        def __init__(self, clips, modalities, config, reference_index=None,
                     reference_seed=None):
            self.clips = clips
            self.modalities = modalities
            self.config = config
            self.reference_index = reference_index
            self.reference_seed = reference_seed

    def score(model, dataset, tokenizer, config, device):
        calls.append(("evaluate", dataset.clips, dataset.reference_index,
                      dataset.reference_seed))
        return {"split": dataset.clips}

    model = object()
    config = {"seed": 7}

    with mock.patch.object(trainer, "MultimodalClipDataset", FakeDataset), \
         mock.patch.object(trainer, "evaluate", score):
        result = _evaluate_test_split(
            model, test_clips=["test"],
            modalities=["v"], use_reference=True,
            reference_index={"id01": ["ref"]}, tokenizer="tokenizer", config=config,
            device="cpu")

    assert result == {"split": ["test"]}
    assert calls == [("evaluate", ["test"], {"id01": ["ref"]}, 7)]


def test_evaluate_test_split_disables_references_for_i3():
    reference_indices = []

    def score(model, dataset, tokenizer, config, device):
        reference_indices.append(dataset.reference_index)
        return {"split": dataset}

    with mock.patch.object(trainer, "MultimodalClipDataset") as dataset_class, \
         mock.patch.object(trainer, "evaluate", side_effect=score):
        dataset_class.side_effect = lambda clips, modalities, config, **kwargs: \
            types.SimpleNamespace(reference_index=kwargs.get("reference_index"))
        _evaluate_test_split(
            object(), test_clips=["test"],
            modalities=["a"], use_reference=False, reference_index={}, tokenizer=None,
            config={"seed": 0}, device="cpu")

    assert reference_indices == [None]


def test_visible_bit_mask_per_modality():
    # Synthesis bits are visible in their own stream...
    assert _visible_bit_mask(["v"]).tolist() == [1, 1, 0, 0, 0, 0, 0]
    assert _visible_bit_mask(["a"]).tolist() == [0, 0, 1, 1, 0, 0, 0]
    assert _visible_bit_mask(["t"]).tolist() == [0, 0, 0, 0, 0, 1, 0]
    # ...but the shuffle bits splice genuine content, so they need a second
    # stream to compare against. Measured: audio_shuffle is visible to a+v
    # (94.3 AP) but NOT to a+t (35.0, below audio-only 46.4) because the
    # shuffled speech and transcript share a donor; text_shuffle is visible
    # to a+t (85.7).
    assert _visible_bit_mask(["v", "a"]).tolist() == [1, 1, 1, 1, 1, 0, 0]
    assert _visible_bit_mask(["a", "t"]).tolist() == [0, 0, 1, 1, 0, 1, 1]
    assert _visible_bit_mask(["v", "t"]).tolist() == [1, 1, 0, 0, 0, 1, 0]
    assert _visible_bit_mask(["v", "a", "t"]).tolist() == [1] * 7


def test_softmax_mlc_loss_zeroes_pristine_rows():
    logits = torch.tensor([[2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    targets = torch.tensor([[1.0, 0, 1, 0, 0, 0, 0],   # 2-bit fake
                            [0.0, 0, 0, 0, 0, 0, 0]])  # pristine -> sifir katki
    outputs = {"binary_logits": torch.zeros(2, 2),
               "technique_logits": logits}
    batch = {"is_fake": torch.tensor([1, 0]), "identity": ["id00", "id01"]}
    zero = lambda *args, **kwargs: torch.tensor(0.0)
    with mock.patch.object(trainer, "cross_modal_consistency_loss", zero), \
         mock.patch.object(trainer, "identity_aware_contrastive_loss", zero):
        loss = _compute_training_loss(outputs, batch, targets, ["v"], 1.0, 0.0, 0.0,
                                      None, 1, mlc_loss_mode="softmax")
    expected_mlc = -(targets * F.log_softmax(logits, 1)).sum(1).mean()
    expected = F.cross_entropy(outputs["binary_logits"], batch["is_fake"]) + expected_mlc
    assert torch.isclose(loss, expected)
    # pristine satirinin katkisi gercekten sifir mi: sadece fake satiriyla ayni mlc
    only_fake = -(targets[:1] * F.log_softmax(logits[:1], 1)).sum(1)
    assert torch.isclose(expected_mlc * 2, only_fake.sum())


def test_softmax_norm_mlc_loss_removes_k_weight():
    logits = torch.tensor([[2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    targets = torch.tensor([[1.0, 0, 1, 0, 0, 0, 0],   # K=2 fake
                            [0.0, 0, 0, 0, 0, 0, 0]])  # pristine -> sifir katki
    outputs = {"binary_logits": torch.zeros(2, 2),
               "technique_logits": logits}
    batch = {"is_fake": torch.tensor([1, 0]), "identity": ["id00", "id01"]}
    zero = lambda *args, **kwargs: torch.tensor(0.0)
    with mock.patch.object(trainer, "cross_modal_consistency_loss", zero), \
         mock.patch.object(trainer, "identity_aware_contrastive_loss", zero):
        raw = _compute_training_loss(outputs, batch, targets, ["v"], 1.0, 0.0, 0.0,
                                     None, 1, mlc_loss_mode="softmax")
        norm = _compute_training_loss(outputs, batch, targets, ["v"], 1.0, 0.0, 0.0,
                                      None, 1, mlc_loss_mode="softmax_norm")
    bic = F.cross_entropy(outputs["binary_logits"], batch["is_fake"])
    # K=2'lik tek fake satirda: raw mlc = 2 * norm mlc (ayni minimizer, K-kat agirlik)
    assert torch.isclose(raw - bic, 2 * (norm - bic))
    # pristine satiri normalize edilirken 0/0 uretmemeli (clamp korumasi)
    assert torch.isfinite(norm)


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")
    print("all trainer-helper tests passed")


def test_probe_bits_widen_the_visible_set():
    # A probe run asks whether a pair can see a bit the map assigns elsewhere.
    assert _visible_bit_mask(["v", "t"]).tolist() == [1, 1, 0, 0, 0, 1, 0]
    assert _visible_bit_mask(["v", "t"], ("text_shuffle", "audio_shuffle")
                             ).tolist() == [1, 1, 0, 0, 1, 1, 1]
    # already-visible names are idempotent, not duplicated
    assert _visible_bit_mask(["v"], ("lip_sync",)).tolist() == [1, 1, 0, 0, 0, 0, 0]
