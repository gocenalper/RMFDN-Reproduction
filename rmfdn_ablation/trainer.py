"""Single-experiment training loop.

One call to :func:`run_single_experiment` trains and tests one cell of the
ablation grid -- a (modality configuration, reference on/off) pair. Pieces:

* discriminative learning rates (the ImageNet-pretrained ResNet-18 backbone
  trains at ``backbone_lr_scale`` x the base rate),
* linear warmup followed by cosine decay,
* CUDA mixed-precision training,
* total loss = binary CE + per-technique BCE
  + ``cross_modal_beta`` * cross-modal InfoNCE (Eq. 4)
  + ``identity_gamma`` * identity-aware InfoNCE (Eq. 5),
* periodic validation with early stopping on the best validation AUC,
* a periodic **atomic** mid-run checkpoint plus a guarded resume, so a crashed
  or externally-restarted process loses at most ``CHECKPOINT_INTERVAL``
  iterations. The overnight full-dataset session ran under a shell supervisor
  that simply relaunched the process until it printed its completion marker;
  these two pieces are what made that loop self-healing.

The no-reference arm of the ablation is obtained by zeroing both reference
knobs (``identity_gamma`` and ``reference_alpha``) -- the architecture is
otherwise identical, so the comparison isolates the reference mechanism.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import (MANIPULATION_TECHNIQUES, TECHNIQUE_SHORT, TYPE_TEMPLATES,
                     default_device, effective_visible_indices)
from .data_catalog import ClipRecord
from .dataset import MultimodalClipDataset, build_collate_fn
from .evaluation import evaluate
from .losses import cross_modal_consistency_loss, identity_aware_contrastive_loss
from .RMFDN import RMFDN
from .reference import encode_reference_batch

#: Iterations between atomic mid-run checkpoints (crash-resume granularity).
#: 500 caps the redo-after-requeue at ~12 min for the slowest (VA) cells;
#: the extra _ck writes cost well under a minute each.
CHECKPOINT_INTERVAL = 500


@dataclass(frozen=True)
class _CheckpointPaths:
    """Filesystem artifacts owned by one experiment run."""

    checkpoint_path: str
    best_path: str
    best_meta_path: str
    final_path: str

    @classmethod
    def for_run(cls, output_dir: str, run_name: str) -> "_CheckpointPaths":
        run_dir = os.path.join(output_dir, "runs")
        return cls(
            checkpoint_path=os.path.join(run_dir, f"{run_name}_ck.pt"),
            best_path=os.path.join(run_dir, f"{run_name}.pt"),
            best_meta_path=os.path.join(run_dir, f"{run_name}.best.json"),
            final_path=os.path.join(run_dir, f"{run_name}_final.pt"),
        )


def cosine_warmup_lr(iteration: int, config: dict) -> float:
    """Linear warmup to ``peak_lr``, then cosine decay to ``min_lr``."""
    if iteration < config["warmup_iters"]:
        ramp = iteration / max(1, config["warmup_iters"])
        return config["warmup_start_lr"] + ramp * (config["peak_lr"] - config["warmup_start_lr"])
    schedule_iters = config.get("lr_schedule_iters", config["total_iters"])
    progress = min(1.0, (iteration - config["warmup_iters"])
                   / max(1, schedule_iters - config["warmup_iters"]))
    return config["min_lr"] + 0.5 * (config["peak_lr"] - config["min_lr"]) \
        * (1 + math.cos(math.pi * progress))


def _seed_everything(seed: int) -> None:
    """Seed every RNG used by one experiment, in the historical call order."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _build_optimizer(model, modalities: Sequence[str], config: dict):
    """Build AdamW with the visual backbone's discriminative LR group.

    Frozen parameters are left out entirely: staged-fusion cells freeze every
    encoder, and an optimizer that still holds them would carry their state
    for nothing.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if "v" in modalities and any(p.requires_grad
                                 for p in model.encoders["v"].backbone.parameters()):
        backbone_parameter_ids = {id(p) for p in model.encoders["v"].backbone.parameters()}
        parameter_groups = [
            {"params": [p for p in trainable if id(p) not in backbone_parameter_ids],
             "lr_scale": 1.0},
            {"params": [p for p in trainable if id(p) in backbone_parameter_ids],
             "lr_scale": config["backbone_lr_scale"]},
        ]
    else:
        parameter_groups = [{"params": trainable, "lr_scale": 1.0}]
    return torch.optim.AdamW(parameter_groups, lr=config["peak_lr"],
                             weight_decay=config["weight_decay"])


def _build_train_loader(run_name: str, train_clips: Sequence[ClipRecord],
                        modalities: Sequence[str], tokenizer, config: dict,
                        reference_index: Optional[Dict[str, List[str]]] = None) -> DataLoader:
    """Build the training dataset, optional bounded sampler, and DataLoader."""
    train_dataset = MultimodalClipDataset(
        train_clips, modalities, config, reference_index=reference_index)
    train_sampler = None
    if config.get("bounded_sampler_pristine") is not None:
        from fractions import Fraction

        from .bounded_sampler import BoundedForgeryTypeSampler

        effective_batch = config["batch_size"] * max(
            1, int(config.get("grad_accum_steps", 1)))
        total_samples = config["total_iters"] * effective_batch
        override = config.get("fake_type_shares")  # {type: per-1000 int}
        fake_type_shares = ({category: Fraction(int(mils), 1000)
                             for category, mils in override.items()}
                            if override else None)
        train_sampler = BoundedForgeryTypeSampler(
            train_clips, num_samples=total_samples,
            pristine_share=float(config["bounded_sampler_pristine"]),
            seed=config["seed"], fake_type_shares=fake_type_shares)
        repeats = train_sampler.repeat_factors(len(train_dataset))
        bit_shares = train_sampler.technique_positive_shares()
        print(f"[{run_name}] bounded sampler category shares="
              f"{ {k: round(v, 4) for k, v in train_sampler.shares.items()} }",
              flush=True)
        print(f"[{run_name}] bounded sampler positive bit shares="
              f"{ {k: round(v, 4) for k, v in bit_shares.items()} }", flush=True)
        print(f"[{run_name}] expected repeats per dataset-sized window="
              f"{ {k: round(v, 2) for k, v in repeats.items()} }", flush=True)
    # NOTE: persistent_workers must stay False -- long-lived workers leaked
    # RSS over the 90k-iteration runs until the OOM killer fired.
    return DataLoader(train_dataset, batch_size=config["batch_size"],
                      shuffle=(train_sampler is None), sampler=train_sampler,
                      drop_last=True, num_workers=config["num_workers"],
                      collate_fn=build_collate_fn(tokenizer, config),
                      pin_memory=True, persistent_workers=False)


def _move_training_batch(batch: dict, device: str) -> torch.Tensor:
    """Move model inputs and the canonical seven-bit targets to ``device``."""
    keys = ["frames", "mel", "input_ids", "attention_mask", "is_fake"]
    if "reference_frames" in batch:
        keys.extend(["reference_frames", "reference_mel", "reference_input_ids",
                     "reference_attention_mask"])
    for key in keys:
        batch[key] = batch[key].to(device)
    return batch["technique_targets"].to(device)


def _oracle_codes(technique_targets: torch.Tensor, is_fake: torch.Tensor,
                  visible_idx: Sequence[int]) -> torch.Tensor:
    """Integer id of each clip's *observable* equivalence class.

    code = is_fake + 2 * binary-encoding of the modality-visible bits. The
    invisible bits never enter, so two clips share a code exactly when the
    cell's modalities cannot tell their label patterns apart (for the visual
    cell this reproduces the PRISTINE / NONE / LIP / FACE partition).
    """
    code = is_fake.long()
    for position, bit in enumerate(visible_idx):
        code = code + ((technique_targets[:, bit] > 0.5).long() << (position + 1))
    return code


def build_oracle_table(clips, visible_idx: Sequence[int],
                       eps: float = 1e-3) -> torch.Tensor:
    """(2^(n+1), 7) frozen log-odds lookup for oracle-PoE training.

    Row ``code`` holds log(p / (1-p)) with p = P(bit_j = 1 | observable class)
    counted over ``clips`` and clamped to [eps, 1-eps] so structurally-absent
    bits (e.g. audio_shuffle in visually-clean fakes) stay finite. Columns in
    ``visible_idx`` are zeroed -- visible bits keep plain BCE supervision --
    and rows for patterns absent from ``clips`` stay zero (no offset).
    """
    targets = torch.tensor([clip.technique_targets for clip in clips],
                           dtype=torch.float32)
    is_fake = torch.tensor([clip.is_fake for clip in clips])
    codes = _oracle_codes(targets, is_fake, visible_idx)
    table = torch.zeros(2 ** (len(visible_idx) + 1), targets.shape[1])
    for code in codes.unique().tolist():
        probs = targets[codes == code].mean(0).clamp(eps, 1 - eps)
        table[code] = torch.log(probs / (1 - probs))
    table[:, list(visible_idx)] = 0.0
    return table


def build_visible_loo_table(clips, visible_idx: Sequence[int],
                            eps: float = 1e-3,
                            fake_cond: bool = True) -> torch.Tensor:
    """(2^(n+1), 7) leave-one-out log-odds over the *visible* bits.

    Entry ``[code, j]`` is log(p / (1-p)) with
    p = P(bit_j = 1 | the other visible bits of ``code``, real/fake): each
    visible bit is conditioned on its siblings, so a bit that its siblings
    already determine (face_swap implies lip_sync) earns no gradient. Columns
    outside ``visible_idx`` stay zero -- invisible bits are handled by the
    loss mask, not by an offset.

    ``fake_cond=False`` drops the real/fake flag from the conditioning set
    (p = P(bit_j = 1 | sibling visible bits), marginalised over real/fake):
    the sensitivity variant for the ablation that asks how much of the
    oracle's work the fake flag does. Rows are still indexed by the full
    code; the real and fake variants of a sibling pattern simply share p.
    """
    targets = torch.tensor([clip.technique_targets for clip in clips],
                           dtype=torch.float32)
    is_fake = torch.tensor([clip.is_fake for clip in clips])
    codes = _oracle_codes(targets, is_fake, visible_idx)
    table = torch.zeros(2 ** (len(visible_idx) + 1), targets.shape[1])
    for code in range(table.shape[0]):
        for position, bit in enumerate(visible_idx):
            sibling_mask = ~(1 << (position + 1))       # drop bit j from the code
            if not fake_cond:
                sibling_mask &= ~1                      # drop the real/fake flag too
            selected = (codes & sibling_mask) == (code & sibling_mask)
            if selected.any():
                p = targets[selected, bit].mean().clamp(eps, 1 - eps)
                table[code, bit] = torch.log(p / (1 - p))
    return table


def build_frozen_offset_table(csv_path: str, cell: str, variant: str, fold: str,
                              visible_idx: Sequence[int]) -> torch.Tensor:
    """(2^(n+1), 7) offset table from the frozen Phase-0 ``run_matrix.csv``.

    The Phase-1 spec forbids recomputing oracles: every offset is read from
    the shipped table. A row's conditioning bits are the cell's observable
    set (class_oracle) or that set minus the target bit (sibling_oracle), in
    canonical bit order; ``conditioning_pattern`` is aligned to that order.
    The frozen oracles carry no real/fake conditioning, so offsets are
    written to the even (is_fake = 0) codes only and the caller must build
    runtime codes with ``is_fake`` zeroed. Patterns absent from the table
    (never realised in that fold's train split) keep offset 0.
    """
    import csv as _csv
    column = f"logodds_{fold}"
    position_of = {bit: position for position, bit in enumerate(visible_idx)}
    table = torch.zeros(2 ** (len(visible_idx) + 1), len(MANIPULATION_TECHNIQUES))
    rows_used = 0
    with open(csv_path) as handle:
        reader = _csv.DictReader(line for line in handle
                                 if not line.startswith("#"))
        for row in reader:
            if (row.get("cell") != cell or row.get("variant") != variant
                    or row.get("bias_source") not in ("sibling_oracle",
                                                      "class_oracle")):
                continue
            if not row.get(column):
                # a structurally empty row (e.g. a single-observable cell's
                # sibling oracle has no conditioning bit) means "no offset"
                continue
            bit_index = TECHNIQUE_SHORT.index(row["bit"])
            conditioning = ([i for i in visible_idx if i != bit_index]
                            if row["bias_source"] == "sibling_oracle"
                            else list(visible_idx))
            pattern = [int(x) for x in row["conditioning_pattern"].split("|")]
            if len(pattern) != len(conditioning):
                raise ValueError(
                    f"{csv_path}: pattern {row['conditioning_pattern']!r} does "
                    f"not match conditioning set {conditioning} for "
                    f"{cell}/{variant}/{row['bit']}")
            value = float(row[column])
            free_positions = [position_of[i] for i in visible_idx
                              if i not in conditioning]
            base = sum(v << position_of[c]
                       for c, v in zip(conditioning, pattern))
            for fill in range(2 ** len(free_positions)):
                encoding = base
                for k, position in enumerate(free_positions):
                    if (fill >> k) & 1:
                        encoding |= 1 << position
                table[2 * encoding, bit_index] = value
            rows_used += 1
    if rows_used == 0:
        raise ValueError(f"{csv_path}: no offset rows for {cell}/{variant}")
    return table


def _compute_training_loss(outputs: dict, batch: dict, technique_targets: torch.Tensor,
                           modalities: Sequence[str], temperature: float,
                           cross_modal_beta: float, identity_gamma: float,
                           reference_outputs, accum_steps: int,
                           bit_mask: Optional[torch.Tensor] = None,
                           mlc_loss_mode: str = "bce",
                           binary_weight: float = 1.0,
                           mlc_loss_scale: float = 1.0,
                           oracle_table: Optional[torch.Tensor] = None,
                           oracle_visible_idx: Optional[Sequence[int]] = None,
                           oracle_ignore_fake: bool = False) -> torch.Tensor:
    """Compose the paper's supervised and auxiliary losses for one micro-batch."""
    if mlc_loss_mode == "softmax":
        # Literal "H = cross-entropy" reading: softmax over the 7 technique
        # logits against the RAW multi-hot target (no normalisation). Pristine
        # rows (all-zero Y_t) contribute exactly zero loss.
        log_probs = F.log_softmax(outputs["technique_logits"], dim=1)
        technique_loss = -(technique_targets * log_probs).sum(1).mean()
    elif mlc_loss_mode == "softmax_norm":
        # A/B twin of "softmax": identical per-clip minimiser and competition,
        # but targets are normalised to a distribution (y / K), removing the
        # K-fold gradient weight that raw multi-hot gives K-positive clips.
        # Pristine rows keep the raw variant's exact zero-loss behaviour.
        log_probs = F.log_softmax(outputs["technique_logits"], dim=1)
        norm_targets = technique_targets / technique_targets.sum(1, keepdim=True).clamp(min=1)
        technique_loss = -(norm_targets * log_probs).sum(1).mean()
    elif mlc_loss_mode == "type11":
        # 11-way forgery-TYPE softmax: the 11 types are mutually exclusive, so each
        # clip has exactly one type label -> plain cross-entropy over 11 logits.
        # The type index is recovered by matching the 7-bit target to its template.
        _tpl = torch.tensor(TYPE_TEMPLATES, dtype=technique_targets.dtype,
                            device=technique_targets.device)          # (11, 7)
        _match = (technique_targets.unsqueeze(1) == _tpl.unsqueeze(0)).all(dim=2)
        type_idx = _match.float().argmax(dim=1)                       # (B,)
        technique_loss = F.cross_entropy(outputs["technique_logits"], type_idx)
    elif bit_mask is None:
        technique_logits = outputs["technique_logits"]
        if oracle_table is not None:
            # Oracle-PoE: add the frozen class-statistics log-odds to the
            # invisible columns (visible columns of the table are zero), so
            # gradients vanish wherever the class statistic already explains
            # the target; evaluation reads the raw head (evidence) alone.
            codes = _oracle_codes(technique_targets,
                                  torch.zeros_like(batch["is_fake"])
                                  if oracle_ignore_fake else batch["is_fake"],
                                  oracle_visible_idx)
            technique_logits = technique_logits + oracle_table[codes]
        technique_loss = F.binary_cross_entropy_with_logits(
            technique_logits, technique_targets)
    else:  # supervise only the modality-visible bits (masked mean)
        technique_logits = outputs["technique_logits"]
        if oracle_table is not None:
            # Visible-bit leave-one-out PoE: each supervised bit is offset by
            # its siblings' log-odds, so structure the other visible bits
            # already determine earns no gradient; evaluation reads the raw
            # head alone.
            codes = _oracle_codes(technique_targets,
                                  torch.zeros_like(batch["is_fake"])
                                  if oracle_ignore_fake else batch["is_fake"],
                                  oracle_visible_idx)
            technique_logits = technique_logits + oracle_table[codes]
        per_bit = F.binary_cross_entropy_with_logits(
            technique_logits, technique_targets, reduction="none")
        technique_loss = ((per_bit * bit_mask).sum()
                          / (bit_mask.sum() * per_bit.shape[0]))
        if "aux_technique_logits" in outputs:
            # Training-only plain-BCE head: restores the trunk gradient that the
            # oracle's saturated regions withhold (all genuine clips, the
            # sibling-determined fake regions) without altering the measured
            # head's objective -- evaluation never reads this head.
            aux_per_bit = F.binary_cross_entropy_with_logits(
                outputs["aux_technique_logits"], technique_targets, reduction="none")
            technique_loss = technique_loss + ((aux_per_bit * bit_mask).sum()
                                               / (bit_mask.sum() * aux_per_bit.shape[0]))
    binary_term = (binary_weight * F.cross_entropy(outputs["binary_logits"],
                                                   batch["is_fake"])
                   if binary_weight and "binary_logits" in outputs else 0.0)
    identity_term = (identity_gamma * identity_aware_contrastive_loss(
        outputs, batch["identity"], batch["is_fake"], reference_outputs,
        modalities, temperature) if identity_gamma else 0.0)
    return (
        binary_term
        + mlc_loss_scale * technique_loss
        + cross_modal_beta * cross_modal_consistency_loss(
            outputs, batch["is_fake"], modalities, temperature)
        + identity_term
    ) / accum_steps


def _restore_checkpoint(paths: _CheckpointPaths, model, optimizer, scaler,
                        device: str, run_name: str):
    """Restore periodic and persisted-best state, or return a fresh run state."""
    iteration, best_score, best_model_state, bad_evals = 0, -1.0, None, 0
    if os.path.exists(paths.checkpoint_path):
        try:
            checkpoint = torch.load(paths.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            iteration = checkpoint["iteration"]
            best_score = checkpoint["best_score"]
            best_model_state = checkpoint["best_model_state"]
            bad_evals = checkpoint["bad_evals"]
            print(f"[{run_name}] resumed from checkpoint at iteration {iteration}", flush=True)
        except Exception as error:
            print(f"[{run_name}] checkpoint unreadable ({error}) -- restarting run", flush=True)
            iteration, best_score, best_model_state, bad_evals = 0, -1.0, None, 0

    # A periodic checkpoint can predate a newer best-AUC sidecar. Prefer the
    # persisted best score so a later, worse validation cannot overwrite it.
    if os.path.exists(paths.best_meta_path):
        with open(paths.best_meta_path) as handle:
            persisted_best = json.load(handle)
        if persisted_best["best_score"] > best_score:
            best_score = persisted_best["best_score"]
            best_model_state = torch.load(paths.best_path, map_location="cpu")
            bad_evals = 0
            print(f"[{run_name}] restored persisted best (score {best_score:.3f} "
                  f"@ iter {persisted_best['iteration']})", flush=True)
    return iteration, best_score, best_model_state, bad_evals


def _save_periodic_checkpoint(paths: _CheckpointPaths, model, optimizer, scaler,
                              iteration: int, best_score: float,
                              best_model_state, bad_evals: int) -> None:
    """Atomically persist the existing seven-field crash-resume payload."""
    torch.save({"model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "iteration": iteration + 1,
                "best_score": best_score,
                "best_model_state": best_model_state,
                "bad_evals": bad_evals}, paths.checkpoint_path + ".tmp")
    os.replace(paths.checkpoint_path + ".tmp", paths.checkpoint_path)


def _evaluate_test_split(model, test_clips: Sequence[ClipRecord],
                         modalities: Sequence[str], use_reference: bool,
                         reference_index: Dict[str, List[str]], tokenizer,
                         config: dict, device: str):
    """Score the complete held-out split with live shared reference encoders."""
    return evaluate(
        model, MultimodalClipDataset(
            test_clips, modalities, config,
            reference_index=reference_index if use_reference else None,
            reference_seed=config["seed"]),
        tokenizer, config, device)


def run_single_experiment(run_name: str, modalities: Sequence[str], use_reference: bool,
                          train_clips: Sequence[ClipRecord], val_clips: Sequence[ClipRecord],
                          test_clips: Sequence[ClipRecord],
                          reference_index: Dict[str, List[str]], tokenizer, config: dict,
                          output_dir: str, device: Optional[str] = None,
                          model_class=RMFDN, skip_test_eval: bool = False) -> dict:
    """Train one configuration and report complete held-out test metrics.

    ``model_class`` selects the fusion architecture: the default ``RMFDN``
    (m1, progressive cross-attention) or ``MulGRmfdn`` (m2, MulG directed
    pairwise cross-modal attention + GRU fusion). Both expose the same
    constructor and forward contract, so the rest of the loop is unchanged.
    """
    device = device or default_device()
    _seed_everything(config["seed"])

    # Reference ablation: the no-reference arm zeroes both knobs.
    identity_gamma = config["identity_gamma"] if use_reference else 0.0
    run_config = {**config,
                  "reference_alpha": config["reference_alpha"] if use_reference else 0.0}
    cross_modal_beta = config["cross_modal_beta"] if len(modalities) >= 2 else 0.0

    model = model_class(run_config, modalities).to(device)
    if config.get("freeze_experts"):
        # Same encoders as the m0 cell, same frozen state: the only thing that
        # differs between the two is the fusion mechanism itself.
        from .staged import freeze_encoders, load_expert_encoders
        load_expert_encoders(model, config.get("expert_checkpoints") or {})
        total = sum(p.numel() for p in model.parameters())
        trainable = freeze_encoders(model)
        model.to(device)
        print(f"[{run_name}] experts frozen -- trainable {trainable:,} / "
              f"{total:,} parameters ({100*trainable/total:.2f}%)", flush=True)

    # Only the ImageNet ResNet backbone uses the discriminative LR scale. AST
    # and BERT are fine-tuned end to end at the full rate.
    optimizer = _build_optimizer(model, modalities, config)
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    else:  # torch < 2.3 (old HPC containers): same class under torch.cuda.amp
        scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))

    active_reference_index = reference_index if use_reference else None
    train_loader = _build_train_loader(
        run_name, train_clips, modalities, tokenizer, config,
        reference_index=active_reference_index)
    val_dataset = MultimodalClipDataset(
        val_clips, modalities, config,
        reference_index=active_reference_index,
        reference_seed=config["seed"])

    num_fake = sum(clip.is_fake for clip in train_clips)
    num_real = len(train_clips) - num_fake
    trainable_millions = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[{run_name}] modalities={list(modalities)} reference={use_reference} "
          f"techniques={MANIPULATION_TECHNIQUES} "
          f"train fake/real={num_fake}/{num_real} val={len(val_clips)} | "
          f"trainable {trainable_millions:.1f}M | "
          f"max/schedule iters={config['total_iters']}/"
          f"{config.get('lr_schedule_iters', config['total_iters'])}", flush=True)

    # Guarded resume: pick up the atomic checkpoint left by a previous
    # (crashed / supervisor-restarted) process; an unreadable checkpoint
    # restarts the run instead of skipping it.
    os.makedirs(os.path.join(output_dir, "runs"), exist_ok=True)
    paths = _CheckpointPaths.for_run(output_dir, run_name)
    iteration, best_score, best_model_state, bad_evals = _restore_checkpoint(
        paths, model, optimizer, scaler, device, run_name)

    accum_steps = max(1, int(config.get("grad_accum_steps", 1)))
    train_sampler = getattr(train_loader, "sampler", None)
    if hasattr(train_sampler, "set_range"):
        effective_batch = config["batch_size"] * accum_steps
        consumed_samples = iteration * effective_batch
        remaining_samples = (config["total_iters"] - iteration) * effective_batch
        train_sampler.set_range(consumed_samples, remaining_samples)
        print(f"[{run_name}] sampler resume offset={consumed_samples}", flush=True)

    # AST/BERT gradient checkpointing only frees activations in training mode.
    # Pretrained transformer constructors leave the encoder in eval mode, so
    # force the complete model into training mode before the loop.
    model.train()
    start_time = time.time()
    stop = False
    # Gradient accumulation: each optimizer step averages the gradients of
    # ``accum_steps`` micro-batches (each of size ``batch_size``), giving an
    # effective batch of ``batch_size * accum_steps`` at the memory cost of one
    # micro-batch. ``accum_steps == 1`` reproduces the original per-batch loop
    # exactly. One ``iteration`` == one optimizer step. ``total_iters`` limits
    # runtime, while ``lr_schedule_iters`` controls the independent cosine horizon.
    micro = 0
    bit_mask = None
    if config.get("mask_invisible_bits"):
        bit_mask = torch.zeros(len(MANIPULATION_TECHNIQUES))
        bit_mask[list(effective_visible_indices(modalities, config))] = 1.0
        bit_mask = bit_mask.to(device)
        print(f"[{run_name}] invisible-bit BCE masking ON -- supervised bits="
              f"{[MANIPULATION_TECHNIQUES[i] for i in range(len(MANIPULATION_TECHNIQUES)) if bit_mask[i] > 0]}",
              flush=True)
    mlc_loss_mode = config.get("mlc_loss", "bce")
    oracle_table, oracle_visible_idx = None, None
    if config.get("oracle_poe") and config.get("oracle_poe_visible"):
        raise ValueError("oracle_poe and oracle_poe_visible are mutually exclusive")
    if config.get("oracle_poe"):
        if config.get("mask_invisible_bits"):
            raise ValueError("oracle_poe and mask_invisible_bits are mutually "
                             "exclusive: both alter invisible-bit supervision")
        if mlc_loss_mode != "bce":
            raise ValueError("oracle_poe is defined for mlc_loss='bce' only")
        oracle_visible_idx = tuple(sorted(effective_visible_indices(modalities, config)))
        oracle_table = build_oracle_table(train_clips, oracle_visible_idx).to(device)
        active_rows = int((oracle_table.abs().sum(1) > 0).sum())
        print(f"[{run_name}] oracle-PoE ON -- visible bits kept plain="
              f"{[MANIPULATION_TECHNIQUES[i] for i in oracle_visible_idx]}, "
              f"offset rows={active_rows}/{oracle_table.shape[0]}", flush=True)
    if config.get("oracle_poe_visible"):
        if not config.get("mask_invisible_bits"):
            raise ValueError("oracle_poe_visible requires mask_invisible_bits: "
                             "it debiases the supervised (visible) bits only")
        if mlc_loss_mode != "bce":
            raise ValueError("oracle_poe_visible is defined for mlc_loss='bce' only")
        oracle_visible_idx = tuple(sorted(effective_visible_indices(modalities, config)))
        oracle_table = build_visible_loo_table(
            train_clips, oracle_visible_idx,
            fake_cond=not config.get("poe_no_fake_cond", False)).to(device)
        names = [MANIPULATION_TECHNIQUES[i] for i in oracle_visible_idx]
        spread = [round(float(oracle_table[:, i].abs().max()), 2) for i in oracle_visible_idx]
        print(f"[{run_name}] visible-bit leave-one-out PoE ON -- debiased bits={names}, "
              f"max|log-odds| per bit={spread}", flush=True)
    oracle_ignore_fake = False
    if config.get("frozen_poe_csv"):
        if config.get("oracle_poe") or config.get("oracle_poe_visible"):
            raise ValueError("frozen_poe_csv is exclusive with the oracle_poe "
                             "modes: the offsets come from the shipped table")
        if mlc_loss_mode != "bce":
            raise ValueError("frozen_poe_csv is defined for mlc_loss='bce' only")
        # the offset patterns live over the cell's *observable* set, which in
        # v1/v2 is narrower than the trained (in-head) set the loss mask uses
        obs_names = config.get("frozen_poe_obs_bits")
        if not obs_names:
            raise ValueError("frozen_poe_csv requires frozen_poe_obs_bits "
                             "(the cell's observable bits, short names)")
        oracle_visible_idx = tuple(sorted(TECHNIQUE_SHORT.index(n)
                                          for n in obs_names))
        oracle_table = build_frozen_offset_table(
            config["frozen_poe_csv"], config["frozen_poe_cell"],
            config["frozen_poe_variant"], config["frozen_poe_fold"],
            oracle_visible_idx).to(device)
        oracle_ignore_fake = True
        offset_bits = [MANIPULATION_TECHNIQUES[i] for i in
                       range(len(MANIPULATION_TECHNIQUES))
                       if bool((oracle_table[:, i].abs() > 0).any())]
        print(f"[{run_name}] FROZEN PoE offsets ON -- "
              f"cell={config['frozen_poe_cell']} variant={config['frozen_poe_variant']} "
              f"fold={config['frozen_poe_fold']} offset bits={offset_bits}, "
              f"rows={int((oracle_table.abs().sum(1) > 0).sum())}/{oracle_table.shape[0]}",
              flush=True)
    mlc_loss_scale = float(config.get("mlc_loss_scale", 1.0))
    if mlc_loss_scale != 1.0:
        print(f"[{run_name}] L_mlc SCALED by {mlc_loss_scale:.4f}", flush=True)
    if mlc_loss_mode == "softmax":
        print(f"[{run_name}] L_mlc = SOFTMAX cross-entropy (raw multi-hot; "
              f"pristine rows zero-loss); eval scores = softmax", flush=True)
    elif mlc_loss_mode == "softmax_norm":
        print(f"[{run_name}] L_mlc = SOFTMAX cross-entropy (NORMALISED target y/K; "
              f"pristine rows zero-loss); eval scores = softmax", flush=True)
    elif mlc_loss_mode == "type11":
        print(f"[{run_name}] L_mlc = 11-CLASS forgery-TYPE softmax-CE "
              f"(mutually-exclusive types; eval marginalised to 7-bit for mAP)", flush=True)
    optimizer.zero_grad()
    while iteration < config["total_iters"] and not stop:
        for batch in train_loader:
            technique_targets = _move_training_batch(batch, device)

            with (torch.amp.autocast("cuda", enabled=str(device).startswith("cuda"))
                  if hasattr(torch.amp, "autocast")
                  else torch.cuda.amp.autocast(enabled=str(device).startswith("cuda"))):
                reference_outputs = (encode_reference_batch(model, batch)
                                     if active_reference_index is not None else None)
                reference_features = (reference_outputs["fused"]
                                      if reference_outputs is not None else None)
                outputs = model(batch, reference_fused=reference_features)
                loss = _compute_training_loss(
                    outputs, batch, technique_targets, modalities, config["temperature"],
                    cross_modal_beta, identity_gamma, reference_outputs, accum_steps,
                    bit_mask=bit_mask, mlc_loss_mode=mlc_loss_mode,
                    binary_weight=0.0 if config.get("drop_binary_loss") else 1.0,
                    mlc_loss_scale=mlc_loss_scale,
                    oracle_table=oracle_table,
                    oracle_visible_idx=oracle_visible_idx,
                    oracle_ignore_fake=oracle_ignore_fake)
            scaler.scale(loss).backward()
            micro += 1
            if micro < accum_steps:
                continue                      # accumulate more before stepping
            micro = 0

            base_lr = cosine_warmup_lr(iteration, config)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * group["lr_scale"]
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if iteration % config["log_interval"] == 0:
                minutes = (time.time() - start_time) / 60
                display_loss = loss.item() * accum_steps
                print(f"[{run_name}] iter {iteration}/{config['total_iters']} "
                      f"loss {display_loss:.3f} ({minutes:.0f} min)", flush=True)

            # Atomic mid-run checkpoint (write-then-rename so a kill cannot
            # leave a truncated file behind).
            if iteration > 0 and iteration % CHECKPOINT_INTERVAL == 0:
                _save_periodic_checkpoint(
                    paths, model, optimizer, scaler, iteration, best_score,
                    best_model_state, bad_evals)

            # Periodic validation with early stopping on the best AUC.
            if iteration > 0 and iteration % config["eval_interval"] == 0:
                if config.get("select_bits"):
                    select_indices = tuple(sorted(
                        TECHNIQUE_SHORT.index(n) for n in config["select_bits"]))
                elif config.get("select_visible_only"):
                    select_indices = tuple(sorted(effective_visible_indices(
                        modalities, config)))
                else:
                    select_indices = None
                val_metrics = evaluate(model, val_dataset, tokenizer, config,
                                       device,
                                       technique_indices=select_indices)
                # The eval pass allocates/frees its own forward buffers, leaving
                # the caching allocator fragmented; without reclaiming it the next
                # training iteration can OOM on a heavy (visual+audio+text) config.
                torch.cuda.empty_cache()
                print(f"[{run_name}] VAL iter {iteration} ACC={val_metrics['ACC']:.2f} "
                      f"AUC={val_metrics['AUC']:.2f} mAP={val_metrics['mAP']:.2f}", flush=True)
                select_key = config.get("select_metric", "AUC")
                score = val_metrics[select_key] if not math.isnan(val_metrics[select_key]) \
                    else val_metrics["ACC"]
                if score > best_score:
                    best_score = score
                    # Keep the best weights on the *host* -- a GPU deepcopy of the
                    # full state dict (~0.9 GB) on top of the live training memory
                    # is enough to OOM the heaviest (visual+audio+text) configs on
                    # the next iteration. CPU storage is free here and loaded back
                    # to the device at the end.
                    best_model_state = {k: v.detach().cpu()
                                        for k, v in model.state_dict().items()}
                    bad_evals = 0
                    # Persist the improvement now, atomically: a walltime kill
                    # or requeue before the next periodic checkpoint would
                    # otherwise lose it (the end-of-run save never happens).
                    torch.save(best_model_state, paths.best_path + ".tmp")
                    os.replace(paths.best_path + ".tmp", paths.best_path)
                    with open(paths.best_meta_path + ".tmp", "w") as handle:
                        json.dump({"best_score": best_score, "iteration": iteration},
                                  handle)
                    os.replace(paths.best_meta_path + ".tmp", paths.best_meta_path)
                else:
                    bad_evals += 1
                    if bad_evals >= config["early_stop_patience"]:
                        print(f"[{run_name}] early stopping", flush=True)
                        stop = True
                        break

            iteration += 1
            if iteration >= config["total_iters"]:
                break

    # --final-eval: the loop above never validates at iteration == total_iters
    # (it increments and breaks first), so a cosine schedule that decays fully
    # inside the budget is never scored. When enabled, score the final weights
    # once, let them compete for best-checkpoint selection under the same rule,
    # and persist them as {run_name}_final.pt for a deferred "final" test pass.
    if config.get("final_eval"):
        final_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(final_state, paths.final_path + ".tmp")
        os.replace(paths.final_path + ".tmp", paths.final_path)
        if config.get("select_bits"):
            final_select = tuple(sorted(
                TECHNIQUE_SHORT.index(n) for n in config["select_bits"]))
        elif config.get("select_visible_only"):
            final_select = tuple(sorted(effective_visible_indices(modalities, config)))
        else:
            final_select = None
        final_metrics = evaluate(model, val_dataset, tokenizer, config, device,
                                 technique_indices=final_select)
        torch.cuda.empty_cache()
        print(f"[{run_name}] VAL iter {iteration} ACC={final_metrics['ACC']:.2f} "
              f"AUC={final_metrics['AUC']:.2f} mAP={final_metrics['mAP']:.2f} (final)",
              flush=True)
        final_key = config.get("select_metric", "AUC")
        final_score = (final_metrics[final_key] if not math.isnan(final_metrics[final_key])
                       else final_metrics["ACC"])
        if final_score > best_score:
            best_score = final_score
            best_model_state = final_state
            print(f"[{run_name}] final weights selected as best ({final_key}={final_score:.3f})",
                  flush=True)

    # Load the best validation weights and persist the checkpoint (always), so
    # the deferred test pass can score it even when test eval is skipped here.
    if best_model_state:
        model.load_state_dict(best_model_state)
    torch.save(best_model_state, paths.best_path + ".tmp")
    os.replace(paths.best_path + ".tmp", paths.best_path)
    # The run completed; its crash-resume state is no longer needed.
    for stale in (paths.checkpoint_path, paths.best_meta_path):
        if os.path.exists(stale):
            os.remove(stale)
    minutes = (time.time() - start_time) / 60

    if skip_test_eval:
        # Train-only phase: the held-out test split is not extracted yet.
        # run_test_eval.py loads {run_name}.pt and scores it later.
        print(f"[{run_name}] *** test eval SKIPPED -- checkpoint saved | {minutes:.0f} min",
              flush=True)
        return {"test": None, "test_eval": "skipped",
                "best_val_score": round(best_score, 4),
                "train_fake": num_fake, "train_real": num_real,
                "techniques": MANIPULATION_TECHNIQUES,
                "config": dict(config),
                "minutes": round(minutes, 1)}

    test_metrics = _evaluate_test_split(
        model, test_clips, modalities, use_reference, reference_index,
        tokenizer, config, device)
    print(f"[{run_name}] *** TEST ACC={test_metrics['ACC']:.2f} "
          f"AUC={test_metrics['AUC']:.2f} mAP={test_metrics['mAP']:.2f} "
          f"CF1={test_metrics['CF1']:.2f} OF1={test_metrics['OF1']:.2f} "
          f"| {minutes:.0f} min", flush=True)
    return {"test": test_metrics,
            "train_fake": num_fake, "train_real": num_real,
            "techniques": MANIPULATION_TECHNIQUES,
            "config": dict(config),
            "minutes": round(minutes, 1)}
