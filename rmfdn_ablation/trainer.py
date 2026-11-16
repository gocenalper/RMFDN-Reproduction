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

from .config import (MANIPULATION_TECHNIQUES, default_device,
                     visible_technique_indices)
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
    """Build AdamW with the visual backbone's discriminative LR group."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if "v" in modalities:
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
    """Build the training dataset and its shuffled DataLoader."""
    train_dataset = MultimodalClipDataset(
        train_clips, modalities, config, reference_index=reference_index)
    # NOTE: persistent_workers must stay False -- long-lived workers leaked
    # RSS over the 90k-iteration runs until the OOM killer fired.
    return DataLoader(train_dataset, batch_size=config["batch_size"],
                      shuffle=True,
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


def _visible_bit_mask(modalities: Sequence[str]) -> torch.Tensor:
    """(7,) 0/1 mask of the technique bits observable from ``modalities``."""
    mask = torch.zeros(len(MANIPULATION_TECHNIQUES))
    mask[list(visible_technique_indices(modalities))] = 1.0
    return mask


def _compute_training_loss(outputs: dict, batch: dict, technique_targets: torch.Tensor,
                           modalities: Sequence[str], temperature: float,
                           cross_modal_beta: float, identity_gamma: float,
                           reference_outputs, accum_steps: int,
                           bit_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compose the paper's supervised and auxiliary losses for one micro-batch."""
    if bit_mask is None:
        technique_loss = F.binary_cross_entropy_with_logits(
            outputs["technique_logits"], technique_targets)
    else:  # supervise only the modality-visible bits (masked mean)
        per_bit = F.binary_cross_entropy_with_logits(
            outputs["technique_logits"], technique_targets, reduction="none")
        technique_loss = ((per_bit * bit_mask).sum()
                          / (bit_mask.sum() * per_bit.shape[0]))
    binary_loss = F.cross_entropy(outputs["binary_logits"], batch["is_fake"])
    cross_modal_term = cross_modal_beta * cross_modal_consistency_loss(
        outputs, batch["is_fake"], modalities, temperature)
    identity_term = (identity_gamma * identity_aware_contrastive_loss(
        outputs, batch["identity"], batch["is_fake"], reference_outputs,
        modalities, temperature) if identity_gamma else 0.0)
    return (binary_loss + technique_loss + cross_modal_term + identity_term) / accum_steps


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
                          skip_test_eval: bool = False) -> dict:
    """Train one configuration and report complete held-out test metrics."""
    device = device or default_device()
    _seed_everything(config["seed"])

    # Reference ablation: the no-reference arm zeroes both knobs.
    identity_gamma = config["identity_gamma"] if use_reference else 0.0
    run_config = {**config,
                  "reference_alpha": config["reference_alpha"] if use_reference else 0.0}
    cross_modal_beta = config["cross_modal_beta"] if len(modalities) >= 2 else 0.0

    model = RMFDN(run_config, modalities).to(device)

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
        bit_mask = _visible_bit_mask(modalities).to(device)
        print(f"[{run_name}] invisible-bit BCE masking ON -- supervised bits="
              f"{[MANIPULATION_TECHNIQUES[i] for i in range(len(MANIPULATION_TECHNIQUES)) if bit_mask[i] > 0]}",
              flush=True)
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
                    bit_mask=bit_mask)
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
                val_metrics = evaluate(model, val_dataset, tokenizer, config, device)
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
        final_metrics = evaluate(model, val_dataset, tokenizer, config, device)
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
