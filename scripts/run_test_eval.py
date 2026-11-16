#!/usr/bin/env python3
"""Phase 2: evaluate Phase-1 checkpoints on the held-out test split.

Phase 1 trains with ``--skip-test-eval`` (test split unextracted to save
inodes). Once the test split is extracted, this loads each cell's best-val
checkpoint and scores the complete test split with live shared reference encoders
with ACC/AUC/mAP/CF1/OF1 plus per-technique accuracy. Results are written
incrementally to ``--output`` so a crash/resume re-evaluates only missing cells.

The model architecture is canonical and the matrix ``spec`` restores its
modalities, fusion type, protocol, and reference arm.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from rmfdn_ablation.config import (MANIPULATION_TECHNIQUES, MODALITY_CONFIGS,
                                   default_device, effective_visible_indices)
from rmfdn_ablation.data_catalog import scan_reference_clips
from rmfdn_ablation.dataset import MultimodalClipDataset
from rmfdn_ablation.evaluation import evaluate
from rmfdn_ablation.protocols import build_protocol_splits

_MODALITIES_BY_NAME = dict(MODALITY_CONFIGS)


def _result_directory(phase1_dir: Path) -> Path:
    """Validate and return the canonical suite output directory."""
    if not (phase1_dir / "results.json").is_file():
        raise FileNotFoundError(f"missing suite results: {phase1_dir / 'results.json'}")
    return phase1_dir


def _evaluation_config(run: dict, args) -> dict:
    """Restore training config and apply only explicit runtime/path overrides."""
    if "config" not in run:
        raise ValueError("phase-1 result has no saved effective config")
    config = dict(run["config"])
    for key in ("bert_model", "ast_checkpoint", "batch_size", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def _model_class(cross_modal):
    from rmfdn_ablation.experiment import model_class_for
    return model_class_for(cross_modal)


def _load_evaluation_checkpoint(suite_dir: Path, name: str, device: str,
                                kind: str = "best_validation"):
    """Load validation-best weights, falling back to an interrupted run state.
    kind="final" loads the last-iteration weights saved by --final-eval."""
    if kind == "final":
        final_path = suite_dir / "runs" / f"{name}_final.pt"
        if not final_path.exists():
            raise FileNotFoundError(f"no final checkpoint (train with --final-eval): {final_path}")
        return torch.load(final_path, map_location=device), {
            "kind": "final", "path": str(final_path),
        }
    best_path = suite_dir / "runs" / f"{name}.pt"
    if best_path.exists():
        return torch.load(best_path, map_location=device), {
            "kind": "best_validation", "path": str(best_path),
        }

    periodic_path = suite_dir / "runs" / f"{name}_ck.pt"
    if not periodic_path.exists():
        return None, None

    payload = torch.load(periodic_path, map_location=device)
    embedded_best = payload.get("best_model_state")
    if embedded_best is not None:
        return embedded_best, {
            "kind": "periodic_embedded_best",
            "path": str(periodic_path),
            "iteration": payload.get("iteration"),
            "best_score": payload.get("best_score"),
        }
    return payload["model_state"], {
        "kind": "periodic_current",
        "path": str(periodic_path),
        "iteration": payload.get("iteration"),
        "best_score": payload.get("best_score"),
    }


def main():
    ap = argparse.ArgumentParser(description="Phase-2 held-out test evaluation")
    ap.add_argument("--checkpoint", choices=["best_validation", "final"],
                    default="best_validation",
                    help="which trained weights to score (final needs --final-eval training)")
    ap.add_argument("--phase1-dir", required=True,
                    help="suite dir containing results.json and runs/*.pt")
    ap.add_argument("--data-root", required=True, help="root with on-disk train/val/test")
    ap.add_argument("--reference-root", required=True)
    ap.add_argument("--p2-split", default=None)
    ap.add_argument("--bert-model", default=None)
    ap.add_argument("--ast-checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--visible-bits-only", action="store_true",
                    help="compute mAP/CF1/OF1 only over technique bits directly "
                         "observable from each run's modalities")
    ap.add_argument("--metric-implementation",
                    choices=["sklearn", "reference_apmeter"],
                    default="sklearn",
                    help="multi-label metric formulas; reference_apmeter reproduces "
                         "MultiModal-DeepFake for BCE checkpoints")
    ap.add_argument("--dump-scores", default=None,
                    help="directory for per-clip score dumps (test_<run>.npz with "
                         "raw logits under reference_apmeter)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import BertTokenizer
    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")

    suite_dir = _result_directory(Path(args.phase1_dir))
    results_path = suite_dir / "results.json"
    training_results = json.load(open(results_path))
    first_runnable = next(
        run for run in training_results.values() if run.get("status") != "not_implemented"
    )
    base_config = _evaluation_config(first_runnable, args)

    device = default_device()
    tokenizer = BertTokenizer.from_pretrained(base_config["bert_model"])
    # scanned lazily: i3-only suites never touch the reference bank
    reference_index = None

    out = json.load(open(args.output)) if os.path.exists(args.output) else {}
    split_cache = {}

    for name, run in training_results.items():
        spec = run.get("spec")
        if not spec or name in out:
            continue
        if run.get("status") == "not_implemented":
            out[name] = run
            json.dump(out, open(args.output, "w"), indent=1)
            continue
        checkpoint_state, checkpoint_meta = _load_evaluation_checkpoint(
            suite_dir, name, device, args.checkpoint)
        if checkpoint_state is None:
            print(f"[{name}] no checkpoint -> skip", flush=True)
            continue

        modalities = _MODALITIES_BY_NAME[spec["modality"]]
        use_reference = spec["identity"] == "i1"
        if use_reference and reference_index is None:
            reference_index = scan_reference_clips(args.reference_root)
        saved_config = _evaluation_config(run, args)
        run_cfg = {**saved_config,
                   "reference_alpha": (saved_config["reference_alpha"]
                                       if use_reference else 0.0)}

        model = _model_class(spec["cross_modal"])(run_cfg, modalities).to(device)
        model.load_state_dict(checkpoint_state)

        if spec["protocol"] not in split_cache:
            split_cache[spec["protocol"]] = build_protocol_splits(
                spec["protocol"], args.data_root, args.p2_split)
        test_clips = split_cache[spec["protocol"]]["test"]

        dataset = MultimodalClipDataset(
            test_clips, modalities, run_cfg,
            reference_index=reference_index if use_reference else None,
            reference_seed=run_cfg["seed"])
        # A probe run widened its own visible set, and a Phase-1 run may pin
        # it outright; score exactly what the run supervised, so the eval
        # matches the training contract.
        technique_indices = (effective_visible_indices(modalities, saved_config)
                             if args.visible_bits_only else None)
        metrics = evaluate(model, dataset, tokenizer, run_cfg, device,
                           technique_indices=technique_indices,
                           metric_implementation=args.metric_implementation,
                           collect_scores=bool(args.dump_scores))
        arrays = metrics.pop("_arrays", None)
        if arrays is not None:
            os.makedirs(args.dump_scores, exist_ok=True)
            score_key = ("logits" if args.metric_implementation == "reference_apmeter"
                         else "scores")
            np.savez_compressed(
                os.path.join(args.dump_scores, f"test_{name}.npz"),
                **{score_key: arrays["technique_scores"]},
                targets=arrays["technique_labels"],
                pfake=arrays["fake_scores"],
                is_fake=arrays["fake_labels"],
                ftype=np.array([c.forgery_type for c in test_clips]),
                ident=np.array([c.identity for c in test_clips]),
                directory=np.array([c.directory for c in test_clips]),
                techniques=np.array(MANIPULATION_TECHNIQUES))
            print(f"[{name}] scores dumped -> {args.dump_scores}/test_{name}.npz",
                  flush=True)
        out[name] = {"test": metrics, "spec": spec, "config": saved_config,
                     "checkpoint": checkpoint_meta,
                     "metric_implementation": args.metric_implementation,
                     "metric_scope": ("visible_bits_only"
                                      if args.visible_bits_only else "all_bits")}
        json.dump(out, open(args.output, "w"), indent=1)
        print(f"[{name}] *** TEST [{checkpoint_meta['kind']}] "
              f"ACC={metrics['ACC']:.2f} AUC={metrics['AUC']:.2f} "
              f"mAP={metrics['mAP']:.2f} CF1={metrics['CF1']:.2f} "
              f"OF1={metrics['OF1']:.2f}", flush=True)

    print("TEST_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
