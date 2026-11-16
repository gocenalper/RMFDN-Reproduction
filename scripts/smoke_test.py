#!/usr/bin/env python3
"""End-to-end smoke test on synthetic clips. No dataset and no GPU required.

The script builds a tiny synthetic dataset that follows the on-disk data
contract (frame arrays, wav audio, transcripts, a per-identity reference
bank), then runs the two real entry points on it:

1. ``run_experiment_sequence`` -- trains a few optimizer steps per cell,
   validates, checkpoints and scores the test split;
2. ``scripts/run_test_eval.py`` -- reloads every checkpoint and re-scores it.

It exercises data loading, the fusion model for three, two and one
modalities, the identity-reference path (i1), both split protocols (p1, p2),
the losses, the checkpoint logic and the metrics. The numbers it prints are meaningless: the
clips are random noise. It only proves that the pipeline runs.

By default the encoders are small randomly initialised models, so the test
needs no network access and finishes in about a minute on a laptop CPU. Pass
``--pretrained`` to build the real ImageNet ResNet-18, AudioSet AST and
BERT-base instead (downloads roughly 1 GB from the Hugging Face Hub once).

    python scripts/smoke_test.py
    python scripts/smoke_test.py --pretrained --keep ./smoke_out
"""

import argparse
import json
import math
import shutil
import sys
import tempfile
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np

#: Forgery types written to the synthetic dataset. Together they switch every
#: one of the seven technique bits on and off at least once.
SMOKE_TYPES = ("pristine", "face_tts_textgen", "lip_rvc_textmismatch",
               "face_audiomismatch_textmismatch")

#: identity -> split used for the identity-disjoint (p2) cell.
P2_ASSIGNMENT = {"train": ["id01", "id02"], "val": ["id03"], "test": ["id04"]}
IDENTITIES = tuple(i for ids in P2_ASSIGNMENT.values() for i in ids)

#: Cells trained by the smoke test: identity x protocol x modalities.
SMOKE_CELLS = [
    {"cross_modal": "m1", "identity": "i1", "protocol": "p1", "modalities": ["VAT"]},
    {"cross_modal": "m1", "identity": "i3", "protocol": "p2", "modalities": ["VA", "T"]},
]

WORDS = ("the", "speaker", "said", "that", "this", "is", "a", "test", "of",
         "synthetic", "speech", "and", "video", "today", "we", "talk")


def _write_clip(directory: Path, rng: np.random.Generator) -> None:
    """Write one clip: 16 small RGB frames, 2 s of 16 kHz audio, a transcript."""
    directory.mkdir(parents=True, exist_ok=True)
    frames = rng.integers(0, 256, size=(16, 64, 64, 3), dtype=np.uint8)
    np.save(directory / "frames_ndarray.npy", frames)
    samples = (rng.standard_normal(32000) * 3000).astype("<i2")
    with wave.open(str(directory / f"{directory.name}.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples.tobytes())
    text = " ".join(rng.choice(WORDS, size=8))
    (directory / f"{directory.name}.txt").write_text(text, encoding="utf-8")


def build_synthetic_dataset(root: Path, seed: int = 0) -> dict:
    """Create ``data/``, ``reference/`` and a p2 split file under ``root``."""
    rng = np.random.default_rng(seed)
    data_root, reference_root = root / "data", root / "reference"
    count = 0
    for split in ("train", "val", "test"):
        for forgery_type in SMOKE_TYPES:
            for identity in IDENTITIES:
                clip = f"{split}_{forgery_type}_{identity}"
                _write_clip(data_root / split / forgery_type / identity / clip, rng)
                count += 1
    for identity in IDENTITIES:
        for index in range(2):
            _write_clip(reference_root / identity / f"ref_{identity}_{index}", rng)
    split_path = root / "p2_split.json"
    split_path.write_text(json.dumps({"splits": P2_ASSIGNMENT}, indent=1))
    vocab_path = root / "vocab.txt"
    vocab_path.write_text("\n".join(("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]") + WORDS) + "\n")
    print(f"[smoke] wrote {count} clips and {2 * len(IDENTITIES)} reference clips to {root}",
          flush=True)
    return {"data_root": str(data_root), "reference_root": str(reference_root),
            "p2_split": str(split_path), "vocab": str(vocab_path)}


def use_small_random_encoders(vocab_path: str) -> None:
    """Swap every pretrained download for a small randomly initialised model.

    Only the weight sources change. The model code, data pipeline, trainer
    and evaluation that run afterwards are the real ones.
    """
    import timm
    from transformers import (ASTConfig, ASTFeatureExtractor, ASTModel, BertConfig,
                              BertModel, BertTokenizer)

    vocab_size = len(Path(vocab_path).read_text().split())
    create_model = timm.create_model
    timm.create_model = lambda name, pretrained=False, **kwargs: create_model(
        name, pretrained=False, **kwargs)
    ASTModel.from_pretrained = classmethod(lambda cls, *_args, **_kwargs: cls(ASTConfig(
        hidden_size=64, num_hidden_layers=1, num_attention_heads=2, intermediate_size=128)))
    BertModel.from_pretrained = classmethod(lambda cls, *_args, **_kwargs: cls(BertConfig(
        vocab_size=vocab_size, hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=128)))
    BertTokenizer.from_pretrained = classmethod(
        lambda cls, *_args, **_kwargs: cls(vocab_file=vocab_path))
    ASTFeatureExtractor.from_pretrained = classmethod(lambda cls, *_args, **_kwargs: cls())


def _check_metrics(name: str, metrics: dict) -> None:
    for key in ("ACC", "AUC", "mAP", "CF1", "OF1"):
        value = metrics[key]
        if not isinstance(value, float) or math.isnan(value) or not 0.0 <= value <= 100.0:
            raise AssertionError(f"{name}: metric {key} is not a finite percentage: {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pretrained", action="store_true",
                        help="use the real pretrained ResNet-18 / AST / BERT "
                             "(downloads about 1 GB) instead of small random encoders")
    parser.add_argument("--iters", type=int, default=2,
                        help="optimizer steps per cell (default 2)")
    parser.add_argument("--keep", default=None,
                        help="write the dataset and outputs here and keep them "
                             "(default: a temporary directory that is deleted)")
    args = parser.parse_args()

    import torch

    from rmfdn_ablation.config import DEFAULT_CONFIG, default_device
    from rmfdn_ablation.data_catalog import clip_basename, scan_split_clips
    from rmfdn_ablation.experiment import expand_cells, run_experiment_sequence
    from rmfdn_ablation.media_io import load_ast_input

    root = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="rmfdn_smoke_"))
    if args.keep and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        paths = build_synthetic_dataset(root)
        config = dict(DEFAULT_CONFIG)
        config.update(total_iters=args.iters, lr_schedule_iters=args.iters, warmup_iters=1,
                      eval_interval=1, early_stop_patience=10, log_interval=1,
                      batch_size=4, num_workers=0, mask_invisible_bits=True,
                      skip_test_eval=False)
        if not args.pretrained:
            use_small_random_encoders(paths["vocab"])
            config["bert_model"] = paths["vocab"]
        print(f"[smoke] device={default_device()} torch={torch.__version__} "
              f"encoders={'pretrained' if args.pretrained else 'small random'}", flush=True)

        # A missing audio backend degrades every clip to silence without
        # raising, so check explicitly that real audio reaches the model.
        first_clip = scan_split_clips(paths["data_root"], "train")[0]
        mel = load_ast_input(first_clip.directory, clip_basename(first_clip.directory), config)
        if tuple(mel.shape) != (1024, 128) or float(mel.abs().sum()) == 0.0:
            raise AssertionError("audio decoding failed: the AST input is empty")

        specs = expand_cells(SMOKE_CELLS)
        output_dir = root / "suite_out"
        results = run_experiment_sequence(
            specs, config, paths["data_root"], paths["reference_root"], str(output_dir),
            p2_split_path=paths["p2_split"])
        for spec in specs:
            if spec.name not in results:
                raise AssertionError(f"{spec.name}: training failed (see the traceback above)")
            _check_metrics(spec.name, results[spec.name]["test"])
            if not (output_dir / "runs" / f"{spec.name}.pt").is_file():
                raise AssertionError(f"{spec.name}: best checkpoint was not written")

        # Second entry point: reload each checkpoint and score the test split.
        import run_test_eval

        test_output = output_dir / "test_results.json"
        sys.argv = ["run_test_eval.py", "--phase1-dir", str(output_dir),
                    "--data-root", paths["data_root"],
                    "--reference-root", paths["reference_root"],
                    "--p2-split", paths["p2_split"], "--visible-bits-only",
                    "--output", str(test_output)]
        run_test_eval.main()
        reloaded = json.loads(test_output.read_text())
        for spec in specs:
            _check_metrics(f"{spec.name} (reloaded)", reloaded[spec.name]["test"])
            # Same weights, same clips, same seed: the detection score must match.
            first, second = results[spec.name]["test"]["AUC"], reloaded[spec.name]["test"]["AUC"]
            if abs(first - second) > 1e-3:
                raise AssertionError(
                    f"{spec.name}: test AUC changed after reloading the checkpoint "
                    f"({first:.4f} -> {second:.4f})")

        print(f"\n[smoke] OK: {len(specs)} cells trained, checkpointed, "
              f"evaluated and re-evaluated from disk.", flush=True)
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
