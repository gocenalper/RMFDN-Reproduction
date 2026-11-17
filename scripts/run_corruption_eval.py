#!/usr/bin/env python3
"""Score trained checkpoints on clean and corrupted test inputs.

Test-time robustness evaluation of the improvement experiment (report
Section 6). For every trained run in a suite directory and every condition in
``rmfdn_ablation.corruptions.CONDITIONS`` this script scores the complete test
split and writes

* ``--output``: one JSON with the metrics per run and condition, rewritten
  after every (run, condition) so a relaunch only fills in what is missing;
* ``--dump-scores``: one ``<run>__<condition>.npz`` per pair with the per-clip
  fake probability, label, forgery type and, for gated models, the mean gate
  value. ``scripts/analyze_improvement.py`` reads these files.

Corruptions use fixed seeds (``--corruption-seed``), so every arm and every
training seed sees exactly the same corrupted clips. Nothing here is tuned on
the test set: the checkpoint is chosen on validation during training.

    python scripts/run_corruption_eval.py --phase1-dir runs/seed0 \\
        --data-root /data/idforge --reference-root /data/idforge_ref \\
        --output runs/seed0/corruption_results.json --dump-scores runs/seed0/scores
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from rmfdn_ablation.config import MODALITY_CONFIGS, default_device
from rmfdn_ablation.corruptions import CONDITIONS
from rmfdn_ablation.data_catalog import scan_reference_clips
from rmfdn_ablation.dataset import MultimodalClipDataset
from rmfdn_ablation.evaluation import evaluate
from rmfdn_ablation.protocols import build_protocol_splits
from rmfdn_ablation.RMFDN import RMFDN
from run_test_eval import _evaluation_config, _load_evaluation_checkpoint, _result_directory

_MODALITIES_BY_NAME = dict(MODALITY_CONFIGS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase1-dir", required=True, help="suite dir with results.json and runs/*.pt")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--reference-root", required=True)
    ap.add_argument("--p2-split", default=None)
    ap.add_argument("--checkpoint", choices=["best_validation", "final"], default="best_validation")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    choices=list(CONDITIONS))
    ap.add_argument("--runs", nargs="+", default=None, help="only these run names")
    ap.add_argument("--corruption-seed", type=int, default=0)
    ap.add_argument("--bert-model", default=None)
    ap.add_argument("--ast-checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--dump-scores", default=None, help="directory for per-clip .npz dumps")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import torch.multiprocessing as mp
    from transformers import BertTokenizer
    mp.set_sharing_strategy("file_system")

    suite_dir = _result_directory(Path(args.phase1_dir))
    training_results = json.load(open(suite_dir / "results.json"))
    base_config = _evaluation_config(next(iter(training_results.values())), args)
    device = default_device()
    tokenizer = BertTokenizer.from_pretrained(base_config["bert_model"])
    reference_index = None
    split_cache = {}
    out = json.load(open(args.output)) if os.path.exists(args.output) else {}
    if args.dump_scores:
        os.makedirs(args.dump_scores, exist_ok=True)

    for name, run in training_results.items():
        spec = run.get("spec")
        if not spec or (args.runs and name not in args.runs):
            continue
        todo = [c for c in args.conditions if c not in out.get(name, {}).get("conditions", {})]
        if not todo:
            continue
        state, meta = _load_evaluation_checkpoint(suite_dir, name, device, args.checkpoint)
        if state is None:
            print(f"[{name}] no checkpoint -> skip", flush=True)
            continue
        modalities = _MODALITIES_BY_NAME[spec["modality"]]
        use_reference = spec["identity"] == "i1"
        if use_reference and reference_index is None:
            reference_index = scan_reference_clips(args.reference_root)
        config = _evaluation_config(run, args)
        config = {**config, "reference_alpha": config["reference_alpha"] if use_reference else 0.0}
        model = RMFDN(config, modalities).to(device)
        model.load_state_dict(state)
        if spec["protocol"] not in split_cache:
            split_cache[spec["protocol"]] = build_protocol_splits(
                spec["protocol"], args.data_root, args.p2_split)
        test_clips = split_cache[spec["protocol"]]["test"]

        entry = out.setdefault(name, {"spec": spec, "checkpoint": meta,
                                      "variant": spec.get("variant", "base"),
                                      "seed": config.get("seed"), "conditions": {}})
        for condition_name in todo:
            dataset = MultimodalClipDataset(
                test_clips, modalities, config,
                reference_index=reference_index if use_reference else None,
                reference_seed=config["seed"],
                corruption=condition_name, corruption_seed=args.corruption_seed)
            metrics = evaluate(model, dataset, tokenizer, config, device, collect_scores=True)
            arrays = metrics.pop("_arrays")
            if "gate" in arrays:
                metrics["gate_mean"] = float(arrays["gate"].mean())
            if args.dump_scores:
                payload = {"pfake": arrays["fake_scores"], "is_fake": arrays["fake_labels"],
                           "ftype": np.array([c.forgery_type for c in test_clips]),
                           "directory": np.array([c.directory for c in test_clips])}
                if "gate" in arrays:
                    payload["gate"] = arrays["gate"]
                np.savez_compressed(os.path.join(args.dump_scores,
                                                 f"{name}__{condition_name}.npz"), **payload)
            entry["conditions"][condition_name] = metrics
            json.dump(out, open(args.output, "w"), indent=1)
            print(f"[{name}] {condition_name:17s} ACC={metrics['ACC']:.2f} "
                  f"AUC={metrics['AUC']:.2f}"
                  + (f" gate={metrics['gate_mean']:.3f}" if "gate_mean" in metrics else ""),
                  flush=True)
    print("CORRUPTION_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
