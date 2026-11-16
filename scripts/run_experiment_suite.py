#!/usr/bin/env python3
"""Train the cells of an experiment matrix.

Reads a declarative matrix (a JSON config with a ``cells`` list) and runs
every cell in order, writing all metrics, including per-technique accuracy,
to ``{output-dir}/results.json`` after each cell. Finished cells are skipped
when the command is launched again.

Example::

    python scripts/run_experiment_suite.py \\
        --config experiments/experiment_matrix.json \\
        --data-root /path/to/idforge --reference-root /path/to/idforge_reference \\
        --p2-split protocol_p2/p2_split.json \\
        --output-dir ./suite_out --iters 6000 --lr-schedule-iters 90000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmfdn_ablation.config import DEFAULT_CONFIG
from rmfdn_ablation.experiment import expand_cells, run_experiment_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configurable experiment matrix")
    parser.add_argument("--config", required=True,
                        help="JSON config with a 'cells' list")
    parser.add_argument("--data-root", required=True,
                        help="dataset root with on-disk train/ val/ test/")
    parser.add_argument("--reference-root", required=True,
                        help="pristine per-identity reference set root")
    parser.add_argument("--p2-split", default=None,
                        help="identity -> split JSON, required if any cell uses protocol p2")
    parser.add_argument("--output-dir", default="./suite_out",
                        help="where results.json and runs/*.pt are written")
    parser.add_argument("--iters", type=int, default=None,
                        help="maximum optimizer steps to execute (e.g. 12000 for a pilot)")
    parser.add_argument("--lr-schedule-iters", type=int, default=None,
                        help="cosine-decay horizon in optimizer steps; keep at 90000 "
                             "for paper-schedule pilots")
    parser.add_argument("--val-cap", type=int, default=None,
                        help="cap validation to N clips (fixed seed-0 subsample)")
    parser.add_argument("--eval-interval", type=int, default=None,
                        help="validate every N iters (paper 90k protocol: 3000)")
    parser.add_argument("--patience", type=int, default=None,
                        help="early-stop patience (non-improving validations)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override training/eval batch size (per micro-batch)")
    parser.add_argument("--grad-accum-steps", type=int, default=None,
                        help="accumulate this many micro-batches per optimizer step; "
                             "effective batch = batch-size * grad-accum-steps (default 1)")
    parser.add_argument("--peak-lr", type=float, default=None,
                        help="override peak learning rate")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (tune to the box; default %d)"
                             % DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--bert-model", default=None,
                        help="local path or HF id for BERT; pass the local model "
                             "directory when running offline (no internet on node)")
    parser.add_argument("--skip-test-eval", action="store_true",
                        help="train and checkpoint only; skip the final filtered/full "
                             "test evaluation (run it later via run_test_eval.py when "
                             "the test split is extracted)")
    parser.add_argument("--ast-checkpoint", default=None,
                        help="local path or HF id for the AST encoder (pass the local "
                             "directory when running offline)")
    parser.add_argument("--final-eval", action="store_true",
                        help="also validate the weights at the final iteration (eligible "
                             "for best-checkpoint selection) and save runs/<cell>_final.pt")
    parser.add_argument("--select-metric", choices=["ACC", "AUC", "mAP"], default=None,
                        help="validation metric for best-checkpoint/early-stop (default AUC)")
    parser.add_argument("--blank-modalities", default=None,
                        help="train/eval the unchanged (VAT) graph with these "
                             "modalities' raw inputs blanked, e.g. 'at' -> "
                             "black frames kept real, zero mel, empty text")
    parser.add_argument("--mask-invisible-bits", action="store_true",
                        help="supervise the technique BCE only on bits directly "
                             "perceivable by the cell's modalities (v: lip/face, "
                             "a: tts/rvc, t: textgen, v+a: +aud_sh, a+t: +text_sh)")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the training seed (default: config seed 0)")
    parser.add_argument("--list", action="store_true",
                        help="print the expanded run list and exit (no training)")
    args = parser.parse_args()

    cells = json.load(open(args.config))["cells"]
    specs = expand_cells(cells)

    if args.list:
        print(f"{len(specs)} runs:")
        for spec in specs:
            print(f"  {spec.name}")
        return

    config = dict(DEFAULT_CONFIG)
    if args.iters is not None:
        config["total_iters"] = args.iters
    if args.lr_schedule_iters is not None:
        config["lr_schedule_iters"] = args.lr_schedule_iters
    if args.val_cap is not None:
        config["val_subsample_cap"] = args.val_cap
    if args.eval_interval is not None:
        config["eval_interval"] = args.eval_interval
    if args.patience is not None:
        config["early_stop_patience"] = args.patience
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        config["grad_accum_steps"] = args.grad_accum_steps
    if args.peak_lr is not None:
        config["peak_lr"] = args.peak_lr
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.bert_model is not None:
        config["bert_model"] = args.bert_model
    if args.select_metric is not None:
        config["select_metric"] = args.select_metric
    if args.final_eval:
        config["final_eval"] = True
    if args.mask_invisible_bits:
        config["mask_invisible_bits"] = True
    if args.seed is not None:
        config["seed"] = args.seed
    if args.blank_modalities is not None:
        blank = set(args.blank_modalities.lower())
        if not blank or not blank.issubset({"v", "a", "t"}):
            parser.error("--blank-modalities must be a non-empty subset of 'vat'")
        config["blank_modalities"] = "".join(sorted(blank))
    config["skip_test_eval"] = args.skip_test_eval
    if args.ast_checkpoint is not None:
        config["ast_checkpoint"] = args.ast_checkpoint

    # Large test-set evaluation opens many files per worker: keep cv2's own
    # thread pool off (workers otherwise explode into thousands of threads) and
    # use the file_system fd-sharing strategy -- the container fd limit is 1024
    # and not raisable, so this keeps fds worker-bound for the 77k test eval.
    try:
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        pass  # cv2 is only used to decode JPEG frame archives (frames.jpz);
              # datasets stored as frames_ndarray.npy never import it.
    import torch
    torch.multiprocessing.set_sharing_strategy("file_system")

    run_experiment_sequence(specs, config, args.data_root, args.reference_root,
                            args.output_dir, p2_split_path=args.p2_split)


if __name__ == "__main__":
    main()
