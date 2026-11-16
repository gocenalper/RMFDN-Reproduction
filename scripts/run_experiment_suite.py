#!/usr/bin/env python3
"""Entry point for the configurable experiment suite.

Reads a declarative matrix (a JSON config of cells) and runs every cell in
order, resume-safely, writing all metrics -- including per-bit (per-technique)
accuracy -- to ``{output}/results.json``.

The protocol is configurable per cell, so the advisor's full study runs with a
single command. Both fusion variants run -- ``m1`` (progressive cross-attention)
and ``m2`` (MulG directed pairwise cross-modal attention + GRU). Cells using the
not-yet-built ``i2`` verification identity are recorded as ``not_implemented``
and skipped without stopping the runnable cells.

Examples::

    # Run the advisor's matrix; p2 cells use the identity-disjoint split.
    python scripts/run_experiment_suite.py \
        --config experiments/experiment_matrix.json \
        --data-root /path/to/jpz --reference-root /path/to/jpz/ref \
        --p2-split protocol_p2/p2_split.json \
        --output-dir ./suite_out --iters 12000 --lr-schedule-iters 90000 \
        --val-cap 4000

    # A custom subset of cells in a one-off config file works the same way.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmfdn_ablation.config import (DEFAULT_CONFIG, MANIPULATION_TECHNIQUES,
                                   TECHNIQUE_SHORT)
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
    parser.add_argument("--bounded-sampler-pristine", type=float, default=None,
                        help="sample this pristine share and use bounded max-entropy "
                             "weights across all available fake types")
    parser.add_argument("--fake-type-shares", default=None,
                        help="JSON file mapping fake type -> per-1000 integer share "
                             "(must sum to 1000); overrides the built-in DECOR table")
    parser.add_argument("--mlc-loss", choices=["bce", "softmax", "softmax_norm", "type11"], default=None,
                        help="technique-head loss: 'bce' (per-bit sigmoid, default), "
                             "'softmax' (literal H reading: softmax-CE against the raw "
                             "multi-hot target; pristine rows zero-loss; eval scores softmax) "
                             "or 'softmax_norm' (A/B twin: target normalised to y/K, "
                             "removing the K-fold weight of multi-bit clips); "
                             "'type11' trains an 11-class forgery-type softmax head, "
                             "marginalised back to the 7 bits at eval")
    parser.add_argument("--no-binary-head", action="store_true",
                        help="do not build the real/fake detection head at all "
                             "(pure multi-label model; detection ACC/AUC "
                             "reported as placeholders)")
    parser.add_argument("--drop-binary-loss", action="store_true",
                        help="zero the binary-head loss term (multi-label-only training); "
                             "pair with --select-metric mAP since val AUC becomes noise")
    parser.add_argument("--no-pristine-train", action="store_true",
                        help="drop pristine clips from the TRAINING split (val/test unchanged)")
    parser.add_argument("--final-eval", action="store_true",
                        help="also validate the weights at the final iteration (eligible "
                             "for best-checkpoint selection) and save runs/<cell>_final.pt")
    parser.add_argument("--select-metric", choices=["ACC", "AUC", "mAP"], default=None,
                        help="validation metric for best-checkpoint/early-stop (default AUC)")
    parser.add_argument("--attention-heads", type=int, default=None,
                        help="override attention head count for all our MHA blocks "
                             "(embed_dim must be divisible by it)")
    parser.add_argument("--audio-encoder", choices=["ast", "melcnn"], default=None,
                        help="audio stream: 'ast' (paper, default when omitted) or "
                             "'melcnn' (compact from-scratch CNN on the same fbank); "
                             "recorded in the config only when given")
    parser.add_argument("--text-encoder",
                        choices=["bert", "embedbag", "minitf", "bertmini", "bertsmall",
                                 "bertmedium"],
                        default=None,
                        help="text stream: 'bert' (paper, default when omitted), "
                             "'embedbag' (compact from-scratch embedding bag on the "
                             "same WordPiece ids), 'minitf' (the bag plus a 2-layer "
                             "transformer), 'bertmini' / 'bertsmall' / 'bertmedium' "
                             "(pretrained BERT-mini / -small / -medium from "
                             "--bert-mini-model / --bert-small-model / --bert-medium-model); "
                             "recorded in the config only when given")
    parser.add_argument("--bert-mini-model", default=None,
                        help="local BERT-mini checkpoint dir for --text-encoder bertmini "
                             "(same WordPiece vocab as --bert-model)")
    parser.add_argument("--bert-small-model", default=None,
                        help="local BERT-small checkpoint dir for --text-encoder bertsmall "
                             "(same WordPiece vocab as --bert-model)")
    parser.add_argument("--bert-medium-model", default=None,
                        help="local BERT-medium checkpoint dir for --text-encoder bertmedium "
                             "(same WordPiece vocab as --bert-model)")
    parser.add_argument("--no-temporal-transformer", action="store_true",
                        help="drop the visual temporal transformer E_v entirely; "
                             "group features = plain mean of per-frame embeddings")
    parser.add_argument("--raw-contrastive", action="store_true",
                        help="run Eq. 4/5 InfoNCE on the raw encoder features "
                             "(no projection heads) — the paper's literal notation; "
                             "the alignment gradient then reshapes the encoders")
    parser.add_argument("--cross-modal-beta", type=float, default=None,
                        help="override the Eq. 4 loss weight (0 disables L_modal, "
                             "the paper's w/o L_modal ablation)")
    parser.add_argument("--mlc-loss-scale", type=float, default=None,
                        help="multiply the 7-bit MLC loss by this factor "
                             "(e.g. visible/total bit ratio: V-only = 2/7)")
    parser.add_argument("--blank-modalities", default=None,
                        help="train/eval the unchanged (VAT) graph with these "
                             "modalities' raw inputs blanked, e.g. 'at' -> "
                             "black frames kept real, zero mel, empty text")
    parser.add_argument("--mask-invisible-bits", action="store_true",
                        help="supervise the technique BCE only on bits directly "
                             "perceivable by the cell's modalities (v: lip/face, "
                             "a: tts/rvc, t: textgen, v+a: +aud_sh, a+t: +text_sh)")
    parser.add_argument("--expert-checkpoints", default=None,
                        help="for m0 (staged fusion) cells: comma-separated "
                             "'<modality>=<checkpoint>' pairs whose encoders are "
                             "loaded and frozen, e.g. 'v=runs/V.pt,a=runs/A.pt'. "
                             "Only a readout and the output heads are trained.")
    parser.add_argument("--freeze-experts", action="store_true",
                        help="load --expert-checkpoints into an m1/m2 cell and "
                             "freeze every encoder, so the cell differs from the "
                             "m0 staged cell only in its fusion mechanism. m0 "
                             "always freezes and needs no flag.")
    parser.add_argument("--poe-aux-head", action="store_true",
                        help="with --oracle-poe-visible: train a second, plain-BCE "
                             "technique head that is discarded at evaluation. The "
                             "trunk regains the full supervision the oracle's "
                             "saturated regions withhold; the measured head's "
                             "objective (and its scores) are untouched.")
    parser.add_argument("--extra-visible-bits", default=None,
                        help="comma-separated technique names to force into the "
                             "visible set, e.g. 'text_shuffle,audio_shuffle'. For "
                             "probe runs that ask whether a modality pair can see "
                             "a bit assigned elsewhere; recorded in results.json "
                             "so the test evaluation scores the same set.")
    parser.add_argument("--oracle-poe", action="store_true",
                        help="harvest-aware BCE: during training add the frozen "
                             "class-conditional log-odds table to the invisible "
                             "bits' logits (product-of-experts); evaluation uses "
                             "the raw head alone. Excludes --mask-invisible-bits")
    parser.add_argument("--oracle-poe-visible", action="store_true",
                        help="debias the SUPERVISED (visible) bits against each "
                             "other: add frozen leave-one-out log-odds "
                             "P(bit | sibling visible bits, real/fake) during "
                             "training; evaluation uses the raw head alone. "
                             "Requires --mask-invisible-bits")
    parser.add_argument("--visible-bits", default=None,
                        help="comma-separated short bit names (lip,face,tts,rvc,"
                             "a_sh,llm,tsh) that pin the observable set exactly, "
                             "overriding the built-in modality map (Phase-1 "
                             "frozen measurability map)")
    parser.add_argument("--select-bits", default=None,
                        help="comma-separated short bit names for checkpoint "
                             "selection (val mAP over exactly these bits); "
                             "overrides --select-visible-only")
    parser.add_argument("--select-visible-only", action="store_true",
                        help="checkpoint selection uses val mAP over the "
                             "observable bits only, independent of which bits "
                             "the loss trains")
    parser.add_argument("--frozen-poe-csv", default=None,
                        help="path to the frozen Phase-0 run_matrix.csv; enables "
                             "PoE with shipped offsets (no oracle recomputation)")
    parser.add_argument("--frozen-poe-cell", default=None,
                        help="cell name in run_matrix.csv (V/A/T/VA/VT/AT/VAT)")
    parser.add_argument("--frozen-poe-variant", default=None,
                        choices=["v2", "v4"],
                        help="variant column in run_matrix.csv")
    parser.add_argument("--frozen-poe-fold", default=None,
                        help="fold tag for the logodds column (F0..F3)")
    parser.add_argument("--frozen-poe-obs-bits", default=None,
                        help="comma-separated short names of the cell's "
                             "observable bits -- the pattern space of the "
                             "frozen offsets (may be narrower than "
                             "--visible-bits in v1/v2)")
    parser.add_argument("--staged-trainable", action="store_true",
                        help="m0 concat cell with encoders trained jointly "
                             "instead of frozen (fusion ablation fm_c)")
    parser.add_argument("--poe-no-fake-cond", action="store_true",
                        help="with --oracle-poe-visible: drop the real/fake flag "
                             "from the oracle's conditioning set (sensitivity "
                             "variant: P(bit | sibling bits) marginalised over "
                             "real/fake)")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the training seed (default: config seed 0)")
    parser.add_argument("--list", action="store_true",
                        help="print the expanded run list and exit (no training)")
    args = parser.parse_args()
    if args.oracle_poe and args.mask_invisible_bits:
        parser.error("--oracle-poe and --mask-invisible-bits are mutually exclusive")
    if args.oracle_poe_visible and not args.mask_invisible_bits:
        parser.error("--oracle-poe-visible requires --mask-invisible-bits")
    if args.oracle_poe_visible and args.oracle_poe:
        parser.error("--oracle-poe-visible and --oracle-poe are mutually exclusive")
    if args.poe_aux_head and not args.oracle_poe_visible:
        parser.error("--poe-aux-head only makes sense with --oracle-poe-visible")
    if args.poe_no_fake_cond and not args.oracle_poe_visible:
        parser.error("--poe-no-fake-cond only makes sense with --oracle-poe-visible")
    experts = {}
    if args.expert_checkpoints:
        for item in args.expert_checkpoints.split(","):
            if "=" not in item:
                parser.error("--expert-checkpoints wants '<modality>=<path>' pairs")
            modality, path = item.split("=", 1)
            if modality.strip() not in ("v", "a", "t"):
                parser.error(f"--expert-checkpoints: unknown modality {modality!r}")
            experts[modality.strip()] = path.strip()

    probe_bits = ()
    if args.extra_visible_bits:
        probe_bits = tuple(n.strip() for n in args.extra_visible_bits.split(",") if n.strip())
        unknown = [n for n in probe_bits if n not in MANIPULATION_TECHNIQUES]
        if unknown:
            parser.error(f"--extra-visible-bits: unknown technique(s) {unknown}")
        if not args.mask_invisible_bits:
            parser.error("--extra-visible-bits only has an effect with "
                         "--mask-invisible-bits")

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
    if args.bounded_sampler_pristine is not None:
        config["bounded_sampler_pristine"] = args.bounded_sampler_pristine
    if args.fake_type_shares is not None:
        shares = json.load(open(args.fake_type_shares))
        if sum(shares.values()) != 1000:
            parser.error("--fake-type-shares values must sum to exactly 1000")
        config["fake_type_shares"] = shares
    if args.no_binary_head:
        config["no_binary_head"] = True
    if args.drop_binary_loss:
        config["drop_binary_loss"] = True
    if args.no_pristine_train:
        config["no_pristine_train"] = True
    if args.select_metric is not None:
        config["select_metric"] = args.select_metric
    if args.final_eval:
        config["final_eval"] = True
    if args.attention_heads is not None:
        if DEFAULT_CONFIG["embed_dim"] % args.attention_heads != 0:
            parser.error("--attention-heads must divide embed_dim "
                         f"({DEFAULT_CONFIG['embed_dim']})")
        config["attention_heads"] = args.attention_heads
    if args.audio_encoder is not None:
        config["audio_encoder"] = args.audio_encoder
    if args.text_encoder is not None:
        config["text_encoder"] = args.text_encoder
    if args.bert_mini_model is not None:
        config["bert_mini_model"] = args.bert_mini_model
    if args.bert_small_model is not None:
        config["bert_small_model"] = args.bert_small_model
    if args.bert_medium_model is not None:
        config["bert_medium_model"] = args.bert_medium_model
    if args.no_temporal_transformer:
        config["no_temporal_transformer"] = True
    if args.raw_contrastive:
        config["raw_contrastive"] = True
    if args.cross_modal_beta is not None:
        config["cross_modal_beta"] = args.cross_modal_beta
    if args.mask_invisible_bits:
        config["mask_invisible_bits"] = True
    if args.seed is not None:
        config["seed"] = args.seed
    if args.oracle_poe:
        config["oracle_poe"] = True
    if args.oracle_poe_visible:
        config["oracle_poe_visible"] = True
    if args.poe_no_fake_cond:
        config["poe_no_fake_cond"] = True
    if args.visible_bits:
        names = [n.strip() for n in args.visible_bits.split(",") if n.strip()]
        unknown = [n for n in names if n not in TECHNIQUE_SHORT]
        if unknown:
            parser.error(f"--visible-bits: unknown bit names {unknown}; "
                         f"valid: {list(TECHNIQUE_SHORT)}")
        config["override_visible_bits"] = names
    if args.select_visible_only:
        config["select_visible_only"] = True
    if args.select_bits:
        names = [n.strip() for n in args.select_bits.split(",") if n.strip()]
        unknown = [n for n in names if n not in TECHNIQUE_SHORT]
        if unknown:
            parser.error(f"--select-bits: unknown bit names {unknown}")
        config["select_bits"] = names
    if args.staged_trainable:
        config["staged_trainable"] = True
    frozen_args = (args.frozen_poe_csv, args.frozen_poe_cell,
                   args.frozen_poe_variant, args.frozen_poe_fold,
                   args.frozen_poe_obs_bits)
    if any(frozen_args):
        if not all(frozen_args):
            parser.error("--frozen-poe-csv/-cell/-variant/-fold/-obs-bits must "
                         "be given together")
        obs_names = [n.strip() for n in args.frozen_poe_obs_bits.split(",")
                     if n.strip()]
        unknown = [n for n in obs_names if n not in TECHNIQUE_SHORT]
        if unknown:
            parser.error(f"--frozen-poe-obs-bits: unknown bit names {unknown}")
        config["frozen_poe_csv"] = args.frozen_poe_csv
        config["frozen_poe_cell"] = args.frozen_poe_cell
        config["frozen_poe_variant"] = args.frozen_poe_variant
        config["frozen_poe_fold"] = args.frozen_poe_fold
        config["frozen_poe_obs_bits"] = obs_names
    if args.poe_aux_head:
        config["aux_plain_head"] = True
    if probe_bits:
        config["extra_visible_bits"] = list(probe_bits)
    if experts:
        config["expert_checkpoints"] = experts
    if args.freeze_experts:
        if not experts:
            parser.error("--freeze-experts needs --expert-checkpoints")
        config["freeze_experts"] = True
    if args.blank_modalities is not None:
        blank = set(args.blank_modalities.lower())
        if not blank or not blank.issubset({"v", "a", "t"}):
            parser.error("--blank-modalities must be a non-empty subset of 'vat'")
        config["blank_modalities"] = "".join(sorted(blank))
    if args.mlc_loss_scale is not None:
        config["mlc_loss_scale"] = args.mlc_loss_scale
    if args.mlc_loss is not None:
        config["mlc_loss"] = args.mlc_loss
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
