"""Two-stage hypothesis probe: train ONLY the 7-bit head on a frozen encoder.

Stage-1 model (typically trained binary-only via ``--mlc-loss-scale 0``)
is loaded exactly the way ``run_test_eval`` rebuilds it; every parameter
is frozen except a freshly re-initialised ``technique_head``, which is
then trained with plain BCE for a few thousand iterations. The updated
checkpoint + results.json are written to ``--out`` in phase-1 layout so
the standard test evaluation runs on it unchanged.

This simulates the "detector first, attribution head bolted on after"
training scheme -- the leading candidate for the paper's single-modality
operating point (AUC intact, mAP ~56, CF1 ~46).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import BertTokenizer

from rmfdn_ablation.config import MODALITY_CONFIGS
from rmfdn_ablation.data_catalog import scan_reference_clips
from rmfdn_ablation.dataset import MultimodalClipDataset, build_collate_fn
from rmfdn_ablation.protocols import build_protocol_splits
from rmfdn_ablation.reference import encode_reference_batch
from rmfdn_ablation.trainer import _move_training_batch

_MODALITIES_BY_NAME = dict(MODALITY_CONFIGS)


def _model_class(cross_modal):
    from rmfdn_ablation.experiment import model_class_for
    return model_class_for(cross_modal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--bert-model", default=None)
    parser.add_argument("--ast-checkpoint", default=None)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--source-ckpt", default=None,
                        help="copy shared-prefix weights (e.g. encoders.v.*) from "
                             "this checkpoint instead of the phase-1 run's own; "
                             "simulates probing a jointly-trained model's encoder")
    parser.add_argument("--train-binary-head", action="store_true",
                        help="also re-init and train the binary head (needed when "
                             "the encoder comes from a differently-trained model)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    phase1 = Path(args.phase1_dir)
    results = json.load(open(phase1 / "results.json"))
    name, run = next((k, v) for k, v in results.items()
                     if v.get("status") != "not_implemented")
    spec, config = run["spec"], dict(run["config"])
    for key, val in (("bert_model", args.bert_model),
                     ("ast_checkpoint", args.ast_checkpoint)):
        if val is not None:
            config[key] = val

    modalities = _MODALITIES_BY_NAME[spec["modality"]]
    use_reference = spec["identity"] == "i1"
    if not use_reference:
        config["reference_alpha"] = 0.0

    model = _model_class(spec["cross_modal"])(config, modalities).to(device)
    if args.source_ckpt:
        src = torch.load(args.source_ckpt, map_location="cpu")
        src = src["model"] if "model" in src else src
        own = model.state_dict()
        copied = {k: v for k, v in src.items()
                  if k in own and own[k].shape == v.shape}
        own.update(copied)
        model.load_state_dict(own)
        enc = [k for k in copied if k.startswith("encoders.")]
        print(f"[probe] source ckpt: {len(copied)} tensors copied "
              f"({len(enc)} encoder tensors) from {args.source_ckpt}", flush=True)
    else:
        state = torch.load(phase1 / "runs" / f"{name}.pt", map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"[probe] stage-1 checkpoint loaded: {name}", flush=True)

    for p in model.parameters():
        p.requires_grad = False
    model.technique_head.reset_parameters()
    for p in model.technique_head.parameters():
        p.requires_grad = True
    trainable = list(model.technique_head.parameters())
    if args.train_binary_head:
        model.binary_head.reset_parameters()
        for p in model.binary_head.parameters():
            p.requires_grad = True
        trainable += list(model.binary_head.parameters())
    model.eval()  # frozen encoder in eval mode (BN stats untouched)
    model.technique_head.train()
    if args.train_binary_head:
        model.binary_head.train()

    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
    reference_index = scan_reference_clips(args.reference_root) if use_reference else None
    splits = build_protocol_splits(spec["protocol"], args.data_root, None)
    dataset = MultimodalClipDataset(splits["train"], modalities, config,
                                    reference_index=reference_index,
                                    reference_seed=config["seed"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        collate_fn=build_collate_fn(tokenizer, config),
                        pin_memory=True)

    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    iteration, t0 = 0, time.time()
    while iteration < args.iters:
        for batch in loader:
            if iteration >= args.iters:
                break
            technique_targets = _move_training_batch(batch, device)
            reference_outputs = (encode_reference_batch(model, batch)
                                 if use_reference else None)
            reference_features = (reference_outputs["fused"]
                                  if reference_outputs is not None else None)
            outputs = model(batch, reference_fused=reference_features)
            loss = F.binary_cross_entropy_with_logits(
                outputs["technique_logits"], technique_targets)
            if args.train_binary_head:
                loss = loss + F.cross_entropy(outputs["binary_logits"],
                                              batch["is_fake"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            iteration += 1
            if iteration % 100 == 0:
                print(f"[probe] iter {iteration}/{args.iters} "
                      f"bce {loss.item():.3f} ({(time.time()-t0)/60:.0f} min)",
                      flush=True)

    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "runs" / f"{name}.pt")
    trimmed = {name: {**run, "test": None}}
    json.dump(trimmed, open(out / "results.json", "w"), indent=1)
    print(f"[probe] head trained ({args.iters} iters) -> {out}", flush=True)


if __name__ == "__main__":
    main()
