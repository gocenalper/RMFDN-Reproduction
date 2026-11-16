"""Cache frozen per-modality features of a trained (VAT) model.

One GPU pass over (a subsample of) train + the full test split stores each
clip's (f_v, f_a, f_t) from ``model.encode_modalities`` plus labels. All
subset probes (single AND pair heads) and score-ensembles are then trained
offline from the cache in seconds -- the encoders are frozen, so features
are constant.

Shard-resumable: features are written in shards of ``--shard`` clips;
existing shards are skipped on requeue.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import BertTokenizer

from rmfdn_ablation.config import MODALITY_CONFIGS
from rmfdn_ablation.dataset import MultimodalClipDataset, build_collate_fn
from rmfdn_ablation.protocols import build_protocol_splits
from rmfdn_ablation.trainer import _move_training_batch

_MODALITIES_BY_NAME = dict(MODALITY_CONFIGS)


def _model_class(cross_modal):
    from rmfdn_ablation.experiment import model_class_for
    return model_class_for(cross_modal)


def extract_split(model, clips, config, tokenizer, device, out_dir, tag,
                  batch_size, num_workers, shard):
    os.makedirs(out_dir, exist_ok=True)
    for start in range(0, len(clips), shard):
        shard_path = os.path.join(out_dir, f"{tag}_{start:07d}.npz")
        if os.path.exists(shard_path):
            continue
        part = clips[start:start + shard]
        dataset = MultimodalClipDataset(part, model.modalities, config)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,
                            collate_fn=build_collate_fn(tokenizer, config),
                            pin_memory=True)
        fv, fa, ft_, bits = [], [], [], []
        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                targets = _move_training_batch(batch, device)
                feats = model.encode_modalities(batch)
                fv.append(feats["v"].half().cpu().numpy())
                fa.append(feats["a"].half().cpu().numpy())
                ft_.append(feats["t"].half().cpu().numpy())
                bits.append(targets.cpu().numpy())
        np.savez_compressed(
            shard_path,
            f_v=np.concatenate(fv), f_a=np.concatenate(fa),
            f_t=np.concatenate(ft_), bits=np.concatenate(bits),
            is_fake=np.array([c.is_fake for c in part]),
            ftype=np.array([c.forgery_type for c in part]),
            directory=np.array([c.directory for c in part]),
        )
        print(f"[extract] {tag} shard {start} ({len(part)} clips, "
              f"{(time.time()-t0)/60:.1f} min)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vat-dir", required=True,
                        help="phase-1 dir of the trained VAT run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--bert-model", default=None)
    parser.add_argument("--ast-checkpoint", default=None)
    parser.add_argument("--train-cap", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--shard", type=int, default=8000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vat = Path(args.vat_dir)
    results = json.load(open(vat / "results.json"))
    name, run = next((k, v) for k, v in results.items()
                     if v.get("status") != "not_implemented")
    spec, config = run["spec"], dict(run["config"])
    for key, val in (("bert_model", args.bert_model),
                     ("ast_checkpoint", args.ast_checkpoint)):
        if val is not None:
            config[key] = val
    config["reference_alpha"] = 0.0

    modalities = _MODALITIES_BY_NAME[spec["modality"]]
    model = _model_class(spec["cross_modal"])(config, modalities).to(device)
    state = torch.load(vat / "runs" / f"{name}.pt", map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print(f"[extract] VAT model yuklendi: {name} ({spec['modality']})", flush=True)

    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
    splits = build_protocol_splits(spec["protocol"], args.data_root, None)

    train_clips = list(splits["train"])
    random.Random(0).shuffle(train_clips)
    train_clips = train_clips[: args.train_cap]
    print(f"[extract] train alt-orneklem: {len(train_clips)} | "
          f"test: {len(splits['test'])}", flush=True)

    extract_split(model, train_clips, config, tokenizer, device, args.out,
                  "train", args.batch_size, args.num_workers, args.shard)
    extract_split(model, list(splits["test"]), config, tokenizer, device,
                  args.out, "test", args.batch_size, args.num_workers, args.shard)
    print("[extract] TAMAM", flush=True)


if __name__ == "__main__":
    main()
