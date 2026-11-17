#!/usr/bin/env python3
"""Measure parameter count and inference latency of each improvement arm.

Builds the VA model for every arm in ``config.VARIANTS`` (gate on or off; the
dropout arms have the same network as their counterparts), counts parameters
and times the forward pass on random inputs of the real shape: batch size 16,
16 frames at 224 x 224 and a 1024 x 128 AST input. Latency is reported in
milliseconds per clip as the median over ``--repeats`` timed batches after
``--warmup`` untimed ones. Peak training memory and training time are recorded
by the trainer itself (``resources`` and ``minutes`` in ``results.json``).

    python scripts/measure_resources.py --output results/resources.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rmfdn_ablation.config import DEFAULT_CONFIG, VARIANTS, default_device
from rmfdn_ablation.RMFDN import RMFDN


def measure(config: dict, batch_size: int, repeats: int, warmup: int, device: str) -> dict:
    model = RMFDN(config, ["v", "a"]).to(device).eval()
    parameters = sum(p.numel() for p in model.parameters())
    gate_parameters = (sum(p.numel() for p in model.fusion_gate.parameters())
                       if getattr(model, "use_gate", False) else 0)
    batch = {
        "frames": torch.randint(0, 256, (batch_size, config["frames_per_clip"], 3,
                                         config["image_size"], config["image_size"]),
                                dtype=torch.uint8, device=device),
        "mel": torch.randn(batch_size, 1024, 128, device=device),
        "input_ids": torch.zeros(batch_size, 1, dtype=torch.long, device=device),
        "attention_mask": torch.ones(batch_size, 1, dtype=torch.long, device=device),
    }
    cuda = str(device).startswith("cuda")
    timings = []
    with torch.no_grad(), torch.autocast("cuda", enabled=cuda):
        for step in range(warmup + repeats):
            if cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(batch)
            if cuda:
                torch.cuda.synchronize()
            if step >= warmup:
                timings.append((time.perf_counter() - start) * 1000 / batch_size)
    return {"parameters": parameters, "gate_parameters": gate_parameters,
            "latency_ms_per_clip": round(statistics.median(timings), 3),
            "batch_size": batch_size, "device": str(device),
            "device_name": torch.cuda.get_device_name() if cuda else "cpu"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--ast-checkpoint", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = default_device()
    out = {}
    for variant, overrides in VARIANTS.items():
        config = {**DEFAULT_CONFIG, **overrides}
        if args.ast_checkpoint:
            config["ast_checkpoint"] = args.ast_checkpoint
        out[variant] = measure(config, args.batch_size, args.repeats, args.warmup, device)
        print(f"{variant:13s} params={out[variant]['parameters'] / 1e6:.2f}M "
              f"(gate {out[variant]['gate_parameters'] / 1e6:.2f}M) "
              f"latency={out[variant]['latency_ms_per_clip']:.2f} ms/clip", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)


if __name__ == "__main__":
    main()
