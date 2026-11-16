#!/usr/bin/env python3
"""Where does the visual pathway look, and does debiasing move it?

The visual encoder is a ResNet-18 followed by a transformer over four frame
tokens. There is no spatial attention to read out, so "where in the frame"
has to be answered with Grad-CAM on the last convolutional block; the only
genuine attention in this pathway is temporal and spans four tokens.

The interesting comparison is not the pictures but their mass. An earlier
control found that feeding full frames beats feeding face crops by 4.5 mAP,
which means the model is using something outside the face. If the debiasing
objective removes label-structure shortcuts, the share of Grad-CAM mass
falling inside the face should rise. That is a falsifiable prediction and
this script measures it: for each model, bit and package group it reports
the fraction of CAM mass inside the detected face box and inside the mouth
region, and writes side-by-side overlays for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmfdn_ablation.config import MANIPULATION_TECHNIQUES, DEFAULT_CONFIG, default_device
from rmfdn_ablation.dataset import MultimodalClipDataset
from rmfdn_ablation.protocols import build_protocol_splits
from rmfdn_ablation.RMFDN import RMFDN


def _package_group(clip) -> str:
    """FACE / LIP / NONE -- the visually distinguishable equivalence classes."""
    if not clip.is_fake:
        return "pristine"
    targets = clip.technique_targets
    if targets[1]:
        return "FACE"
    if targets[0]:
        return "LIP"
    return "NONE"


class GradCAM:
    """Grad-CAM on the ResNet's last block, one map per frame."""

    def __init__(self, model):
        self.activations = None
        self.gradients = None
        layer = model.encoders["v"].backbone.layer4
        layer.register_forward_hook(self._save_activations)
        layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output):
        self.activations = output

    def _save_gradients(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def maps(self) -> torch.Tensor:
        """(N, H, W) non-negative maps, each normalised to sum 1."""
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))
        cam = F.interpolate(cam.unsqueeze(1), size=(224, 224),
                            mode="bilinear", align_corners=False).squeeze(1)
        flat = cam.flatten(1)
        total = flat.sum(1, keepdim=True).clamp_min(1e-8)
        return (flat / total).view_as(cam)


def detect_faces(frames_uint8, detector):
    """MTCNN boxes and mouth regions for a (T, 3, 224, 224) uint8 tensor."""
    images = frames_uint8.permute(0, 2, 3, 1).cpu().numpy()
    boxes, landmarks = detector.detect(list(images), landmarks=True)[0::2]
    out = []
    for box, marks in zip(boxes, landmarks):
        if box is None or len(box) == 0:
            out.append(None)
            continue
        face = box[0]
        # MTCNN landmark order: left eye, right eye, nose, mouth left, mouth right
        mouth = marks[0][3:5] if marks is not None and len(marks) else None
        out.append((face, mouth))
    return out


def mass_inside(cam, box, pad=0.0):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, y0 = max(0, int(x0 - pad * w)), max(0, int(y0 - pad * h))
    x1, y1 = min(224, int(x1 + pad * w)), min(224, int(y1 + pad * h))
    if x1 <= x0 or y1 <= y0:
        return None
    return float(cam[y0:y1, x0:x1].sum())


def mouth_box(mouth_marks, face_box):
    """A square around the mouth corners, sized relative to the face."""
    (lx, ly), (rx, ry) = mouth_marks
    cx, cy = (lx + rx) / 2, (ly + ry) / 2
    half = max(abs(rx - lx), 20.0) * 0.9
    return (cx - half, cy - half * 0.7, cx + half, cy + half * 0.7)


def overlay(frame_uint8, cam, path):
    import cv2
    image = frame_uint8.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()
    heat = (cam / cam.max().clamp_min(1e-8) * 255).byte().cpu().numpy()
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    cv2.imwrite(path, cv2.addWeighted(image, 0.55, heat, 0.45, 0))


def load_model(checkpoint_path, config, device):
    model = RMFDN(config, ["v"]).to(device)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain-checkpoint", required=True)
    parser.add_argument("--debiased-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--p2-split", required=True)
    parser.add_argument("--bert-model", default=None)
    parser.add_argument("--ast-checkpoint", default=None)
    parser.add_argument("--clips-per-group", type=int, default=60)
    parser.add_argument("--overlays", type=int, default=6,
                        help="clips per group to write side-by-side overlays for")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = default_device()
    config = dict(DEFAULT_CONFIG)
    if args.bert_model:
        config["bert_model"] = args.bert_model
    if args.ast_checkpoint:
        config["ast_checkpoint"] = args.ast_checkpoint

    splits = build_protocol_splits("p2", args.data_root, args.p2_split)
    by_group = defaultdict(list)
    for clip in splits["test"]:
        by_group[_package_group(clip)].append(clip)
    selected = []
    for group in ("FACE", "LIP", "NONE", "pristine"):
        selected += by_group[group][:args.clips_per_group]
    print(f"[cam] {len(selected)} clips: "
          f"{ {g: min(len(by_group[g]), args.clips_per_group) for g in by_group} }",
          flush=True)

    dataset = MultimodalClipDataset(selected, ["v"], config)
    from facenet_pytorch import MTCNN
    detector = MTCNN(keep_all=False, device=device)

    models = {"plain": load_model(args.plain_checkpoint, config, device),
              "debiased": load_model(args.debiased_checkpoint, config, device)}
    cams = {name: GradCAM(model) for name, model in models.items()}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    image_dir = os.path.splitext(args.output)[0] + "_overlays"
    os.makedirs(image_dir, exist_ok=True)

    stats = defaultdict(lambda: defaultdict(list))
    written = defaultdict(int)
    for index, clip in enumerate(selected):
        sample = dataset[index]
        frames = sample["frames"].unsqueeze(0).to(device)
        group = _package_group(clip)
        regions = detect_faces(sample["frames"], detector)

        for bit_name in ("lip_sync", "face_swap"):
            bit = MANIPULATION_TECHNIQUES.index(bit_name)
            per_model_maps = {}
            for name, model in models.items():
                model.zero_grad(set_to_none=True)
                outputs = model({"frames": frames})
                outputs["technique_logits"][0, bit].backward()
                per_model_maps[name] = cams[name].maps().detach()

            for name, maps in per_model_maps.items():
                for frame_index, region in enumerate(regions):
                    if region is None:
                        continue
                    face, mouth = region
                    cam = maps[frame_index]
                    face_mass = mass_inside(cam, face)
                    if face_mass is None:
                        continue
                    stats[(name, bit_name, group)]["face"].append(face_mass)
                    if mouth is not None:
                        mouth_mass = mass_inside(cam, mouth_box(mouth, face))
                        if mouth_mass is not None:
                            stats[(name, bit_name, group)]["mouth"].append(mouth_mass)

            if written[(group, bit_name)] < args.overlays:
                written[(group, bit_name)] += 1
                tag = f"{group}_{bit_name}_{written[(group, bit_name)]}"
                for name, maps in per_model_maps.items():
                    overlay(sample["frames"][0], maps[0],
                            os.path.join(image_dir, f"{tag}_{name}.png"))
        if (index + 1) % 25 == 0:
            print(f"[cam] {index + 1}/{len(selected)}", flush=True)

    summary = {}
    for (name, bit_name, group), values in stats.items():
        summary[f"{name}|{bit_name}|{group}"] = {
            "face_mass_mean": float(np.mean(values["face"])),
            "face_mass_std": float(np.std(values["face"])),
            "mouth_mass_mean": (float(np.mean(values["mouth"]))
                                if values["mouth"] else None),
            "frames": len(values["face"]),
        }
    json.dump(summary, open(args.output, "w"), indent=1, sort_keys=True)

    print("\nGrad-CAM mass inside the detected face box (share of total map mass)")
    print(f"{'bit':11s} {'group':9s} {'plain':>8s} {'debiased':>9s} {'delta':>8s}")
    for bit_name in ("lip_sync", "face_swap"):
        for group in ("FACE", "LIP", "NONE", "pristine"):
            a = summary.get(f"plain|{bit_name}|{group}")
            b = summary.get(f"debiased|{bit_name}|{group}")
            if a and b:
                print(f"{bit_name:11s} {group:9s} {a['face_mass_mean']:8.4f} "
                      f"{b['face_mass_mean']:9.4f} "
                      f"{b['face_mass_mean'] - a['face_mass_mean']:+8.4f}")
    print(f"\noverlays -> {image_dir}\nsummary  -> {args.output}")


if __name__ == "__main__":
    main()
