"""Evaluation: binary detection plus multi-label technique attribution.

Two heads, two metric families:

* **Binary** (real/fake, from ``binary_logits``):
  - ACC -- accuracy at the P(fake) > 0.5 threshold,
  - AUC -- area under the ROC curve (threshold-free ranking quality).

* **Multi-label** (per-technique, from ``technique_logits``; every technique
  is an independent binary class -- there is *no* argmax over named forgery
  types):
  - mAP -- macro average precision: per-technique area under the
    precision-recall curve of the raw sigmoid scores, averaged with equal
    weight per technique (threshold-free),
  - CF1 -- per-Class F1. The default implementation averages per-technique
    F1, while the reference APMeter compatibility mode first averages
    per-technique precision/recall and then takes their harmonic mean,
  - OF1 -- Overall F1: pool every (clip, technique) decision into one
    TP/FP/FN count (micro), which weights techniques by frequency.

Techniques that are single-valued in the evaluated split (all-0 or all-1
ground truth) are excluded -- AP and F1 are undefined for them; ``nkeep``
reports how many techniques were actually scored.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from .config import MANIPULATION_TECHNIQUES, TYPE_TEMPLATES, default_device
from .dataset import MultimodalClipDataset, build_collate_fn
from .reference import encode_reference_batch


def _harmonic_mean(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 0.0 if denominator == 0 else 2.0 * precision * recall / denominator


def _reference_apmeter_metrics(labels: np.ndarray, logits: np.ndarray,
                               metric_indices: Sequence[int]) -> Dict[str, float]:
    """Reproduce MultiModal-DeepFake's AveragePrecisionMeter formulas."""
    selected_labels = labels[:, metric_indices].astype(np.int64, copy=False)
    selected_logits = logits[:, metric_indices].astype(np.float32, copy=False)
    positives = selected_labels.sum(axis=0)
    sample_count = selected_labels.shape[0]
    invalid = [metric_indices[j] for j, count in enumerate(positives)
               if count == 0 or count == sample_count]
    if invalid:
        names = [MANIPULATION_TECHNIQUES[j] for j in invalid]
        raise ValueError(
            "reference APMeter requires positive and negative examples for every "
            f"evaluated bit; constant bits: {names}")

    per_bit_ap = []
    for bit in range(selected_logits.shape[1]):
        scores = torch.from_numpy(selected_logits[:, bit])
        targets = torch.from_numpy(selected_labels[:, bit])
        order = torch.sort(scores, descending=True).indices
        ranked_targets = targets[order]
        cumulative_positives = ranked_targets.cumsum(0, dtype=torch.float64)
        ranks = torch.arange(1, len(ranked_targets) + 1, dtype=torch.float64)
        precision_at_rank = cumulative_positives / ranks
        per_bit_ap.append(float(precision_at_rank[ranked_targets == 1].mean()))

    predicted = selected_logits >= 0
    nc = (selected_labels * predicted).sum(axis=0).astype(np.float64)
    npred = predicted.sum(axis=0).astype(np.float64)
    ngt = positives.astype(np.float64)
    npred[npred == 0] = 1

    overall_precision = float(nc.sum() / npred.sum())
    overall_recall = float(nc.sum() / ngt.sum())
    class_precision = float(np.mean(nc / npred))
    class_recall = float(np.mean(nc / ngt))
    return {
        "mAP": float(np.mean(per_bit_ap) * 100),
        "OP": overall_precision * 100,
        "OR": overall_recall * 100,
        "OF1": _harmonic_mean(overall_precision, overall_recall) * 100,
        "CP": class_precision * 100,
        "CR": class_recall * 100,
        "CF1": _harmonic_mean(class_precision, class_recall) * 100,
        "per_bit_ap": {
            MANIPULATION_TECHNIQUES[index]: ap * 100
            for index, ap in zip(metric_indices, per_bit_ap)
        },
    }


@torch.no_grad()
def evaluate(model, dataset: MultimodalClipDataset, tokenizer, config: dict,
             device: Optional[str] = None,
             technique_indices: Optional[Sequence[int]] = None,
             metric_implementation: str = "sklearn",
             collect_scores: bool = False) -> Dict[str, float]:
    """Run the model and score all technique bits or an explicit subset."""
    device = device or default_device()
    metric_indices = (list(range(len(MANIPULATION_TECHNIQUES)))
                      if technique_indices is None else list(technique_indices))
    if not metric_indices:
        raise ValueError("technique_indices must contain at least one bit")
    if len(set(metric_indices)) != len(metric_indices) or any(
            index < 0 or index >= len(MANIPULATION_TECHNIQUES)
            for index in metric_indices):
        raise ValueError("technique_indices must be unique valid bit indices")
    if metric_implementation not in {"sklearn", "reference_apmeter"}:
        raise ValueError(f"unknown metric implementation: {metric_implementation}")
    if (metric_implementation == "reference_apmeter"
            and config.get("mlc_loss", "bce") != "bce"):
        raise ValueError("reference_apmeter evaluation is defined for BCE logits")
    model.eval()
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"],
                        collate_fn=build_collate_fn(tokenizer, config))

    fake_scores, fake_labels = [], []
    technique_scores, technique_labels = [], []
    for batch in loader:
        keys = ["frames", "mel", "input_ids", "attention_mask"]
        if "reference_frames" in batch:
            keys.extend(["reference_frames", "reference_mel", "reference_input_ids",
                         "reference_attention_mask"])
        for key in keys:
            batch[key] = batch[key].to(device)
        with (torch.amp.autocast("cuda", enabled=str(device).startswith("cuda"))
              if hasattr(torch.amp, "autocast")
              else torch.cuda.amp.autocast(enabled=str(device).startswith("cuda"))):
            reference_outputs = (encode_reference_batch(model, batch)
                                 if "reference_frames" in batch else None)
            reference_features = (reference_outputs["fused"]
                                  if reference_outputs is not None else None)
            outputs = model(batch, reference_fused=reference_features)

        fake_scores.append(
            torch.softmax(outputs["binary_logits"].float(), 1)[:, 1].cpu().numpy()
            if "binary_logits" in outputs
            else np.zeros(len(batch["is_fake"]), dtype=np.float32))
        fake_labels.append(batch["is_fake"].numpy())
        mlc_mode = config.get("mlc_loss", "bce")
        if mlc_mode == "type11":  # 11-class type-softmax -> marginalise to 7-bit
            _tpl = torch.tensor(TYPE_TEMPLATES, dtype=torch.float32,
                                device=outputs["technique_logits"].device)
            _type_probs = torch.softmax(outputs["technique_logits"].float(), 1)
            technique_scores.append((_type_probs @ _tpl).cpu().numpy())
        elif mlc_mode.startswith("softmax"):  # softmax-trained 7-bit head
            technique_scores.append(
                torch.softmax(outputs["technique_logits"].float(), 1).cpu().numpy())
        elif metric_implementation == "reference_apmeter":
            technique_scores.append(
                outputs["technique_logits"].float().cpu().numpy())
        else:
            technique_scores.append(
                torch.sigmoid(outputs["technique_logits"].float()).cpu().numpy())
        technique_labels.append(batch["technique_targets"].numpy())
    model.train()

    fake_scores = np.concatenate(fake_scores)
    fake_labels = np.concatenate(fake_labels)
    technique_scores = np.concatenate(technique_scores)
    technique_labels = np.concatenate(technique_labels)

    # ----- binary detection ------------------------------------------------ #
    accuracy = ((fake_scores > 0.5).astype(int) == fake_labels).mean() * 100
    try:
        auc = roc_auc_score(fake_labels, fake_scores) * 100
    except ValueError:  # split contains a single class
        auc = float("nan")

    # ----- multi-label attribution ----------------------------------------- #
    threshold = 0.0 if metric_implementation == "reference_apmeter" else 0.5
    predictions = (technique_scores >= threshold).astype(int)

    # Per-bit accuracy: each active technique scored independently at the 0.5
    # threshold (fraction of clips whose predicted presence matches the truth).
    # Defined for every active bit, including single-valued ones.
    per_bit_acc = {
        name: float((predictions[:, j] == technique_labels[:, j]).mean() * 100)
        for j, name in enumerate(MANIPULATION_TECHNIQUES) if j in metric_indices
    }

    if metric_implementation == "reference_apmeter":
        attribution = _reference_apmeter_metrics(
            technique_labels, technique_scores, metric_indices)
        scoreable = metric_indices
        mean_ap = attribution["mAP"]
        per_class_f1 = attribution["CF1"]
        overall_f1 = attribution["OF1"]
    else:
        # mAP / F1 are only defined for bits that vary in this split.
        scoreable = [j for j in metric_indices
                     if 0 < technique_labels[:, j].sum() < len(technique_labels)]
    if metric_implementation == "sklearn" and scoreable:
        mean_ap = average_precision_score(
            technique_labels[:, scoreable], technique_scores[:, scoreable],
            average="macro") * 100
        per_class_f1 = f1_score(technique_labels[:, scoreable], predictions[:, scoreable],
                                average="macro", zero_division=0) * 100
        overall_f1 = f1_score(technique_labels[:, scoreable], predictions[:, scoreable],
                              average="micro", zero_division=0) * 100
    elif metric_implementation == "sklearn":
        mean_ap = per_class_f1 = overall_f1 = float("nan")

    metrics = {"ACC": accuracy, "AUC": auc, "mAP": mean_ap,
               "CF1": per_class_f1, "OF1": overall_f1,
               "nkeep": len(scoreable),
               "metric_implementation": metric_implementation,
               "metric_bits": [MANIPULATION_TECHNIQUES[j] for j in metric_indices],
               "per_bit_acc": per_bit_acc}
    if metric_implementation == "reference_apmeter":
        metrics.update({key: value for key, value in attribution.items()
                        if key not in {"mAP", "CF1", "OF1"}})
    if collect_scores:
        # Per-clip arrays for offline analyses (ensemble/late-fusion studies).
        # technique_scores are raw logits under reference_apmeter, otherwise
        # the mode's native scores (sigmoid / softmax / type-marginalised).
        metrics["_arrays"] = {
            "technique_scores": technique_scores.astype(np.float32),
            "technique_labels": technique_labels.astype(np.uint8),
            "fake_scores": fake_scores.astype(np.float32),
            "fake_labels": fake_labels.astype(np.uint8),
        }
    return metrics
