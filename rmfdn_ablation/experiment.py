"""Experiment matrix: run a declared list of setups in sequence.

A run is one cell of the matrix, defined by

* **protocol**    -- ``p1`` (official IDForge split) or ``p2`` (identity-disjoint),
* **identity**    -- ``i1`` (identity reference on) or ``i3`` (reference off),
* **cross_modal** -- always ``m1``, the R-MFDN progressive cross-attention fusion,
* **modality**    -- ``VA`` or ``VAT``,
* **variant**     -- arm of the improvement experiment (``base``, ``dropout``,
  ``gate``, ``gate_dropout``; see ``config.VARIANTS``), default ``base``.

and named ``<cross_modal>_<identity>_<protocol>_<modality>[_<variant>]``, for
example ``m1_i1_p1_VAT`` or ``m1_i3_p1_VA_gate_dropout``. The matrix is declared as data (see
``experiments/experiment_matrix.json``), so a new setup needs no code change.
:func:`run_experiment_sequence` executes the declared specs in order and can
be relaunched: finished runs are skipped.
"""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .config import MODALITY_CONFIGS, VARIANTS, default_device
from .protocols import build_protocol_splits
from .trainer import run_single_experiment

#: "VAT" -> ["v", "a", "t"]
_MODALITIES_BY_NAME = dict(MODALITY_CONFIGS)

#: identity axis -> whether the identity-reference branch is active.
_IDENTITY_USES_REFERENCE = {"i1": True, "i3": False}


@dataclass(frozen=True)
class ExperimentSpec:
    """One runnable cell of the matrix."""

    cross_modal: str   # 'm1'
    identity: str      # 'i1' | 'i3'
    protocol: str      # 'p1' | 'p2'
    modality: str      # 'VA' | 'VAT'
    variant: str = "base"

    @property
    def name(self) -> str:
        name = f"{self.cross_modal}_{self.identity}_{self.protocol}_{self.modality}"
        return name if self.variant == "base" else f"{name}_{self.variant}"


def expand_cells(cells: List[dict]) -> List[ExperimentSpec]:
    """Expand declarative cells (each listing several modalities) into specs."""
    specs: List[ExperimentSpec] = []
    for cell in cells:
        for variant in cell.get("variants", ["base"]):
            if variant not in VARIANTS:
                raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
            for modality in cell["modalities"]:
                specs.append(ExperimentSpec(
                    cross_modal=cell["cross_modal"], identity=cell["identity"],
                    protocol=cell["protocol"], modality=modality, variant=variant))
    return specs


def resolve_reference_flag(spec: ExperimentSpec) -> bool:
    """Map a spec's identity axis to the trainer's ``use_reference`` flag."""
    if spec.cross_modal != "m1":
        raise ValueError(f"unknown cross_modal: {spec.cross_modal!r}")
    if spec.identity not in _IDENTITY_USES_REFERENCE:
        raise ValueError(f"unknown identity: {spec.identity!r}")
    return _IDENTITY_USES_REFERENCE[spec.identity]


def run_experiment_sequence(specs: List[ExperimentSpec], config: dict,
                            data_root: str, reference_root: str, output_dir: str,
                            p2_split_path: Optional[str] = None,
                            device: Optional[str] = None) -> dict:
    """Run a list of specs in order, writing results.json after each.

    Protocol splits are scanned once and cached (scanning is expensive), so
    grouping specs by protocol in the config keeps the suite fast. Completed
    runs (present in results.json) are skipped on relaunch.
    """
    from transformers import BertTokenizer

    from .data_catalog import scan_reference_clips

    device = device or default_device()
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
    # The reference bank is only consulted by i1 cells; scanning it is an
    # expensive directory walk, so skip it outright when no spec needs it.
    reference_index = (scan_reference_clips(reference_root)
                       if any(resolve_reference_flag(spec) for spec in specs)
                       else None)

    results_path = os.path.join(output_dir, "results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    if results:
        print(f"resume: {len(results)} run(s) already recorded -> skipped", flush=True)

    split_cache: Dict[str, Dict[str, list]] = {}

    def splits_for(protocol: str):
        if protocol not in split_cache:
            print(f"[protocol {protocol}] scanning clips...", flush=True)
            split_cache[protocol] = build_protocol_splits(protocol, data_root, p2_split_path)
            sizes = {s: len(c) for s, c in split_cache[protocol].items()}
            print(f"[protocol {protocol}] {sizes}", flush=True)
        return split_cache[protocol]

    val_cap = config.get("val_subsample_cap")

    for spec in specs:
        if spec.name in results:
            print(f"[{spec.name}] already recorded -> skipped", flush=True)
            continue

        use_reference = resolve_reference_flag(spec)
        modalities = _MODALITIES_BY_NAME[spec.modality]
        splits = splits_for(spec.protocol)
        train_clips = splits["train"]
        val_clips = splits["val"]
        if val_cap and len(val_clips) > val_cap:                 # fixed seed-0 subsample
            import random
            val_clips = random.Random(0).sample(val_clips, val_cap)

        print(f"\n{'=' * 70}\nRUN {spec.name}\n{'=' * 70}", flush=True)
        run_config = {**config, **VARIANTS[spec.variant]}
        if run_config.get("fusion_gate") and len(modalities) != 2:
            print(f"[{spec.name}] FAILED: the fusion gate is defined for VA only", flush=True)
            continue
        try:
            run_result = run_single_experiment(
                spec.name, modalities, use_reference,
                train_clips, val_clips, splits["test"],
                reference_index, tokenizer, run_config, output_dir, device,
                skip_test_eval=config.get("skip_test_eval", False))
            run_result["spec"] = asdict(spec)
            results[spec.name] = run_result
        except Exception as error:
            traceback.print_exc()
            print(f"[{spec.name}] FAILED: {error}", flush=True)
        json.dump(results, open(results_path, "w"), indent=1)

    print("\nSUITE_DONE", flush=True)
    return results
