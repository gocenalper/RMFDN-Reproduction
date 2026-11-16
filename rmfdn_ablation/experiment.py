"""Configurable experiment suite: run a declared matrix of setups in sequence.

The advisor's study spans three independent axes:

* **protocol**   -- ``p1`` (original split) or ``p2`` (identity-disjoint),
* **cross_modal**-- ``m0`` (frozen experts + concatenation readout),
  ``m1`` (original cross-attention fusion) or ``m2`` (MulG),
* **identity**   -- ``i1`` (closed-set reference), ``i2`` (verification-based,
  open-set) or ``i3`` (no identity pipeline).

A run is one (axis combination, modality configuration) cell, e.g.
``m1_i1_p1_VAT``. The matrix is declared as data (see
``experiments/experiment_matrix.json``) so new setups need no code change --
just another entry. :func:`run_experiment_sequence` executes the declared
specs in order, resume-safely.

Current implementation status:

* ``m1`` (original progressive-cross-attention fusion) and ``m2`` (MulG
  directed pairwise cross-modal attention + GRU fusion) are both wired to the
  trainer via :func:`resolve_model_class`.
* ``i1`` / ``i3`` (reference on/off) are wired; ``i2`` (verification-based
  identity) is recognised but raises :class:`NotImplementedError`, so the
  suite records those cells as ``not_implemented`` and continues.
"""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .config import MODALITY_CONFIGS, default_device
from .protocols import build_protocol_splits
from .trainer import run_single_experiment

#: "VAT" -> ["v", "a", "t"]
_MODALITIES_BY_NAME = dict(MODALITY_CONFIGS)

#: identity axis -> whether the identity-reference branch is active.
#: i2 (verification-based) is intentionally absent -- it raises until designed.
_IDENTITY_USES_REFERENCE = {"i1": True, "i3": False}


@dataclass(frozen=True)
class ExperimentSpec:
    """One runnable cell of the matrix."""

    cross_modal: str   # 'm1' | 'm2'
    identity: str      # 'i1' | 'i2' | 'i3'
    protocol: str      # 'p1' | 'p2'
    modality: str      # 'V' | 'A' | 'T' | 'VA' | 'VT' | 'AT' | 'VAT'

    @property
    def name(self) -> str:
        return f"{self.cross_modal}_{self.identity}_{self.protocol}_{self.modality}"


def expand_cells(cells: List[dict]) -> List[ExperimentSpec]:
    """Expand declarative cells (each listing several modalities) into specs."""
    specs: List[ExperimentSpec] = []
    for cell in cells:
        for modality in cell["modalities"]:
            specs.append(ExperimentSpec(
                cross_modal=cell["cross_modal"], identity=cell["identity"],
                protocol=cell["protocol"], modality=modality))
    return specs


def resolve_reference_flag(spec: ExperimentSpec) -> bool:
    """Map a spec's axes to the trainer's ``use_reference`` flag.

    The reference flag depends only on the identity axis (it is independent of
    the cross_modal fusion choice). Raises NotImplementedError for ``i2``
    (verification-based identity), which is declared but not yet built.
    """
    if spec.cross_modal not in ("m0", "m1", "m2"):
        raise ValueError(f"unknown cross_modal: {spec.cross_modal!r}")
    if spec.identity == "i2":
        raise NotImplementedError(
            "identity 'i2' (verification-based, open-set) is not implemented yet")
    if spec.identity not in _IDENTITY_USES_REFERENCE:
        raise ValueError(f"unknown identity: {spec.identity!r}")
    return _IDENTITY_USES_REFERENCE[spec.identity]


def model_class_for(cross_modal: str):
    """Map a cross_modal axis value to its fusion architecture class.

    ``m0`` -> :class:`~rmfdn_ablation.staged.StagedFusion` (frozen unimodal
    experts, concatenation readout -- no cross-modal interaction),
    ``m1`` -> :class:`~rmfdn_ablation.RMFDN.RMFDN` (progressive cross-attention),
    ``m2`` -> :class:`~rmfdn_ablation.mulG_RMFDN.MulGRmfdn` (MulG directed pairwise
    cross-modal attention + GRU fusion). Imports are deferred so this module
    stays importable without the deep-learning stack.
    """
    from .RMFDN import RMFDN
    from .mulG_RMFDN import MulGRmfdn
    from .staged import StagedFusion
    return {"m0": StagedFusion, "m1": RMFDN, "m2": MulGRmfdn}[cross_modal]


def resolve_model_class(spec: ExperimentSpec):
    """Select the fusion architecture for a spec's cross_modal axis."""
    return model_class_for(spec.cross_modal)


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
    # expensive GPFS walk, so skip it outright when no spec needs it.
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

        # Axes that aren't built yet are recorded and skipped, not fatal.
        try:
            use_reference = resolve_reference_flag(spec)
        except NotImplementedError as error:
            print(f"[{spec.name}] SKIPPED -- {error}", flush=True)
            results[spec.name] = {"status": "not_implemented", "reason": str(error),
                                  "spec": asdict(spec)}
            json.dump(results, open(results_path, "w"), indent=1)
            continue

        model_class = resolve_model_class(spec)
        modalities = _MODALITIES_BY_NAME[spec.modality]
        splits = splits_for(spec.protocol)
        train_clips = splits["train"]
        if config.get("no_pristine_train"):
            train_clips = [clip for clip in train_clips if clip.is_fake]
            print(f"[{spec.name}] pristine TRAIN'den cikarildi -> "
                  f"{len(train_clips)} klip (yalniz fake)", flush=True)
        val_clips = splits["val"]
        if val_cap and len(val_clips) > val_cap:                 # fixed seed-0 subsample
            import random
            val_clips = random.Random(0).sample(val_clips, val_cap)

        print(f"\n{'=' * 70}\nRUN {spec.name}\n{'=' * 70}", flush=True)
        try:
            run_result = run_single_experiment(
                spec.name, modalities, use_reference,
                train_clips, val_clips, splits["test"],
                reference_index, tokenizer, config, output_dir, device,
                model_class=model_class,
                skip_test_eval=config.get("skip_test_eval", False))
            run_result["spec"] = asdict(spec)
            results[spec.name] = run_result
        except Exception as error:
            traceback.print_exc()
            print(f"[{spec.name}] FAILED: {error}", flush=True)
        json.dump(results, open(results_path, "w"), indent=1)

    print("\nSUITE_DONE", flush=True)
    return results
