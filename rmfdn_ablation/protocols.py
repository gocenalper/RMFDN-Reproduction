"""Dataset split protocols.

A *protocol* turns the on-disk clips into train / val / test
:class:`~rmfdn_ablation.data_catalog.ClipRecord` lists. Two are supported:

* **p1** -- the original IDForge_v2 split: the on-disk ``train/ val/ test``
  directories are used as-is. Identities are shared across splits (the same
  speaker may appear in train and test).
* **p2** -- identity-disjoint, gender-stratified: every clip is reassigned to
  the split its *identity* belongs to, following a precomputed
  identity -> split map (``protocol_p2/p2_split.json``). No speaker appears in
  more than one split, so test identities are unseen during training.

Only the train/val/test grouping changes between protocols; everything
downstream (filtering, dataset, model, trainer) is identical. The reference
set is per-identity and is not split, so it is protocol-agnostic.
"""

from __future__ import annotations

import json
from typing import Dict, List

from .data_catalog import ClipRecord, scan_split_clips

#: On-disk split directories that physically hold the clips.
ON_DISK_SPLITS = ("train", "val", "test")


def build_protocol_splits(protocol: str, data_root: str,
                          p2_split_path: str = None) -> Dict[str, List[ClipRecord]]:
    """Return ``{"train": [...], "val": [...], "test": [...]}`` for a protocol.

    Args:
        protocol: ``"p1"`` (on-disk split) or ``"p2"`` (identity-disjoint).
        data_root: dataset root containing the on-disk ``train/ val/ test``.
        p2_split_path: path to the identity -> split JSON (required for p2).
    """
    if protocol == "p1":
        return {split: scan_split_clips(data_root, split) for split in ON_DISK_SPLITS}

    if protocol == "p2":
        if not p2_split_path:
            raise ValueError("protocol 'p2' requires p2_split_path (identity -> split map)")
        identity_to_split = load_identity_assignment(p2_split_path)
        splits: Dict[str, List[ClipRecord]] = {s: [] for s in ON_DISK_SPLITS}
        unknown_identities = set()
        # Pool every clip regardless of where it physically sits, then regroup
        # by identity. p1 mixes identities across the on-disk splits, so we must
        # scan all of them to recover each identity's full set of clips.
        for on_disk_split in ON_DISK_SPLITS:
            for clip in scan_split_clips(data_root, on_disk_split):
                target = identity_to_split.get(clip.identity)
                if target is None:
                    unknown_identities.add(clip.identity)
                    continue
                splits[target].append(clip)
        if unknown_identities:
            raise ValueError(
                f"identities present on disk but absent from the p2 split map: "
                f"{sorted(unknown_identities)}")
        return splits

    raise ValueError(f"unknown protocol: {protocol!r} (expected 'p1' or 'p2')")


def load_identity_assignment(p2_split_path: str) -> Dict[str, str]:
    """Flatten ``p2_split.json`` into an ``identity -> split`` mapping."""
    data = json.load(open(p2_split_path, encoding="utf-8"))
    assignment: Dict[str, str] = {}
    for split, identities in data["splits"].items():
        for identity in identities:
            assignment[identity] = split
    return assignment
