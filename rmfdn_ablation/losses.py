"""Auxiliary contrastive losses (paper Eq. 4-5).

Both losses operate on the per-modality contrastive projections produced by
the model (``outputs["contrastive_<m>"]``) and use **pristine samples as
anchors** -- the intuition being that for a genuine clip all modalities tell
one consistent story about one identity, while manipulations break either the
cross-modal agreement or the identity match.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor,
                  negatives: torch.Tensor, temperature: float) -> torch.Tensor:
    """InfoNCE over one anchor row: ``(1, d)`` vs ``(1, d)`` and ``(1, N, d)``."""
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)
    positive_logit = (anchor * positive).sum(-1, keepdim=True) / temperature
    negative_logits = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / temperature
    logits = torch.cat([positive_logit, negative_logits], 1)
    target = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, target)


def cross_modal_consistency_loss(outputs: Dict[str, torch.Tensor], is_fake: torch.Tensor,
                                 modalities: Sequence[str], temperature: float) -> torch.Tensor:
    """Eq. 4 -- align the modalities of *pristine* clips within the batch.

    For each ordered modality pair, a pristine sample's projection in the
    query modality must retrieve its own row in the key modality against all
    other rows in the batch (in-batch negatives). Fakes participate only as
    negatives. Returns 0 when fewer than two modalities are active or the
    batch contains no pristine sample.
    """
    if len(modalities) < 2:
        return outputs["fused"].new_zeros(())
    pristine_mask = (is_fake == 0)
    if is_fake.size(0) < 2 or pristine_mask.sum() < 1:
        return outputs["fused"].new_zeros(())
    pristine_indices = pristine_mask.nonzero(as_tuple=True)[0]

    # Audio is the hub with three modalities (it pairs with both others);
    # with two modalities both directions of the single pair are used.
    pairs = ([("a", "v"), ("a", "t"), ("v", "a"), ("t", "a")] if len(modalities) == 3
             else [(modalities[0], modalities[1]), (modalities[1], modalities[0])])

    def pair_loss(query_modality: str, key_modality: str) -> torch.Tensor:
        query = F.normalize(outputs[f"contrastive_{query_modality}"], dim=-1)
        key = F.normalize(outputs[f"contrastive_{key_modality}"], dim=-1)
        logits = (query[pristine_indices] @ key.T) / temperature
        return F.cross_entropy(logits, pristine_indices)

    return sum(pair_loss(q, k) for q, k in pairs) / len(pairs)


def identity_aware_contrastive_loss(outputs: Dict[str, torch.Tensor], identities: Sequence[str],
                                    is_fake: torch.Tensor,
                                    reference_outputs: Optional[Dict[str, torch.Tensor]],
                                    modalities: Sequence[str], temperature: float) -> torch.Tensor:
    """Eq. 5 -- pull pristine samples toward their identity's reference entry.

    Every row has a freshly sampled same-identity pristine reference encoded
    by the live shared encoder. For a pristine anchor, that aligned row is the
    positive; different-identity references and same-identity fakes in the
    suspect batch are negatives.
    """
    if reference_outputs is None or (is_fake == 0).sum() < 1:
        return outputs["fused"].new_zeros(())
    fake_positions = [j for j in range(len(identities)) if is_fake[j] == 1]

    loss_terms = []
    for modality in modalities:
        key = f"contrastive_{modality}"
        for i in range(len(identities)):
            if is_fake[i] != 0:
                continue
            anchor = outputs[key][i:i + 1]
            positive = reference_outputs[key][i:i + 1]
            negatives = (
                [reference_outputs[key][j] for j in range(len(identities))
                 if identities[j] != identities[i]]
                + [outputs[key][j] for j in fake_positions if identities[j] == identities[i]]
            )
            if negatives:
                loss_terms.append(
                    info_nce_loss(anchor, positive, torch.stack(negatives).unsqueeze(0), temperature))
    return torch.stack(loss_terms).mean() if loss_terms else outputs["fused"].new_zeros(())
