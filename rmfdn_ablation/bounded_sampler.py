"""Deterministic sampler with DECOR (visual-class decorrelating) type shares."""

from __future__ import annotations

import random
import zlib
from collections import Counter
from fractions import Fraction
from math import lcm
from typing import Dict, List, Sequence

from torch.utils.data import Sampler

from .config import FORGERY_TYPE_TO_TECHNIQUES, MANIPULATION_TECHNIQUES


# Relative shares inside the fake portion of the stream. These weights are the
# DECOR (decorrelating) solution: minimise the spread of P(bit | visual class)
# across FACE / LIP / CLEAN for the visually-invisible bits (tts, rvc,
# llm_textgen, text_shuffle), subject to every fake type keeping >= 2%
# coverage (SLSQP over the type-bit table, rounded to /1000). Pumping the
# visually-clean fake types makes the visual evidence uninformative about the
# invisible bits, which is what deflates V-only attribution-mAP leakage; the
# audio_shuffle bit is structurally exempt (it only occurs in face_/lip_
# types, so P(aud_sh | CLEAN) = 0 under any share table).
FAKE_TYPE_SHARES: Dict[str, Fraction] = {
    "face_audiomismatch_textmismatch": Fraction(24, 1000),
    "face_rvc_textmismatch": Fraction(34, 1000),
    "face_tts": Fraction(20, 1000),
    "face_tts_textgen": Fraction(30, 1000),
    "lip_audiomismatch_textmismatch": Fraction(20, 1000),
    "lip_rvc_textmismatch": Fraction(25, 1000),
    "lip_tts_textgen": Fraction(35, 1000),
    "rvc_textmismatch": Fraction(320, 1000),
    "tts_textgen": Fraction(319, 1000),
    "tts_textmismatch": Fraction(173, 1000),
}


def _smooth_cycle(shares: Dict[str, Fraction]) -> List[str]:
    """Build the shortest smooth weighted round-robin category cycle."""
    cycle_length = lcm(*(share.denominator for share in shares.values()))
    counts = {category: int(share * cycle_length)
              for category, share in shares.items()}
    current = {category: 0 for category in shares}
    cycle: List[str] = []
    for _ in range(cycle_length):
        for category, count in counts.items():
            current[category] += count
        category = max(current, key=current.get)
        current[category] -= cycle_length
        cycle.append(category)
    if Counter(cycle) != Counter(counts):
        raise RuntimeError("failed to construct exact sampler cycle")
    return cycle


class BoundedForgeryTypeSampler(Sampler):
    """Sample an exact pristine share and bounded shares for all fake types.

    The category schedule is a short deterministic weighted-round-robin cycle.
    Inside each category, every clip is visited once in a seeded shuffled order
    before the next cycle starts. ``start_index`` makes the stream reproducible
    across checkpoint resume without serialising DataLoader prefetch state.
    """

    def __init__(self, clips: Sequence, num_samples: int,
                 pristine_share: float = 0.25, seed: int = 0,
                 start_index: int = 0,
                 fake_type_shares: Dict[str, Fraction] = None):
        if not 0.0 < pristine_share < 1.0:
            raise ValueError("pristine_share must be strictly between 0 and 1")
        if num_samples < 0 or start_index < 0:
            raise ValueError("num_samples and start_index must be non-negative")
        type_shares = dict(fake_type_shares or FAKE_TYPE_SHARES)
        if sum(type_shares.values()) != 1:
            raise ValueError("fake_type_shares must sum to exactly one")

        self.category_indices: Dict[str, List[int]] = {}
        for index, clip in enumerate(clips):
            category = "pristine" if clip.is_fake == 0 else clip.forgery_type
            self.category_indices.setdefault(category, []).append(index)

        expected_categories = {"pristine", *type_shares}
        actual_categories = set(self.category_indices)
        if actual_categories != expected_categories:
            missing = sorted(expected_categories - actual_categories)
            unexpected = sorted(actual_categories - expected_categories)
            raise ValueError(
                "bounded sampling requires every IDForge category; "
                f"missing={missing}, unexpected={unexpected}")

        pristine = Fraction(str(pristine_share)).limit_denominator(10_000)
        fractional_shares = {
            "pristine": pristine,
            **{category: (1 - pristine) * relative_share
               for category, relative_share in type_shares.items()},
        }
        if sum(fractional_shares.values()) != 1:
            raise RuntimeError("sampler category shares must sum to one")

        self.shares = {category: float(share)
                       for category, share in fractional_shares.items()}
        self.category_cycle = _smooth_cycle(fractional_shares)
        self._cycle_counts = Counter(self.category_cycle)
        self.seed = int(seed)
        self.start_index = int(start_index)
        self.num_samples = int(num_samples)
        self._pool_cache: Dict[str, tuple[int, List[int]]] = {}

    def set_range(self, start_index: int, num_samples: int) -> None:
        """Set the deterministic stream range used by the next iterator."""
        if start_index < 0 or num_samples < 0:
            raise ValueError("start_index and num_samples must be non-negative")
        self.start_index = int(start_index)
        self.num_samples = int(num_samples)
        self._pool_cache.clear()

    def repeat_factors(self, window_size: int) -> Dict[str, float]:
        """Expected visits per unique clip in a window of ``window_size``."""
        return {
            category: window_size * share / len(self.category_indices[category])
            for category, share in self.shares.items()
        }

    def technique_positive_shares(self) -> Dict[str, float]:
        """Expected positive rate of each attribution bit in the stream."""
        return {
            technique: sum(
                self.shares[category] * FORGERY_TYPE_TO_TECHNIQUES[category][index]
                for category in self.shares
            )
            for index, technique in enumerate(MANIPULATION_TECHNIQUES)
        }

    def _occurrences_before(self, sample_index: int) -> Dict[str, int]:
        full_cycles, remainder = divmod(sample_index, len(self.category_cycle))
        occurrences = {
            category: full_cycles * count
            for category, count in self._cycle_counts.items()
        }
        for category in self.category_cycle[:remainder]:
            occurrences[category] += 1
        return occurrences

    def _index_for_occurrence(self, category: str, occurrence: int) -> int:
        indices = self.category_indices[category]
        pool_cycle, offset = divmod(occurrence, len(indices))
        cached = self._pool_cache.get(category)
        if cached is None or cached[0] != pool_cycle:
            pool = list(indices)
            category_seed = zlib.crc32(category.encode("utf-8"))
            random.Random(self.seed + category_seed + pool_cycle * 1_000_003).shuffle(pool)
            self._pool_cache[category] = (pool_cycle, pool)
        return self._pool_cache[category][1][offset]

    def __iter__(self):
        occurrences = self._occurrences_before(self.start_index)
        for sample_index in range(self.start_index,
                                  self.start_index + self.num_samples):
            category = self.category_cycle[sample_index % len(self.category_cycle)]
            occurrence = occurrences[category]
            yield self._index_for_occurrence(category, occurrence)
            occurrences[category] += 1

    def __len__(self):
        return self.num_samples
