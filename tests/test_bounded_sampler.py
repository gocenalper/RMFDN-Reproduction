"""Contract tests for the bounded forgery-type sampler."""

from collections import Counter

import pytest

from rmfdn_ablation.bounded_sampler import BoundedForgeryTypeSampler
from rmfdn_ablation.config import FORGERY_TYPE_TO_TECHNIQUES


class Clip:
    def __init__(self, forgery_type: str, serial: int):
        self.forgery_type = forgery_type
        self.is_fake = int(forgery_type != "pristine")
        self.serial = serial


def _clips():
    clips = [Clip("pristine", serial) for serial in range(4)]
    for forgery_type in FORGERY_TYPE_TO_TECHNIQUES:
        if forgery_type != "pristine":
            clips.extend(Clip(forgery_type, serial) for serial in range(4))
    return clips


def test_quarter_pristine_cycle_has_decor_fake_type_shares():
    clips = _clips()
    sampler = BoundedForgeryTypeSampler(
        clips, num_samples=4000, pristine_share=0.25, seed=3)
    sampled_types = [clips[index].forgery_type for index in sampler]

    assert len(sampler.category_cycle) == 4000
    assert Counter(sampled_types) == {
        "pristine": 1000,
        "face_audiomismatch_textmismatch": 72,
        "face_rvc_textmismatch": 102,
        "face_tts": 60,
        "face_tts_textgen": 90,
        "lip_audiomismatch_textmismatch": 60,
        "lip_rvc_textmismatch": 75,
        "lip_tts_textgen": 105,
        "rvc_textmismatch": 960,
        "tts_textgen": 957,
        "tts_textmismatch": 519,
    }


def test_positive_bit_shares_match_the_decor_distribution():
    sampler = BoundedForgeryTypeSampler(
        _clips(), num_samples=4000, pristine_share=0.25, seed=3)

    assert sampler.technique_positive_shares() == pytest.approx({
        "lip_sync": 0.141,
        "face_swap": 0.081,
        "tts": 0.43275,
        "rvc": 0.28425,
        "audio_shuffle": 0.033,
        "llm_textgen": 0.288,
        "text_shuffle": 0.447,
    })


def test_category_clips_cycle_before_repeating():
    clips = _clips()
    # rarest DECOR share is 1.5% -> 4000 samples give every type >= 4 visits
    sampler = BoundedForgeryTypeSampler(
        clips, num_samples=4000, pristine_share=0.25, seed=3)
    sampled = [clips[index] for index in sampler]

    for forgery_type in FORGERY_TYPE_TO_TECHNIQUES:
        first_four = [clip.serial for clip in sampled
                      if clip.forgery_type == forgery_type][:4]
        assert len(first_four) == len(set(first_four)) == 4


def test_resume_range_matches_the_uninterrupted_stream():
    clips = _clips()
    uninterrupted = list(BoundedForgeryTypeSampler(
        clips, num_samples=160, pristine_share=0.25, seed=11))
    resumed = BoundedForgeryTypeSampler(
        clips, num_samples=1, pristine_share=0.25, seed=11)
    resumed.set_range(start_index=37, num_samples=123)

    assert list(resumed) == uninterrupted[37:]
