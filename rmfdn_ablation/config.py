"""Canonical experiment configuration and seven-bit manipulation taxonomy.

Everything that *defines* an experiment is centralised here so that the rest
of the package contains only mechanism:

* the seven-technique manipulation taxonomy shared by the dataset labels and
  the model's multi-label attribution head,
* the mapping from named IDForge_v2 forgery types to 7-bit technique vectors
  (used **only** to derive ground truth -- the model never predicts the named
  combination types themselves, it predicts each technique independently),
* the paper-scale training hyper-parameters used by every matrix cell.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Manipulation-technique taxonomy
# --------------------------------------------------------------------------- #

#: The seven atomic manipulation techniques. Index order is significant: the
#: ground-truth vectors below and the model's attribution head share it.
MANIPULATION_TECHNIQUES: List[str] = [
    "lip_sync",       # 0  visual: mouth re-animation
    "face_swap",      # 1  visual: identity replacement
    "tts",            # 2  audio:  synthetic speech (text-to-speech)
    "rvc",            # 3  audio:  voice conversion
    "audio_shuffle",  # 4  audio:  mismatched / shuffled audio track
    "llm_textgen",    # 5  text:   LLM-generated transcript
    "text_shuffle",   # 6  text:   mismatched / shuffled transcript
]

#: Named IDForge_v2 forgery types -> multi-hot technique vector
#: (lip_sync, face_swap, tts, rvc, audio_shuffle, llm_textgen, text_shuffle).
#: These names exist only as a labelling convenience: each is a fixed
#: *combination* of atomic techniques and supplies the 7-bit ground truth.
#: These match IDForge's official 8-technique table (github xyyandxyy/IDForge):
#: lip_sync (Wav2Lip) is set on BOTH lip_* AND face_* types (face swaps also run
#: Wav2Lip). We fold the official "Stand-in Use" column into face_swap here
#: (they are column-identical on this dataset -- both mark exactly the 4 face_*).
FORGERY_TYPE_TO_TECHNIQUES: Dict[str, Tuple[int, ...]] = {
    "pristine":                        (0, 0, 0, 0, 0, 0, 0),
    "face_audiomismatch_textmismatch": (1, 1, 0, 0, 1, 0, 1),
    "face_rvc_textmismatch":           (1, 1, 0, 1, 0, 0, 1),
    "face_tts":                        (1, 1, 1, 0, 0, 0, 0),
    "face_tts_textgen":                (1, 1, 1, 0, 0, 1, 0),
    "lip_audiomismatch_textmismatch":  (1, 0, 0, 0, 1, 0, 1),
    "lip_rvc_textmismatch":            (1, 0, 0, 1, 0, 0, 1),
    "lip_tts_textgen":                 (1, 0, 1, 0, 0, 1, 0),
    "rvc_textmismatch":                (0, 0, 0, 1, 0, 0, 1),
    "tts_textgen":                     (0, 0, 1, 0, 0, 1, 0),
    "tts_textmismatch":                (0, 0, 1, 0, 0, 0, 1),
}

TECHNIQUE_INDICES = tuple(range(len(MANIPULATION_TECHNIQUES)))


def visible_technique_indices(modalities: Sequence[str]) -> Tuple[int, ...]:
    """Technique bits directly observable from a modality combination.

    A bit is observable only if the manipulation leaves evidence *inside* the
    signals at hand. Synthesis leaves such evidence in its own stream
    (tts/rvc in audio, lip_sync/face_swap in video, llm_textgen in text), but
    the two *shuffle* manipulations splice genuine content into the wrong
    place: the shuffled stream itself is authentic, and only a second stream
    reveals the mismatch. audio_shuffle is exposed by audio + video (the lip
    track disagrees with the speech), text_shuffle by audio + text (the
    speech disagrees with the transcript).
    """
    visible = set()
    if "v" in modalities:
        visible.update((0, 1))          # lip_sync, face_swap
    if "a" in modalities:
        visible.update((2, 3))          # tts, rvc
    if "t" in modalities:
        visible.add(5)                  # llm_textgen
    if "a" in modalities and "v" in modalities:
        visible.add(4)                  # audio_shuffle: lip-speech desync only
    if "a" in modalities and "t" in modalities:
        visible.add(6)                  # text_shuffle: speech <-> transcript
    return tuple(sorted(visible))


#: The seven modality configurations of the ablation grid, ordered light to
#: heavy. Each is run twice: with and without the identity-reference branch.
MODALITY_CONFIGS: List[Tuple[str, List[str]]] = [
    ("T", ["t"]), ("A", ["a"]), ("AT", ["a", "t"]),
    ("V", ["v"]), ("VT", ["v", "t"]), ("VA", ["v", "a"]),
    ("VAT", ["v", "a", "t"]),
]


# --------------------------------------------------------------------------- #
# Training hyper-parameters (paper 90k protocol defaults)
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG: Dict = dict(
    # --- model dimensions -------------------------------------------------- #
    attention_heads=8,             # heads in every multi-head attention block
    contrastive_dim=128,           # projection width for the InfoNCE heads
    embed_dim=512,                 # shared feature width of all encoders
    # --- visual input ------------------------------------------------------ #
    frame_groups=4,                # frames are split into groups of 4 for the
                                   # within-group temporal transformer
    frames_per_clip=16,            # frames consumed per clip
    image_size=224,
    # --- audio input -------------------------------------------------------- #
    ast_checkpoint="MIT/ast-finetuned-audioset-10-10-0.4593",  # paper's audio encoder
    max_audio_seconds=8.0,
    sample_rate=16000,
    # --- text input --------------------------------------------------------- #
    bert_model="bert-base-uncased",
    max_text_tokens=50,            # BERT tokenizer max length
    # --- optimisation (paper 90k schedule) ----------------------------------- #
    backbone_lr_scale=1.0,         # paper: ResNet-18 fine-tuned jointly at the full rate
    batch_size=128,                # paper-faithful effective batch (authors: multi-GPU 4090s)
    grad_clip_norm=1.0,
    min_lr=1e-8,
    peak_lr=5e-5,                  # paper schedule
    total_iters=90_000,            # maximum optimizer steps to execute
    lr_schedule_iters=90_000,      # independent cosine horizon (paper schedule)
    warmup_iters=1000,             # paper schedule
    warmup_start_lr=2e-7,
    weight_decay=0.01,
    # --- auxiliary losses (paper Eq. 4-6) ------------------------------------ #
    cross_modal_beta=0.2,          # weight of the cross-modal consistency loss
    identity_gamma=0.2,            # weight of the identity-aware loss (ref runs)
    reference_alpha=0.001,         # Eq. 6: fused' = fused + alpha * reference
    temperature=1.0,               # InfoNCE temperature (tau); paper uses tau=1
    # --- evaluation & early stopping ---------------------------------------- #
    early_stop_patience=10,        # stop after N non-improving validations
    eval_interval=1000,            # validation every N iterations
    # --- runtime -------------------------------------------------------------- #
    log_interval=100,
    num_workers=8,
    seed=0,
)


def default_device() -> str:
    """'cuda' when available, else 'cpu' (import deferred so that this module
    stays importable in torch-free tooling)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
