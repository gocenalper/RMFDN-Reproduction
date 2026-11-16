"""R-MFDN modality x reference ablation on IDForge_v2.

Package layout (data flows top to bottom):

    config.py           experiment configuration & manipulation-technique taxonomy
    data_catalog.py     clip discovery and ground-truth labelling on disk
    media_io.py         frame / pretrained-AST audio / transcript loading
    dataset.py          PyTorch dataset and batch collation
    bounded_sampler.py fixed-pristine, bounded fake-type sample stream
    encoders.py         per-modality encoders (visual / audio / text), one each
    attention.py        shared attention blocks
    RMFDN.py            R-MFDN: three encoders, progressive fusion, two heads
    mulG_RMFDN.py       m2 cross-modal variant (MulG fusion blocks)
    losses.py           InfoNCE-based auxiliary losses (paper Eq. 4-5)
    reference.py        live shared-encoder reference path (paper Eq. 5-6)
    protocols.py        p1 / p2 (identity-disjoint) split construction
    evaluation.py       binary detection + multi-label attribution metrics
    trainer.py          one training run: schedule, AMP, checkpoints, early stop
    experiment.py       cell expansion + sequencing for the experiment matrix

Entry points live in ``scripts/``: ``run_experiment_suite.py`` (trains the
12 runnable cells of the matrix), ``run_test_eval.py`` (held-out test
evaluation), ``train_probe.py`` (frozen-encoder 7-bit head probe) and
``extract_features.py`` (frozen per-modality feature cache for offline
probes). The eight
i2 cells remain explicit ``not_implemented`` entries.
"""
