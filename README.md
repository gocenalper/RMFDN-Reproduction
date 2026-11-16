# IDForge R-MFDN Experiment Matrix

Clean training and evaluation code for the declared experiments in
`experiments/experiment_matrix.json`.

## Scope

The matrix expands to 20 named cells:

- 12 runnable cells covering P1/P2, M1/M2, I1/I3, and the declared modality sets.
- 8 I2 cells retained as explicit `not_implemented` results.

No standalone backbone experiments, input degradation diagnostics, alternate
taxonomies, or visual augmentation paths are included.

## Canonical Model

- Visual: ImageNet-pretrained ResNet-18 per frame, followed by the temporal
  Transformer over four groups of four frames.
- Audio: AudioSet-pretrained AST, fine-tuned end to end.
- Text: pretrained BERT with attention-mask-aware mean pooling.
- M1: R-MFDN progressive multimodal fusion.
- M2: MulG fusion over the same visual, audio, and text encoders as M1.
- Attribution: one fixed seven-output multi-label head for every modality cell.
- Unimodal cells: encoder output feeds the heads directly; no attention,
  normalization, FFN, or other fusion module is constructed or executed.
- Visual input: the dataset's prepared 16 face frames, resized to 224x224 when
  necessary, with ImageNet normalization and no data augmentation.

I1 samples a same-identity pristine clip for every suspect sample and sends
both through the live shared encoders for Eq. 5/6. I3 disables both reference
contributions while keeping the rest of the cell unchanged.

## Layout

```text
experiments/experiment_matrix.json  experiment declarations
protocol_p2/p2_split.json           fixed identity-disjoint P2 assignment
rmfdn_ablation/                     model, data, training, and evaluation package
rmfdn_ablation/bounded_sampler.py   fixed-pristine, bounded fake-type sampler
rmfdn_ablation/reference.py         live shared-encoder reference path (Eq. 5/6)
scripts/run_experiment_suite.py     matrix training entry point
scripts/run_test_eval.py            deferred held-out test evaluation
scripts/train_probe.py              frozen-encoder 7-bit head probe
scripts/extract_features.py         frozen per-modality feature cache
tests/                              local regression and contract tests
```

## Installation

Python 3.9 or newer is required. `pyproject.toml` is the single dependency
source.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . pytest
python -m pytest -q
```

Compute nodes may use local Hugging Face directories through `--bert-model`
and `--ast-checkpoint`.

## Inspect The Matrix

This command performs no model construction and consumes no GPU time:

```bash
python scripts/run_experiment_suite.py \
  --config experiments/experiment_matrix.json \
  --data-root /unused \
  --reference-root /unused \
  --list
```

It must print 20 specs. The 8 I2 specs are recorded as `not_implemented` when
the suite runs; the other 12 are trainable.

## Train

```bash
python scripts/run_experiment_suite.py \
  --config experiments/experiment_matrix.json \
  --data-root /path/to/idforge_root \
  --reference-root /path/to/idforge_reference \
  --p2-split protocol_p2/p2_split.json \
  --output-dir ./suite_out \
  --bert-model /path/to/bert-base-uncased \
  --ast-checkpoint /path/to/ast-finetuned-audioset \
  --iters 90000 \
  --lr-schedule-iters 90000 \
  --eval-interval 3000 \
  --patience 10 \
  --val-cap 4000 \
  --batch-size 16 \
  --grad-accum-steps 8 \
  --bounded-sampler-pristine 0.25 \
  --skip-test-eval
```

`batch-size * grad-accum-steps` is the effective batch size. The example uses
an effective batch of 128 while limiting activation memory.

`--iters` limits the number of optimizer steps actually executed, while
`--lr-schedule-iters` independently controls the cosine-decay horizon. For a
short diagnostic run that still follows the paper's learning-rate curve, use
`--iters 12000 --lr-schedule-iters 90000`.

The suite writes `results.json` after every cell. Completed cells are skipped
on relaunch. During an active cell, `runs/<cell>_ck.pt` is the crash-resume
checkpoint; the best validation model is `runs/<cell>.pt`.

With `--bounded-sampler-pristine 0.25`, exactly 25% of the training stream is
pristine. The fake portion uses fixed 5%-15% bounded max-entropy weights that
preserve every forgery type while reducing the attainable seven-bit imbalance.
Clips cycle without replacement inside each type, and checkpoint resume
continues from the corresponding global sample offset. The effective category
and positive-bit shares are printed before training.

Other active training flags:

- `--select-metric {ACC,AUC,mAP}`: validation metric for best-checkpoint
  selection and early stopping (default AUC).
- `--blank-modalities <subset of vat>`: keep the full (VAT) graph but blank
  the listed modalities' raw inputs (black frames, zero mel, empty text).
- `--mask-invisible-bits`: supervise the technique BCE only on bits directly
  perceivable by the cell's modalities.
- `--mlc-loss {bce,softmax,softmax_norm,type11}`: technique-head loss variant
  (default BCE).

## Evaluate

```bash
python scripts/run_test_eval.py \
  --phase1-dir ./suite_out \
  --data-root /path/to/idforge_root \
  --reference-root /path/to/idforge_reference \
  --p2-split protocol_p2/p2_split.json \
  --bert-model /path/to/bert-base-uncased \
  --ast-checkpoint /path/to/ast-finetuned-audioset \
  --batch-size 16 \
  --num-workers 8 \
  --output ./suite_out/test_results.json
```

Evaluation restores the effective training configuration saved in
`results.json`; model paths and runtime batch/worker counts may be overridden
when the environment changes. It reports binary ACC/AUC and seven-bit
mAP/CF1/OF1 plus per-bit accuracy, writes incrementally, and skips cells
already present in the output file.

Other active evaluation flags:

- `--metric-implementation {sklearn,reference_apmeter}`: multi-label metric
  formulas; `reference_apmeter` reproduces MultiModal-DeepFake for BCE
  checkpoints.
- `--dump-scores <dir>`: write per-clip score dumps (`test_<run>.npz`) for
  offline analysis.

## Data Contract

The data root contains `train/`, `val/`, and `test/`. A clip directory contains
one frame archive (`frames.jpz.npz`, `frames.jpz`, or `frames_ndarray.npy`) and
optional audio/transcript sidecars sharing the clip basename. Missing audio or
text is represented by zeros or an empty string so one broken sidecar does not
abort a long matrix run.

P1 uses the physical dataset splits. P2 pools those clips and reassigns them by
identity using `protocol_p2/p2_split.json`.
