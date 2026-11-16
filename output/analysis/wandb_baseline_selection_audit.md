# Paper-Exact Single-Factor Experiment Audit

Source: private [W&B project](https://wandb.ai/<entity>/idforge-rmfdn-ablation), exported on 2026-07-16.

## Decision

Architecture is not an experimental variable. Every accepted experiment must use the same paper-exact R-MFDN architecture and the same paper-reported hyperparameters. A condition is admissible only when exactly one unresolved protocol, supervision, or evaluation parameter differs from its declared reference condition.

The W&B project contains 93 runs: 69 finished and 24 crashed. After the architecture lock, **no existing W&B run qualifies as a final single-factor baseline comparison**. Architecture-changing runs remain in the inventory only as rejected historical diagnostics. They must not be used to select the reproduction baseline or populate the controlled result table.

| W&B disposition | Runs | Use |
|---|---:|---|
| Accepted final comparison | 0 | None of the historical runs has complete, current, one-factor provenance. |
| Diagnostic only | 51 | May explain prior observations, but cannot support a causal claim. |
| Excluded | 42 | Crashed, metric-free, duplicated, architecture-changing, paper-hyperparameter-changing, or confounded. |

The normalized run-level record is preserved in `wandb_run_inventory.csv`.

## Immutable Paper Contract

### Architecture lock

The following components are identical in every training condition:

| Component | Locked value |
|---|---|
| Visual encoder | ImageNet-pretrained ResNet-18 |
| Visual adaptation | All ResNet layers trainable end-to-end |
| Temporal aggregation | Paper temporal Transformer enabled |
| Audio encoder | AudioSet-pretrained AST |
| Text encoder | Paper BERT encoder; attention-mask mean pooling kept as the fixed implementation choice |
| Fusion | Exact modality-specific R-MFDN fusion graph |
| Classification heads | Two-class binary head plus the same seven-logit technique head |
| Identity path | Paper reference-assisted identity path enabled |
| Visual input | Released 16 prepared face-cropped frames, grouped 4x4 |
| Additional preprocessing | No extra face crop and no visual augmentation |

Consequently, the following historical W&B families are excluded: `noattnV`, `noffnV`, `bareV`, all frozen/stem/layer-depth runs, truncation/bottleneck runs, ResNet-10 or pruned encoders, GroupNorm/SyncBN substitutions, altered fusion runs, and AST-initialization ablations.

### Reported hyperparameter lock

| Hyperparameter | Locked value |
|---|---:|
| Temperature `tau` | 1.0 |
| Reference fusion `alpha` | 0.001 |
| Cross-modal loss `beta` | 0.2 |
| Identity loss `gamma` | 0.2 |
| Peak learning rate | `5e-5` |
| Warm-up | 1,000 optimizer steps |
| Warm-up start LR | `2e-7` |
| Scheduler | Cosine decay |
| Minimum LR | `1e-8` |
| Schedule horizon | 90,000 optimizer steps |

LR, warm-up, cosine horizon, `tau`, `alpha`, `beta`, and `gamma` sweeps are therefore excluded. Operational constants not specified by the paper are fixed across the entire screening matrix: batch 64, gradient accumulation 2, evaluation every 1,500 optimizer steps, patience 8, full validation, and one common seed. They can be studied only in their own explicitly named single-factor condition.

## Permitted Experimental Factors

Only these unresolved factors may vary. No two are combined in the first-stage matrix.

| Factor | Reference value | Controlled alternatives | Architecture changed? |
|---|---|---|---|
| Training sampler | Natural released distribution | Historical 25% pristine balance; bounded DECOR | No |
| Seven-logit MLC formulation | BCE + sigmoid | Raw multi-hot softmax CE; normalized-target softmax CE | No |
| Label supervision scope | All seven bits | Modality-visible-bit loss mask | No |
| Per-process micro-batch size | 64 | 24; 128, with gradient accumulation fixed at 2 | No |
| Validation scope | Full 17,305 | Fixed 4,000 subset | No |
| Checkpoint objective | Validation AUC | Validation mAP | No; selection only |
| F1 threshold | 0.5 | Threshold selected on validation | No; evaluation only |
| Attribution metric scope | All seven bits | Visible bits only | No; evaluation only, but changes the reported task |
| Pristine evaluation scope | Included | Excluded from attribution metric only | No; evaluation only, but changes the reported population |

`type11` is excluded because changing the technique head from seven outputs to eleven package classes changes the architecture and output task. Protocol p2, identity-off i3, extra crop, augmentation, degradation, and alternate reference-fusion strategies are also outside the paper-exact reproduction matrix.

## Single-Factor Training Matrix

Every row below inherits the complete immutable contract. The `Reference` column is part of the experimental definition.

Each training condition is ultimately instantiated for the paper's complete modality lattice: V, A, T, VA, VT, AT, and VAT. A condition cannot be declared successful from V alone. The changed factor must retain the same value across all seven modality cells, and each cell is compared only with the matching C0 modality.

| ID | Reference | Single changed parameter | Control value | Status |
|---|---|---|---|---|
| C0 | - | None | Natural sampler, BCE+sigmoid, all-bit labels | Baseline |
| S1 | C0 | `sampler` | Historical 25% pristine balance | Needs clean rerun |
| S2 | C0 | `sampler` | Bounded DECOR | Needs clean rerun |
| L1 | C0 | `mlc_formulation` | Raw multi-hot softmax CE + softmax scores | Needs clean rerun |
| L2 | C0 | `mlc_formulation` | Normalized-target softmax CE + softmax scores | Needs clean rerun |
| M1 | C0 | `mask_invisible_bits` | On | VT complete; VA pair in progress |
| B1 | C0 | `batch_size` | 24, accumulation remains 2 | Pending, lower priority |
| B2 | C0 | `batch_size` | 128, accumulation remains 2 | Pending, lower priority |
| V1 | C0 | `validation_scope` | Fixed 4,000 validation clips | Pending, selection diagnostic |

Training order under the GPU budget is `C0 -> L1 -> L2 -> S1 -> S2`, completing one factor family before starting the next. `M1` already has a controlled VT result and the VA comparison is in progress. Batch and validation-scope conditions are deferred until the loss/sampler ambiguity is resolved.

## Evaluation-Only Matrix

These conditions reuse saved scores or checkpoints; they must not trigger retraining.

| ID | Reference | Single changed parameter | Rule |
|---|---|---|---|
| E1 | L1 | `f1_threshold` | Select one global threshold on full validation; freeze it for test |
| E2 | L2 | `f1_threshold` | Same procedure for normalized-target softmax |
| E3 | M1 | `metric_bits` | Score visible bits only; report separately from the paper's seven-bit table |
| E4 | C0 | `pristine_metric_scope` | Exclude pristine only from attribution evaluation |
| E5 | C0 | `checkpoint_objective` | Select by validation mAP instead of AUC |

Standard mAP is always computed from continuous seven-bit scores and is threshold-free. A second `mAP(binary predictions)` may be logged only as a non-standard diagnostic to test whether the authors thresholded before AP; it must never replace the primary mAP.

## Accepted Controlled Results

### Label masking: VT

This is the only completed Level-A pair currently available: same paper-exact architecture, natural sampling, BCE, full validation/test, schedule, and checkpoint rule. The only training change is `mask_invisible_bits`.

| VT condition | ACC | AUC | mAP | CF1 | OF1 |
|---|---:|---:|---:|---:|---:|
| C0: unmasked all-bit BCE | 82.93 | 92.39 | 86.31 | 78.99 | 82.04 |
| M1: visible-bit masked BCE | 81.06 | 91.46 | 53.79 | 51.80 | 51.03 |
| C0 - M1 | +1.87 | +0.93 | +32.52 | +27.19 | +31.01 |

Interpretation: masking changes attribution supervision dramatically while binary detection changes by less than two points. It is a valid shortcut-learning diagnostic, but M1 is not a paper-equivalent replacement because the paper reports an all-seven-bit task.

### Label masking: VA

| VA condition | ACC | AUC | mAP | CF1 | OF1 | Status |
|---|---:|---:|---:|---:|---:|---|
| M1: visible-bit masked BCE | 91.31 | 98.96 | 77.57 | 75.71 | 72.51 | Complete |
| C0: unmasked all-bit BCE | - | - | - | - | - | Training in progress |

No masking effect is claimed for VA until the C0 full-test result is available.

## Historical Results Not Accepted as Controlled Evidence

The following results remain useful for deciding what to rerun, but they are not copied into the causal result table because their training/evaluation lineage differs.

| Historical condition | ACC | AUC | mAP | Reason not accepted |
|---|---:|---:|---:|---|
| Balanced-25 V | 75.68 | 84.81 | 55.29 | Closest paper match, but not a clean C0-vs-S1 rerun |
| Bounded DECOR V | 70.47 | 75.78 | 51.94 | Sampler result exists, but detector collapsed and protocol is not fully matched |
| Softmax-7 V | - | - | 54.55 | Incomplete metric record; threshold and pristine handling unresolved |
| W&B Colab softmax V | 70.67 | 78.96 | 46.63 | Crashed, validation-only |

These numbers motivate S1, S2, L1, and L2. They do not determine the baseline.

## Reporting Rule

The report accepts a result only when:

1. Its architecture hash/configuration matches C0.
2. All paper-reported hyperparameters match C0.
3. Exactly one named parameter differs from its reference condition.
4. It uses the same split, full-test evaluator, metric scope, and best-checkpoint rule unless evaluation scope itself is the named factor.
5. The changed parameter was selected without looking at test results.
6. ACC, AUC, mAP, CF1, OF1, per-bit AP, and predicted label cardinality are recorded together.

Metric closeness to the paper is never sufficient to admit a condition. Protocol validity comes first.
