#!/bin/bash
# Train one arm of the improvement experiment for one seed, then score its
# best-validation and final checkpoints under all seven test conditions.
# Recipe of report Section 6: VA, official split (p1), identity reference off,
# 6,000 optimizer steps at effective batch 128 on one GPU, cosine schedule over
# the same 6,000 steps, validation every 500 steps.
#
# Usage (12 jobs: 4 arms x 3 seeds):
#   for ARM in base dropout gate gate_dropout; do for SEED in 0 1 2; do
#     sbatch --account=<account> --export=ALL,ARM=$ARM,SEED=$SEED,WORK=/path/to/workdir \
#            scripts/slurm/train_improvement.slurm
#   done; done
#
# Each job writes its own suite directory $WORK/improvement/<ARM>_s<SEED>; pass
# all twelve to scripts/analyze_improvement.py --suites. The layout expected
# under $WORK is the same as for train_cell.slurm.
#
#SBATCH --job-name=rmfdn-impr
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=20 --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
: "${WORK:?set WORK to the working directory}" "${ARM:?set ARM to base, dropout, gate or gate_dropout}"
SEED=${SEED:-0}
RUN=$WORK/improvement/${ARM}_s${SEED}
mkdir -p "$RUN"

run() {
  singularity exec --nv --bind "$WORK:$WORK" \
    --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$WORK/rmfdn.sif" "$@"
}

cd "$WORK/code"
printf '{"cells":[{"cross_modal":"m1","identity":"i3","protocol":"p1","modalities":["VA"],"variants":["%s"]}]}' \
  "$ARM" > "$RUN/cfg.json"

run python scripts/run_experiment_suite.py \
  --config "$RUN/cfg.json" \
  --data-root "$WORK/data/idforge" --reference-root "$WORK/data/idforge_ref" \
  --output-dir "$RUN" \
  --bert-model "$WORK/models/bert-base-uncased" --ast-checkpoint "$WORK/models/ast" \
  --iters 6000 --lr-schedule-iters 6000 --eval-interval 500 --patience 100 \
  --batch-size 64 --grad-accum-steps 2 --num-workers 16 --seed "$SEED" \
  --mask-invisible-bits --final-eval --skip-test-eval

for CHECKPOINT in best_validation final; do
  SUFFIX=$([ "$CHECKPOINT" = final ] && echo "_final" || echo "")
  run python scripts/run_corruption_eval.py \
    --phase1-dir "$RUN" --checkpoint "$CHECKPOINT" \
    --data-root "$WORK/data/idforge" --reference-root "$WORK/data/idforge_ref" \
    --bert-model "$WORK/models/bert-base-uncased" --ast-checkpoint "$WORK/models/ast" \
    --batch-size 64 --num-workers 4 \
    --output "$RUN/corruption_results$SUFFIX.json" --dump-scores "$RUN/scores$SUFFIX"
done
