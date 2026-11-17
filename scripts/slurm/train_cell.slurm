#!/bin/bash
# Train one matrix cell on a Slurm cluster through Singularity/Apptainer, then
# score its best-validation checkpoint on the test split. This is the recipe
# behind the numbers in results/ (6,000 optimizer steps, effective batch 128,
# cosine schedule over the same 6,000 steps, one GPU).
#
# Usage:
#   sbatch --account=<account> --export=ALL,MOD=VAT,WORK=/path/to/workdir \
#          scripts/slurm/train_cell.slurm
#
# Expected under $WORK:
#   code/                 this repository
#   rmfdn.sif             image converted from the Docker image (see README)
#   data/idforge          dataset root with train/ val/ test/
#   data/idforge_ref      per-identity pristine reference clips
#   models/bert-base-uncased, models/ast   local Hugging Face checkpoints
#   split.json            identity -> split map for the identity-disjoint protocol
#
#SBATCH --job-name=rmfdn
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=20 --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
: "${WORK:?set WORK to the working directory}" "${MOD:?set MOD to VA or VAT}"
IDENTITY=${IDENTITY:-i3}      # i1 = identity reference on, i3 = off
PROTOCOL=${PROTOCOL:-p2}      # p1 = official split, p2 = identity-disjoint
SEED=${SEED:-0}
RUN=$WORK/runs/m1_${IDENTITY}_${PROTOCOL}_${MOD}_s${SEED}
mkdir -p "$RUN"

run() {
  singularity exec --nv --bind "$WORK:$WORK" \
    --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$WORK/rmfdn.sif" "$@"
}

cd "$WORK/code"
printf '{"cells":[{"cross_modal":"m1","identity":"%s","protocol":"%s","modalities":["%s"]}]}' \
  "$IDENTITY" "$PROTOCOL" "$MOD" > "$RUN/cfg.json"

run python scripts/run_experiment_suite.py \
  --config "$RUN/cfg.json" --p2-split "$WORK/split.json" \
  --data-root "$WORK/data/idforge" --reference-root "$WORK/data/idforge_ref" \
  --output-dir "$RUN" \
  --bert-model "$WORK/models/bert-base-uncased" --ast-checkpoint "$WORK/models/ast" \
  --iters 6000 --lr-schedule-iters 6000 --eval-interval 1500 --patience 8 \
  --batch-size 64 --grad-accum-steps 2 --num-workers 16 --seed "$SEED" \
  --mask-invisible-bits --skip-test-eval

run python scripts/run_test_eval.py \
  --phase1-dir "$RUN" --p2-split "$WORK/split.json" \
  --data-root "$WORK/data/idforge" --reference-root "$WORK/data/idforge_ref" \
  --bert-model "$WORK/models/bert-base-uncased" --ast-checkpoint "$WORK/models/ast" \
  --batch-size 64 --num-workers 4 --metric-implementation reference_apmeter \
  --visible-bits-only --output "$RUN/test_results.json"
