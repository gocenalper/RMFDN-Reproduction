#!/bin/bash
#SBATCH --job-name=c0cell
#SBATCH --account=<account>
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=20 --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scratch/<account>/<user>/runs_c0/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/<account>/<user>/runs_c0/logs/%x-%j.err
# C0 single-factor cell. Same recipe as runs4f/fold_cell.slurm (P2, i3, 6k iters,
# 90k cosine horizon, eval 1500, patience 8, b64x2, BCE, mask-invisible-bits),
# generalised: EXP = plan id (run dir + plan file), EXTRA = the plan's flags,
# and the plan gate runs on the box before training.
set -uo pipefail
: "${EXP:?}" "${FOLD:?}" "${MOD:?}"
EXTRA="${EXTRA:-}"
ARCH="${ARCH:-m1}"
S=/gpfs/scratch/<account>/<user>
CODE=$S/code/idforge-final
E=$S/env
G=$S/runs_c0/f$FOLD
SPLIT=$S/folds/fold_F$FOLD.json
RUN=$G/$EXP
PLAN=$CODE/experiments/plans/$EXP.json
mkdir -p "$RUN" "$S/runs_c0/logs"
module load singularity
cd "$CODE"
sexec() { singularity exec --nv --bind /gpfs:/gpfs --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH="$E/pylibs:$CODE" --env HF_HOME="$E/models/hf_cache" \
  --env TORCH_HOME="$E/models/torch_cache" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  "$E/pytorch.sif" "$@"; }
python3 scripts/exp_plan_check.py check "$PLAN" || { echo "PLAN GATE FAILED: $PLAN"; exit 1; }
printf '{"cells":[{"cross_modal":"%s","identity":"i3","protocol":"p2","modalities":["%s"]}]}' "$ARCH" "$MOD" > "$RUN/cfg.json"
if [ ! -f "$RUN/DONE" ]; then
  for attempt in 1 2 3; do
    echo "===== START $EXP f$FOLD TRAIN attempt $attempt $(date) ====="
    set +e
    sexec python3 scripts/run_experiment_suite.py \
      --config "$RUN/cfg.json" --p2-split "$SPLIT" \
      --data-root "$S/data/idforge_root" --reference-root "$S/data/idforge_ref54/ref" --output-dir "$RUN" \
      --bert-model "$E/models/bert-base-uncased" --ast-checkpoint "$E/models/ast" \
      --iters "${ITERS:-6000}" --lr-schedule-iters "${LRH:-90000}" --eval-interval "${EVAL:-1500}" --patience 8 \
      --batch-size 64 --grad-accum-steps 2 --num-workers 16 --skip-test-eval \
      --mlc-loss bce --mask-invisible-bits $EXTRA
    set -e
    if [ -s "$RUN/results.json" ] && [ "$(wc -c < "$RUN/results.json")" -gt 100 ]; then
      touch "$RUN/DONE"; break
    fi
    echo "attempt $attempt dustu; 60s"; sleep 60
  done
fi
if [ -f "$RUN/DONE" ] && [ ! -f "$RUN/EVALDONE" ]; then
  sexec python3 scripts/run_test_eval.py \
    --phase1-dir "$RUN" --p2-split "$SPLIT" \
    --data-root "$S/data/idforge_root" --reference-root "$S/data/idforge_ref54/ref" \
    --bert-model "$E/models/bert-base-uncased" --ast-checkpoint "$E/models/ast" \
    --batch-size 64 --num-workers 4 --metric-implementation reference_apmeter \
    --visible-bits-only --dump-scores "$RUN/dumps" --output "$RUN/test_results.json"
  [ -s "$RUN/test_results.json" ] && touch "$RUN/EVALDONE"
fi
echo "CELL_DONE f$FOLD/$EXP"
