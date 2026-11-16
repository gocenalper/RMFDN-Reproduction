#!/bin/bash
#SBATCH --job-name=mask_vt_va
#SBATCH --account=<account>
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/scratch/<account>/<user>/runs/mask_ablation_90kcos_12k/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/<account>/<user>/runs/mask_ablation_90kcos_12k/logs/%x-%j.err

set -euo pipefail

SCRATCH=/gpfs/scratch/<account>/<user>
CODE=$SCRATCH/code/idforge-clean
ENV_ROOT=$SCRATCH/env
DATA=$SCRATCH/data/idforge_root
REFERENCE=$SCRATCH/data/idforge_ref54/ref
OUTPUT_ROOT=$SCRATCH/runs/mask_ablation_90kcos_12k
MAX_ITERS=${MAX_ITERS:-12000}
LR_SCHEDULE_ITERS=${LR_SCHEDULE_ITERS:-90000}
CELLS=(VT VA)
TRAIN_PID=""

requeue_before_timeout() {
  echo "===== REQUEUE job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} $(date) ====="
  if [ -n "$TRAIN_PID" ]; then
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_before_timeout USR1

module load singularity
cd "$CODE"

for MOD in "${CELLS[@]}"; do
  RUN=$OUTPUT_ROOT/masked/$MOD
  mkdir -p "$RUN" "$OUTPUT_ROOT/logs"
  printf '{"cells":[{"cross_modal":"m1","identity":"i1","protocol":"p1","modalities":["%s"]}]}' \
    "$MOD" > "$RUN/cfg.json"

  echo "===== START masked $MOD full-val max/schedule=$MAX_ITERS/$LR_SCHEDULE_ITERS restart=${SLURM_RESTART_COUNT:-0} $(date) ====="
  singularity exec --nv --bind /gpfs:/gpfs \
    --env PYTHONPATH="$ENV_ROOT/pylibs:$CODE" \
    --env HF_HOME="$ENV_ROOT/models/hf_cache" \
    --env TORCH_HOME="$ENV_ROOT/models/torch_cache" \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    "$ENV_ROOT/pytorch.sif" python3 scripts/run_experiment_suite.py \
      --config "$RUN/cfg.json" \
      --data-root "$DATA" \
      --reference-root "$REFERENCE" \
      --output-dir "$RUN" \
      --bert-model "$ENV_ROOT/models/bert-base-uncased" \
      --ast-checkpoint "$ENV_ROOT/models/ast" \
      --iters "$MAX_ITERS" \
      --lr-schedule-iters "$LR_SCHEDULE_ITERS" \
      --eval-interval 1500 \
      --patience 8 \
      --batch-size 64 \
      --grad-accum-steps 2 \
      --num-workers 16 \
      --skip-test-eval \
      --mlc-loss bce \
      --mask-invisible-bits &
  TRAIN_PID=$!
  wait "$TRAIN_PID"
  TRAIN_PID=""
  echo "===== DONE masked $MOD $(date) ====="
done
