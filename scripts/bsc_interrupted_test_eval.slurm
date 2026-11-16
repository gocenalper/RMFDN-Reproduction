#!/bin/bash
#SBATCH --job-name=old_test
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
#SBATCH --output=/gpfs/scratch/<account>/<user>/runs/clean_compare_eval_stage/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/<account>/<user>/runs/clean_compare_eval_stage/logs/%x-%j.err

set -euo pipefail

SCRATCH=/gpfs/scratch/<account>/<user>
CODE=$SCRATCH/code/idforge-clean
ENV_ROOT=$SCRATCH/env
STAGE=$SCRATCH/runs/clean_compare_eval_stage

requeue_before_timeout() {
  echo "===== REQUEUE job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} $(date) ====="
  scontrol requeue "$SLURM_JOB_ID"
}
trap requeue_before_timeout USR1

module load singularity
cd "$CODE"
echo "===== START interrupted checkpoint test restart=${SLURM_RESTART_COUNT:-0} $(date) ====="
singularity exec --nv --bind /gpfs:/gpfs \
  --env PYTHONPATH="$ENV_ROOT/pylibs:$CODE" \
  --env HF_HOME="$ENV_ROOT/models/hf_cache" \
  --env TORCH_HOME="$ENV_ROOT/models/torch_cache" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  "$ENV_ROOT/pytorch.sif" python3 scripts/run_test_eval.py \
    --phase1-dir "$STAGE" \
    --data-root "$SCRATCH/data/idforge_root" \
    --reference-root "$SCRATCH/data/idforge_ref54/ref" \
    --bert-model "$ENV_ROOT/models/bert-base-uncased" \
    --ast-checkpoint "$ENV_ROOT/models/ast" \
    --batch-size 64 \
    --num-workers 16 \
    --output "$STAGE/test_results.json"
echo "===== DONE interrupted checkpoint test $(date) ====="
