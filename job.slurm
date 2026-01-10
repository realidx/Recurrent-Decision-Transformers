#!/bin/bash
#SBATCH --job-name=ut-gcdt
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2

set -euo pipefail

# Optional: load modules / activate env
# module purge
# module load cuda/12.2
# module load cudnn/9.1
if [ -n "${CONDA_BASE:-}" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "Conda not found. Set CONDA_BASE or load your environment modules."
  exit 1
fi
conda activate "${CONDA_ENV:-mytrain}"

# JAX runtime knobs - enable for better memory efficiency
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

mkdir -p outputs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$SUBMIT_DIR/outputs}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"

export DATASET="${DATASET:-ant_Umaze_FAST}"
export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$DATASET}"
export OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

mkdir -p "$OUTPUT_DIR"

export SEEDS="${SEEDS:-42}"
export SKIP_BASELINE="${SKIP_BASELINE:-1}"
export LOG_EVERY="${LOG_EVERY:-1000}"
export EVAL_EVERY="${EVAL_EVERY:-10000}"
export SAVE_EVERY="${SAVE_EVERY:-50000}"

# FAST_MODE: Set to 1 for quick iteration (~1h/model instead of ~4h)
# Uses: 25k steps (vs 100k), 20 eval episodes (vs 100), single seed
export FAST_MODE="${FAST_MODE:-0}"

# Additional overrides for fine-grained control
export MAX_STEPS="${MAX_STEPS:-}"
export EVAL_EPISODES="${EVAL_EPISODES:-}"


bash run_experiments_v3.sh

echo "All experiments complete."
