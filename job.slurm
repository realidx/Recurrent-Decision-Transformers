#!/bin/bash
#SBATCH --job-name=ut-gcdt
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

set -euo pipefail

# Optional: load modules / activate env
# module purge
# module load cuda/12.2
# module load cudnn/9.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mytrain

# JAX runtime knobs (optional)
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

mkdir -p outputs

DATASETS=("antmaze-medium-stitch-v0")
SEEDS=(42 123 456)
CONFIGS=("gcdt_baseline" "ut_gcdt" "ut_gcdt_plan" "ut_gcdt_full")

OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"

for dataset in "${DATASETS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "Running ${config} on ${dataset} (seed=${seed})"
      python scripts/train.py \
        --config "${config}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --output_dir "${OUTPUT_ROOT}/${dataset}"
    done
  done
done

echo "All experiments complete."
