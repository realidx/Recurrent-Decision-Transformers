#!/bin/bash
# Run all UT-GCDT experiments on OGBench

# Configuration
DATASET="antmaze-medium-stitch-v0"
 #OUTPUT_DIR="./outputs"
SEEDS="42 123 456"

echo "========================================"
echo "UT-GCDT Experiments on OGBench"
echo "Dataset: $DATASET"
echo "========================================"

# Create output directory
# mkdir -p $OUTPUT_DIR

# Experiment 1: GCDT Baseline (untied layers, no plan token)
echo ""
echo "Running GCDT Baseline..."
for seed in $SEEDS; do
    echo "  Seed: $seed"
    python scripts/train.py \
        --config gcdt_baseline \
        --dataset $DATASET \
        --seed $seed \
        # --output_dir $OUTPUT_DIR
done

# Experiment 2: UT-GCDT (weight tying only)
echo ""
echo "Running UT-GCDT (weight tying only)..."
for seed in $SEEDS; do
    echo "  Seed: $seed"
    python scripts/train.py \
        --config ut_gcdt \
        --dataset $DATASET \
        --seed $seed \
        # --output_dir $OUTPUT_DIR
done

# Experiment 3: UT-GCDT + Plan Token
echo ""
echo "Running UT-GCDT + Plan Token..."
for seed in $SEEDS; do
    echo "  Seed: $seed"
    python scripts/train.py \
        --config ut_gcdt_plan \
        --dataset $DATASET \
        --seed $seed \
        # --output_dir $OUTPUT_DIR
done

# Experiment 4: UT-GCDT Full (Plan Token + Waypoint Loss)
echo ""
echo "Running UT-GCDT Full..."
for seed in $SEEDS; do
    echo "  Seed: $seed"
    python scripts/train.py \
        --config ut_gcdt_full \
        --dataset $DATASET \
        --seed $seed \
        # --output_dir $OUTPUT_DIR
done

echo ""
echo "========================================"
echo "All experiments complete!"
# echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

# Test-time scaling experiments (run after training)
echo ""
echo "To run test-time scaling experiments:"
echo "  python scripts/eval.py --checkpoint <path> --test_iterations 2 4 6 8"