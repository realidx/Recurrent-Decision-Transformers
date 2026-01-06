#!/bin/bash
# Run all U-GCDT experiments on OGBench
# Following the "Steel-Manned" Protocol for rigorous scientific comparison

# Configuration
# Primary: antmaze-large-stitch-v0 (The Crucible - hardest stitching task per protocol)
# Contingency: antmaze-medium-play-v0 or kitchen-mixed-v0 if standard DT fails
DATASET="antmaze-large-stitch-v0"
OUTPUT_DIR="./outputs"
SEEDS="42 123 456"

echo "========================================"
echo "U-GCDT Experiments on OGBench"
echo "Dataset: $DATASET"
echo "Following Steel-Manned Protocol"
echo "========================================"

# Create output directory
mkdir -p $OUTPUT_DIR

# ============================================
# Phase 1: Establish Baselines (Day 1-3)
# ============================================

# Step 0: Run HIQL baseline (target score)
echo ""
echo "[Phase 1] Running HIQL baseline (target score)..."
for seed in $SEEDS; do
    echo "  Seed: $seed"
    python scripts/run_hiql_baseline.py \
        --dataset $DATASET \
        --seed $seed \
        --output_dir $OUTPUT_DIR/baselines
done

# Step 1: Run GCBC (floor score)
echo ""
echo "[Phase 1] Running GCBC baseline (floor score)..."
for seed in $SEEDS; do
    echo "  Seed: $seed"
    python scripts/run_gcbc_baseline.py \
        --dataset $DATASET \
        --seed $seed \
        --output_dir $OUTPUT_DIR/baselines 2>/dev/null || echo "  GCBC script not found, skipping..."
done

# Step 2: Stacked GCDT Baseline (Independent Heads + Deep Supervision)
# Per protocol: "Each layer l has its own projection head H_l to predict actions"
echo ""
echo "[Phase 1] Running Stacked GCDT Baseline (Independent Heads)..."
for seed in $SEEDS; do
    echo "  Seed: $seed"
    python scripts/train.py \
        --config gcdt_baseline \
        --dataset $DATASET \
        --seed $seed \
        --output_dir $OUTPUT_DIR
done

# ============================================
# Phase 2: U-GCDT Experiments
# ============================================

# Experiment A: U-GCDT with weight tying only (Shared Head)
# Per protocol: "One head H is used for all steps k"
echo ""
echo "[Phase 2] Running U-GCDT (weight tying, shared head)..."
for seed in $SEEDS; do
    echo "  Seed: $seed"
    python scripts/train.py \
        --config ut_gcdt \
        --dataset $DATASET \
        --seed $seed \
        --output_dir $OUTPUT_DIR
done

# Experiment B: U-GCDT + Plan Token
echo ""
echo "[Phase 2] Running U-GCDT + Plan Token..."
for seed in $SEEDS; do
    echo "  Seed: $seed"
    python scripts/train.py \
        --config ut_gcdt_plan \
        --dataset $DATASET \
        --seed $seed \
        --output_dir $OUTPUT_DIR
done

# Experiment C: U-GCDT Full (Plan Token + Waypoint Loss + Deep Supervision)
echo ""
echo "[Phase 2] Running U-GCDT Full..."
for seed in $SEEDS; do
    echo "  Seed: $seed"
    python scripts/train.py \
        --config ut_gcdt_full \
        --dataset $DATASET \
        --seed $seed \
        --output_dir $OUTPUT_DIR
done

echo ""
echo "========================================"
echo "Training experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

# ============================================
# Phase 3: Test-Time Scaling Experiments
# ============================================
echo ""
echo "[Phase 3] Test-Time Scaling Experiments"
echo "Train with K=4, test with K'=2,4,6,8,12"
echo ""
echo "Run after training completes:"
echo "  python scripts/eval.py --checkpoint outputs/ut_gcdt_full_*/best_* --test_iterations 2 4 6 8 12"
echo ""
echo "Expected result: Curve should rise steeply initially and continue rising or plateau (not crash) beyond K=4"

# ============================================
# Realizability Check (Go/No-Go)
# ============================================
echo ""
echo "========================================"
echo "REALIZABILITY CHECK"
echo "========================================"
echo "After Phase 1 completes, verify:"
echo "  1. HIQL score = Target to approach"
echo "  2. GCBC score = Floor to beat"
echo "  3. Stacked GCDT > GCBC? "
echo "     YES -> Proceed to Phase 2 (U-GCDT)"
echo "     NO  -> Switch loss from MSE to IQL-style Expectile Loss"
echo "========================================"