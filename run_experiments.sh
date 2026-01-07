#!/bin/bash
# Run all U-GCDT experiments on OGBench
# Following the "Steel-Manned" Protocol for rigorous scientific comparison

# Configuration
# Primary: antmaze-large-stitch-v0 (The Crucible - hardest stitching task per protocol)
# Contingency: antmaze-medium-play-v0 or kitchen-mixed-v0 if standard DT fails
DATASET="${DATASET:-antmaze-large-stitch-v0}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
SEEDS="${SEEDS:-42 123 456}"
SKIP_BASELINES="${SKIP_BASELINES:-0}"

# ============================================
# FAST_MODE: For quick iteration and debugging
# ============================================
# Set FAST_MODE=1 to run with reduced settings:
#   - 25k steps instead of 100k (4x faster training)
#   - 20 eval episodes instead of 100 (5x faster eval)
#   - 1 seed instead of 3 (3x fewer runs)
# Estimated time: ~1h per model instead of ~4h
FAST_MODE="${FAST_MODE:-0}"

if [ "$FAST_MODE" -eq 1 ]; then
    echo ">>> FAST_MODE enabled - using reduced settings for quick iteration <<<"
    export MAX_STEPS="${MAX_STEPS:-25000}"
    export EVAL_EPISODES="${EVAL_EPISODES:-20}"
    export EVAL_EVERY="${EVAL_EVERY:-5000}"
    export SAVE_EVERY="${SAVE_EVERY:-10000}"
    SEEDS="${SEEDS:-42}"  # Single seed in fast mode
fi

# Allow explicit overrides even without FAST_MODE
export MAX_STEPS="${MAX_STEPS:-}"
export EVAL_EPISODES="${EVAL_EPISODES:-}"
export EVAL_EVERY="${EVAL_EVERY:-}"
export SAVE_EVERY="${SAVE_EVERY:-}"

echo "========================================"
echo "U-GCDT Experiments on OGBench"
echo "Dataset: $DATASET"
echo "Following Steel-Manned Protocol"
echo "========================================"
echo "Configuration:"
echo "  Seeds: $SEEDS"
echo "  Output: $OUTPUT_DIR"
if [ -n "$MAX_STEPS" ]; then echo "  Max steps: $MAX_STEPS"; fi
if [ -n "$EVAL_EPISODES" ]; then echo "  Eval episodes: $EVAL_EPISODES"; fi
if [ -n "$EVAL_EVERY" ]; then echo "  Eval every: $EVAL_EVERY"; fi
echo "========================================"

# Create output directory
mkdir -p $OUTPUT_DIR

# ============================================
# Phase 1: Establish Baselines (Day 1-3)
# ============================================

if [ "$SKIP_BASELINES" -ne 1 ]; then
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
else
    echo ""
    echo "[Phase 1] Skipping HIQL/GCBC baselines (SKIP_BASELINES=1)"
fi

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
# Phase 3: Evaluation and Test-Time Scaling
# ============================================
echo ""
echo "[Phase 3] Evaluation"
echo ""
echo "========================================"
echo "MODEL COMPARISON (run after training)"
echo "========================================"
echo "Compare all trained models to understand ablation results:"
echo ""
echo "1. Standard evaluation (compare all models):"
echo "   for model in gcdt_baseline ut_gcdt_tied ut_gcdt_plan ut_gcdt_full; do"
echo "     python scripts/eval.py --checkpoint outputs/\${model}_*/best_* --test_iterations 4"
echo "   done"
echo ""
echo "2. Test-time scaling (UT models only - can use more iterations at test time):"
echo "   python scripts/eval.py --checkpoint outputs/ut_gcdt_tied_*/best_* --test_iterations 2 4 6 8 12"
echo "   python scripts/eval.py --checkpoint outputs/ut_gcdt_plan_*/best_* --test_iterations 2 4 6 8 12"
echo "   python scripts/eval.py --checkpoint outputs/ut_gcdt_full_*/best_* --test_iterations 2 4 6 8 12"
echo ""
echo "Expected results:"
echo "  - ut_gcdt_* should benefit from more iterations (curve rises then plateaus)"
echo "  - gcdt_baseline cannot do test-time scaling (fixed architecture)"
echo ""
echo "Ablation questions answered by comparing models:"
echo "  Q1: Does weight tying help?     -> Compare gcdt_baseline vs ut_gcdt_tied"
echo "  Q2: Does plan token help?       -> Compare ut_gcdt_tied vs ut_gcdt_plan"
echo "  Q3: Does waypoint loss help?    -> Compare ut_gcdt_plan vs ut_gcdt_full"
echo "  Q4: Does test-time scaling work? -> Compare ut_* at K=4 vs K=8"
echo "========================================"

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

# ============================================
# Tips for faster iteration
# ============================================
echo ""
echo "========================================"
echo "TIPS FOR FASTER ITERATION"
echo "========================================"
echo "If experiments are taking too long:"
echo ""
echo "1. Use FAST_MODE for quick validation:"
echo "   FAST_MODE=1 ./run_experiments.sh"
echo ""
echo "2. Run single models with custom settings:"
echo "   MAX_STEPS=25000 EVAL_EPISODES=20 python scripts/train.py --config ut_gcdt --seed 42"
echo ""
echo "3. Skip baselines if already computed:"
echo "   SKIP_BASELINES=1 ./run_experiments.sh"
echo ""
echo "4. Use a single seed for initial exploration:"
echo "   SEEDS=42 ./run_experiments.sh"
echo ""
echo "Note: 'best_' checkpoints are only saved when success_rate improves."
echo "If no best checkpoint exists, the model may need more training time"
echo "or the task is too difficult for the current architecture."
echo "========================================"
