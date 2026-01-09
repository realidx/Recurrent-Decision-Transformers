#!/bin/bash
# V3 Protocol: U-GCDT Experiments on D4RL
# Following the "Steel-Manned" Protocol with Gated Recurrence

set -e

# ============================================
# V3 Protocol Configuration
# ============================================
# Task progression:
#   1. antmaze-umaze-v2 (sanity check, target: >50% success)
#   2. antmaze-medium-play-v2 (primary stitching, target: beat baseline by >10%)
#   3. kitchen-mixed-v0 (secondary, long-horizon manipulation)

TASK="${TASK:-antmaze-umaze-v2}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
SEEDS="${SEEDS:-42 123 456}"
SKIP_BASELINE="${SKIP_BASELINE:-1}"

# V3 Architecture: L=12 layers, K=12 iterations
# Gated recurrence enabled for U-GCDT

# ============================================
# FAST_MODE: For quick iteration and debugging
# ============================================
FAST_MODE="${FAST_MODE:-1}"

if [ "$FAST_MODE" -eq 1 ]; then
    echo ">>> FAST_MODE enabled - reduced settings for quick iteration <<<"
    export MAX_STEPS="${MAX_STEPS:-30000}"
    export EVAL_EPISODES="${EVAL_EPISODES:-20}"
    export EVAL_EVERY="${EVAL_EVERY:-10000}"
    export SAVE_EVERY="${SAVE_EVERY:-10000}"
    export SEEDS="${SEEDS:-42}"
fi

# Export environment overrides
export MAX_STEPS="${MAX_STEPS:-}"
export EVAL_EPISODES="${EVAL_EPISODES:-}"
export EVAL_EVERY="${EVAL_EVERY:-}"
export SAVE_EVERY="${SAVE_EVERY:-}"

echo "========================================"
echo "V3 Protocol: U-GCDT on D4RL"
echo "========================================"
echo "Task: $TASK"
echo "Seeds: $SEEDS"
echo "Output: $OUTPUT_DIR"
if [ -n "$MAX_STEPS" ]; then echo "Max steps: $MAX_STEPS"; fi
if [ -n "$EVAL_EPISODES" ]; then echo "Eval episodes: $EVAL_EPISODES"; fi
echo "========================================"

mkdir -p "$OUTPUT_DIR"

# ============================================
# Phase 1: Steel-Manned Baseline (L=12 layers)
# ============================================
# Architecture: Standard GPT-2 style stack
# - L=12 layers with independent action heads
# - Loss: Sum of MSE from all heads
# - Eval: Only use Head_12 output

if [ "$SKIP_BASELINE" -ne 1 ]; then
    echo ""
    echo "[Phase 1] Training Steel-Manned Baseline (L=12 layers, independent heads)"
    echo "Loss: sum_{l=1}^{12} MSE(Head_l(x_l), a_true)"
    echo ""

    for seed in $SEEDS; do
        echo "  Seed: $seed"
        python scripts/train.py \
            --config gcdt_baseline \
            --dataset "$TASK" \
            --seed "$seed" \
            --output_dir "$OUTPUT_DIR"
    done
else
    echo ""
    echo "[Phase 1] Skipping baseline (SKIP_BASELINE=1)"
fi

# ============================================
# Phase 2: U-GCDT with Gated Recurrence (K=12)
# ============================================
# Architecture: Single Transformer Block looped K=12 times
# - Gated update: H_{k+1} = (1-g) * H_k + g * Block(H_k)
# - Sinusoidal step embeddings (no learned params)
# - Shared action head applied at every step k
# - Loss: Sum of MSE from shared head at each step

echo ""
echo "[Phase 2] Training U-GCDT with Gated Recurrence (K=12)"
echo "Gated update: H_{k+1} = (1-g) * H_k + g * Block(H_k)"
echo ""

for seed in $SEEDS; do
    echo "  Seed: $seed"
    python scripts/train.py \
        --config ut_gcdt_gated \
        --dataset "$TASK" \
        --seed "$seed" \
        --output_dir "$OUTPUT_DIR"
done

# ============================================
# Phase 2b (Optional): U-GCDT Full (with waypoint loss)
# ============================================
echo ""
echo "[Phase 2b] Training U-GCDT Gated Full (+ waypoint loss)"
echo ""

for seed in $SEEDS; do
    echo "  Seed: $seed"
    python scripts/train.py \
        --config ut_gcdt_gated_full \
        --dataset "$TASK" \
        --seed "$seed" \
        --output_dir "$OUTPUT_DIR"
done

echo ""
echo "========================================"
echo "Training complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

# ============================================
# Phase 3: Evaluation Instructions
# ============================================
echo ""
echo "========================================"
echo "EVALUATION (run after training)"
echo "========================================"
echo ""
echo "1. Compare baseline vs U-GCDT:"
echo "   python scripts/eval.py --checkpoint $OUTPUT_DIR/gcdt_baseline_*/best_* --test_iterations 12"
echo "   python scripts/eval.py --checkpoint $OUTPUT_DIR/ut_gcdt_gated_*/best_* --test_iterations 12"
echo ""
echo "2. Test-time scaling (U-GCDT only):"
echo "   python scripts/eval.py --checkpoint $OUTPUT_DIR/ut_gcdt_gated_*/best_* --test_iterations 6 12 18 24"
echo ""
echo "========================================"

# ============================================
# Go/No-Go Decision (V3 Protocol)
# ============================================
echo ""
echo "========================================"
echo "V3 PROTOCOL DECISION RULES"
echo "========================================"
echo ""
echo "Step 1: Umaze Check (antmaze-umaze-v2)"
echo "  - Success > 20%: PROCEED to medium-play"
echo "  - Success < 20%: Switch to IQL-Expectile loss and re-run"
echo ""
echo "Step 2: Medium Comparison (antmaze-medium-play-v2)"
echo "  - Target: U-GCDT beats baseline by >10%"
echo "  - Standard DTs typically get 0-20% on this task"
echo ""
echo "Step 3: Kitchen (kitchen-mixed-v0)"
echo "  - Long-horizon sequential manipulation"
echo "  - Tests compositional reasoning"
echo "========================================"
