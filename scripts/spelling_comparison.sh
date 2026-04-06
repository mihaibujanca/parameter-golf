#!/usr/bin/env bash
# spelling_comparison.sh — 1000-step comparative battery for fast spelling
# acquisition hypotheses H1 (aux encoder loss), H2 (continuation loss
# upweighting), and H4 (encoder depth).
#
# Each variant runs ~15 min on workhorse (M2 Max). Fixed seed so architectural
# differences are the only moving parts. Analyze each run afterwards with
#   python -m pgolf.analyze <run_id>
# and compare end-of-run val_bpb across variants.
#
# Baseline here is 10L/640d/3x — matches the "standard M4 local config" block
# in CLAUDE.md so times are predictable. H4 (encoder depth) is a pure env-var
# flip, no code changes.

set -euo pipefail

cd "$(dirname "$0")/.."

# Shared config — keep identical across every run so only the hypothesis
# variable moves.
export SEED=1337
export ITERATIONS=1000
export WARMDOWN_ITERS=0
export VAL_LOSS_EVERY=100
export TRAIN_LOG_EVERY=50
export TRAIN_BATCH_TOKENS=16384
export GRAD_ACCUM_STEPS=2
export MLX_MAX_MICROBATCH_TOKENS=8192
export MAX_WALLCLOCK_SECONDS=0
export VAL_BATCH_SIZE=65536
export VAL_MAX_TOKENS=1048576
export NUM_LAYERS=10
export MODEL_DIM=640
export MLP_MULT=3
export SWA_ENABLED=0
export CHECKPOINT_EVERY=0

PY=.venv/bin/python3

run_variant() {
    local run_id=$1
    shift
    echo "========================================"
    echo "[spelling_comparison] launching: $run_id"
    echo "  extra env: $*"
    echo "========================================"
    env RUN_ID="$run_id" "$@" "$PY" train_gpt_mlx.py 2>&1 | tee "logs/${run_id}.console.log"
    "$PY" -m pgolf.analyze "$run_id" --desc "spelling_comparison: $run_id" || true
}

# --- Baseline (no modifications) ---
run_variant spelling_baseline_1k

# --- H1: auxiliary encoder loss ---
run_variant spelling_h1_aux015_1k AUX_LOSS_WEIGHT=0.15
run_variant spelling_h1_aux030_1k AUX_LOSS_WEIGHT=0.3

# --- H2: continuation loss upweighting ---
run_variant spelling_h2_cont2x_1k CONTINUATION_LOSS_WEIGHT=2.0

# --- H4: encoder depth (zero code changes) ---
run_variant spelling_h4_enc7_1k NUM_ENCODER_LAYERS=7
run_variant spelling_h4_enc8_1k NUM_ENCODER_LAYERS=8

echo "========================================"
echo "[spelling_comparison] done. Compare runs:"
echo "  $PY -m pgolf.analyze --list"
echo "========================================"
