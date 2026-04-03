#!/usr/bin/env bash
# run_post_training_pipeline.sh — Standardized post-training pipeline.
#
# Fast tier (default, ~5-10 min):
#   1. Float eval (eval_bpb)
#   2. Sensitivity analysis (quant_analysis.py)
#   3. Quantize + weight permutation
#   4. Compress with zpaq
#   5. Verify artifact (eval_bpb)
#
# Expensive tier (opt-in):
#   --polish: Gradient polish (~500 steps, ~12-30 min)
#   --corrections: PTQ corrections (~10-20 min)
#
# Usage:
#   # Fast (default):
#   ./scripts/run_post_training_pipeline.sh logs/checkpoint_best.npz
#
#   # With polish:
#   ./scripts/run_post_training_pipeline.sh logs/checkpoint_best.npz --polish
#
#   # Full pipeline:
#   ./scripts/run_post_training_pipeline.sh logs/checkpoint_best.npz --polish --corrections
#
# Environment: set model arch env vars (NUM_LAYERS, MLP_MULT, etc.) before running.
# QUANT_BITS env var controls mixed-precision allocation.

set -euo pipefail

CHECKPOINT="${1:?Usage: $0 <checkpoint.npz> [--polish] [--corrections]}"
shift

DO_POLISH=0
DO_CORRECTIONS=0
POLISH_STEPS=500
POLISH_LR="1e-4"
CORRECTION_LAYERS=""
COMPRESSOR="zpaq"
BATCH_SEQS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --polish) DO_POLISH=1; shift ;;
        --corrections) DO_CORRECTIONS=1; shift ;;
        --polish-steps) POLISH_STEPS="$2"; shift 2 ;;
        --polish-lr) POLISH_LR="$2"; shift 2 ;;
        --correction-layers) CORRECTION_LAYERS="$2"; shift 2 ;;
        --compressor) COMPRESSOR="$2"; shift 2 ;;
        --batch-seqs) BATCH_SEQS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PYTHON="${PYTHON:-.venv/bin/python3}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASENAME=$(basename "$CHECKPOINT" .npz)
LOG_DIR="logs/pipeline_${BASENAME}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=== Post-Training Pipeline ==="
echo "Checkpoint: $CHECKPOINT"
echo "Polish: $DO_POLISH  Corrections: $DO_CORRECTIONS"
echo "Compressor: $COMPRESSOR"
echo "Log dir: $LOG_DIR"
echo ""

# Step 1: Float eval
echo "--- Step 1: Float BPB eval ---"
$PYTHON -c "
import sys; sys.path.insert(0, '.')
from scripts.eval_commons import build_model, eval_bpb
from train_gpt_mlx import Hyperparameters
import mlx.core as mx
from mlx.utils import tree_unflatten

hparams = Hyperparameters()
model = build_model(hparams)
flat = dict(mx.load('$CHECKPOINT'))
model.update(tree_unflatten(list(flat.items())))
mx.eval(model.parameters())
val_loss, val_bpb = eval_bpb(model, hparams)
print(f'Float: val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}')
" 2>&1 | tee "$LOG_DIR/01_float_eval.txt"
echo ""

# Step 2: Sensitivity analysis
echo "--- Step 2: Sensitivity analysis ---"
$PYTHON scripts/quant_analysis.py "$CHECKPOINT" \
    --save-analysis "$LOG_DIR/analysis.json" \
    --skip-compound \
    2>&1 | tee "$LOG_DIR/02_sensitivity.txt"
echo ""

# Step 3 (optional): Gradient polish
ARTIFACT_CHECKPOINT="$CHECKPOINT"
if [[ $DO_POLISH -eq 1 ]]; then
    echo "--- Step 3: Gradient polish ($POLISH_STEPS steps) ---"
    $PYTHON scripts/float_polish.py "$CHECKPOINT" \
        --steps "$POLISH_STEPS" --lr "$POLISH_LR" \
        --batch-seqs "$BATCH_SEQS" \
        --log-file "$LOG_DIR/03_polish.txt" \
        2>&1 | tee -a "$LOG_DIR/03_polish.txt"
    # Find the polished checkpoint
    POLISHED=$(ls -t logs/*polished*.npz 2>/dev/null | head -1 || true)
    if [[ -n "$POLISHED" ]]; then
        ARTIFACT_CHECKPOINT="$POLISHED"
        echo "Using polished checkpoint: $ARTIFACT_CHECKPOINT"
    else
        echo "WARNING: No polished checkpoint found, using original"
    fi
    echo ""
fi

# Step 4: Build artifact (quantize + compress)
echo "--- Step 4: Build artifact ---"
BUILD_ARGS="--compressor $COMPRESSOR"
if [[ $DO_CORRECTIONS -eq 1 ]]; then
    if [[ -n "$CORRECTION_LAYERS" ]]; then
        BUILD_ARGS="$BUILD_ARGS --correction-layers $CORRECTION_LAYERS"
    fi
else
    BUILD_ARGS="$BUILD_ARGS --no-correction"
fi
$PYTHON scripts/build_artifact.py "$ARTIFACT_CHECKPOINT" $BUILD_ARGS \
    2>&1 | tee "$LOG_DIR/04_build_artifact.txt"
echo ""

# Step 5: Verify artifact
ARTIFACT=$(ls -t *.zpaq *.br *.zs 2>/dev/null | head -1 || true)
if [[ -n "$ARTIFACT" ]]; then
    echo "--- Step 5: Verify artifact ---"
    $PYTHON scripts/verify_artifact.py "$ARTIFACT" \
        --float-checkpoint "$CHECKPOINT" \
        2>&1 | tee "$LOG_DIR/05_verify.txt"
else
    echo "WARNING: No artifact found to verify"
fi

echo ""
echo "=== Pipeline complete ==="
echo "Logs: $LOG_DIR/"
