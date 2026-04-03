#!/usr/bin/env bash
# Overnight gradient polish experiment suite
# Run from project root: bash scripts/overnight_polish.sh
# Estimated: ~3-4 hours total on M4 Pro
set -euo pipefail

PYTHON=".venv/bin/python3"
CKPT="logs/warmdown_11L_45x_best.npz"
SCRIPT="scripts/float_polish.py"

# Common model env vars
export NUM_LAYERS=11 MLP_MULT=4.5 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16
export BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=128 LRELU_SLOPE=0.5
export LOGIT_SOFTCAP=30 QK_GAIN_INIT=1.5 NUM_KV_HEADS=4

# Mixed precision QUANT_BITS strings
BITS_A="attn.0:5,mlp.0:5,attn.4:5,mlp.4:5,attn.9:5,mlp.9:5,attn.1:4,mlp.1:4,attn.2:4,mlp.2:4,attn.3:4,mlp.3:4,attn.5:3,mlp.5:3,attn.6:3,mlp.6:3,attn.7:3,mlp.7:3,attn.8:3,mlp.8:3,attn.10:3,mlp.10:3,attn:4,mlp:4"
BITS_B="attn.0:5,mlp.0:5,attn.4:4,mlp.4:4,attn.9:4,mlp.9:4,attn.1:3,mlp.1:3,attn.2:3,mlp.2:3,attn.3:3,mlp.3:3,attn.5:3,mlp.5:3,attn.6:3,mlp.6:3,attn.7:3,mlp.7:3,attn.8:3,mlp.8:3,attn.10:2,mlp.10:2,attn:3,mlp:3"
BITS_C="attn.0:4,mlp.0:4,attn.4:3,mlp.4:3,attn.9:3,mlp.9:3,attn.1:3,mlp.1:3,attn.2:3,mlp.2:3,attn.3:3,mlp.3:3,attn.5:2,mlp.5:2,attn.6:2,mlp.6:2,attn.7:2,mlp.7:2,attn.8:2,mlp.8:2,attn.10:2,mlp.10:2,attn:3,mlp:3"

run_exp() {
    local name="$1"; shift
    echo ""
    echo "============================================================"
    echo "  $name — $(date)"
    echo "============================================================"
    if "$PYTHON" "$SCRIPT" "$CKPT" "$@"; then
        echo "  ✓ $name completed at $(date)"
    else
        echo "  ✗ $name FAILED (exit $?) at $(date)"
    fi
}

echo "Overnight polish sweep started at $(date)"
echo "Checkpoint: $CKPT"
echo ""

# --- Exp 2: Uniform int4 polish (sanity check) ---
QUANT_ATTN_BITS=4 QUANT_MLP_BITS=4 \
run_exp "Exp2: uniform int4, 500s, lr=1e-4" \
    --steps 500 --lr 1e-4 --log-file logs/polish_uniform_int4_500s.txt

# --- Exp 3: Uniform int3 polish (sanity check) ---
QUANT_ATTN_BITS=3 QUANT_MLP_BITS=3 \
run_exp "Exp3: uniform int3, 500s, lr=1e-4" \
    --steps 500 --lr 1e-4 --log-file logs/polish_uniform_int3_500s.txt

# --- Exp 4a: Mixed Config A (conservative), 500 steps ---
QUANT_BITS="$BITS_A" \
run_exp "Exp4a: Config A (conservative), 500s" \
    --steps 500 --lr 1e-4 --log-file logs/polish_mixed_A_500s.txt

# --- Exp 4a-long: Config A, 1000 steps ---
QUANT_BITS="$BITS_A" \
run_exp "Exp4a-long: Config A (conservative), 1000s" \
    --steps 1000 --lr 1e-4 --log-file logs/polish_mixed_A_1000s.txt

# --- Exp 4b: Mixed Config B (aggressive), 1000 steps ---
QUANT_BITS="$BITS_B" \
run_exp "Exp4b: Config B (aggressive), 1000s" \
    --steps 1000 --lr 1e-4 --log-file logs/polish_mixed_B_1000s.txt

# --- Exp 4c: Mixed Config C (max compression), 1000 steps ---
QUANT_BITS="$BITS_C" \
run_exp "Exp4c: Config C (max compression), 1000s" \
    --steps 1000 --lr 1e-4 --log-file logs/polish_mixed_C_1000s.txt

# --- Exp 5: LR sweep on Config A ---
QUANT_BITS="$BITS_A" \
run_exp "Exp5a: Config A, lr=3e-5" \
    --steps 500 --lr 3e-5 --log-file logs/polish_mixed_A_500s_lr3e5.txt

QUANT_BITS="$BITS_A" \
run_exp "Exp5b: Config A, lr=3e-4" \
    --steps 500 --lr 3e-4 --log-file logs/polish_mixed_A_500s_lr3e4.txt

QUANT_BITS="$BITS_A" \
run_exp "Exp5c: Config A, lr=1e-3" \
    --steps 500 --lr 1e-3 --log-file logs/polish_mixed_A_500s_lr1e3.txt

# --- Exp 6: Config A with --filter ---
QUANT_BITS="$BITS_A" \
run_exp "Exp6a: Config A, MLP only" \
    --steps 500 --lr 1e-4 --filter "mlp" --log-file logs/polish_mixed_A_500s_mlp_only.txt

QUANT_BITS="$BITS_A" \
run_exp "Exp6b: Config A, int3 layers only (L5,6,7,8,10)" \
    --steps 500 --lr 1e-4 --filter 'blocks\.(5|6|7|8|10)\.' --log-file logs/polish_mixed_A_500s_int3_layers.txt

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE — $(date)"
echo "============================================================"
echo ""

# Collect summary from all log files
echo "=== RESULTS SUMMARY ==="
for f in logs/polish_*.txt; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .txt)
        recovery=$(grep -o "Quant gap recovery: [0-9.-]*%" "$f" 2>/dev/null || echo "N/A")
        baseline=$(grep "Baseline quant loss:" "$f" 2>/dev/null | grep -o "[0-9.]*" || echo "N/A")
        polished=$(grep "Polished quant loss:" "$f" 2>/dev/null | grep -o "[0-9.]*" || echo "N/A")
        echo "$name | baseline=$baseline | polished=$polished | $recovery"
    fi
done
