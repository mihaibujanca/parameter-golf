#!/bin/bash
# Launch script for 8xH100 RunPod submission
# Template: runpod/parameter-golf (https://console.runpod.io/hub/template/parameter-golf?id=y5cejece4j)
#
# SETUP (run once after pod starts):
#   git clone https://github.com/mihaibujanca/parameter-golf.git
#   cd parameter-golf
#   python3 data/cached_challenge_fineweb.py --variant sp1024
#   pip install flash-attn --no-build-isolation  # if FA3 not pre-installed
#   cp records/track_10min_16mb/2026-03-30_MixedPrec_PTQCorrection/train_gpt.py .
#
# IMPORTANT: Stop the pod immediately after runs complete to avoid idle charges.

set -e
cd /workspace/parameter-golf

# Phase 1: Smoke test (1 min wallclock cap, 1 seed)
# Uncomment to verify infra works before burning budget:
# MAX_WALLCLOCK_SECONDS=60 SEED=1337 \
#   torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/smoke_test.log

# Phase 2: Architecture sweep (1 seed each, full 600s)
# Config A: 11L/3x + XSA=11 (minimal change from baseline)
run_config_a() {
  NUM_LAYERS=11 MLP_MULT=3 MLP_ACT=lrelu2 LRELU_SLOPE=0.5 \
  XSA_LAST_N=11 ROPE_DIMS=16 LN_SCALE=1 \
  EMA_ENABLED=1 EMA_DECAY=0.997 \
  SWA_ENABLED=1 SWA_EVERY=50 \
  VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
  MUON_WD=0.04 ADAM_WD=0.04 \
  MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
  MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
  ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
  QUANT_BITS="attn:4,mlp:5" \
  NGRAM_CACHE=0 \
  RUN_ID=sweep_A_11L_3x SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/sweep_A.log
}

# Config B: 11L/4x + XSA=11 + slope=0.75
run_config_b() {
  NUM_LAYERS=11 MLP_MULT=4 MLP_ACT=lrelu2 LRELU_SLOPE=0.75 \
  XSA_LAST_N=11 ROPE_DIMS=16 LN_SCALE=1 \
  EMA_ENABLED=1 EMA_DECAY=0.997 \
  SWA_ENABLED=1 SWA_EVERY=50 \
  VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
  MUON_WD=0.04 ADAM_WD=0.04 \
  MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
  MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
  ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
  QUANT_BITS="attn:4,mlp:5,mlp.10:4" \
  NGRAM_CACHE=0 \
  RUN_ID=sweep_B_11L_4x SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/sweep_B.log
}

# Config C: 13L/3x + XSA=13 + slope=0.75
run_config_c() {
  NUM_LAYERS=13 MLP_MULT=3 MLP_ACT=lrelu2 LRELU_SLOPE=0.75 \
  XSA_LAST_N=13 ROPE_DIMS=16 LN_SCALE=1 \
  EMA_ENABLED=1 EMA_DECAY=0.997 \
  SWA_ENABLED=1 SWA_EVERY=50 \
  VE_ENABLED=1 VE_DIM=128 VE_LAYERS=11,12 \
  MUON_WD=0.04 ADAM_WD=0.04 \
  MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
  MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
  ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
  QUANT_BITS="attn:4,mlp:5" \
  NGRAM_CACHE=0 \
  RUN_ID=sweep_C_13L_3x SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/sweep_C.log
}

# Phase 4: Three-seed validation (run with winning config)
run_3seed() {
  CONFIG=$1  # "A", "B", or "C"
  for SEED in 42 1337 2024; do
    echo "=== Running $CONFIG seed=$SEED ==="
    run_config_${CONFIG,,} # calls the function, override SEED below
  done
}

echo "Usage: source launch.sh && run_config_a  (or run_config_b, run_config_c)"
echo "After picking winner: run each seed manually with SEED=42/1337/2024"
