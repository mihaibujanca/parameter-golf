#!/usr/bin/env bash
# smoke.sh — fast end-to-end sanity check on M4 Pro, target <2 min
# Validates: model trains, loss falls, val_bpb is finite, int8 roundtrip works.
# Not a quality benchmark — val_bpb will be noisy at 1M tokens.
set -euo pipefail

RUN_ID=smoke \
ITERATIONS=20 \
WARMUP_STEPS=5 \
TRAIN_BATCH_TOKENS=8192 \
GRAD_ACCUM_STEPS=1 \
TRAIN_LOG_EVERY=5 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=65536 \
VAL_MAX_TOKENS=1048576 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/python3 train_gpt_mlx.py
