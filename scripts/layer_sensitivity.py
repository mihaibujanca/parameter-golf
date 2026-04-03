#!/usr/bin/env python3
"""
layer_sensitivity.py — Per-layer isolated quantization sensitivity.

For each layer i, quantize ONLY that layer's weights (attn+mlp) to a target
bitwidth while keeping everything else at a baseline bitwidth. Measures the
BPB delta caused by each layer in isolation.

Usage:
    NUM_LAYERS=11 MLP_MULT=4.5 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \
    BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=128 LRELU_SLOPE=0.5 \
    LOGIT_SOFTCAP=30 QK_GAIN_INIT=1.5 NUM_KV_HEADS=4 \
    .venv/bin/python3 scripts/layer_sensitivity.py logs/warmdown_11L_45x_best.npz \
        --target-bits 3 --baseline-bits 6

    # Also test int2 on cheapest layers:
    ... --target-bits 2 --baseline-bits 6
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gpt_mlx import (
    Hyperparameters, load_validation_tokens,
    quantize_state_dict_int8, dequantize_state_dict_int8,
)
from scripts.eval_commons import build_model, quick_ce


def main():
    parser = argparse.ArgumentParser(description="Per-layer quantization sensitivity")
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument("--target-bits", type=int, default=3, help="Bitwidth for the target layer (default: 3)")
    parser.add_argument("--baseline-bits", type=int, default=6, help="Bitwidth for all other layers (default: 6)")
    parser.add_argument("--n-eval-seqs", type=int, default=64, help="Number of eval sequences (default: 64)")
    parser.add_argument("--components", type=str, default="both",
                        choices=["both", "attn", "mlp"],
                        help="Which components to quantize at target bits (default: both)")
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    hparams = Hyperparameters()
    n_layers = hparams.num_layers

    # Logging
    import logging
    log = logging.getLogger("layer_sensitivity")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)
    if args.log_file is None:
        args.log_file = f"logs/layer_sensitivity_int{args.target_bits}_base{args.baseline_bits}.txt"
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    fh = logging.FileHandler(args.log_file, mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    log.info(f"Per-layer sensitivity: target=int{args.target_bits}, baseline=int{args.baseline_bits}")
    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"Model: {n_layers}L/{hparams.model_dim}d, MLP {hparams.mlp_mult}x")
    log.info(f"Components: {args.components}, eval seqs: {args.n_eval_seqs}")

    # Build model
    model = build_model(hparams)

    ckpt = dict(mx.load(args.checkpoint))
    model.update(tree_unflatten(list(ckpt.items())))
    mx.eval(model.state)
    original_state = {k: v for k, v in tree_flatten(model.state)}

    # Load val data
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    log.info(f"Val tokens: {len(val_tokens):,}")

    # Float baseline
    log.info("\n--- Float baseline (no quantization) ---")
    float_loss = quick_ce(model, val_tokens, args.n_eval_seqs)
    float_bpt = float_loss / math.log(2)  # NB: bits-per-token, not BPB
    log.info(f"  float: loss={float_loss:.6f}  bpt={float_bpt:.4f}")

    # Uniform baseline at baseline bits
    log.info(f"\n--- Uniform int{args.baseline_bits} baseline ---")
    cat_bits_uniform = {"attn": args.baseline_bits, "mlp": args.baseline_bits}
    flat = {k: v for k, v in tree_flatten(model.state)}
    qobj, _ = quantize_state_dict_int8(flat, cat_bits=cat_bits_uniform)
    qflat = dequantize_state_dict_int8(qobj)
    model.update(tree_unflatten(list(qflat.items())))
    mx.eval(model.state)
    baseline_loss = quick_ce(model, val_tokens, args.n_eval_seqs)
    baseline_bpt = baseline_loss / math.log(2)  # NB: bits-per-token, not BPB
    log.info(f"  int{args.baseline_bits}: loss={baseline_loss:.6f}  bpt={baseline_bpt:.4f}  gap={baseline_bpt - float_bpt:.6f}")

    # Per-layer sweep
    log.info(f"\n--- Per-layer sensitivity: one layer at int{args.target_bits}, rest int{args.baseline_bits} ---")
    results = []
    for layer_i in range(n_layers):
        # Restore original weights
        model.update(tree_unflatten(list(original_state.items())))
        mx.eval(model.state)

        # Build cat_bits: all at baseline, target layer at target_bits
        cat_bits = {"attn": args.baseline_bits, "mlp": args.baseline_bits}
        if args.components in ("both", "attn"):
            cat_bits[f"attn.{layer_i}"] = args.target_bits
        if args.components in ("both", "mlp"):
            cat_bits[f"mlp.{layer_i}"] = args.target_bits

        flat = {k: v for k, v in tree_flatten(model.state)}
        qobj, _ = quantize_state_dict_int8(flat, cat_bits=cat_bits)
        qflat = dequantize_state_dict_int8(qobj)
        model.update(tree_unflatten(list(qflat.items())))
        mx.eval(model.state)

        loss = quick_ce(model, val_tokens, args.n_eval_seqs)
        bpt = loss / math.log(2)  # NB: bits-per-token, not BPB
        delta_from_float = bpt - float_bpt
        delta_from_baseline = bpt - baseline_bpt

        results.append({
            "layer": layer_i,
            "loss": loss,
            "bpt": bpt,
            "delta_float": delta_from_float,
            "delta_baseline": delta_from_baseline,
        })
        log.info(f"  L{layer_i:2d}: bpt={bpt:.4f}  Δfloat={delta_from_float:+.6f}  Δbase={delta_from_baseline:+.6f}")

    # Summary sorted by sensitivity (most sensitive first)
    log.info(f"\n{'='*60}")
    log.info(f"SENSITIVITY RANKING (most sensitive → least)")
    log.info(f"{'='*60}")
    log.info(f"{'Layer':>5s} {'BPT':>8s} {'Δ float':>10s} {'Δ baseline':>12s}")
    log.info(f"{'-'*5} {'-'*8} {'-'*10} {'-'*12}")
    for r in sorted(results, key=lambda x: -x["delta_float"]):
        log.info(f"  L{r['layer']:2d}  {r['bpt']:8.4f}  {r['delta_float']:+10.6f}  {r['delta_baseline']:+12.6f}")

    log.info(f"\nReference: float={float_bpt:.4f}, int{args.baseline_bits}={baseline_bpt:.4f} (bits-per-token, NOT BPB)")
    log.info(f"Logged to {args.log_file}")


if __name__ == "__main__":
    main()
