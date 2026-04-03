#!/usr/bin/env python3
"""
layer_ablation.py — Per-layer importance via ablation.

For each layer, skip its block (identity on residual stream) and measure BPB.
Skip connections are preserved: encoder still saves states, decoder still receives them.
Also measures attn-only and MLP-only ablation per layer.

Usage:
    NUM_LAYERS=11 MLP_MULT=5.0 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \
    GRAD_ACCUM_STEPS=2 VAL_BATCH_SIZE=65536 VAL_MAX_TOKENS=1048576 \
    .venv/bin/python3 scripts/layer_ablation.py logs/wd70_11L_5x_best.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gpt_mlx import (
    GPT, COMPUTE_DTYPE, Hyperparameters, load_validation_tokens,
    build_sentencepiece_luts, eval_val, rms_norm,
)


def build_model(hparams):
    per_layer = None
    if hparams.mlp_mult_per_layer:
        per_layer = [int(x) for x in hparams.mlp_mult_per_layer.split(",")]
    return GPT(
        vocab_size=hparams.vocab_size, num_layers=hparams.num_layers,
        dim=hparams.model_dim, num_heads=hparams.num_heads,
        num_kv_heads=hparams.num_kv_heads, mlp_mult=hparams.mlp_mult,
        logit_chunk_tokens=0, logit_softcap=hparams.logit_softcap,
        rope_base=hparams.rope_base, tied_embed_init_std=hparams.tied_embed_init_std,
        qk_gain_init=hparams.qk_gain_init, mlp_act=hparams.mlp_act,
        mlp_mult_per_layer=per_layer, bigram_vocab_size=hparams.bigram_vocab_size,
        bigram_dim=hparams.bigram_dim, logit_temp=hparams.logit_temp,
        lrelu_slope=hparams.lrelu_slope,
        xsa_last_n=hparams.xsa_last_n, rope_dims=hparams.rope_dims,
    )


class IdentityBlock(nn.Module):
    """A block that passes input through unchanged."""
    def __call__(self, x, x0):
        return x


def eval_with_ablation(model, skip_layer, hparams, val_tokens, luts):
    """Replace block with identity, recompile, eval, restore."""
    original_block = model.blocks[skip_layer]
    model.blocks[skip_layer] = IdentityBlock()
    compiled_loss = mx.compile(
        lambda x, y: model.loss(x, y),
        inputs=model.state, outputs=model.state,
    )
    _, bpb = eval_val(hparams, compiled_loss, val_tokens, *luts)
    model.blocks[skip_layer] = original_block
    return bpb


def main():
    parser = argparse.ArgumentParser(description="Per-layer ablation analysis")
    parser.add_argument("checkpoint", help="Path to .npz float checkpoint")
    parser.add_argument("--n-eval-seqs", type=int, default=64)
    parser.add_argument("--log-file", type=str, default=None)
    args_cli = parser.parse_args()

    hparams = Hyperparameters()
    n_seqs = args_cli.n_eval_seqs
    if n_seqs > 0:
        hparams.val_max_tokens = n_seqs * hparams.train_seq_len

    # Load model
    model = build_model(hparams)
    flat = dict(mx.load(args_cli.checkpoint))
    model.update(tree_unflatten(list(flat.items())))
    mx.eval(model.state)

    # Load eval data
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    sp = spm.SentencePieceProcessor(model_file=hparams.tokenizer_path)
    luts = build_sentencepiece_luts(sp, hparams.vocab_size)

    n_layers = hparams.num_layers
    n_enc = n_layers // 2

    # Baseline
    compiled_loss = mx.compile(
        lambda x, y: model.loss(x, y),
        inputs=model.state, outputs=model.state,
    )
    _, baseline_bpb = eval_val(hparams, compiled_loss, val_tokens, *luts)
    print(f"Baseline BPB: {baseline_bpb:.4f}  ({n_seqs} seqs)\n")

    # Full layer ablation
    print(f"{'L':>2s} {'role':>7s} {'skip':>6s} {'BPB':>7s} {'Δ mBPB':>8s}")
    print("-" * 38)

    results = []
    for i in range(n_layers):
        role = "enc" if i < n_enc else "dec"
        # Skip connection mapping
        if i < n_enc:
            skip_target = n_layers - 1 - i  # encoder L0 → decoder last, etc.
            # Actually: skips are popped, so L0 saved first, popped last
            # L0→L(n_enc + n_skip - 1), L1→L(n_enc + n_skip - 2), etc.
            skip_to = n_enc + (n_enc - 1 - i) if (n_enc - 1 - i) < model.num_skip_weights else -1
            skip_str = f"→L{skip_to}" if skip_to >= 0 else ""
        else:
            dec_j = i - n_enc
            if dec_j < model.num_skip_weights:
                skip_from = n_enc - 1 - dec_j
                skip_str = f"���L{skip_from}"
            else:
                skip_str = ""

        bpb = eval_with_ablation(model, i, hparams, val_tokens, luts)
        delta = (bpb - baseline_bpb) * 1000
        print(f"{i:>2d} {role:>7s} {skip_str:>6s} {bpb:7.4f} {delta:+8.1f}")
        results.append((i, role, skip_str, bpb, delta))

    # Summary
    print(f"\n{'='*40}")
    enc_avg = np.mean([r[4] for r in results if r[1] == "enc"])
    dec_avg = np.mean([r[4] for r in results if r[1] == "dec"])
    print(f"Avg encoder ablation impact: {enc_avg:+.1f} mBPB")
    print(f"Avg decoder ablation impact: {dec_avg:+.1f} mBPB")

    # Rank by importance
    ranked = sorted(results, key=lambda r: -r[4])
    print(f"\nRanked by importance (most → least):")
    for rank, (i, role, skip, bpb, delta) in enumerate(ranked):
        print(f"  #{rank+1}: L{i} ({role}{' '+skip if skip else ''}) {delta:+.1f} mBPB")


if __name__ == "__main__":
    main()
