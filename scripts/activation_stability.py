#!/usr/bin/env python3
"""
activation_stability.py — Analyze neuron activation stability across checkpoints.

For each MLP layer, runs a batch of val data through the float model and measures:
- Fraction of neurons that are "always off" (pre-activation always <= 0)
- Fraction of neurons that are "always on" (pre-activation always > 0)
- Fraction "unstable" (sometimes on, sometimes off)

For ReLU² networks, stable neurons = structured weight rows that compress better.
SUGAR should increase stable fraction vs lrelu2 (where every neuron is partially active).

Usage:
    .venv/bin/python3 scripts/activation_stability.py logs/sugar_s05_2k_best.npz
    .venv/bin/python3 scripts/activation_stability.py logs/lrelu2_s05_2k_best.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gpt_mlx import (
    GPT, COMPUTE_DTYPE, Hyperparameters, load_validation_tokens,
    build_sentencepiece_luts, validate_dataset_tokenizer_pair, rms_norm,
)


def collect_mlp_preacts(model, tokens, seq_len=1024, max_seqs=64):
    """Run forward pass and collect MLP pre-activations (before relu/lrelu/sugar squaring)."""
    # Hook into each block's MLP to capture pre-activations
    preacts_per_layer = {i: [] for i in range(len(model.blocks))}

    # Process sequences
    n_tokens = len(tokens)
    n_seqs = min(max_seqs, n_tokens // seq_len)

    for s in range(n_seqs):
        x_np = tokens[s * seq_len : (s + 1) * seq_len]
        x = mx.array(x_np[np.newaxis, :])  # (1, seq_len)

        # Manual forward to capture pre-activations
        tok_emb = model.tok_emb(x).astype(COMPUTE_DTYPE)
        if model.bigram is not None:
            tok_emb = tok_emb + model.bigram(x)
        x0 = model.smear(rms_norm(tok_emb))
        h = x0

        n_enc = model.num_encoder_layers
        n_skip = model.num_skip_weights
        encoder_outputs = []

        for i, block in enumerate(model.blocks):
            if i >= n_enc and (i - n_enc) < n_skip:
                skip_idx = n_enc - 1 - (i - n_enc)
                skip_h = encoder_outputs[skip_idx]
                w = model.skip_weights[i - n_enc]
                h = h + w.astype(h.dtype)[None, None, :] * skip_h

            mix = block.resid_mix.astype(h.dtype)
            h = mix[0][None, None, :] * h + mix[1][None, None, :] * x0
            attn_out = block.attn(block.attn_norm(h))
            h = h + block.attn_scale.astype(h.dtype)[None, None, :] * attn_out
            # MLP: get pre-activation
            mlp_input = block.mlp_norm(h)
            pre_act = block.mlp.fc(mlp_input)  # (1, seq_len, hidden)
            mx.eval(pre_act)
            preacts_per_layer[i].append(np.array(pre_act))
            # Continue normal forward for residual
            h = h + block.mlp_scale.astype(h.dtype)[None, None, :] * block.mlp(mlp_input, slope=block.lrelu_slope)

            if i < n_enc:
                encoder_outputs.append(h)

    # Concatenate across sequences
    return {i: np.concatenate(v, axis=1) for i, v in preacts_per_layer.items()}  # (1, total_tokens, hidden)


def analyze_stability(preacts_per_layer):
    """Compute per-layer neuron stability stats."""
    print(f"\n{'Layer':>6s} {'Hidden':>7s} {'AlwaysOff':>10s} {'AlwaysOn':>10s} {'Unstable':>10s} "
          f"{'%Off':>7s} {'%On':>7s} {'%Unstable':>10s} {'MeanAct':>10s}")
    print("-" * 95)

    for layer_idx in sorted(preacts_per_layer.keys()):
        pa = preacts_per_layer[layer_idx].reshape(-1, preacts_per_layer[layer_idx].shape[-1])  # (tokens, hidden)
        n_tokens, hidden = pa.shape

        # For each neuron: is it ever positive? ever non-positive?
        ever_positive = (pa > 0).any(axis=0)   # (hidden,)
        ever_nonpos = (pa <= 0).any(axis=0)     # (hidden,)

        always_off = (~ever_positive).sum()
        always_on = (~ever_nonpos).sum()
        unstable = hidden - always_off - always_on

        mean_act = pa.mean()

        print(f"{layer_idx:>6d} {hidden:>7d} {always_off:>10d} {always_on:>10d} {unstable:>10d} "
              f"{100*always_off/hidden:>6.1f}% {100*always_on/hidden:>6.1f}% {100*unstable/hidden:>9.1f}% "
              f"{mean_act:>10.4f}")

    # Summary
    all_pa = np.concatenate([v.reshape(-1, v.shape[-1]) for v in preacts_per_layer.values()], axis=1)
    total_neurons = sum(v.shape[-1] for v in preacts_per_layer.values())
    print(f"\nTotal neurons across all layers: {total_neurons}")
    print(f"Mean pre-activation: {all_pa.mean():.6f}")
    print(f"Std pre-activation: {all_pa.std():.6f}")
    print(f"Fraction negative: {(all_pa < 0).mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument("--max-seqs", type=int, default=64, help="Max sequences to process")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    args_cli = parser.parse_args()

    # Load hyperparams from environment (or defaults)
    hparams = Hyperparameters()

    # Build model
    per_layer = None
    if hparams.mlp_mult_per_layer:
        per_layer = [int(x) for x in hparams.mlp_mult_per_layer.split(",")]

    model = GPT(
        vocab_size=hparams.vocab_size,
        num_layers=hparams.num_layers,
        dim=hparams.model_dim,
        num_heads=hparams.num_heads,
        num_kv_heads=hparams.num_kv_heads,
        mlp_mult=hparams.mlp_mult,
        logit_chunk_tokens=hparams.logit_chunk_tokens,
        logit_softcap=hparams.logit_softcap,
        rope_base=hparams.rope_base,
        tied_embed_init_std=hparams.tied_embed_init_std,
        qk_gain_init=hparams.qk_gain_init,
        mlp_act=hparams.mlp_act,
        mlp_mult_per_layer=per_layer,
        bigram_vocab_size=hparams.bigram_vocab_size,
        bigram_dim=hparams.bigram_dim,
        logit_temp=hparams.logit_temp,
        lrelu_slope=hparams.lrelu_slope,
        xsa_last_n=hparams.xsa_last_n,
        rope_dims=hparams.rope_dims,
        num_encoder_layers=hparams.num_encoder_layers,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args_cli.checkpoint}")
    flat = dict(mx.load(args_cli.checkpoint))
    model.update(tree_unflatten(list(flat.items())))
    mx.eval(model.parameters())

    # Load validation data
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    print(f"Val tokens: {len(val_tokens):,}, using {args_cli.max_seqs} seqs of {args_cli.seq_len}")
    print(f"Model: {hparams.num_layers}L/{hparams.model_dim}d, MLP {hparams.mlp_mult}x, act={hparams.mlp_act}, slope={hparams.lrelu_slope}")

    # Collect and analyze
    preacts = collect_mlp_preacts(model, val_tokens, seq_len=args_cli.seq_len, max_seqs=args_cli.max_seqs)
    analyze_stability(preacts)


if __name__ == "__main__":
    main()
