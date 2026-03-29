#!/usr/bin/env python3
"""
error_decomposition.py — Canonical error analysis for quantized transformers.

For each transformer block, decomposes quantization error into:
- Local error: this layer's quantization applied to its input
- Propagated error: accumulated error from all previous layers amplified by this layer's weights

Also measures ReLU disagreement (where float and quant models make different activation decisions)
and per-layer quantization sensitivity.

Usage:
    NUM_LAYERS=13 MLP_MULT=3 MLP_ACT=lrelu2 QUANT_ATTN_BITS=5 QUANT_MLP_BITS=5 \
    .venv/bin/python3 scripts/error_decomposition.py logs/overnight_13L_3x_lrelu2_best.npz
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
    quantize_state_dict_int8, dequantize_state_dict_int8,
)


def forward_collecting_activations(model, x_tokens, slope):
    """Run forward pass collecting intermediate activations after each block.

    Returns list of (pre_block_hidden, post_block_hidden) for each block,
    plus MLP pre-activations for disagreement analysis.
    """
    x = mx.array(x_tokens[np.newaxis, :])
    tok_emb = model.tok_emb(x)
    if model.bigram is not None:
        tok_emb = tok_emb + model.bigram(x)
    x0 = model.smear(tok_emb)
    h = x0

    block_activations = []  # (pre, post) hidden states
    mlp_preacts = []  # pre-activation before relu/lrelu squaring

    n_enc = model.num_encoder_layers
    n_skip = model.num_skip_weights
    encoder_outputs = []

    for i, block in enumerate(model.blocks):
        pre_h = h

        # U-Net skip connections (decoder side)
        if i >= n_enc and (i - n_enc) < n_skip:
            skip_idx = n_enc - 1 - (i - n_enc)
            skip_h = encoder_outputs[skip_idx]
            w = model.skip_weights[i - n_enc]
            h = h + w.astype(h.dtype)[None, None, :] * skip_h

        mix = block.resid_mix.astype(h.dtype)
        h = mix[0][None, None, :] * h + mix[1][None, None, :] * x0

        # Attention
        attn_out = block.attn(block.attn_norm(h))
        h = h + block.attn_scale.astype(h.dtype)[None, None, :] * attn_out

        # MLP - capture pre-activation
        mlp_input = block.mlp_norm(h)
        pre_act = block.mlp.fc(mlp_input)
        mlp_preacts.append(pre_act)
        h = h + block.mlp_scale.astype(h.dtype)[None, None, :] * block.mlp(mlp_input, slope=slope)

        # Store for U-Net
        if i < n_enc:
            encoder_outputs.append(h)

        block_activations.append((pre_h, h))
        mx.eval(h, pre_act)

    # Final logits
    logits = model._apply_logit_processing(model.tok_emb.as_linear(model.final_norm(h)))
    mx.eval(logits)

    return block_activations, mlp_preacts, logits


def analyze_errors(float_acts, quant_acts, float_preacts, quant_preacts, float_logits, quant_logits):
    """Decompose quantization error per layer."""
    n_layers = len(float_acts)

    print(f"\n{'Layer':>6s} {'HiddenErr':>10s} {'RelErr':>8s} {'LogitErr':>10s} "
          f"{'ReLUDisagree':>13s} {'PreActShift':>12s} {'HiddenNorm':>11s}")
    print("-" * 85)

    cumulative_hidden_err = None

    for i in range(n_layers):
        f_pre, f_post = float_acts[i]
        q_pre, q_post = quant_acts[i]

        # Hidden state error after this block
        err = np.array(q_post) - np.array(f_post)
        err_norm = np.sqrt((err ** 2).mean())
        hidden_norm = np.sqrt((np.array(f_post) ** 2).mean())
        rel_err = err_norm / max(hidden_norm, 1e-12)

        # MLP pre-activation disagreement (where relu decisions differ)
        f_pa = np.array(float_preacts[i])
        q_pa = np.array(quant_preacts[i])
        disagree = ((f_pa > 0) != (q_pa > 0)).mean()

        # Pre-activation shift (how much quantization moves pre-activations)
        pa_shift = np.sqrt(((q_pa - f_pa) ** 2).mean())

        # Track cumulative error
        if cumulative_hidden_err is None:
            cumulative_hidden_err = err_norm
        else:
            cumulative_hidden_err = err_norm  # just track latest

        print(f"{i:>6d} {err_norm:>10.6f} {rel_err:>7.4f}% {0:>10.6f} "
              f"{100*disagree:>12.2f}% {pa_shift:>12.6f} {hidden_norm:>11.4f}")

    # Logit-level error
    f_logits = np.array(float_logits)
    q_logits = np.array(quant_logits)
    logit_err = np.sqrt(((q_logits - f_logits) ** 2).mean())
    logit_norm = np.sqrt((f_logits ** 2).mean())
    print(f"\nLogit RMSE: {logit_err:.6f} (norm: {logit_norm:.4f}, rel: {logit_err/max(logit_norm,1e-12):.4f})")

    # Per-token KL divergence
    f_probs = np.exp(f_logits - f_logits.max(axis=-1, keepdims=True))
    f_probs = f_probs / f_probs.sum(axis=-1, keepdims=True)
    q_probs = np.exp(q_logits - q_logits.max(axis=-1, keepdims=True))
    q_probs = q_probs / q_probs.sum(axis=-1, keepdims=True)
    kl = (f_probs * np.log(np.maximum(f_probs, 1e-12) / np.maximum(q_probs, 1e-12))).sum(axis=-1).mean()
    print(f"Mean KL(float || quant): {kl:.6f} ({kl / np.log(2):.6f} bits)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument("--n-seqs", type=int, default=8, help="Number of sequences to average over")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    args_cli = parser.parse_args()

    hparams = Hyperparameters()
    per_layer = None
    if hparams.mlp_mult_per_layer:
        per_layer = [int(x) for x in hparams.mlp_mult_per_layer.split(",")]

    # Build float model
    model = GPT(
        vocab_size=hparams.vocab_size, num_layers=hparams.num_layers,
        dim=hparams.model_dim, num_heads=hparams.num_heads,
        num_kv_heads=hparams.num_kv_heads, mlp_mult=hparams.mlp_mult,
        logit_chunk_tokens=0, logit_softcap=hparams.logit_softcap,
        rope_base=hparams.rope_base, tied_embed_init_std=hparams.tied_embed_init_std,
        qk_gain_init=hparams.qk_gain_init, mlp_act=hparams.mlp_act,
        mlp_mult_per_layer=per_layer, bigram_vocab_size=hparams.bigram_vocab_size,
        bigram_dim=hparams.bigram_dim, logit_temp=hparams.logit_temp,
        lrelu_slope=hparams.lrelu_slope,
    )

    print(f"Loading checkpoint: {args_cli.checkpoint}")
    flat = dict(mx.load(args_cli.checkpoint))
    model.update(tree_unflatten(list(flat.items())))
    mx.eval(model.parameters())

    # Load val data
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    print(f"Model: {hparams.num_layers}L/{hparams.model_dim}d, MLP {hparams.mlp_mult}x, "
          f"act={hparams.mlp_act}, quant=a{hparams.quant_attn_bits}m{hparams.quant_mlp_bits}")

    # Collect float activations
    print(f"\nCollecting float activations ({args_cli.n_seqs} seqs)...")
    float_acts_all, float_preacts_all, float_logits_all = [], [], []
    for s in range(args_cli.n_seqs):
        tokens = val_tokens[s * args_cli.seq_len : (s + 1) * args_cli.seq_len + 1]
        acts, preacts, logits = forward_collecting_activations(model, tokens[:args_cli.seq_len], hparams.lrelu_slope)
        float_acts_all.append(acts)
        float_preacts_all.append(preacts)
        float_logits_all.append(logits)

    # Quantize and dequantize
    print("Quantizing model...")
    cat_bits = {"attn": hparams.quant_attn_bits, "mlp": hparams.quant_mlp_bits}
    quant_obj, _ = quantize_state_dict_int8(dict(tree_flatten(model.state)), cat_bits=cat_bits)
    quant_flat = dequantize_state_dict_int8(quant_obj)
    model.update(tree_unflatten(list(quant_flat.items())))
    mx.eval(model.parameters())

    # Collect quantized activations
    print(f"Collecting quantized activations ({args_cli.n_seqs} seqs)...")
    quant_acts_all, quant_preacts_all, quant_logits_all = [], [], []
    for s in range(args_cli.n_seqs):
        tokens = val_tokens[s * args_cli.seq_len : (s + 1) * args_cli.seq_len + 1]
        acts, preacts, logits = forward_collecting_activations(model, tokens[:args_cli.seq_len], hparams.lrelu_slope)
        quant_acts_all.append(acts)
        quant_preacts_all.append(preacts)
        quant_logits_all.append(logits)

    # Average error analysis across sequences
    print(f"\n=== Error Decomposition ({args_cli.n_seqs} seqs averaged) ===")
    # Use first seq for now (averaging activations doesn't make sense, but errors do)
    analyze_errors(
        float_acts_all[0], quant_acts_all[0],
        float_preacts_all[0], quant_preacts_all[0],
        float_logits_all[0], quant_logits_all[0],
    )

    # Also show per-seq variance
    print(f"\n=== Per-sequence logit KL divergence ===")
    for s in range(args_cli.n_seqs):
        f_log = np.array(float_logits_all[s])
        q_log = np.array(quant_logits_all[s])
        f_p = np.exp(f_log - f_log.max(axis=-1, keepdims=True))
        f_p = f_p / f_p.sum(axis=-1, keepdims=True)
        q_p = np.exp(q_log - q_log.max(axis=-1, keepdims=True))
        q_p = q_p / q_p.sum(axis=-1, keepdims=True)
        kl = (f_p * np.log(np.maximum(f_p, 1e-12) / np.maximum(q_p, 1e-12))).sum(axis=-1).mean()
        print(f"  seq {s}: KL = {kl:.6f} ({kl/np.log(2):.6f} bits)")


if __name__ == "__main__":
    main()
