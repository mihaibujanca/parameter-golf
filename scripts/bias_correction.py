#!/usr/bin/env python3
"""
bias_correction.py — Statistical bias correction for quantized models.

Computes E[h_float - h_quant] per block on calibration data and injects as
additive bias. This is the simplest possible data-dependent correction:
a static vector per layer (~22KB total for 11L/512d).

Usage:
    NUM_LAYERS=11 MLP_MULT=4.5 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \
    QUANT_ATTN_BITS=5 QUANT_MLP_BITS=5 \
    .venv/bin/python3 scripts/bias_correction.py logs/wd50_11L_5x_best.npz \
        --n-seqs 64 --log-file logs/bias_correction_int5.txt
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gpt_mlx import (
    GPT, COMPUTE_DTYPE, Hyperparameters, load_validation_tokens,
    quantize_state_dict_int8, dequantize_state_dict_int8, rms_norm,
)

log = logging.getLogger("bias_correction")


# ============================================================================
# Hidden state collection (same pattern as error_decomposition.py)
# ============================================================================

def forward_collect_hidden(model, tokens):
    """Full forward collecting post-block hidden states at every layer.

    Returns: list[mx.array] of length n_layers, each (1, T, D).
    """
    n_enc = model.num_encoder_layers
    n_skip = model.num_skip_weights
    n_layers = len(model.blocks)

    x = mx.array(tokens[np.newaxis, :])
    tok_emb = model.tok_emb(x).astype(COMPUTE_DTYPE)
    if model.bigram is not None:
        tok_emb = tok_emb + model.bigram(x)
    x0 = model.smear(rms_norm(tok_emb))
    h = x0
    mx.eval(x0)

    encoder_outputs = [None] * n_enc
    hidden = []

    for i in range(n_layers):
        if i >= n_enc:
            dec_j = i - n_enc
            if dec_j < n_skip:
                enc_j = n_enc - 1 - dec_j
                if encoder_outputs[enc_j] is not None:
                    h = h + model.skip_weights[dec_j].astype(h.dtype)[None, None, :] * encoder_outputs[enc_j]
        h = model.blocks[i](h, x0)
        mx.eval(h)
        hidden.append(h)
        if i < n_enc:
            encoder_outputs[i] = h

    return hidden


# ============================================================================
# Bias computation
# ============================================================================

def compute_bias_vectors(model_float, model_quant, val_tokens, n_seqs, seq_len):
    """Compute mean(h_float - h_quant) per layer on calibration data.

    Returns: list of np.ndarray (dim,) for each layer, plus diagnostics.
    """
    n_layers = len(model_float.blocks)
    dim = model_float.blocks[0].attn_scale.shape[0]

    # Accumulators
    bias_sum = [np.zeros(dim, dtype=np.float64) for _ in range(n_layers)]
    bias_sq_sum = [np.zeros(dim, dtype=np.float64) for _ in range(n_layers)]
    total_tokens = 0

    for s in range(n_seqs):
        start = s * seq_len
        toks = val_tokens[start:start + seq_len]
        if len(toks) < seq_len:
            break

        h_float = forward_collect_hidden(model_float, toks)
        h_quant = forward_collect_hidden(model_quant, toks)

        for i in range(n_layers):
            diff = np.array(h_float[i]).reshape(-1, dim).astype(np.float64) - \
                   np.array(h_quant[i]).reshape(-1, dim).astype(np.float64)
            bias_sum[i] += diff.sum(axis=0)
            bias_sq_sum[i] += (diff ** 2).sum(axis=0)

        total_tokens += seq_len

    # Mean and std
    biases = []
    diagnostics = []
    for i in range(n_layers):
        mean = bias_sum[i] / total_tokens
        var = bias_sq_sum[i] / total_tokens - mean ** 2
        std = np.sqrt(np.maximum(var, 0))
        biases.append(mean)
        diagnostics.append({
            "layer": i,
            "bias_norm": float(np.linalg.norm(mean)),
            "bias_max": float(np.abs(mean).max()),
            "bias_mean_abs": float(np.abs(mean).mean()),
            "error_std_mean": float(std.mean()),
            "snr": float(np.abs(mean).mean() / max(std.mean(), 1e-15)),
        })

    return biases, diagnostics


# ============================================================================
# Inject bias and evaluate
# ============================================================================

class BiasInjectedGPT:
    """Wrapper that adds per-layer biases to forward pass."""

    def __init__(self, model, biases):
        self.model = model
        # Convert to mx arrays
        self.biases = [mx.array(b.astype(np.float32)) for b in biases]

    def forward_with_bias(self, tokens):
        """Forward pass with bias correction injected after each block."""
        n_enc = self.model.num_encoder_layers
        n_skip = self.model.num_skip_weights
        n_layers = len(self.model.blocks)

        x = self.model.tok_emb(tokens).astype(COMPUTE_DTYPE)
        if self.model.bigram is not None:
            x = x + self.model.bigram(tokens)
        x0 = self.model.smear(rms_norm(x))
        h = x0

        encoder_outputs = [None] * n_enc

        for i in range(n_layers):
            if i >= n_enc:
                dec_j = i - n_enc
                if dec_j < n_skip:
                    enc_j = n_enc - 1 - dec_j
                    if encoder_outputs[enc_j] is not None:
                        h = h + self.model.skip_weights[dec_j].astype(h.dtype)[None, None, :] * encoder_outputs[enc_j]
            h = self.model.blocks[i](h, x0)
            # Inject bias correction
            h = h + self.biases[i].astype(h.dtype)[None, None, :]
            if i < n_enc:
                encoder_outputs[i] = h

        return self.model.final_norm(h)

    def loss(self, x, y):
        """CE loss with bias correction."""
        h = self.forward_with_bias(x)
        logits = self.model._apply_logit_processing(h @ self.model.tok_emb.weight.T)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        return loss


def quick_eval(model_or_wrapper, hparams, n_seqs=32, seq_len=1024, label=""):
    """Quick CE evaluation."""
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    total_loss = 0.0
    total_tokens = 0
    for s in range(n_seqs):
        tokens = val_tokens[s * seq_len : (s + 1) * seq_len + 1]
        if len(tokens) < seq_len + 1:
            break
        x = mx.array(tokens[:seq_len].reshape(1, seq_len))
        y = mx.array(tokens[1:seq_len + 1].reshape(1, seq_len))
        loss = model_or_wrapper.loss(x, y)
        mx.eval(loss)
        total_loss += loss.item() * seq_len
        total_tokens += seq_len
    avg = total_loss / total_tokens
    bpb_approx = avg / math.log(2)
    log.info(f"  {label}val_loss={avg:.6f} ({bpb_approx:.4f} bits/tok, {n_seqs} seqs)")
    return avg


# ============================================================================
# Main
# ============================================================================

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


def main():
    parser = argparse.ArgumentParser(description="Statistical bias correction")
    parser.add_argument("checkpoint", help="Path to .npz float checkpoint")
    parser.add_argument("--n-seqs", type=int, default=64,
                        help="Number of calibration sequences")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-eval-seqs", type=int, default=64,
                        help="Number of sequences for evaluation")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=0,
                        help="Only apply bias to top-k layers by bias norm (0=all)")
    args_cli = parser.parse_args()

    hparams = Hyperparameters()

    # Logging
    global log
    log = logging.getLogger("bias_correction")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if args_cli.log_file is None:
        args_cli.log_file = f"logs/bias_correction_int{hparams.quant_attn_bits}.txt"
    os.makedirs(os.path.dirname(args_cli.log_file), exist_ok=True)
    fh = logging.FileHandler(args_cli.log_file, mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.info(f"Logging to {args_cli.log_file}")

    cat_bits = {"attn": hparams.quant_attn_bits, "mlp": hparams.quant_mlp_bits}

    log.info(f"Config: {hparams.num_layers}L/{hparams.model_dim}d, MLP {hparams.mlp_mult}x, "
             f"act={hparams.mlp_act}, quant=a{hparams.quant_attn_bits}m{hparams.quant_mlp_bits}")
    log.info(f"Calibration: {args_cli.n_seqs} seqs × {args_cli.seq_len} tokens")

    # Load float model
    log.info(f"\nLoading checkpoint: {args_cli.checkpoint}")
    model_float = build_model(hparams)
    flat = dict(mx.load(args_cli.checkpoint))
    model_float.update(tree_unflatten(list(flat.items())))
    mx.eval(model_float.parameters())

    # Create quantized model
    log.info("Quantizing model...")
    model_quant = build_model(hparams)
    model_quant.update(tree_unflatten(list(flat.items())))
    mx.eval(model_quant.parameters())
    flat_q = {k: v for k, v in tree_flatten(model_quant.state)}
    quant_obj, stats = quantize_state_dict_int8(flat_q, cat_bits=cat_bits)
    quant_flat = dequantize_state_dict_int8(quant_obj)
    model_quant.update(tree_unflatten(list(quant_flat.items())))
    mx.eval(model_quant.parameters())

    # Baseline evals
    log.info("\n=== Float baseline ===")
    float_loss = quick_eval(model_float, hparams, args_cli.n_eval_seqs, args_cli.seq_len, "float: ")
    log.info("\n=== Quant baseline (no correction) ===")
    quant_loss = quick_eval(model_quant, hparams, args_cli.n_eval_seqs, args_cli.seq_len, "quant: ")
    quant_gap = quant_loss - float_loss
    log.info(f"Quant gap: {quant_gap:.6f} CE ({quant_gap / math.log(2):.6f} bits)")

    # Compute bias vectors
    log.info(f"\n=== Computing bias vectors ({args_cli.n_seqs} calibration sequences) ===")
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    t0 = time.time()
    biases, diagnostics = compute_bias_vectors(
        model_float, model_quant, val_tokens, args_cli.n_seqs, args_cli.seq_len
    )
    t_bias = time.time() - t0
    log.info(f"Bias computation: {t_bias:.1f}s")

    # Print diagnostics
    log.info(f"\n{'Layer':>5s} {'bias_norm':>10s} {'bias_max':>10s} {'bias_mean':>10s} {'err_std':>10s} {'SNR':>8s}")
    log.info("-" * 58)
    for d in diagnostics:
        log.info(f"  {d['layer']:>3d}  {d['bias_norm']:10.6f} {d['bias_max']:10.6f} "
                 f"{d['bias_mean_abs']:10.6f} {d['error_std_mean']:10.6f} {d['snr']:8.4f}")

    # Hidden state scale reference
    log.info(f"\nBias storage: {len(biases)} layers × {biases[0].shape[0]} dims × 4 bytes = "
             f"{len(biases) * biases[0].shape[0] * 4:,} bytes ({len(biases) * biases[0].shape[0] * 4 / 1024:.1f} KB)")

    # Evaluate with bias correction (all layers)
    log.info(f"\n=== Eval: bias correction (all layers) ===")
    wrapper_all = BiasInjectedGPT(model_quant, biases)
    corrected_loss_all = quick_eval(wrapper_all, hparams, args_cli.n_eval_seqs, args_cli.seq_len, "corrected: ")
    improvement_all = quant_loss - corrected_loss_all
    recovery_all = improvement_all / max(quant_gap, 1e-15) * 100
    log.info(f"Improvement: {improvement_all:.6f} CE ({recovery_all:.1f}% of gap)")

    # Evaluate with top-k layers only
    if args_cli.top_k > 0:
        # Sort layers by bias norm
        sorted_layers = sorted(range(len(biases)),
                               key=lambda i: diagnostics[i]["bias_norm"], reverse=True)
        top_k_layers = set(sorted_layers[:args_cli.top_k])
        biases_topk = [b if i in top_k_layers else np.zeros_like(b) for i, b in enumerate(biases)]
        log.info(f"\n=== Eval: bias correction (top-{args_cli.top_k} layers: {sorted(top_k_layers)}) ===")
        wrapper_topk = BiasInjectedGPT(model_quant, biases_topk)
        corrected_loss_topk = quick_eval(wrapper_topk, hparams, args_cli.n_eval_seqs, args_cli.seq_len, "corrected: ")
        improvement_topk = quant_loss - corrected_loss_topk
        recovery_topk = improvement_topk / max(quant_gap, 1e-15) * 100
        log.info(f"Improvement: {improvement_topk:.6f} CE ({recovery_topk:.1f}% of gap)")

    # Summary
    log.info(f"\n=== Summary ===")
    log.info(f"Float val_loss:     {float_loss:.6f} ({float_loss / math.log(2):.4f} bits/tok)")
    log.info(f"Quant val_loss:     {quant_loss:.6f} ({quant_loss / math.log(2):.4f} bits/tok)")
    log.info(f"Corrected val_loss: {corrected_loss_all:.6f} ({corrected_loss_all / math.log(2):.4f} bits/tok)")
    log.info(f"Quant gap:          {quant_gap:.6f} CE")
    log.info(f"Bias improvement:   {improvement_all:.6f} CE ({recovery_all:.1f}% recovery)")


if __name__ == "__main__":
    main()
