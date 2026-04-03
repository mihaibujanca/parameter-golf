#!/usr/bin/env python3
"""
error_attribution.py — Principled error attribution for quantized transformers.

Computes Logit KL Impact per layer: run quant model through layers 0..i, inject
the resulting hidden state into the float model at layer i+1, and measure the
KL divergence between float and injected logits.

Metrics:
  logit_kl[i]   — cumulative KL damage if quantization error stops after layer i
  marginal_kl[i] — per-layer marginal contribution (delta between consecutive KLs)
  delta_err[i]  — RMSE(h_quant[i] - h_float[i]) under the actual quant config
  masking[i]    — marginal_kl / delta_err (proxy for how well hidden error translates to logit damage)

Usage:
    NUM_LAYERS=13 MLP_MULT=3 MLP_ACT=lrelu2 QUANT_ATTN_BITS=5 QUANT_MLP_BITS=5 \\
    .venv/bin/python3 scripts/error_attribution.py logs/overnight_13L_3x_lrelu2_best.npz

    # Mechanism decomposition only:
    ... scripts/error_attribution.py checkpoint.npz --mode decompose
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
    COMPUTE_DTYPE, Hyperparameters, load_validation_tokens,
    quantize_state_dict_int8, dequantize_state_dict_int8, rms_norm,
)
from scripts.eval_commons import build_model


# =============================================================================
# Forward passes
# =============================================================================

def forward_collect_all(model, x_tokens):
    """Full forward, collecting hidden states at every layer boundary.

    Returns:
        logits: mx.array (1, T, V)
        hidden: list[mx.array]  — post-block hidden at each layer (length n_layers)
        x0: mx.array            — clean embedding after smear
        encoder_outputs: list[mx.array|None]  — post-block for encoder layers (length n_enc)
    """
    n_enc = model.num_encoder_layers
    n_skip = model.num_skip_weights
    n_layers = len(model.blocks)

    x = mx.array(x_tokens[np.newaxis, :])
    tok_emb = model.tok_emb(x).astype(COMPUTE_DTYPE)
    if model.bigram is not None:
        tok_emb = tok_emb + model.bigram(x)
    x0 = model.smear(rms_norm(tok_emb))
    mx.eval(x0)
    h = x0

    encoder_outputs: list = [None] * n_enc
    hidden: list = []

    for i in range(n_layers):
        if i >= n_enc:
            dec_j = i - n_enc
            if dec_j < n_skip:
                enc_j = n_enc - 1 - dec_j
                h = h + model.skip_weights[dec_j].astype(h.dtype)[None, None, :] * encoder_outputs[enc_j]
        h = model.blocks[i](h, x0)
        mx.eval(h)
        hidden.append(h)
        if i < n_enc:
            encoder_outputs[i] = h

    logits = model._apply_logit_processing(model.tok_emb.as_linear(model.final_norm(h)))
    mx.eval(logits)
    return logits, hidden, x0, encoder_outputs


def forward_from_layer(model, h_start, x0, start_layer, encoder_outputs):
    """Run float model from start_layer onward, injecting h_start as initial state.

    encoder_outputs: list of length n_enc. Entries for indices < start_layer hold
    quant encoder outputs; entries for indices >= start_layer are None and will be
    computed fresh as this function processes encoder layers.

    Returns: logits mx.array (1, T, V)
    """
    n_enc = model.num_encoder_layers
    n_skip = model.num_skip_weights
    n_layers = len(model.blocks)

    # Copy so we can fill in fresh encoder outputs without mutating the caller's list
    enc_out = list(encoder_outputs)
    h = h_start

    for i in range(start_layer, n_layers):
        if i >= n_enc:
            dec_j = i - n_enc
            if dec_j < n_skip:
                enc_j = n_enc - 1 - dec_j
                if enc_out[enc_j] is not None:
                    h = h + model.skip_weights[dec_j].astype(h.dtype)[None, None, :] * enc_out[enc_j]
        h = model.blocks[i](h, x0)
        mx.eval(h)
        if i < n_enc:
            enc_out[i] = h

    logits = model._apply_logit_processing(model.tok_emb.as_linear(model.final_norm(h)))
    mx.eval(logits)
    return logits


def kl_div(f_logits, g_logits):
    """Mean KL(softmax(f) || softmax(g)) over tokens. Inputs: np arrays (1, T, V)."""
    f = np.array(f_logits).reshape(-1, f_logits.shape[-1])
    g = np.array(g_logits).reshape(-1, g_logits.shape[-1])
    fp = np.exp(f - f.max(axis=-1, keepdims=True))
    fp /= fp.sum(axis=-1, keepdims=True)
    gp = np.exp(g - g.max(axis=-1, keepdims=True))
    gp /= gp.sum(axis=-1, keepdims=True)
    return float((fp * np.log(np.maximum(fp, 1e-12) / np.maximum(gp, 1e-12))).sum(axis=-1).mean())


# =============================================================================
# Experiment 1: Logit KL Impact
# =============================================================================

def compute_logit_kl_impact(model_float, model_quant, val_tokens, n_seqs=4, seq_len=512):
    """Compute per-layer Logit KL Impact profile."""
    n_layers = len(model_float.blocks)
    n_enc = model_float.num_encoder_layers

    kl_cumulative = np.zeros(n_layers)   # kl_cumulative[i] = KL after injecting at layer i
    delta_err = np.zeros(n_layers)
    kl_full_sum = 0.0

    print(f"\nComputing Logit KL Impact ({n_seqs} seqs × {seq_len} tokens)...")

    for s in range(n_seqs):
        tokens = val_tokens[s * seq_len : (s + 1) * seq_len]
        if len(tokens) < seq_len:
            break

        float_logits, float_hidden, x0, float_enc_out = forward_collect_all(model_float, tokens)
        quant_logits, quant_hidden, _, quant_enc_out = forward_collect_all(model_quant, tokens)

        kl_full_sum += kl_div(float_logits, quant_logits)

        for i in range(n_layers):
            # Build hybrid encoder outputs: quant for 0..i, float for i+1..n_enc-1
            # (entries >= start_layer inside forward_from_layer are computed fresh)
            hybrid_enc = [None] * n_enc
            for j in range(n_enc):
                if j <= i:
                    hybrid_enc[j] = quant_enc_out[j]  # quant encoder output
                else:
                    hybrid_enc[j] = float_enc_out[j]  # float encoder output (used if i < n_enc)

            inject_logits = forward_from_layer(model_float, quant_hidden[i], x0, i + 1, hybrid_enc)
            kl_cumulative[i] += kl_div(float_logits, inject_logits)

            diff = np.array(quant_hidden[i]) - np.array(float_hidden[i])
            delta_err[i] += np.sqrt((diff ** 2).mean())

        print(f"  seq {s}: kl_full={kl_full_sum/(s+1):.5f}  kl_last_layer={kl_cumulative[-1]/(s+1):.5f}")

    kl_cumulative /= n_seqs
    delta_err /= n_seqs
    kl_full = kl_full_sum / n_seqs

    # Per-layer marginal KL
    marginal_kl = np.diff(kl_cumulative, prepend=0.0)

    return {
        "n_layers": n_layers,
        "n_enc": n_enc,
        "kl_cumulative": kl_cumulative,
        "marginal_kl": marginal_kl,
        "delta_err": delta_err,
        "kl_full": kl_full,
    }


def print_kl_impact_table(res):
    n = res["n_layers"]
    n_enc = res["n_enc"]

    print(f"\n{'Layer':>6s} {'Type':>5s} {'DeltaErr':>10s} "
          f"{'CumulKL':>10s} {'MarginalKL':>11s} {'Masking':>8s}")
    print("-" * 58)

    for i in range(n):
        ltype = "enc" if i < n_enc else "dec"
        de = res["delta_err"][i]
        ck = res["kl_cumulative"][i]
        mk = res["marginal_kl"][i]
        masking = mk / max(de, 1e-12)
        print(f"{i:>6d} {ltype:>5s} {de:>10.6f} {ck:>10.6f} {mk:>11.6f} {masking:>8.4f}")

    print(f"\nFull-model KL (both quant):    {res['kl_full']:.6f}")
    print(f"KL at last-layer injection:    {res['kl_cumulative'][-1]:.6f}  "
          f"({'matches' if abs(res['kl_full'] - res['kl_cumulative'][-1]) / max(res['kl_full'], 1e-9) < 0.05 else 'MISMATCH'})")

    # Top-3 layers by marginal KL
    top3 = [int(x) for x in np.argsort(res["marginal_kl"])[::-1][:3]]
    top3_de = [int(x) for x in np.argsort(res["delta_err"])[::-1][:3]]
    print(f"\nTop-3 layers by marginal KL impact: {top3}")
    print(f"  (current compound-DeltaErr top-3:  {top3_de})")


# =============================================================================
# Experiment 3: Mechanism Decomposition (last layer)
# =============================================================================

def mechanism_decomposition(model_float, model_quant, val_tokens, hparams,
                             n_seqs=4, seq_len=512):
    """Trace quantization error through each masking stage at the final layer."""
    print(f"\nMechanism decomposition (last layer, {n_seqs} seqs)...")
    cap = hparams.logit_softcap

    def rms_norm_np(x):
        return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + 1e-5)

    W = np.array(model_float.tok_emb.weight.astype(mx.float32))  # (V, D)

    err_h, err_normed, err_pre, err_capped, kl_vals = [], [], [], [], []
    for s in range(n_seqs):
        tokens = val_tokens[s * seq_len : (s + 1) * seq_len]
        if len(tokens) < seq_len:
            break

        float_logits, float_hidden, _, _ = forward_collect_all(model_float, tokens)
        quant_logits, quant_hidden, _, _ = forward_collect_all(model_quant, tokens)

        fh = np.array(float_hidden[-1]).reshape(-1, W.shape[1])   # (T, D)
        qh = np.array(quant_hidden[-1]).reshape(-1, W.shape[1])

        err_h.append(np.sqrt(((qh - fh) ** 2).mean()))

        fn = rms_norm_np(fh)
        qn = rms_norm_np(qh)
        err_normed.append(np.sqrt(((qn - fn) ** 2).mean()))

        fl_pre = fn @ W.T
        ql_pre = qn @ W.T
        err_pre.append(np.sqrt(((ql_pre - fl_pre) ** 2).mean()))

        fl_cap = cap * np.tanh(fl_pre / cap)
        ql_cap = cap * np.tanh(ql_pre / cap)
        err_capped.append(np.sqrt(((ql_cap - fl_cap) ** 2).mean()))

        kl_vals.append(kl_div(
            mx.array(fl_cap[np.newaxis]),
            mx.array(ql_cap[np.newaxis]),
        ))

    def av(lst):
        return float(np.mean(lst))

    eh = av(err_h)
    en = av(err_normed)
    ep = av(err_pre)
    ec = av(err_capped)
    kl = av(kl_vals)

    print(f"\n  Hidden state error:          {eh:.6f}")
    print(f"  After RMSNorm:               {en:.6f}  ({eh/max(en,1e-12):.1f}x reduction)")
    print(f"  After logit projection:      {ep:.6f}")
    print(f"  After softcap (cap={cap}):  {ec:.6f}  ({ep/max(ec,1e-12):.1f}x reduction)")
    print(f"  KL divergence (bits):        {kl/np.log(2):.6f}")
    return {"err_hidden": eh, "err_normed": en, "err_logits_pre": ep,
            "err_logits_capped": ec, "kl": kl}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--n-seqs", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--mode", choices=["all", "kl-impact", "decompose"], default="all")
    args_cli = parser.parse_args()

    hparams = Hyperparameters()

    print(f"Loading: {args_cli.checkpoint}")
    flat = dict(mx.load(args_cli.checkpoint))

    model_float = build_model(hparams)
    model_float.update(tree_unflatten(list(flat.items())))
    mx.eval(model_float.parameters())

    print("Quantizing model...")
    quant_bits_str = os.environ.get("QUANT_BITS", "")
    if quant_bits_str:
        cat_bits = {}
        for part in quant_bits_str.split(","):
            k, v = part.strip().rsplit(":", 1)
            cat_bits[k.strip()] = int(v.strip())
        print(f"  Per-layer quant: {cat_bits}")
    else:
        cat_bits = {"attn": hparams.quant_attn_bits, "mlp": hparams.quant_mlp_bits}
    quant_obj, _ = quantize_state_dict_int8(flat, cat_bits=cat_bits)
    quant_flat = dequantize_state_dict_int8(quant_obj)
    model_quant = build_model(hparams)
    model_quant.update(tree_unflatten(list(quant_flat.items())))
    mx.eval(model_quant.parameters())

    print(f"Config: {hparams.num_layers}L/{hparams.model_dim}d  MLP {hparams.mlp_mult}x  "
          f"act={hparams.mlp_act}  quant=a{hparams.quant_attn_bits}m{hparams.quant_mlp_bits}")

    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)

    if args_cli.mode in ("all", "kl-impact"):
        res = compute_logit_kl_impact(
            model_float, model_quant, val_tokens,
            n_seqs=args_cli.n_seqs, seq_len=args_cli.seq_len,
        )
        print_kl_impact_table(res)

    if args_cli.mode in ("all", "decompose"):
        mechanism_decomposition(
            model_float, model_quant, val_tokens, hparams,
            n_seqs=args_cli.n_seqs, seq_len=args_cli.seq_len,
        )


if __name__ == "__main__":
    main()
