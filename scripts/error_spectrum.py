#!/usr/bin/env python3
"""
error_spectrum.py — SVD analysis of quantization error at each layer.

Measures:
1. Error spectrum (SVD of δh = h_float - h_quant across tokens) — tells us
   the effective rank / bottleneck size needed for correction
2. Activation disagreement rate — fraction of neurons that flip sign (lrelu2),
   the irreversible nonlinear error component
3. Oracle ceiling — if we inject float hidden at layer L and run quant forward
   from L+1, how much error remains? Tests whether correction at layer L
   can in principle recover all downstream error
4. Cross-layer mutual information — does h_quant at uncorrected layers carry
   information about δh at the corrected layer?

Usage:
    NUM_LAYERS=11 MLP_MULT=5.0 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \\
    BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=128 LOGIT_SOFTCAP=30.0 \\
    QUANT_ATTN_BITS=4 QUANT_MLP_BITS=4 \\
    .venv/bin/python3 scripts/error_spectrum.py logs/wd50_11L_5x_best.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gpt_mlx import (
    COMPUTE_DTYPE, Hyperparameters, load_validation_tokens,
    quantize_state_dict_int8, dequantize_state_dict_int8, rms_norm,
)
from scripts.eval_commons import build_model
from scripts.error_attribution import forward_collect_all, forward_from_layer, kl_div


def collect_hidden_and_preacts(model, x_tokens, collect_preacts=True):
    """Forward pass collecting hidden states AND pre-activations (before MLP activation).

    Returns:
        hidden: list[np.array] — post-block hidden at each layer (n_layers)
        preacts: list[np.array] — pre-activation values at MLP (before lrelu2), (n_layers)
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

    encoder_outputs = [None] * n_enc
    hidden = []
    preacts = []

    for i in range(n_layers):
        if i >= n_enc:
            dec_j = i - n_enc
            if dec_j < n_skip:
                enc_j = n_enc - 1 - dec_j
                h = h + model.skip_weights[dec_j].astype(h.dtype)[None, None, :] * encoder_outputs[enc_j]

        block = model.blocks[i]
        mix = block.resid_mix.astype(h.dtype)
        h = mix[0][None, None, :] * h + mix[1][None, None, :] * x0
        attn_out = block.attn(block.attn_norm(h))
        h = h + block.attn_scale.astype(h.dtype)[None, None, :] * attn_out

        # Collect pre-activation (input to MLP activation function)
        if collect_preacts:
            mlp_input = block.mlp_norm(h)
            fc_out = block.mlp.fc(mlp_input)
            mx.eval(fc_out)
            preacts.append(np.array(fc_out.astype(mx.float32)))

        h = h + block.mlp_scale.astype(h.dtype)[None, None, :] * block.mlp(block.mlp_norm(h), slope=block.lrelu_slope)
        mx.eval(h)
        hidden.append(np.array(h.astype(mx.float32)))
        if i < n_enc:
            encoder_outputs[i] = h

    return hidden, preacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument("--n-seqs", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()

    hparams = Hyperparameters()
    n_layers = hparams.num_layers
    dim = hparams.model_dim

    print(f"Config: {n_layers}L/{dim}d, MLP {hparams.mlp_mult}x, "
          f"quant=a{hparams.quant_attn_bits}m{hparams.quant_mlp_bits}")

    # Build float model
    print(f"\nLoading float model: {args.checkpoint}")
    model_float = build_model(hparams)
    flat = dict(mx.load(args.checkpoint))
    model_float.update(tree_unflatten(list(flat.items())))
    mx.eval(model_float.parameters())

    # Build quant model
    print("Quantizing...")
    cat_bits = {"attn": hparams.quant_attn_bits, "mlp": hparams.quant_mlp_bits}
    quant_bits_str = os.environ.get("QUANT_BITS", "")
    if quant_bits_str:
        cat_bits = {}
        for part in quant_bits_str.split(","):
            k, v = part.strip().rsplit(":", 1)
            cat_bits[k.strip()] = int(v.strip())
    quant_obj, _ = quantize_state_dict_int8(flat, cat_bits=cat_bits)
    quant_flat = dequantize_state_dict_int8(quant_obj)
    model_quant = build_model(hparams)
    model_quant.update(tree_unflatten(list(quant_flat.items())))
    mx.eval(model_quant.parameters())

    # Load val data
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    print(f"Val tokens: {len(val_tokens):,}")

    # =========================================================================
    # Collect hidden states and pre-activations
    # =========================================================================
    print(f"\nCollecting hidden states ({args.n_seqs} seqs × {args.seq_len} tokens)...")

    all_delta_h = [[] for _ in range(n_layers)]  # δh = h_float - h_quant
    all_h_quant = [[] for _ in range(n_layers)]
    all_preact_disagree = [[] for _ in range(n_layers)]

    for s in range(args.n_seqs):
        tokens = val_tokens[s * args.seq_len : (s + 1) * args.seq_len]
        if len(tokens) < args.seq_len:
            break

        h_float, preact_float = collect_hidden_and_preacts(model_float, tokens)
        h_quant, preact_quant = collect_hidden_and_preacts(model_quant, tokens)

        for l in range(n_layers):
            delta = h_float[l].reshape(-1, dim) - h_quant[l].reshape(-1, dim)
            all_delta_h[l].append(delta)
            all_h_quant[l].append(h_quant[l].reshape(-1, dim))

            # Activation disagreement: where sign differs for lrelu2
            # lrelu2(x, slope) = x if x > 0 else slope * x
            # The "disagreement" is where x changes sign
            pf = preact_float[l].reshape(-1, preact_float[l].shape[-1])
            pq = preact_quant[l].reshape(-1, preact_quant[l].shape[-1])
            disagree = ((pf > 0) != (pq > 0)).mean()
            all_preact_disagree[l].append(disagree)

        if (s + 1) % 4 == 0:
            print(f"  seq {s+1}/{args.n_seqs}")

    # =========================================================================
    # 1. SVD of error at each layer
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. ERROR SPECTRUM (SVD of δh = h_float - h_quant)")
    print("=" * 70)

    print(f"\n{'Layer':>5s} {'RMSE':>8s} {'‖h_f‖':>8s} {'RelErr':>8s} "
          f"{'Rank90':>7s} {'Rank95':>7s} {'Rank99':>7s} {'Top1%':>7s} {'Top10%':>8s}")
    print("-" * 80)

    layer_ranks = {}
    for l in range(n_layers):
        delta = np.concatenate(all_delta_h[l], axis=0)  # (n_tokens, dim)
        h_q = np.concatenate(all_h_quant[l], axis=0)

        rmse = np.sqrt((delta ** 2).mean())
        h_norm = np.sqrt((h_q ** 2).mean())
        rel_err = rmse / max(h_norm, 1e-12)

        # SVD of the error matrix
        # Center first (remove mean error — a bias that could be corrected trivially)
        delta_centered = delta - delta.mean(axis=0, keepdims=True)
        U, S, Vh = np.linalg.svd(delta_centered, full_matrices=False)

        # Cumulative energy
        energy = (S ** 2).cumsum() / (S ** 2).sum()
        rank_90 = int(np.searchsorted(energy, 0.90)) + 1
        rank_95 = int(np.searchsorted(energy, 0.95)) + 1
        rank_99 = int(np.searchsorted(energy, 0.99)) + 1
        top1_pct = float(energy[0]) * 100
        top10_pct = float(energy[min(9, len(energy)-1)]) * 100

        layer_ranks[l] = {"rank90": rank_90, "rank95": rank_95, "rank99": rank_99,
                          "rmse": rmse, "singular_values": S}

        ltype = "enc" if l < hparams.num_layers // 2 else "dec"
        print(f"{l:>5d} {rmse:>8.3f} {h_norm:>8.1f} {rel_err:>8.5f} "
              f"{rank_90:>7d} {rank_95:>7d} {rank_99:>7d} "
              f"{top1_pct:>6.1f}% {top10_pct:>7.1f}%")

    # Print singular value spectra for a few key layers
    print("\nSingular value spectra (top 20):")
    for l in [0, n_layers // 2, n_layers - 2, n_layers - 1]:
        S = layer_ranks[l]["singular_values"]
        s_str = ", ".join(f"{s:.2f}" for s in S[:20])
        print(f"  L{l}: [{s_str}, ...]")

    # =========================================================================
    # 2. Activation disagreement
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. ACTIVATION DISAGREEMENT (fraction of neurons flipping sign)")
    print("=" * 70)

    print(f"\n{'Layer':>5s} {'Disagree%':>10s} {'Interpretation':>40s}")
    print("-" * 60)
    for l in range(n_layers):
        disagree = np.mean(all_preact_disagree[l]) * 100
        interp = ""
        if disagree > 30:
            interp = "HIGH — significant nonlinear error"
        elif disagree > 15:
            interp = "moderate — some irreversible error"
        else:
            interp = "low — mostly linear regime"
        print(f"{l:>5d} {disagree:>9.1f}% {interp:>40s}")

    # =========================================================================
    # 3. Oracle ceiling: inject float hidden at layer L, run quant rest
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. ORACLE CEILING (inject float h at layer L, quant forward from L+1)")
    print("=" * 70)

    # Get full-model KL as reference
    total_kl_full = 0.0
    total_kl_oracle = np.zeros(n_layers)
    n_oracle_seqs = min(args.n_seqs, 4)

    print(f"\nRunning oracle injection ({n_oracle_seqs} seqs)...")
    for s in range(n_oracle_seqs):
        tokens = val_tokens[s * args.seq_len : (s + 1) * args.seq_len]
        if len(tokens) < args.seq_len:
            break

        f_logits, f_hidden, x0, f_enc = forward_collect_all(model_float, tokens)
        q_logits, q_hidden, _, q_enc = forward_collect_all(model_quant, tokens)
        total_kl_full += kl_div(f_logits, q_logits)

        for l in range(n_layers):
            # Inject float hidden at layer l, run QUANT model from l+1
            n_enc = model_quant.num_encoder_layers
            hybrid_enc = [None] * n_enc
            for j in range(n_enc):
                if j <= l:
                    hybrid_enc[j] = f_hidden[j]  # float hidden
                else:
                    hybrid_enc[j] = q_enc[j]

            injected_logits = forward_from_layer(
                model_quant, f_hidden[l], x0, l + 1, hybrid_enc)
            total_kl_oracle[l] += kl_div(f_logits, injected_logits)

    total_kl_full /= n_oracle_seqs
    total_kl_oracle /= n_oracle_seqs

    print(f"\nFull-model KL (float vs quant): {total_kl_full:.6f}")
    print(f"\n{'Layer':>5s} {'OracleKL':>10s} {'Recovery%':>10s} {'Residual%':>10s}")
    print("-" * 40)
    for l in range(n_layers):
        recovery = (1.0 - total_kl_oracle[l] / max(total_kl_full, 1e-12)) * 100
        residual = total_kl_oracle[l] / max(total_kl_full, 1e-12) * 100
        print(f"{l:>5d} {total_kl_oracle[l]:>10.6f} {recovery:>9.1f}% {residual:>9.1f}%")

    # =========================================================================
    # 4. Cross-layer correlation of errors
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. CROSS-LAYER ERROR CORRELATION")
    print("   (Can h_quant at other layers predict δh at target layer?)")
    print("=" * 70)

    # For each layer, compute correlation between δh[l] and h_quant[l-1], h_quant[l+1]
    # Use canonical correlation (top singular value of cross-covariance / product of norms)
    print(f"\n{'Target':>6s} {'Corr(δh_l, h_q_{l-1})':>22s} {'Corr(δh_l, h_q_l)':>20s} {'Corr(δh_l, h_q_{l+1})':>22s}")
    print("-" * 75)

    for l in range(n_layers):
        delta = np.concatenate(all_delta_h[l], axis=0)  # (n_tok, dim)
        delta_c = delta - delta.mean(axis=0, keepdims=True)

        results = {}
        for other, label in [(l-1, "l-1"), (l, "l"), (l+1, "l+1")]:
            if other < 0 or other >= n_layers:
                results[label] = "   —"
                continue
            h_q = np.concatenate(all_h_quant[other], axis=0)
            h_q_c = h_q - h_q.mean(axis=0, keepdims=True)

            # Top canonical correlation via SVD of cross-covariance
            # C = delta_c^T @ h_q_c / n
            n_tok = delta_c.shape[0]
            # Normalize columns
            d_norms = np.sqrt((delta_c ** 2).sum(axis=0, keepdims=True) + 1e-12)
            h_norms = np.sqrt((h_q_c ** 2).sum(axis=0, keepdims=True) + 1e-12)
            cross_cov = (delta_c / d_norms).T @ (h_q_c / h_norms) / n_tok
            top_sv = np.linalg.svd(cross_cov, compute_uv=False)[0]
            results[label] = f"{top_sv:>6.4f}"

        print(f"{l:>6d} {results.get('l-1', '   —'):>22s} "
              f"{results.get('l', '   —'):>20s} {results.get('l+1', '   —'):>22s}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    median_rank95 = int(np.median([v["rank95"] for v in layer_ranks.values()]))
    mean_disagree = np.mean([np.mean(d) for d in all_preact_disagree]) * 100
    best_oracle_layer = int(np.argmax([(1.0 - k / max(total_kl_full, 1e-12))
                                        for k in total_kl_oracle]))
    best_oracle_recovery = (1.0 - total_kl_oracle[best_oracle_layer] / max(total_kl_full, 1e-12)) * 100

    print(f"\n  Median rank_95 across layers: {median_rank95}")
    print(f"  → Bottleneck size should be ≥ {median_rank95} for 95% error energy")
    print(f"  Mean activation disagreement: {mean_disagree:.1f}%")
    print(f"  → ~{mean_disagree:.0f}% of error is from nonlinear activation flips (irreversible)")
    print(f"  Best single-layer oracle: L{best_oracle_layer} ({best_oracle_recovery:.1f}% KL recovery)")
    print(f"  Full-model KL: {total_kl_full:.6f}")


if __name__ == "__main__":
    main()
