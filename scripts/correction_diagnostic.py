#!/usr/bin/env python3
"""
correction_diagnostic.py — Diagnostic investigation: why MSE recovery ≠ BPB recovery.

Runs 7 experiments to determine whether linear quantization correction is feasible
and why 25-47% MSE recovery gives 0/negative BPB improvement.

Core test (Exp 2): OLS oracle linear regression — fits the mathematically optimal
linear correction on train data and evaluates BPB on val.

Usage:
    NUM_LAYERS=11 MLP_MULT=4.5 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \\
    BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=128 LOGIT_SOFTCAP=30.0 \\
    QUANT_ATTN_BITS=5 QUANT_MLP_BITS=5 \\
    .venv/bin/python3 scripts/correction_diagnostic.py logs/wd50_11L_5x_best.npz \\
        --n-corrections 3 --log-file logs/correction_diagnostic.txt
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
    build_sentencepiece_luts, eval_val,
)
from scripts.error_attribution import (
    forward_collect_all, forward_from_layer, kl_div,
    compute_logit_kl_impact, print_kl_impact_table,
)
from scripts.correction_mse_standalone import (
    forward_with_corrections, forward_collect_hidden,
    build_model, load_train_tokens,
)

log = logging.getLogger("correction_diag")


# =============================================================================
# Linear correction wrapper (for OLS / bias-only, compatible with forward_with_corrections)
# =============================================================================

class LinearCorrection:
    """Wraps numpy W, b as an MLX-callable correction for forward_with_corrections.

    forward_with_corrections does: x = x + correction(x)
    So __call__ returns the DELTA, not the corrected state.
    """

    def __init__(self, W: np.ndarray | None, b: np.ndarray):
        """W: (D, D) or None for bias-only. b: (D,)."""
        self.W = mx.array(W.astype(np.float32)) if W is not None else None
        self.b = mx.array(b.astype(np.float32))

    def __call__(self, h: mx.array) -> mx.array:
        if self.W is not None:
            return (h @ self.W + self.b).astype(h.dtype)
        return mx.broadcast_to(self.b, h.shape).astype(h.dtype)


# =============================================================================
# Helper: forward from layer collecting all downstream hidden states
# =============================================================================

def forward_from_layer_collect_hidden(model, h_start, x0, start_layer, encoder_outputs):
    """Like forward_from_layer but returns hidden states at all layers from start_layer onward."""
    n_enc = model.num_encoder_layers
    n_skip = model.num_skip_weights
    n_layers = len(model.blocks)

    enc_out = list(encoder_outputs)
    h = h_start
    hidden = []

    for i in range(start_layer, n_layers):
        if i >= n_enc:
            dec_j = i - n_enc
            if dec_j < n_skip:
                enc_j = n_enc - 1 - dec_j
                if enc_out[enc_j] is not None:
                    h = h + model.skip_weights[dec_j].astype(h.dtype)[None, None, :] * enc_out[enc_j]
        h = model.blocks[i](h, x0)
        mx.eval(h)
        hidden.append(h)
        if i < n_enc:
            enc_out[i] = h

    return hidden


# =============================================================================
# Helper: manual BPB computation (for oracle/injection experiments)
# =============================================================================

def compute_bpb_with_loss_fn(hparams, loss_fn, val_tokens, sp_luts, log_fn=None):
    """Compute BPB using eval_val with a custom loss function.

    loss_fn: (input_ids: mx.array, target_ids: mx.array) -> scalar loss (mean CE)
    """
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = sp_luts
    val_loss, val_bpb = eval_val(
        hparams, loss_fn, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        log_fn=log_fn,
    )
    return val_loss, val_bpb


def rms_norm_np(x):
    """RMSNorm in numpy (no learnable params)."""
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + 1e-6)


# =============================================================================
# Experiment 2: OLS Oracle Linear Regression (THE KEY TEST)
# =============================================================================

def run_ols_oracle(model_float, model_quant, hparams, train_tokens, val_tokens,
                   sp_luts, correction_layers, n_train_seqs, n_val_seqs, seq_len):
    """Fit optimal linear correction via OLS on train, evaluate BPB on val."""
    dim = hparams.model_dim
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT 2: OLS ORACLE LINEAR REGRESSION")
    log.info("=" * 70)

    # Phase 1: Accumulate normal equations on train data
    log.info(f"\nFitting OLS on train data ({n_train_seqs} seqs × {seq_len} tokens)...")
    ols_results = {}

    for layer in correction_layers:
        cov = np.zeros((dim + 1, dim + 1), dtype=np.float64)   # H_aug^T @ H_aug
        cross = np.zeros((dim + 1, dim), dtype=np.float64)      # H_aug^T @ Delta
        n_tokens_seen = 0

        for s in range(n_train_seqs):
            start = s * seq_len
            tokens = train_tokens[start:start + seq_len]
            if len(tokens) < seq_len:
                break

            fh = forward_collect_hidden(model_float, tokens)
            qh = forward_collect_hidden(model_quant, tokens)

            h_q = np.array(qh[layer].astype(mx.float32)).reshape(-1, dim)
            h_f = np.array(fh[layer].astype(mx.float32)).reshape(-1, dim)
            delta = (h_f - h_q).astype(np.float64)
            h_q_64 = h_q.astype(np.float64)

            # Augment with ones column for bias
            ones = np.ones((h_q_64.shape[0], 1), dtype=np.float64)
            h_aug = np.concatenate([h_q_64, ones], axis=1)  # (T, D+1)

            cov += h_aug.T @ h_aug
            cross += h_aug.T @ delta
            n_tokens_seen += h_q_64.shape[0]

            if (s + 1) % 8 == 0:
                log.info(f"  L{layer}: {s+1}/{n_train_seqs} seqs ({n_tokens_seen:,} tokens)")

        # Solve normal equations with regularization for stability
        reg = 1e-6 * np.trace(cov) / (dim + 1)
        cov += reg * np.eye(dim + 1, dtype=np.float64)
        Beta = np.linalg.solve(cov, cross)  # (D+1, D)
        W_star = Beta[:dim, :].astype(np.float32)   # (D, D)
        b_star = Beta[dim, :].astype(np.float32)     # (D,)

        # Train MSE and R²
        train_mse_total = 0.0
        train_baseline_mse_total = 0.0
        for s in range(min(n_train_seqs, 8)):
            tokens = train_tokens[s * seq_len:(s + 1) * seq_len]
            if len(tokens) < seq_len:
                break
            fh = forward_collect_hidden(model_float, tokens)
            qh = forward_collect_hidden(model_quant, tokens)
            h_q = np.array(qh[layer].astype(mx.float32)).reshape(-1, dim)
            h_f = np.array(fh[layer].astype(mx.float32)).reshape(-1, dim)
            delta = h_f - h_q
            pred = h_q @ W_star + b_star
            residual = delta - pred
            train_mse_total += (residual ** 2).mean()
            train_baseline_mse_total += (delta ** 2).mean()

        n_eval = min(n_train_seqs, 8)
        train_mse = train_mse_total / n_eval
        train_base_mse = train_baseline_mse_total / n_eval
        train_r2 = 1.0 - train_mse / max(train_base_mse, 1e-12)

        log.info(f"\n  L{layer} OLS fit ({n_tokens_seen:,} tokens):")
        log.info(f"    W* norm: {np.linalg.norm(W_star):.4f}  b* norm: {np.linalg.norm(b_star):.4f}")
        log.info(f"    Train MSE: {train_mse:.6f}  baseline: {train_base_mse:.6f}  R²={train_r2:.4f}")

        ols_results[layer] = {
            "W": W_star, "b": b_star,
            "train_r2": train_r2, "train_mse": train_mse,
            "train_base_mse": train_base_mse,
        }

    # Phase 2: Val MSE evaluation
    log.info(f"\nVal MSE evaluation ({n_val_seqs} seqs)...")
    for layer in correction_layers:
        W_star = ols_results[layer]["W"]
        b_star = ols_results[layer]["b"]
        val_mse_total = 0.0
        val_base_mse_total = 0.0
        n_val = 0

        for s in range(n_val_seqs):
            tokens = val_tokens[s * seq_len:(s + 1) * seq_len]
            if len(tokens) < seq_len:
                break
            fh = forward_collect_hidden(model_float, tokens)
            qh = forward_collect_hidden(model_quant, tokens)
            h_q = np.array(qh[layer].astype(mx.float32)).reshape(-1, dim)
            h_f = np.array(fh[layer].astype(mx.float32)).reshape(-1, dim)
            delta = h_f - h_q
            pred = h_q @ W_star + b_star
            residual = delta - pred
            val_mse_total += (residual ** 2).mean()
            val_base_mse_total += (delta ** 2).mean()
            n_val += 1

        val_mse = val_mse_total / n_val
        val_base_mse = val_base_mse_total / n_val
        val_r2 = 1.0 - val_mse / max(val_base_mse, 1e-12)
        ols_results[layer]["val_r2"] = val_r2
        ols_results[layer]["val_mse"] = val_mse
        ols_results[layer]["val_base_mse"] = val_base_mse

        log.info(f"  L{layer}: val MSE={val_mse:.6f}  baseline={val_base_mse:.6f}  "
                 f"R²={val_r2:.4f}  (train R²={ols_results[layer]['train_r2']:.4f})")

    # Phase 3: BPB evaluation with OLS corrections injected
    log.info("\nBPB evaluation with OLS corrections...")
    correction_map = {}
    for layer in correction_layers:
        correction_map[layer] = LinearCorrection(
            ols_results[layer]["W"], ols_results[layer]["b"])

    def ols_corrected_loss(input_ids, target_ids):
        h = forward_with_corrections(model_quant, input_ids, correction_map)
        h = h.reshape(-1, model_quant.tok_emb.weight.shape[1])
        logits = model_quant._apply_logit_processing(
            h @ model_quant.tok_emb.weight.astype(h.dtype).T)
        return nn.losses.cross_entropy(
            logits.astype(mx.float32), target_ids.reshape(-1), reduction="mean")

    ols_loss, ols_bpb = compute_bpb_with_loss_fn(
        hparams, ols_corrected_loss, val_tokens, sp_luts,
        log_fn=lambda msg: log.info(f"  {msg}"))

    ols_results["bpb"] = ols_bpb
    ols_results["loss"] = ols_loss
    log.info(f"\n  OLS corrected: val_loss={ols_loss:.6f}  val_bpb={ols_bpb:.6f}")

    return ols_results


# =============================================================================
# Experiment 1: Oracle Ceiling in BPB
# =============================================================================

def run_oracle_ceiling_bpb(model_float, model_quant, hparams, val_tokens, sp_luts,
                           correction_layers, n_seqs, seq_len):
    """Oracle ceiling: inject h_float at correction layers, measure BPB."""
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT 1: ORACLE CEILING (BPB)")
    log.info("=" * 70)
    n_layers = len(model_float.blocks)
    results = {}

    # Single-layer oracle injections
    for target_layer in correction_layers:
        log.info(f"\n  Oracle injection at L{target_layer}...")

        def make_oracle_loss(inject_layer):
            def oracle_loss(input_ids, target_ids):
                # Process one sequence at a time for oracle injection
                B = input_ids.shape[0]
                total_loss = mx.array(0.0)
                for b in range(B):
                    tokens_np = np.array(input_ids[b])
                    _, f_hidden, _, f_enc = forward_collect_all(model_float, tokens_np)
                    _, _, q_x0, q_enc = forward_collect_all(model_quant, tokens_np)

                    # Build hybrid encoder outputs
                    n_enc = model_quant.num_encoder_layers
                    hybrid_enc = [None] * n_enc
                    for j in range(n_enc):
                        if j <= inject_layer:
                            hybrid_enc[j] = f_hidden[j]
                        else:
                            hybrid_enc[j] = q_enc[j]

                    inj_logits = forward_from_layer(
                        model_quant, f_hidden[inject_layer], q_x0, inject_layer + 1, hybrid_enc)
                    inj_logits = inj_logits.reshape(-1, inj_logits.shape[-1])
                    tgt = target_ids[b].reshape(-1)
                    total_loss = total_loss + nn.losses.cross_entropy(
                        inj_logits.astype(mx.float32), tgt, reduction="mean")
                return total_loss / B
            return oracle_loss

        loss, bpb = compute_bpb_with_loss_fn(
            hparams, make_oracle_loss(target_layer), val_tokens, sp_luts,
            log_fn=lambda msg: log.info(f"    {msg}"))
        results[f"oracle_L{target_layer}"] = {"loss": loss, "bpb": bpb}
        log.info(f"    L{target_layer}: val_bpb={bpb:.6f}")

    # All correction layers simultaneously
    log.info(f"\n  Oracle injection at ALL correction layers {correction_layers}...")

    def oracle_all_loss(input_ids, target_ids):
        B = input_ids.shape[0]
        total_loss = mx.array(0.0)
        for b in range(B):
            tokens_np = np.array(input_ids[b])
            _, f_hidden, _, f_enc = forward_collect_all(model_float, tokens_np)
            _, q_hidden, q_x0, q_enc = forward_collect_all(model_quant, tokens_np)

            # For multi-layer injection, we inject at the earliest correction layer
            # and rebuild forward from there, injecting float hidden at each correction point
            # For simplicity, inject at earliest and propagate through quant model
            # replacing hidden at each correction layer with float hidden
            first_layer = correction_layers[0]
            n_enc = model_quant.num_encoder_layers
            n_skip = model_quant.num_skip_weights

            # Build hybrid enc with float hidden for all correction layers
            hybrid_enc = [None] * n_enc
            for j in range(n_enc):
                if j in correction_layers or j < first_layer:
                    hybrid_enc[j] = f_hidden[j]
                else:
                    hybrid_enc[j] = q_enc[j]

            # Forward from first correction layer, replacing at each correction point
            h = f_hidden[first_layer]
            enc_out = list(hybrid_enc)

            for i in range(first_layer + 1, len(model_quant.blocks)):
                if i >= n_enc:
                    dec_j = i - n_enc
                    if dec_j < n_skip:
                        enc_j = n_enc - 1 - dec_j
                        if enc_out[enc_j] is not None:
                            h = h + model_quant.skip_weights[dec_j].astype(h.dtype)[None, None, :] * enc_out[enc_j]
                h = model_quant.blocks[i](h, q_x0)
                mx.eval(h)
                if i < n_enc:
                    enc_out[i] = h
                # Inject float hidden at correction layers
                if i in correction_layers:
                    h = f_hidden[i]

            logits = model_quant._apply_logit_processing(
                model_quant.tok_emb.as_linear(model_quant.final_norm(h)))
            logits = logits.reshape(-1, logits.shape[-1])
            tgt = target_ids[b].reshape(-1)
            total_loss = total_loss + nn.losses.cross_entropy(
                logits.astype(mx.float32), tgt, reduction="mean")
        return total_loss / B

    loss, bpb = compute_bpb_with_loss_fn(
        hparams, oracle_all_loss, val_tokens, sp_luts,
        log_fn=lambda msg: log.info(f"    {msg}"))
    results["oracle_all"] = {"loss": loss, "bpb": bpb}
    log.info(f"    All layers: val_bpb={bpb:.6f}")

    return results


# =============================================================================
# Experiment 3: Angular Analysis (RMSNorm Effect)
# =============================================================================

def run_angular_analysis(model_float, model_quant, hparams, val_tokens,
                         correction_layers, ols_corrections, n_seqs, seq_len):
    """Measure angular error and whether corrections fix direction or magnitude."""
    dim = hparams.model_dim
    n_layers = len(model_float.blocks)
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT 3: ANGULAR ANALYSIS (RMSNorm Effect)")
    log.info("=" * 70)

    # Measure at correction layers AND final layer
    measure_layers = sorted(set(correction_layers) | {n_layers - 1})

    stats = {l: {"cos_base": [], "cos_ols": [], "mag_err_frac": [],
                 "dir_err_frac": [], "delta_norm": [], "h_q_norm": []}
             for l in measure_layers}

    for s in range(n_seqs):
        tokens = val_tokens[s * seq_len:(s + 1) * seq_len]
        if len(tokens) < seq_len:
            break

        fh = forward_collect_hidden(model_float, tokens)
        qh = forward_collect_hidden(model_quant, tokens)

        for l in measure_layers:
            h_f = np.array(fh[l].astype(mx.float32)).reshape(-1, dim)
            h_q = np.array(qh[l].astype(mx.float32)).reshape(-1, dim)
            delta = h_f - h_q

            # RMSNorm both
            h_f_n = rms_norm_np(h_f)
            h_q_n = rms_norm_np(h_q)

            # Cosine similarity after RMSNorm (per-token, then average)
            norms_f = np.linalg.norm(h_f_n, axis=-1, keepdims=False)
            norms_q = np.linalg.norm(h_q_n, axis=-1, keepdims=False)
            cos_base = (h_f_n * h_q_n).sum(axis=-1) / (norms_f * norms_q + 1e-12)
            stats[l]["cos_base"].append(cos_base.mean())

            # Decompose delta into magnitude component and direction component
            # magnitude component: projection of delta onto h_q direction
            h_q_dir = h_q / (np.linalg.norm(h_q, axis=-1, keepdims=True) + 1e-12)
            mag_proj = (delta * h_q_dir).sum(axis=-1, keepdims=True) * h_q_dir
            dir_comp = delta - mag_proj

            mag_energy = (mag_proj ** 2).sum(axis=-1).mean()
            dir_energy = (dir_comp ** 2).sum(axis=-1).mean()
            total_energy = mag_energy + dir_energy
            stats[l]["mag_err_frac"].append(mag_energy / max(total_energy, 1e-12))
            stats[l]["dir_err_frac"].append(dir_energy / max(total_energy, 1e-12))
            stats[l]["delta_norm"].append(np.sqrt((delta ** 2).mean()))
            stats[l]["h_q_norm"].append(np.sqrt((h_q ** 2).mean()))

            # If OLS correction available, measure angular improvement
            if l in ols_corrections:
                W = ols_corrections[l]["W"]
                b = ols_corrections[l]["b"]
                correction = h_q @ W + b
                h_c = h_q + correction
                h_c_n = rms_norm_np(h_c)
                norms_c = np.linalg.norm(h_c_n, axis=-1, keepdims=False)
                cos_ols = (h_f_n * h_c_n).sum(axis=-1) / (norms_f * norms_c + 1e-12)
                stats[l]["cos_ols"].append(cos_ols.mean())

                # Decompose OLS correction too
                ols_mag = (correction * h_q_dir).sum(axis=-1, keepdims=True) * h_q_dir
                ols_dir = correction - ols_mag
                ols_mag_e = (ols_mag ** 2).sum(axis=-1).mean()
                ols_dir_e = (ols_dir ** 2).sum(axis=-1).mean()
                ols_total = ols_mag_e + ols_dir_e
                if "ols_mag_frac" not in stats[l]:
                    stats[l]["ols_mag_frac"] = []
                    stats[l]["ols_dir_frac"] = []
                stats[l]["ols_mag_frac"].append(ols_mag_e / max(ols_total, 1e-12))
                stats[l]["ols_dir_frac"].append(ols_dir_e / max(ols_total, 1e-12))

        if (s + 1) % 8 == 0:
            log.info(f"  {s+1}/{n_seqs} seqs")

    # Print results
    log.info(f"\n{'Layer':>5s} {'cos(f,q)':>10s} {'cos(f,ols)':>11s} "
             f"{'‖δh‖':>8s} {'‖h_q‖':>8s} {'RelErr':>8s} "
             f"{'%Mag':>6s} {'%Dir':>6s} {'OLS%Mag':>8s} {'OLS%Dir':>8s}")
    log.info("-" * 95)

    for l in measure_layers:
        cos_b = np.mean(stats[l]["cos_base"])
        delta_n = np.mean(stats[l]["delta_norm"])
        hq_n = np.mean(stats[l]["h_q_norm"])
        mag_f = np.mean(stats[l]["mag_err_frac"]) * 100
        dir_f = np.mean(stats[l]["dir_err_frac"]) * 100

        cos_o = np.mean(stats[l]["cos_ols"]) if stats[l]["cos_ols"] else float("nan")
        ols_mag = np.mean(stats[l].get("ols_mag_frac", [float("nan")])) * 100
        ols_dir = np.mean(stats[l].get("ols_dir_frac", [float("nan")])) * 100

        tag = " *CORR" if l in correction_layers else ""
        log.info(f"{l:>5d} {cos_b:>10.6f} {cos_o:>11.6f} "
                 f"{delta_n:>8.4f} {hq_n:>8.1f} {delta_n/max(hq_n,1e-12):>8.5f} "
                 f"{mag_f:>5.1f}% {dir_f:>5.1f}% {ols_mag:>7.1f}% {ols_dir:>7.1f}%{tag}")

    # Interpretation
    avg_cos_base = np.mean([np.mean(stats[l]["cos_base"]) for l in correction_layers])
    avg_mag_frac = np.mean([np.mean(stats[l]["mag_err_frac"]) for l in correction_layers]) * 100
    log.info(f"\n  Mean cos(float, quant) at correction layers: {avg_cos_base:.6f}")
    log.info(f"  Mean magnitude fraction of error: {avg_mag_frac:.1f}%")
    if avg_mag_frac > 60:
        log.info("  → Most error is in MAGNITUDE (RMSNorm erases it). MSE improvements don't help BPB.")
    elif avg_mag_frac < 40:
        log.info("  → Most error is in DIRECTION (affects logits). MSE should correlate with BPB.")
    else:
        log.info("  → Error is mixed magnitude/direction.")

    return stats


# =============================================================================
# Experiment 7: Bias-Only Correction
# =============================================================================

def run_bias_only(model_float, model_quant, hparams, train_tokens, val_tokens,
                  sp_luts, correction_layers, n_train_seqs, seq_len):
    """Simplest correction: per-layer additive bias b* = mean(δh)."""
    dim = hparams.model_dim
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT 7: BIAS-ONLY CORRECTION")
    log.info("=" * 70)

    biases = {}
    for layer in correction_layers:
        bias_sum = np.zeros(dim, dtype=np.float64)
        n_tokens = 0
        for s in range(n_train_seqs):
            tokens = train_tokens[s * seq_len:(s + 1) * seq_len]
            if len(tokens) < seq_len:
                break
            fh = forward_collect_hidden(model_float, tokens)
            qh = forward_collect_hidden(model_quant, tokens)
            h_f = np.array(fh[layer].astype(mx.float32)).reshape(-1, dim)
            h_q = np.array(qh[layer].astype(mx.float32)).reshape(-1, dim)
            delta = h_f - h_q
            bias_sum += delta.sum(axis=0).astype(np.float64)
            n_tokens += h_f.shape[0]

        b_star = (bias_sum / n_tokens).astype(np.float32)
        biases[layer] = b_star
        log.info(f"  L{layer}: bias norm={np.linalg.norm(b_star):.4f}  "
                 f"mean_delta_norm={np.sqrt((b_star ** 2).sum()):.4f}  "
                 f"({n_tokens:,} tokens)")

    # BPB eval
    correction_map = {l: LinearCorrection(None, biases[l]) for l in correction_layers}

    def bias_corrected_loss(input_ids, target_ids):
        h = forward_with_corrections(model_quant, input_ids, correction_map)
        h = h.reshape(-1, model_quant.tok_emb.weight.shape[1])
        logits = model_quant._apply_logit_processing(
            h @ model_quant.tok_emb.weight.astype(h.dtype).T)
        return nn.losses.cross_entropy(
            logits.astype(mx.float32), target_ids.reshape(-1), reduction="mean")

    loss, bpb = compute_bpb_with_loss_fn(
        hparams, bias_corrected_loss, val_tokens, sp_luts,
        log_fn=lambda msg: log.info(f"  {msg}"))
    log.info(f"\n  Bias-only corrected: val_loss={loss:.6f}  val_bpb={bpb:.6f}")

    # Overhead
    total_params = len(correction_layers) * dim
    log.info(f"  Overhead: {total_params:,} params ({total_params/1024:.1f} KB int8)")

    return {"bpb": bpb, "loss": loss, "biases": biases}


# =============================================================================
# Experiment 5: Downstream Error Propagation
# =============================================================================

def run_downstream_propagation(model_float, model_quant, hparams, val_tokens,
                               correction_layers, ols_corrections, n_seqs, seq_len):
    """Trace how OLS corrections propagate through downstream layers."""
    dim = hparams.model_dim
    n_layers = len(model_float.blocks)
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT 5: DOWNSTREAM ERROR PROPAGATION")
    log.info("=" * 70)

    # For each correction layer, inject corrected hidden and trace downstream
    for target_layer in correction_layers:
        if target_layer not in ols_corrections:
            continue
        W = ols_corrections[target_layer]["W"]
        b = ols_corrections[target_layer]["b"]

        log.info(f"\n  Injecting OLS correction at L{target_layer}, tracing downstream...")

        # Accumulate per-layer RMSE: baseline (quant), corrected, float
        rmse_quant = np.zeros(n_layers)
        rmse_corrected = np.zeros(n_layers)
        count = 0

        for s in range(n_seqs):
            tokens = val_tokens[s * seq_len:(s + 1) * seq_len]
            if len(tokens) < seq_len:
                break

            _, f_hidden, _, f_enc = forward_collect_all(model_float, tokens)
            _, q_hidden, q_x0, q_enc = forward_collect_all(model_quant, tokens)

            # Apply OLS correction at target_layer
            h_q_np = np.array(q_hidden[target_layer].astype(mx.float32)).reshape(-1, dim)
            correction = h_q_np @ W + b
            h_corrected = mx.array((h_q_np + correction).astype(np.float32)).reshape(q_hidden[target_layer].shape)

            # Build hybrid encoder outputs for corrected path
            n_enc = model_quant.num_encoder_layers
            hybrid_enc = [None] * n_enc
            for j in range(n_enc):
                if j <= target_layer:
                    hybrid_enc[j] = h_corrected if j == target_layer else q_enc[j]
                else:
                    hybrid_enc[j] = q_enc[j]

            # Collect downstream hidden states: corrected
            corr_downstream = forward_from_layer_collect_hidden(
                model_quant, h_corrected, q_x0, target_layer + 1, hybrid_enc)

            # Collect downstream hidden states: uncorrected (quant baseline)
            hybrid_enc_q = list(q_enc)
            quant_downstream = forward_from_layer_collect_hidden(
                model_quant, q_hidden[target_layer], q_x0, target_layer + 1, hybrid_enc_q)

            # Compare RMSE at each downstream layer
            for k, i in enumerate(range(target_layer + 1, n_layers)):
                f_h = np.array(f_hidden[i].astype(mx.float32)).reshape(-1, dim)

                q_h = np.array(quant_downstream[k].astype(mx.float32)).reshape(-1, dim)
                c_h = np.array(corr_downstream[k].astype(mx.float32)).reshape(-1, dim)

                rmse_quant[i] += np.sqrt(((q_h - f_h) ** 2).mean())
                rmse_corrected[i] += np.sqrt(((c_h - f_h) ** 2).mean())

            count += 1

        rmse_quant /= max(count, 1)
        rmse_corrected /= max(count, 1)

        log.info(f"\n  {'Layer':>5s} {'RMSE_quant':>11s} {'RMSE_corr':>11s} {'Δ%':>8s}")
        log.info("  " + "-" * 40)
        for i in range(target_layer + 1, n_layers):
            rq = rmse_quant[i]
            rc = rmse_corrected[i]
            delta_pct = (rc - rq) / max(rq, 1e-12) * 100
            log.info(f"  {i:>5d} {rq:>11.6f} {rc:>11.6f} {delta_pct:>+7.1f}%")


# =============================================================================
# Experiment 4: Logit-Space Error Structure
# =============================================================================

def run_logit_space_analysis(model_float, model_quant, hparams, val_tokens, n_seqs, seq_len):
    """SVD of logit-space error, softcap effect."""
    dim = hparams.model_dim
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT 4: LOGIT-SPACE ERROR STRUCTURE")
    log.info("=" * 70)

    W = np.array(model_float.tok_emb.weight.astype(mx.float32))  # (V, D)
    cap = hparams.logit_softcap

    all_logit_err_pre = []
    all_logit_err_post = []
    all_kl = []

    for s in range(n_seqs):
        tokens = val_tokens[s * seq_len:(s + 1) * seq_len]
        if len(tokens) < seq_len:
            break

        fh = forward_collect_hidden(model_float, tokens)
        qh = forward_collect_hidden(model_quant, tokens)

        h_f = np.array(fh[-1].astype(mx.float32)).reshape(-1, dim)
        h_q = np.array(qh[-1].astype(mx.float32)).reshape(-1, dim)

        h_f_n = rms_norm_np(h_f)
        h_q_n = rms_norm_np(h_q)

        logits_f = h_f_n @ W.T
        logits_q = h_q_n @ W.T
        err_pre = logits_q - logits_f
        all_logit_err_pre.append(err_pre)

        logits_f_cap = cap * np.tanh(logits_f / cap)
        logits_q_cap = cap * np.tanh(logits_q / cap)
        err_post = logits_q_cap - logits_f_cap
        all_logit_err_post.append(err_post)

        all_kl.append(kl_div(
            mx.array(logits_f_cap[np.newaxis]),
            mx.array(logits_q_cap[np.newaxis])))

    err_pre = np.concatenate(all_logit_err_pre, axis=0)
    err_post = np.concatenate(all_logit_err_post, axis=0)

    # SVD of logit error (pre-softcap)
    err_centered = err_pre - err_pre.mean(axis=0, keepdims=True)
    # Subsample for SVD if too many tokens
    if err_centered.shape[0] > 4096:
        idx = np.random.choice(err_centered.shape[0], 4096, replace=False)
        err_sub = err_centered[idx]
    else:
        err_sub = err_centered
    _, S, _ = np.linalg.svd(err_sub, full_matrices=False)
    energy = (S ** 2).cumsum() / (S ** 2).sum()
    rank_90 = int(np.searchsorted(energy, 0.90)) + 1
    rank_95 = int(np.searchsorted(energy, 0.95)) + 1
    rank_99 = int(np.searchsorted(energy, 0.99)) + 1

    rmse_pre = np.sqrt((err_pre ** 2).mean())
    rmse_post = np.sqrt((err_post ** 2).mean())
    mean_kl = np.mean(all_kl)

    log.info(f"\n  Logit error RMSE (pre-softcap):  {rmse_pre:.4f}")
    log.info(f"  Logit error RMSE (post-softcap): {rmse_post:.4f}  "
             f"({rmse_pre/max(rmse_post,1e-12):.1f}x reduction)")
    log.info(f"  Mean KL: {mean_kl:.6f}")
    log.info(f"\n  Logit error rank (pre-softcap):")
    log.info(f"    rank_90={rank_90}  rank_95={rank_95}  rank_99={rank_99}  (of {S.shape[0]})")
    log.info(f"    Top-1 energy: {energy[0]*100:.1f}%  Top-10: {energy[min(9,len(energy)-1)]*100:.1f}%")

    return {
        "rmse_pre": rmse_pre, "rmse_post": rmse_post,
        "rank_90": rank_90, "rank_95": rank_95, "rank_99": rank_99,
        "kl": mean_kl,
    }


# =============================================================================
# Experiment 6: Train vs Val MSE
# =============================================================================

def run_train_vs_val_mse(model_float, model_quant, hparams, ols_corrections,
                         correction_layers, train_tokens, val_tokens, n_seqs, seq_len):
    """Compare OLS MSE recovery on train vs val data."""
    dim = hparams.model_dim
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT 6: TRAIN vs VAL MSE (OLS)")
    log.info("=" * 70)

    for data_name, tokens in [("train", train_tokens), ("val", val_tokens)]:
        log.info(f"\n  {data_name} data ({n_seqs} seqs):")
        for layer in correction_layers:
            if layer not in ols_corrections:
                continue
            W = ols_corrections[layer]["W"]
            b = ols_corrections[layer]["b"]
            mse_base_total = 0.0
            mse_corr_total = 0.0
            n = 0
            for s in range(n_seqs):
                tok = tokens[s * seq_len:(s + 1) * seq_len]
                if len(tok) < seq_len:
                    break
                fh = forward_collect_hidden(model_float, tok)
                qh = forward_collect_hidden(model_quant, tok)
                h_f = np.array(fh[layer].astype(mx.float32)).reshape(-1, dim)
                h_q = np.array(qh[layer].astype(mx.float32)).reshape(-1, dim)
                delta = h_f - h_q
                pred = h_q @ W + b
                mse_base_total += (delta ** 2).mean()
                mse_corr_total += ((delta - pred) ** 2).mean()
                n += 1
            r2 = 1.0 - (mse_corr_total / n) / max(mse_base_total / n, 1e-12)
            log.info(f"    L{layer}: base_mse={mse_base_total/n:.6f}  "
                     f"corr_mse={mse_corr_total/n:.6f}  R²={r2:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic: why MSE recovery ≠ BPB recovery")
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument("--correction-layers", type=str, default=None,
                        help="Override: comma-separated layer indices")
    parser.add_argument("--n-corrections", type=int, default=3,
                        help="Auto-select top-N layers by marginal KL")
    parser.add_argument("--n-train-seqs", type=int, default=32)
    parser.add_argument("--n-val-seqs", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--experiments", type=str, default="all",
                        help="Comma-separated experiment numbers (1-7) or 'all'")
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    hparams = Hyperparameters()

    # Logging
    global log
    log = logging.getLogger("correction_diag")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if args.log_file is None:
        bits = hparams.quant_attn_bits
        args.log_file = f"logs/correction_diagnostic_int{bits}.txt"
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    fh = logging.FileHandler(args.log_file, mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.info(f"Logging to {args.log_file}")

    # Parse experiment selection
    if args.experiments == "all":
        experiments = {1, 2, 3, 4, 5, 6, 7}
    else:
        experiments = {int(x) for x in args.experiments.split(",")}

    log.info(f"Config: {hparams.num_layers}L/{hparams.model_dim}d, MLP {hparams.mlp_mult}x, "
             f"act={hparams.mlp_act}, quant=a{hparams.quant_attn_bits}m{hparams.quant_mlp_bits}")
    log.info(f"Experiments: {sorted(experiments)}")

    # --- Build models ---
    log.info(f"\nLoading checkpoint: {args.checkpoint}")
    model_float = build_model(hparams)
    flat = dict(mx.load(args.checkpoint))
    model_float.update(tree_unflatten(list(flat.items())))
    mx.eval(model_float.parameters())

    log.info("Quantizing model...")
    cat_bits = {"attn": hparams.quant_attn_bits, "mlp": hparams.quant_mlp_bits}
    quant_bits_str = os.environ.get("QUANT_BITS", "")
    if quant_bits_str:
        cat_bits = {}
        for part in quant_bits_str.split(","):
            k, v = part.strip().rsplit(":", 1)
            cat_bits[k.strip()] = int(v.strip())
        log.info(f"Per-layer quant: {cat_bits}")
    quant_obj, _ = quantize_state_dict_int8(flat, cat_bits=cat_bits)
    quant_flat = dequantize_state_dict_int8(quant_obj)
    model_quant = build_model(hparams)
    model_quant.update(tree_unflatten(list(quant_flat.items())))
    mx.eval(model_quant.parameters())

    # --- Load data ---
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    log.info(f"Val tokens: {len(val_tokens):,}")
    train_tokens = load_train_tokens(hparams, max_tokens=args.n_train_seqs * args.seq_len + 2048)
    log.info(f"Train tokens: {len(train_tokens):,}")

    # --- Sentencepiece LUTs for BPB ---
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=hparams.tokenizer_path)
    sp_luts = build_sentencepiece_luts(sp, hparams.vocab_size)

    # --- Auto-select correction layers ---
    if args.correction_layers is not None:
        correction_layers = [int(x) for x in args.correction_layers.split(",")]
        log.info(f"Override: correction layers = {correction_layers}")
    else:
        log.info(f"\n=== Auto-selecting correction layers (top-{args.n_corrections} by marginal KL) ===")
        res = compute_logit_kl_impact(
            model_float, model_quant, val_tokens,
            n_seqs=4, seq_len=args.seq_len)
        print_kl_impact_table(res)
        correction_layers = sorted(
            [int(x) for x in np.argsort(res["marginal_kl"])[::-1][:args.n_corrections]])
        log.info(f"Selected: {correction_layers}")

    # --- Baselines ---
    log.info("\n" + "=" * 70)
    log.info("BASELINES")
    log.info("=" * 70)

    def float_loss(x, y):
        return model_float.loss(x, y)

    def quant_loss(x, y):
        return model_quant.loss(x, y)

    float_val_loss, float_bpb = compute_bpb_with_loss_fn(
        hparams, float_loss, val_tokens, sp_luts,
        log_fn=lambda msg: log.info(f"  float: {msg}"))
    log.info(f"  Float: val_loss={float_val_loss:.6f}  val_bpb={float_bpb:.6f}")

    quant_val_loss, quant_bpb = compute_bpb_with_loss_fn(
        hparams, quant_loss, val_tokens, sp_luts,
        log_fn=lambda msg: log.info(f"  quant: {msg}"))
    log.info(f"  Quant: val_loss={quant_val_loss:.6f}  val_bpb={quant_bpb:.6f}")

    gap = quant_bpb - float_bpb
    log.info(f"  Gap:   {gap:.6f} BPB")

    # --- Run experiments ---
    ols_results = None

    if 2 in experiments:
        ols_results = run_ols_oracle(
            model_float, model_quant, hparams, train_tokens, val_tokens,
            sp_luts, correction_layers, args.n_train_seqs, args.n_val_seqs, args.seq_len)

    oracle_results = None
    if 1 in experiments:
        oracle_results = run_oracle_ceiling_bpb(
            model_float, model_quant, hparams, val_tokens, sp_luts,
            correction_layers, args.n_val_seqs, args.seq_len)

    if 3 in experiments:
        ols_corr = ols_results if ols_results else {}
        run_angular_analysis(
            model_float, model_quant, hparams, val_tokens,
            correction_layers, ols_corr, min(args.n_val_seqs, 16), args.seq_len)

    if 7 in experiments:
        bias_results = run_bias_only(
            model_float, model_quant, hparams, train_tokens, val_tokens,
            sp_luts, correction_layers, args.n_train_seqs, args.seq_len)

    if 5 in experiments and ols_results:
        run_downstream_propagation(
            model_float, model_quant, hparams, val_tokens,
            correction_layers, ols_results, min(args.n_val_seqs, 8), args.seq_len)

    if 4 in experiments:
        run_logit_space_analysis(
            model_float, model_quant, hparams, val_tokens,
            min(args.n_val_seqs, 8), args.seq_len)

    if 6 in experiments and ols_results:
        run_train_vs_val_mse(
            model_float, model_quant, hparams, ols_results,
            correction_layers, train_tokens, val_tokens,
            min(args.n_val_seqs, 16), args.seq_len)

    # --- Summary ---
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"\n{'Method':>25s} {'BPB':>10s} {'ΔBPB':>10s} {'Recovery%':>10s}")
    log.info("-" * 58)

    def row(name, bpb):
        delta = quant_bpb - bpb
        recov = delta / max(gap, 1e-9) * 100
        log.info(f"{name:>25s} {bpb:>10.6f} {delta:>+10.6f} {recov:>9.1f}%")

    row("Float (oracle ceiling)", float_bpb)
    row("Quant (baseline)", quant_bpb)

    if oracle_results:
        for l in correction_layers:
            key = f"oracle_L{l}"
            if key in oracle_results:
                row(f"Oracle inject L{l}", oracle_results[key]["bpb"])
        if "oracle_all" in oracle_results:
            row("Oracle inject ALL", oracle_results["oracle_all"]["bpb"])

    if ols_results and "bpb" in ols_results:
        row("OLS linear correction", ols_results["bpb"])
        log.info(f"\n  OLS R² (train/val per layer):")
        for l in correction_layers:
            if l in ols_results:
                log.info(f"    L{l}: train={ols_results[l]['train_r2']:.4f}  "
                         f"val={ols_results[l]['val_r2']:.4f}")

    if 7 in experiments:
        row("Bias-only correction", bias_results["bpb"])

    log.info(f"\n  Quant gap: {gap:.6f} BPB")
    log.info(f"  Correction layers: {correction_layers}")


if __name__ == "__main__":
    main()
