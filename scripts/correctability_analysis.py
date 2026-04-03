#!/usr/bin/env python3
"""
correctability_analysis.py — Manifold distortion & pre-nonlinearity correction analysis.

Quantization warps the activation manifold: some directions get amplified, others
compressed by per-row scaling + rounding. This distortion is LINEARLY CORRECTABLE
up until it passes through the nonlinearity (relu2/lrelu2). Once errors cause sign
flips at the nonlinearity boundary, they become irrecoverable by affine maps.

This script measures, at each layer:
  1. Pre-nonlinearity affine correction: fit z_float_pre_act ~ W @ z_quant_pre_act + b
     and measure R² / MSE reduction (how much is linearly correctable BEFORE relu2)
  2. Post-nonlinearity R²: same fit after relu2 squaring (the irrecoverable regime)
  3. ReLU flip rate: fraction of pre-activations with different signs between
     float and quant (the nonlinearity crossing rate — boundary between correctable and not)
  4. Error amplification through layers: how distortion compounds
  5. Correctability score per layer (composite metric from quant_aware_training)

Based on the framework in ~/Projects/quant_aware_training (aleph/qgeom/).

Usage:
    NUM_LAYERS=11 MLP_MULT=4.5 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \
    QUANT_ATTN_BITS=5 QUANT_MLP_BITS=5 \
    .venv/bin/python3 scripts/correctability_analysis.py logs/wd50_11L_5x_best.npz \
        --n-seqs 32 --log-file logs/correctability_int5.txt
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
    COMPUTE_DTYPE, Hyperparameters, load_validation_tokens,
    quantize_state_dict_int8, dequantize_state_dict_int8, rms_norm,
)
from scripts.eval_commons import build_model
from scripts.error_decomposition import block_forward_decomposed

log = logging.getLogger("correctability")


# ============================================================================
# Forward pass collecting pre/post nonlinearity activations
# ============================================================================

def forward_collect_prepost_act(model, tokens):
    """Run forward pass, collecting pre-act and post-act at each MLP.

    Returns per layer:
        pre_act:  output of mlp.fc BEFORE nonlinearity  (1, T, mlp_hidden)
        post_act: AFTER nonlinearity (relu2 squaring)   (1, T, mlp_hidden)
        h_out:    full block output                      (1, T, dim)
        post_attn: hidden state entering MLP norm        (1, T, dim)
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
    results = []

    for i in range(n_layers):
        if i >= n_enc:
            dec_j = i - n_enc
            if dec_j < n_skip:
                enc_j = n_enc - 1 - dec_j
                if encoder_outputs[enc_j] is not None:
                    h = h + model.skip_weights[dec_j].astype(h.dtype)[None, None, :] * encoder_outputs[enc_j]

        decomposed = block_forward_decomposed(model.blocks[i], h, x0)
        mx.eval(decomposed['pre_act'], decomposed['post_act'],
                decomposed['h_out'], decomposed['post_attn'],
                decomposed['mlp_normed'])
        results.append(decomposed)

        h = decomposed['h_out']
        if i < n_enc:
            encoder_outputs[i] = h

    return results


# ============================================================================
# Affine correction fitting (ridge-regularized, following quant_aware_training)
# ============================================================================

def fit_affine_correction(X: np.ndarray, Y: np.ndarray, ridge: float = 1.0
                          ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit affine map Y ≈ X @ W + b via ridge regression.

    Args:
        X: (n, d_in) — quant activations (input features)
        Y: (n, d_out) — float activations (targets)
        ridge: regularization strength

    Returns:
        W: (d_in, d_out) linear map
        b: (d_out,) bias
        mse_before: MSE(Y, X) — error before correction
        mse_after:  MSE(Y, X @ W + b) — error after correction
    """
    n, d_in = X.shape
    _, d_out = Y.shape

    # Augment X with bias column
    X_aug = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)

    # Ridge: (X^T X + λI)^{-1} X^T Y
    XtX = X_aug.T @ X_aug
    XtX[np.diag_indices(d_in + 1)] += ridge * n
    XtY = X_aug.T @ Y

    try:
        sol = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(XtX, XtY, rcond=None)[0]

    W = sol[:d_in]       # (d_in, d_out)
    b = sol[d_in]        # (d_out,)

    Y_pred = X @ W + b
    mse_before = float(np.mean((Y - X) ** 2)) if d_in == d_out else float(np.mean(Y ** 2))
    mse_after = float(np.mean((Y - Y_pred) ** 2))

    return W, b, mse_before, mse_after


# ============================================================================
# Fate metrics (from quant_aware_training aleph/qgeom/metrics.py)
# ============================================================================

def compute_fate_metrics(pre_act_float: np.ndarray, pre_act_quant: np.ndarray
                         ) -> dict[str, float]:
    """Compute nonlinearity fate metrics for pre-activation vectors.

    For relu2/lrelu2: the critical boundary is sign(x). Sign flips mean the
    quantized activation takes a fundamentally different path through the
    nonlinearity.
    """
    n_elements = pre_act_float.size

    sign_float = pre_act_float > 0
    sign_quant = pre_act_quant > 0

    # Sign agreement categories
    both_positive = sign_float & sign_quant       # survive: both active
    both_negative = ~sign_float & ~sign_quant     # dead: both inactive
    flip_to_dead = sign_float & ~sign_quant       # was active, now dead
    flip_to_alive = ~sign_float & sign_quant      # was dead, now active

    survive_rate = float(both_positive.sum()) / n_elements
    dead_rate = float(both_negative.sum()) / n_elements
    flip_to_dead_rate = float(flip_to_dead.sum()) / n_elements
    flip_to_alive_rate = float(flip_to_alive.sum()) / n_elements
    flip_rate = flip_to_dead_rate + flip_to_alive_rate
    collapse_mass = dead_rate + flip_rate  # total nonlinear loss

    return {
        "flip_rate": flip_rate,
        "flip_to_dead_rate": flip_to_dead_rate,
        "flip_to_alive_rate": flip_to_alive_rate,
        "survive_rate": survive_rate,
        "dead_rate": dead_rate,
        "collapse_mass": collapse_mass,
    }


# ============================================================================
# Error covariance / anisotropy
# ============================================================================

def compute_error_geometry(errors: np.ndarray) -> dict[str, float]:
    """Anisotropy and effective rank of the error distribution."""
    n, dim = errors.shape
    errors_c = errors - errors.mean(axis=0, keepdims=True)

    if n < dim:
        _, S, _ = np.linalg.svd(errors_c, full_matrices=False)
        eigenvalues = (S ** 2) / n
    else:
        cov = (errors_c.T @ errors_c) / n
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]

    eigenvalues = np.maximum(eigenvalues, 0)
    total = eigenvalues.sum()
    if total < 1e-15:
        return {"anisotropy_ratio": 1.0, "effective_rank": float(dim),
                "top1_frac": 0.0, "top10_frac": 0.0}

    mean_eig = total / len(eigenvalues)
    anisotropy = float(eigenvalues[0] / max(mean_eig, 1e-15))

    p = eigenvalues / total
    p = p[p > 1e-15]
    entropy = -np.sum(p * np.log(p))
    effective_rank = float(np.exp(entropy))

    top1_frac = float(eigenvalues[0] / total)
    top10_frac = float(eigenvalues[:min(10, len(eigenvalues))].sum() / total)

    return {
        "anisotropy_ratio": anisotropy,
        "effective_rank": effective_rank,
        "top1_frac": top1_frac,
        "top10_frac": top10_frac,
    }


# ============================================================================
# Correctability score (from quant_aware_training)
# ============================================================================

def correctability_score(*, linear_error_norm: float, nonlinear_error_norm: float,
                         flip_rate: float, anisotropy_ratio: float,
                         amplification: float = 1.0) -> float:
    """Composite 0-1 score predicting linear correctability.

    Weights match quant_aware_training/aleph/qgeom/metrics.py.
    """
    eps = 1e-12
    linear_fraction = linear_error_norm / max(linear_error_norm + nonlinear_error_norm, eps)
    anis_penalty = 1.0 / (1.0 + math.log1p(max(anisotropy_ratio - 1.0, 0.0)))
    amp_penalty = 1.0 / (1.0 + max(amplification - 1.0, 0.0))

    score = (
        0.55 * linear_fraction
        + 0.15 * (1.0 - flip_rate)
        + 0.15 * 1.0  # saturation rate — always 0 for symmetric quant
        + 0.10 * anis_penalty
        + 0.05 * amp_penalty
    )
    return float(np.clip(score, 0.0, 1.0))


# ============================================================================
# Model helpers
# ============================================================================

# build_model imported from scripts.eval_commons (see top of file)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Manifold distortion & pre-nonlinearity correction analysis")
    parser.add_argument("checkpoint", help="Path to .npz float checkpoint")
    parser.add_argument("--n-seqs", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--ridge", type=float, default=1.0,
                        help="Ridge regularization for affine fits")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Cap tokens for ridge fits (memory)")
    parser.add_argument("--log-file", type=str, default=None)
    args_cli = parser.parse_args()

    hparams = Hyperparameters()

    # Logging
    global log
    log = logging.getLogger("correctability")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if args_cli.log_file is None:
        args_cli.log_file = f"logs/correctability_int{hparams.quant_attn_bits}.txt"
    os.makedirs(os.path.dirname(args_cli.log_file), exist_ok=True)
    fh = logging.FileHandler(args_cli.log_file, mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.info(f"Logging to {args_cli.log_file}")

    cat_bits = {"attn": hparams.quant_attn_bits, "mlp": hparams.quant_mlp_bits}
    log.info(f"Config: {hparams.num_layers}L/{hparams.model_dim}d, MLP {hparams.mlp_mult}x, "
             f"act={hparams.mlp_act}, quant=a{hparams.quant_attn_bits}m{hparams.quant_mlp_bits}")

    # Load float model
    log.info(f"\nLoading checkpoint: {args_cli.checkpoint}")
    model_float = build_model(hparams)
    flat = dict(mx.load(args_cli.checkpoint))
    model_float.update(tree_unflatten(list(flat.items())))
    mx.eval(model_float.parameters())

    # Create quantized model
    model_quant = build_model(hparams)
    model_quant.update(tree_unflatten(list(flat.items())))
    mx.eval(model_quant.parameters())
    flat_q = {k: v for k, v in tree_flatten(model_quant.state)}
    quant_obj, _ = quantize_state_dict_int8(flat_q, cat_bits=cat_bits)
    quant_flat = dequantize_state_dict_int8(quant_obj)
    model_quant.update(tree_unflatten(list(quant_flat.items())))
    mx.eval(model_quant.parameters())

    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    n_layers = hparams.num_layers

    # ---- Collect pre/post-nonlinearity activations for both models ----
    log.info(f"\n=== Collecting activations ({args_cli.n_seqs} seqs × {args_cli.seq_len} tokens) ===")

    # Per-layer accumulators
    pre_act_float_all = [[] for _ in range(n_layers)]
    pre_act_quant_all = [[] for _ in range(n_layers)]
    post_act_float_all = [[] for _ in range(n_layers)]
    post_act_quant_all = [[] for _ in range(n_layers)]
    h_out_float_all = [[] for _ in range(n_layers)]
    h_out_quant_all = [[] for _ in range(n_layers)]
    post_attn_float_all = [[] for _ in range(n_layers)]
    post_attn_quant_all = [[] for _ in range(n_layers)]
    mlp_normed_float_all = [[] for _ in range(n_layers)]
    mlp_normed_quant_all = [[] for _ in range(n_layers)]

    t0 = time.time()
    for s in range(args_cli.n_seqs):
        start = s * args_cli.seq_len
        toks = val_tokens[start:start + args_cli.seq_len]
        if len(toks) < args_cli.seq_len:
            break

        results_f = forward_collect_prepost_act(model_float, toks)
        results_q = forward_collect_prepost_act(model_quant, toks)

        for i in range(n_layers):
            pre_act_float_all[i].append(
                np.array(results_f[i]['pre_act']).reshape(-1, results_f[i]['pre_act'].shape[-1]))
            pre_act_quant_all[i].append(
                np.array(results_q[i]['pre_act']).reshape(-1, results_q[i]['pre_act'].shape[-1]))
            post_act_float_all[i].append(
                np.array(results_f[i]['post_act']).reshape(-1, results_f[i]['post_act'].shape[-1]))
            post_act_quant_all[i].append(
                np.array(results_q[i]['post_act']).reshape(-1, results_q[i]['post_act'].shape[-1]))
            h_out_float_all[i].append(
                np.array(results_f[i]['h_out']).reshape(-1, results_f[i]['h_out'].shape[-1]))
            h_out_quant_all[i].append(
                np.array(results_q[i]['h_out']).reshape(-1, results_q[i]['h_out'].shape[-1]))
            post_attn_float_all[i].append(
                np.array(results_f[i]['post_attn']).reshape(-1, results_f[i]['post_attn'].shape[-1]))
            post_attn_quant_all[i].append(
                np.array(results_q[i]['post_attn']).reshape(-1, results_q[i]['post_attn'].shape[-1]))
            mlp_normed_float_all[i].append(
                np.array(results_f[i]['mlp_normed']).reshape(-1, results_f[i]['mlp_normed'].shape[-1]))
            mlp_normed_quant_all[i].append(
                np.array(results_q[i]['mlp_normed']).reshape(-1, results_q[i]['mlp_normed'].shape[-1]))

    t_collect = time.time() - t0
    log.info(f"Collection: {t_collect:.1f}s")

    # Concatenate and cap
    cap = args_cli.max_tokens
    for i in range(n_layers):
        pre_act_float_all[i] = np.concatenate(pre_act_float_all[i])[:cap].astype(np.float64)
        pre_act_quant_all[i] = np.concatenate(pre_act_quant_all[i])[:cap].astype(np.float64)
        post_act_float_all[i] = np.concatenate(post_act_float_all[i])[:cap].astype(np.float64)
        post_act_quant_all[i] = np.concatenate(post_act_quant_all[i])[:cap].astype(np.float64)
        h_out_float_all[i] = np.concatenate(h_out_float_all[i])[:cap].astype(np.float64)
        h_out_quant_all[i] = np.concatenate(h_out_quant_all[i])[:cap].astype(np.float64)
        post_attn_float_all[i] = np.concatenate(post_attn_float_all[i])[:cap].astype(np.float64)
        post_attn_quant_all[i] = np.concatenate(post_attn_quant_all[i])[:cap].astype(np.float64)
        mlp_normed_float_all[i] = np.concatenate(mlp_normed_float_all[i])[:cap].astype(np.float64)
        mlp_normed_quant_all[i] = np.concatenate(mlp_normed_quant_all[i])[:cap].astype(np.float64)

    n_tokens = pre_act_float_all[0].shape[0]
    log.info(f"Tokens per layer: {n_tokens}")

    # ============================================================
    # Analysis 1: Pre-nonlinearity affine correction
    # ============================================================
    log.info(f"\n{'='*80}")
    log.info(f"ANALYSIS 1: Pre-nonlinearity affine correction (z_float ~ W @ z_quant + b)")
    log.info(f"{'='*80}")
    log.info(f"  This measures how much of the MLP pre-activation error is linearly")
    log.info(f"  correctable BEFORE relu2 destroys the mapping.\n")

    log.info(f"{'L':>2s} {'MSE_before':>11s} {'MSE_after':>11s} {'R_lin':>7s} "
             f"{'||W-I||_F':>10s} {'||b||':>8s}")
    log.info("-" * 55)

    pre_act_results = []
    for i in range(n_layers):
        W, b, mse_before, mse_after = fit_affine_correction(
            pre_act_quant_all[i], pre_act_float_all[i], ridge=args_cli.ridge)

        # How far is W from identity?
        d = min(W.shape[0], W.shape[1])
        W_eye_delta = W[:d, :d] - np.eye(d, dtype=np.float64)
        frob_delta = float(np.linalg.norm(W_eye_delta, 'fro'))
        b_norm = float(np.linalg.norm(b))

        r_lin = max(1.0 - mse_after / max(mse_before, 1e-15), 0.0)

        pre_act_results.append({
            "mse_before": mse_before, "mse_after": mse_after,
            "r_lin": r_lin, "frob_delta": frob_delta, "b_norm": b_norm,
        })
        log.info(f"{i:>2d} {mse_before:11.2e} {mse_after:11.2e} {r_lin:7.4f} "
                 f"{frob_delta:10.4f} {b_norm:8.4f}")

    # ============================================================
    # Analysis 1b: Local vs propagated error decomposition
    # ============================================================
    log.info(f"\n{'='*80}")
    log.info(f"ANALYSIS 1b: Local vs propagated error at pre-activation")
    log.info(f"{'='*80}")
    log.info(f"  pre_act_error = E_fc @ mlp_normed_quant  +  W_fc_quant @ δ_input")
    log.info(f"                  └── local (this layer) ──┘  └── propagated (upstream) ──┘")
    log.info(f"  Local error is from this layer's fc weight quantization alone.")
    log.info(f"  Propagated error is from upstream quant passing through nonlinearities.\n")

    log.info(f"{'L':>2s} {'total_rmse':>11s} {'local_rmse':>11s} {'propag_rmse':>12s} "
             f"{'local_frac':>10s} {'R²_local':>9s}")
    log.info("-" * 62)

    for i in range(n_layers):
        # Get the fc weight error: E_fc = W_fc_quant - W_fc_float
        W_fc_float = np.array(model_float.blocks[i].mlp.fc.weight).astype(np.float64)
        W_fc_quant = np.array(model_quant.blocks[i].mlp.fc.weight).astype(np.float64)
        E_fc = W_fc_quant - W_fc_float  # (mlp_hidden, dim)

        # Total pre-activation error
        total_err = pre_act_quant_all[i] - pre_act_float_all[i]  # (n, mlp_hidden)

        # Local error: E_fc @ mlp_normed_quant (using quant model's input)
        # pre_act = fc(mlp_normed) = mlp_normed @ W_fc^T, so local error = mlp_normed_quant @ E_fc^T
        local_err = mlp_normed_quant_all[i] @ E_fc.T  # (n, mlp_hidden)

        # Propagated error: total - local
        propag_err = total_err - local_err

        total_rmse = float(np.sqrt(np.mean(total_err ** 2)))
        local_rmse = float(np.sqrt(np.mean(local_err ** 2)))
        propag_rmse = float(np.sqrt(np.mean(propag_err ** 2)))

        # Local fraction (by MSE, not RMSE)
        total_mse = float(np.mean(total_err ** 2))
        local_mse = float(np.mean(local_err ** 2))
        local_frac = local_mse / max(total_mse, 1e-15)

        # R² of affine fit on local error only (should be ~1.0 since it's linear in mlp_normed)
        _, _, mse_before_local, mse_after_local = fit_affine_correction(
            pre_act_quant_all[i], pre_act_quant_all[i] + local_err,  # predict float_local from quant
            ridge=args_cli.ridge)
        r2_local = max(1.0 - mse_after_local / max(mse_before_local, 1e-15), 0.0)

        log.info(f"{i:>2d} {total_rmse:11.5f} {local_rmse:11.5f} {propag_rmse:12.5f} "
                 f"{local_frac:10.3f} {r2_local:9.4f}")

    # ============================================================
    # Analysis 2: Post-nonlinearity affine correction
    # ============================================================
    log.info(f"\n{'='*80}")
    log.info(f"ANALYSIS 2: Post-nonlinearity affine correction (after relu2 squaring)")
    log.info(f"{'='*80}")
    log.info(f"  Same fit but AFTER the nonlinearity. The gap between Analysis 1 and 2")
    log.info(f"  shows how much correctability the nonlinearity destroys.\n")

    log.info(f"{'L':>2s} {'MSE_before':>11s} {'MSE_after':>11s} {'R_lin':>7s} "
             f"{'Δ vs pre':>9s}")
    log.info("-" * 45)

    post_act_results = []
    for i in range(n_layers):
        _, _, mse_before, mse_after = fit_affine_correction(
            post_act_quant_all[i], post_act_float_all[i], ridge=args_cli.ridge)

        r_lin = max(1.0 - mse_after / max(mse_before, 1e-15), 0.0)
        delta_vs_pre = r_lin - pre_act_results[i]["r_lin"]

        post_act_results.append({
            "mse_before": mse_before, "mse_after": mse_after, "r_lin": r_lin,
        })
        log.info(f"{i:>2d} {mse_before:11.2e} {mse_after:11.2e} {r_lin:7.4f} "
                 f"{delta_vs_pre:+9.4f}")

    # ============================================================
    # Analysis 3: Post-attention affine correction (residual stream)
    # ============================================================
    log.info(f"\n{'='*80}")
    log.info(f"ANALYSIS 3: Post-attention residual stream correction")
    log.info(f"{'='*80}")
    log.info(f"  Correction at the residual stream (after attn, before MLP).\n")

    dim = hparams.model_dim
    log.info(f"{'L':>2s} {'MSE_before':>11s} {'MSE_after':>11s} {'R_lin':>7s}")
    log.info("-" * 36)

    for i in range(n_layers):
        _, _, mse_before, mse_after = fit_affine_correction(
            post_attn_quant_all[i], post_attn_float_all[i], ridge=args_cli.ridge)
        r_lin = max(1.0 - mse_after / max(mse_before, 1e-15), 0.0)
        log.info(f"{i:>2d} {mse_before:11.2e} {mse_after:11.2e} {r_lin:7.4f}")

    # ============================================================
    # Analysis 4: Fate metrics — nonlinearity crossings
    # ============================================================
    log.info(f"\n{'='*80}")
    log.info(f"ANALYSIS 4: Nonlinearity fate (sign flips at relu2 boundary)")
    log.info(f"{'='*80}")
    log.info(f"  flip_rate = fraction of pre-activations with different sign in float vs quant.")
    log.info(f"  Once flipped, the error is irrecoverable by affine correction.\n")

    log.info(f"{'L':>2s} {'flip_rate':>9s} {'→dead':>7s} {'→alive':>7s} "
             f"{'survive':>8s} {'dead':>6s} {'collapse':>8s}")
    log.info("-" * 55)

    fate_results = []
    for i in range(n_layers):
        # Use float32 for sign comparison (float64 not needed)
        fate = compute_fate_metrics(
            pre_act_float_all[i].astype(np.float32),
            pre_act_quant_all[i].astype(np.float32))
        fate_results.append(fate)
        log.info(f"{i:>2d} {fate['flip_rate']:9.5f} {fate['flip_to_dead_rate']:7.5f} "
                 f"{fate['flip_to_alive_rate']:7.5f} {fate['survive_rate']:8.4f} "
                 f"{fate['dead_rate']:6.4f} {fate['collapse_mass']:8.4f}")

    # ============================================================
    # Analysis 5: Error geometry & amplification
    # ============================================================
    log.info(f"\n{'='*80}")
    log.info(f"ANALYSIS 5: Error geometry (anisotropy, effective rank)")
    log.info(f"{'='*80}")
    log.info(f"  High anisotropy = error concentrated in few directions (harder to correct).")
    log.info(f"  Low effective rank = degenerate error structure.\n")

    log.info(f"{'L':>2s} {'pre_err_rmse':>12s} {'post_err_rmse':>13s} {'amplification':>13s} "
             f"{'aniso':>7s} {'eff_rank':>8s}")
    log.info("-" * 62)

    geometry_results = []
    for i in range(n_layers):
        pre_err = pre_act_float_all[i] - pre_act_quant_all[i]
        post_err = post_act_float_all[i] - post_act_quant_all[i]
        h_err = h_out_float_all[i] - h_out_quant_all[i]

        pre_rmse = float(np.sqrt(np.mean(pre_err ** 2)))
        post_rmse = float(np.sqrt(np.mean(post_err ** 2)))
        h_rmse = float(np.sqrt(np.mean(h_err ** 2)))

        # Amplification: how much does the nonlinearity amplify the error?
        amplification = post_rmse / max(pre_rmse, 1e-15)

        geom = compute_error_geometry(pre_err.astype(np.float64))

        geometry_results.append({
            "pre_rmse": pre_rmse, "post_rmse": post_rmse, "h_rmse": h_rmse,
            "amplification": amplification, **geom,
        })
        log.info(f"{i:>2d} {pre_rmse:12.6f} {post_rmse:13.6f} {amplification:13.2f}x "
                 f"{geom['anisotropy_ratio']:7.1f} {geom['effective_rank']:8.1f}")

    # ============================================================
    # Analysis 6: Composite correctability score
    # ============================================================
    log.info(f"\n{'='*80}")
    log.info(f"ANALYSIS 6: Composite correctability score per layer")
    log.info(f"{'='*80}\n")

    log.info(f"{'L':>2s} {'R_lin_pre':>9s} {'R_lin_post':>10s} {'flip_rate':>9s} "
             f"{'aniso':>7s} {'score':>6s}")
    log.info("-" * 48)

    scores = []
    for i in range(n_layers):
        # Linear vs nonlinear error norms (from pre-act error)
        pre_err = pre_act_float_all[i] - pre_act_quant_all[i]
        post_err = post_act_float_all[i] - post_act_quant_all[i]
        linear_norm = float(np.sqrt(np.mean(pre_err ** 2)))
        # Nonlinear error = additional error introduced by the nonlinearity
        nonlinear_norm = max(float(np.sqrt(np.mean(post_err ** 2))) - linear_norm, 0.0)

        cscore = correctability_score(
            linear_error_norm=linear_norm,
            nonlinear_error_norm=nonlinear_norm,
            flip_rate=fate_results[i]["flip_rate"],
            anisotropy_ratio=geometry_results[i]["anisotropy_ratio"],
            amplification=geometry_results[i]["amplification"],
        )
        scores.append(cscore)
        log.info(f"{i:>2d} {pre_act_results[i]['r_lin']:9.4f} {post_act_results[i]['r_lin']:10.4f} "
                 f"{fate_results[i]['flip_rate']:9.5f} "
                 f"{geometry_results[i]['anisotropy_ratio']:7.1f} {cscore:6.3f}")

    # ============================================================
    # Summary
    # ============================================================
    log.info(f"\n{'='*80}")
    log.info(f"SUMMARY")
    log.info(f"{'='*80}")

    avg_r_pre = np.mean([r["r_lin"] for r in pre_act_results])
    avg_r_post = np.mean([r["r_lin"] for r in post_act_results])
    avg_flip = np.mean([f["flip_rate"] for f in fate_results])
    avg_score = np.mean(scores)

    log.info(f"  Avg pre-nonlinearity R_lin:  {avg_r_pre:.4f}")
    log.info(f"  Avg post-nonlinearity R_lin: {avg_r_post:.4f}")
    log.info(f"  R_lin lost at nonlinearity:  {avg_r_pre - avg_r_post:+.4f}")
    log.info(f"  Avg ReLU flip rate:          {avg_flip:.5f}")
    log.info(f"  Avg correctability score:    {avg_score:.3f}")

    log.info(f"\n  Interpretation:")
    if avg_r_pre > 0.7:
        log.info(f"  Pre-nonlinearity error is highly affine-correctable (R_lin > 0.7).")
        if avg_r_post < avg_r_pre - 0.1:
            log.info(f"  BUT the nonlinearity destroys {avg_r_pre - avg_r_post:.0%} of that correctability.")
            log.info(f"  → Corrections MUST go before the nonlinearity to be effective.")
        else:
            log.info(f"  The nonlinearity preserves most correctability — post-block corrections viable.")
    elif avg_r_pre > 0.4:
        log.info(f"  Moderate pre-nonlinearity correctability. Worth pursuing if gap is large.")
    else:
        log.info(f"  Low pre-nonlinearity correctability (R_lin < 0.4).")
        log.info(f"  Error has already cascaded too far. Focus on reducing error at source (GPTQ).")

    if avg_flip > 0.05:
        log.info(f"  WARNING: {avg_flip:.1%} sign flip rate means significant nonlinear distortion.")
        log.info(f"  Layers with high flip rates need correction BEFORE the nonlinearity.")
    else:
        log.info(f"  Sign flip rate is low ({avg_flip:.2%}) — most error stays in the linear regime.")

    # Per-layer recommendation
    log.info(f"\n  Per-layer correction priority (highest correctability first):")
    layer_priority = sorted(range(n_layers), key=lambda i: scores[i], reverse=True)
    for rank, i in enumerate(layer_priority[:5]):
        log.info(f"    #{rank+1}: Layer {i} (score={scores[i]:.3f}, "
                 f"R_pre={pre_act_results[i]['r_lin']:.3f}, "
                 f"flip={fate_results[i]['flip_rate']:.4f})")


if __name__ == "__main__":
    main()
