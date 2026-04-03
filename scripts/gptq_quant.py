#!/usr/bin/env python3
"""
gptq_quant.py — GPTQ-style optimal rounding for weight quantization.

Instead of round-to-nearest, choose rounding directions to minimize layer output
error using second-order statistics (Hessian proxy H = X^T X) from calibration data.
This changes quantization quality at source with zero inference cost.

Algorithm per weight matrix W with calibration activations X:
  1. H = X^T X / n (activation covariance = Hessian proxy)
  2. Process columns sequentially: for each column, round optimally and propagate
     the residual error to remaining columns weighted by the Hessian.

Usage:
    NUM_LAYERS=11 MLP_MULT=4.5 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \
    QUANT_ATTN_BITS=5 QUANT_MLP_BITS=5 \
    .venv/bin/python3 scripts/gptq_quant.py logs/wd50_11L_5x_best.npz \
        --n-calib-seqs 32 --log-file logs/gptq_int5.txt

    # Compare int4:
    QUANT_ATTN_BITS=4 QUANT_MLP_BITS=4 \
    .venv/bin/python3 scripts/gptq_quant.py logs/wd50_11L_5x_best.npz
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
    quantize_state_dict_int8, dequantize_state_dict_int8, load_data_shard,
    _classify_param, _QUANT_FN, FP16_KEEP_NAME_PATTERNS,
    CONTROL_TENSOR_NAME_PATTERNS, INT8_KEEP_FLOAT_MAX_NUMEL, rms_norm,
)
from scripts.eval_commons import build_model, quick_ce, load_train_tokens

log = logging.getLogger("gptq")


# ============================================================================
# GPTQ core algorithm
# ============================================================================

def gptq_quantize_weight(W: np.ndarray, H: np.ndarray, qmax: int,
                         block_size: int = 128, damp_pct: float = 0.01
                         ) -> tuple[np.ndarray, np.ndarray, float]:
    """GPTQ-quantize a single weight matrix.

    Args:
        W: float32 weight matrix (out_dim, in_dim) — rows are output channels
        H: float32 Hessian proxy (in_dim, in_dim) = X^T X / n
        qmax: max quantization level (e.g. 15 for int5)
        block_size: columns processed together (larger = faster, slightly less optimal)
        damp_pct: Hessian diagonal damping percentage

    Returns:
        Q: int8 quantized weights (out_dim, in_dim)
        scales: float16 per-row scales (out_dim,)
        mse: mean squared error vs float weights (weighted by H)
    """
    out_dim, in_dim = W.shape
    W = W.copy().astype(np.float64)  # work in float64 for numerical stability

    # Per-row scales (same as round-to-nearest)
    row_max = np.abs(W).max(axis=1)
    scales = np.maximum(row_max / qmax, 1e-12).astype(np.float64)

    # Damped Hessian
    H = H.astype(np.float64).copy()
    diag_mean = np.mean(np.diag(H))
    if diag_mean < 1e-15:
        # Degenerate Hessian — fall back to round-to-nearest
        Q = np.clip(np.round(W / scales[:, None]), -qmax - 1, qmax).astype(np.int8)
        scales_f16 = scales.astype(np.float16)
        mse = float(np.mean((W - Q.astype(np.float64) * scales[:, None]) ** 2))
        return Q, scales_f16, mse

    H[np.diag_indices_from(H)] += damp_pct * diag_mean

    Q = np.zeros_like(W, dtype=np.float64)

    # Cholesky of inverse — needed for the GPTQ update
    # GPTQ needs H_inv, specifically the diagonal and off-diagonal of the Cholesky
    # We compute the Cholesky of H (not H_inv) and use it via back-substitution
    try:
        L = np.linalg.cholesky(H)
        H_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(in_dim, dtype=np.float64)))
    except np.linalg.LinAlgError:
        # Fallback: add more damping
        H[np.diag_indices_from(H)] += 0.1 * diag_mean
        try:
            L = np.linalg.cholesky(H)
            H_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(in_dim, dtype=np.float64)))
        except np.linalg.LinAlgError:
            # Last resort: pseudoinverse
            H_inv = np.linalg.pinv(H)

    # Process in blocks of columns
    for col_start in range(0, in_dim, block_size):
        col_end = min(col_start + block_size, in_dim)
        block_cols = col_end - col_start

        # Error accumulator for this block (propagated to remaining columns after block)
        Err = np.zeros((out_dim, block_cols), dtype=np.float64)
        H_inv_block = H_inv[col_start:col_end, col_start:col_end]
        h_diag = np.diag(H_inv_block).copy()
        h_diag = np.maximum(h_diag, 1e-15)  # safety

        for j_local in range(block_cols):
            j = col_start + j_local
            w_col = W[:, j]  # (out_dim,)

            # Quantize
            q_col = np.clip(np.round(w_col / scales), -qmax - 1, qmax)
            Q[:, j] = q_col
            dequant_col = q_col * scales

            # Error for this column
            err = (w_col - dequant_col) / h_diag[j_local]
            Err[:, j_local] = err

            # Propagate error to remaining columns within this block
            if j_local < block_cols - 1:
                W[:, j + 1:col_end] -= np.outer(err, H_inv_block[j_local, j_local + 1:col_end])

        # Propagate block error to remaining columns outside block
        if col_end < in_dim:
            W[:, col_end:] -= Err @ H_inv[col_start:col_end, col_end:]

    Q = Q.astype(np.int8)
    scales_f16 = scales.astype(np.float16)

    # Compute MSE (H-weighted would be ideal but plain MSE is simpler to interpret)
    dequant = Q.astype(np.float64) * scales[:, None]
    mse = float(np.mean((W.astype(np.float64) - dequant) ** 2))
    # Note: W has been modified in-place during GPTQ, so MSE here measures vs the
    # modified W. For a proper comparison, we compute MSE against original W outside.

    return Q, scales_f16, mse


# ============================================================================
# Calibration data collection
# ============================================================================

def collect_layer_activations(model, tokens: np.ndarray, n_seqs: int,
                              seq_len: int) -> dict[str, np.ndarray]:
    """Collect input activations to each weight matrix on calibration data.

    Returns dict mapping weight name -> activation matrix (n_tokens, in_dim).
    We collect activations for the fc/proj/q/k/v/out_proj weight matrices.
    """
    n_layers = len(model.blocks)
    n_enc = model.num_encoder_layers
    n_skip = model.num_skip_weights

    # We'll collect activations by running the model forward and hooking into
    # the intermediate states. For GPTQ, we need the INPUT to each linear layer.
    # Strategy: run full forward per-block, collecting post-norm activations.
    activations = {}  # name -> list of np arrays

    for s in range(n_seqs):
        start = s * seq_len
        toks = tokens[start:start + seq_len]
        if len(toks) < seq_len:
            break

        x_mx = mx.array(toks[np.newaxis, :])
        tok_emb = model.tok_emb(x_mx).astype(COMPUTE_DTYPE)
        if model.bigram is not None:
            tok_emb = tok_emb + model.bigram(x_mx)
        x0 = model.smear(rms_norm(tok_emb))
        h = x0
        mx.eval(x0)

        encoder_outputs = [None] * n_enc

        for i in range(n_layers):
            if i >= n_enc:
                dec_j = i - n_enc
                if dec_j < n_skip:
                    enc_j = n_enc - 1 - dec_j
                    if encoder_outputs[enc_j] is not None:
                        h = h + model.skip_weights[dec_j].astype(h.dtype)[None, None, :] * encoder_outputs[enc_j]

            block = model.blocks[i]

            # ---- Decomposed block forward to capture activations ----
            # Residual mix
            mixed = block.resid_mix.astype(h.dtype) * h + (1 - block.resid_mix.astype(h.dtype)) * x0

            # Attention input: after attn_norm
            attn_normed = rms_norm(mixed)
            mx.eval(attn_normed)
            attn_in_np = np.array(attn_normed).reshape(-1, attn_normed.shape[-1]).astype(np.float32)
            for wname in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
                key = f"blocks.{i}.attn.{wname}"
                activations.setdefault(key, []).append(attn_in_np)

            # Run attention
            attn_out = block.attn(mixed)
            mx.eval(attn_out)

            # Attention output projection input: this is the attention output pre-proj
            # Actually, out_proj is inside the attn module. We need the multi-head concat.
            # For simplicity, we use attn_normed as the input proxy for all attn weights.
            # The out_proj input is the concatenated attention output — different shape.
            # Skip out_proj for now (it's internal to the attention module).

            post_attn = h + block.attn_scale.astype(h.dtype) * attn_out

            # MLP input: after mlp_norm
            mlp_normed = rms_norm(post_attn)
            mx.eval(mlp_normed)
            mlp_in_np = np.array(mlp_normed).reshape(-1, mlp_normed.shape[-1]).astype(np.float32)
            for wname in ("fc.weight",):
                key = f"blocks.{i}.mlp.{wname}"
                activations.setdefault(key, []).append(mlp_in_np)

            # For MLP proj, we need the post-activation output (input to proj.weight)
            # Run MLP fc
            mlp = block.mlp
            fc_out = mlp.fc(mlp_normed)
            if hasattr(mlp, 'gate'):
                # SwiGLU: proj input = silu(gate) * fc
                gate_out = mlp.gate(mlp_normed)
                gate_key = f"blocks.{i}.mlp.gate.weight"
                activations.setdefault(gate_key, []).append(mlp_in_np)
                pre_proj = nn.silu(gate_out) * fc_out  # silu(gate) * fc
            else:
                # relu2/lrelu2/sugar: proj input = act(fc)^2
                if mlp.act in ("relu2", "sugar"):
                    h_act = mx.maximum(fc_out, 0)
                elif mlp.act == "lrelu2":
                    h_act = mx.where(fc_out > 0, fc_out, block.lrelu_slope * fc_out)
                else:
                    h_act = fc_out
                pre_proj = h_act * h_act
            mx.eval(pre_proj)
            proj_in_np = np.array(pre_proj).reshape(-1, pre_proj.shape[-1]).astype(np.float32)
            proj_key = f"blocks.{i}.mlp.proj.weight"
            activations.setdefault(proj_key, []).append(proj_in_np)

            # Full block output
            mlp_out = block.mlp(mlp_normed)
            h = post_attn + block.mlp_scale.astype(h.dtype) * mlp_out
            mx.eval(h)

            if i < n_enc:
                encoder_outputs[i] = h

    # Concatenate activations
    result = {}
    for key, arrays in activations.items():
        result[key] = np.concatenate(arrays, axis=0)

    return result


def compute_hessian(X: np.ndarray) -> np.ndarray:
    """Compute Hessian proxy H = X^T X / n."""
    n = X.shape[0]
    # Use float64 for numerical stability
    X64 = X.astype(np.float64)
    H = (X64.T @ X64) / n
    return H


# ============================================================================
# Quantize with GPTQ
# ============================================================================

def gptq_quantize_state_dict(flat_state: dict[str, mx.array],
                              activations: dict[str, np.ndarray],
                              cat_bits: dict[str, int],
                              block_size: int = 128,
                              damp_pct: float = 0.01,
                              ) -> tuple[dict, dict, dict]:
    """GPTQ-quantize all weight matrices that have calibration activations.

    Returns: (quant_obj, stats, per_layer_mse)
        quant_obj: same format as quantize_state_dict_int8 output
        stats: quantization statistics
        per_layer_mse: {name: (rtn_mse, gptq_mse)} comparing RTN vs GPTQ
    """
    from train_gpt_mlx import (
        _np_float32, keep_float_array, INT8_CLIP_Q,
        INT8_PER_ROW_SCALE_DTYPE, MX_DTYPE_FROM_NAME,
    )

    quantized = {}
    scales_dict = {}
    dtypes = {}
    passthrough = {}
    passthrough_orig_dtypes = {}
    qmeta = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int8_payload_bytes"), 0,
    )
    per_layer_mse = {}

    _BITS_TO_QMAX = {2: 1, 3: 3, 4: 7, 5: 15, 6: 31, 8: 127}

    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)

        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        if any(pattern in name for pattern in FP16_KEEP_NAME_PATTERNS):
            kept = np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=np.float16))
            passthrough[name] = kept
            passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            if any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
                passthrough[name] = np.ascontiguousarray(_np_float32(arr))
            else:
                passthrough[name] = keep_float_array(name, arr, passthrough_orig_dtypes)
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        stats["num_float_tensors"] += 1
        cat = _classify_param(name)
        bits = (cat_bits or {}).get(cat)
        if bits is None:
            base_cat = cat.split(".")[0]
            bits = (cat_bits or {}).get(base_cat)
        if bits is None:
            bits = 8
        qmax = _BITS_TO_QMAX[bits]

        W = _np_float32(arr)

        if name in activations and W.ndim == 2:
            # GPTQ quantization
            X = activations[name]
            H = compute_hessian(X)

            # RTN baseline for comparison
            row_max_rtn = np.abs(W).max(axis=1)
            scale_rtn = np.maximum(row_max_rtn / qmax, 1e-12)
            q_rtn = np.clip(np.round(W / scale_rtn[:, None]), -qmax - 1, qmax)
            rtn_mse = float(np.mean((W - q_rtn * scale_rtn[:, None]) ** 2))

            q, s, gptq_mse = gptq_quantize_weight(W, H, qmax, block_size, damp_pct)
            # Recompute MSE against original W (not the GPTQ-modified W)
            dequant = q.astype(np.float64) * s.astype(np.float64)[:, None]
            gptq_mse = float(np.mean((W.astype(np.float64) - dequant) ** 2))

            per_layer_mse[name] = (rtn_mse, gptq_mse)
            quantized[name] = np.ascontiguousarray(q)
            scales_dict[name] = np.ascontiguousarray(s)
        else:
            # Fall back to standard RTN quantization
            quant_fn = _QUANT_FN[bits]
            q, s = quant_fn(arr)
            quantized[name] = q
            scales_dict[name] = s

        dtypes[name] = str(arr.dtype).split(".")[-1]
        if bits < 8:
            qmeta[name] = {"scheme": f"int{bits}_per_row", "axis": 0}
        elif scales_dict[name].ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        stats["int8_payload_bytes"] += int(quantized[name].nbytes + scales_dict[name].nbytes)

    obj = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales_dict,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats, per_layer_mse


# ============================================================================
# Evaluation helpers
# ============================================================================

def quick_eval(model, hparams, n_seqs=32, seq_len=1024):
    """Quick CE evaluation on val tokens. NB: reports bits-per-token, NOT BPB."""
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    avg = quick_ce(model, val_tokens, n_seqs, seq_len)
    bpt = avg / math.log(2)  # NB: bits-per-token, not BPB
    log.info(f"  val_loss={avg:.6f} ({bpt:.4f} bits/tok, {n_seqs} seqs)")
    return avg


# build_model and load_train_tokens imported from scripts.eval_commons (see top of file)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPTQ-style optimal rounding")
    parser.add_argument("checkpoint", help="Path to .npz float checkpoint")
    parser.add_argument("--n-calib-seqs", type=int, default=32,
                        help="Number of calibration sequences")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-eval-seqs", type=int, default=64,
                        help="Number of sequences for BPB evaluation")
    parser.add_argument("--block-size", type=int, default=128,
                        help="GPTQ column block size")
    parser.add_argument("--damp-pct", type=float, default=0.01,
                        help="Hessian diagonal damping percentage")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--save", type=str, default=None,
                        help="Save GPTQ-quantized checkpoint to this path")
    args_cli = parser.parse_args()

    hparams = Hyperparameters()

    # Logging
    global log
    log = logging.getLogger("gptq")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if args_cli.log_file is None:
        args_cli.log_file = f"logs/gptq_int{hparams.quant_attn_bits}.txt"
    os.makedirs(os.path.dirname(args_cli.log_file), exist_ok=True)
    fh = logging.FileHandler(args_cli.log_file, mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.info(f"Logging to {args_cli.log_file}")

    cat_bits = {"attn": hparams.quant_attn_bits, "mlp": hparams.quant_mlp_bits}
    quant_bits_str = os.environ.get("QUANT_BITS", "")
    if quant_bits_str:
        cat_bits = {}
        for part in quant_bits_str.split(","):
            k, v = part.strip().rsplit(":", 1)
            cat_bits[k.strip()] = int(v.strip())

    log.info(f"Config: {hparams.num_layers}L/{hparams.model_dim}d, MLP {hparams.mlp_mult}x, "
             f"act={hparams.mlp_act}, quant={cat_bits}")
    log.info(f"GPTQ: block_size={args_cli.block_size}, damp={args_cli.damp_pct}")
    log.info(f"Calibration: {args_cli.n_calib_seqs} seqs × {args_cli.seq_len} tokens")

    # Load model
    log.info(f"\nLoading checkpoint: {args_cli.checkpoint}")
    model = build_model(hparams)
    flat = dict(mx.load(args_cli.checkpoint))
    model.update(tree_unflatten(list(flat.items())))
    mx.eval(model.parameters())

    # ---- Baseline: RTN quantization ----
    log.info("\n=== Baseline: round-to-nearest quantization ===")
    rtn_model = build_model(hparams)
    rtn_model.update(tree_unflatten(list(flat.items())))
    mx.eval(rtn_model.parameters())
    flat_rtn = {k: v for k, v in tree_flatten(rtn_model.state)}
    rtn_quant_obj, rtn_stats = quantize_state_dict_int8(flat_rtn, cat_bits=cat_bits)
    rtn_flat = dequantize_state_dict_int8(rtn_quant_obj)
    rtn_model.update(tree_unflatten(list(rtn_flat.items())))
    mx.eval(rtn_model.parameters())
    log.info("RTN eval:")
    rtn_loss = quick_eval(rtn_model, hparams, args_cli.n_eval_seqs, args_cli.seq_len)
    del rtn_model, rtn_flat

    # ---- Float reference ----
    log.info("\n=== Float reference ===")
    float_loss = quick_eval(model, hparams, args_cli.n_eval_seqs, args_cli.seq_len)
    quant_gap = rtn_loss - float_loss
    log.info(f"RTN quant gap: {quant_gap:.6f} CE ({quant_gap / math.log(2):.6f} bits)")

    # ---- Collect calibration activations ----
    log.info(f"\n=== Collecting calibration activations ({args_cli.n_calib_seqs} seqs) ===")
    max_calib_tokens = args_cli.n_calib_seqs * args_cli.seq_len + args_cli.seq_len
    calib_tokens = load_train_tokens(hparams, max_tokens=max_calib_tokens)
    log.info(f"Calibration tokens: {len(calib_tokens):,}")

    t0 = time.time()
    activations = collect_layer_activations(model, calib_tokens,
                                            args_cli.n_calib_seqs, args_cli.seq_len)
    t_calib = time.time() - t0
    log.info(f"Calibration collection: {t_calib:.1f}s")
    for name, X in sorted(activations.items()):
        log.info(f"  {name}: {X.shape}")

    # ---- GPTQ quantization ----
    log.info(f"\n=== GPTQ quantization ===")
    flat_state = {k: v for k, v in tree_flatten(model.state)}

    t0 = time.time()
    gptq_quant_obj, gptq_stats, per_layer_mse = gptq_quantize_state_dict(
        flat_state, activations, cat_bits,
        block_size=args_cli.block_size, damp_pct=args_cli.damp_pct,
    )
    t_gptq = time.time() - t0
    log.info(f"GPTQ quantization: {t_gptq:.1f}s")

    # Print per-layer MSE comparison
    log.info(f"\n=== Per-layer MSE: RTN vs GPTQ ===")
    log.info(f"{'Layer':<45s} {'RTN MSE':>12s} {'GPTQ MSE':>12s} {'Reduction':>10s}")
    log.info("-" * 82)
    total_rtn_mse = 0.0
    total_gptq_mse = 0.0
    n_layers_compared = 0
    for name in sorted(per_layer_mse.keys()):
        rtn_mse, gptq_mse = per_layer_mse[name]
        reduction = (1 - gptq_mse / max(rtn_mse, 1e-15)) * 100
        log.info(f"  {name:<43s} {rtn_mse:12.2e} {gptq_mse:12.2e} {reduction:+9.1f}%")
        total_rtn_mse += rtn_mse
        total_gptq_mse += gptq_mse
        n_layers_compared += 1
    if n_layers_compared > 0:
        avg_reduction = (1 - total_gptq_mse / max(total_rtn_mse, 1e-15)) * 100
        log.info(f"  {'TOTAL':<43s} {total_rtn_mse:12.2e} {total_gptq_mse:12.2e} {avg_reduction:+9.1f}%")

    # ---- Eval GPTQ-quantized model ----
    log.info(f"\n=== GPTQ eval ===")
    gptq_model = build_model(hparams)
    gptq_flat = dequantize_state_dict_int8(gptq_quant_obj)
    gptq_model.update(tree_unflatten(list(gptq_flat.items())))
    mx.eval(gptq_model.parameters())
    gptq_loss = quick_eval(gptq_model, hparams, args_cli.n_eval_seqs, args_cli.seq_len)

    gptq_gap = gptq_loss - float_loss
    improvement = rtn_loss - gptq_loss
    recovery_pct = improvement / max(quant_gap, 1e-15) * 100

    log.info(f"\n=== Summary ===")
    log.info(f"Float val_loss:    {float_loss:.6f} ({float_loss / math.log(2):.4f} bpt)")
    log.info(f"RTN val_loss:      {rtn_loss:.6f} ({rtn_loss / math.log(2):.4f} bpt)")
    log.info(f"GPTQ val_loss:     {gptq_loss:.6f} ({gptq_loss / math.log(2):.4f} bpt)")
    log.info(f"(NB: above are bits-per-token, not BPB)")
    log.info(f"RTN quant gap:     {quant_gap:.6f} CE")
    log.info(f"GPTQ quant gap:    {gptq_gap:.6f} CE")
    log.info(f"GPTQ improvement:  {improvement:.6f} CE ({recovery_pct:.1f}% of RTN gap recovered)")
    log.info(f"GPTQ layers with activations: {n_layers_compared}")

    # Save if requested
    if args_cli.save:
        import pickle
        try:
            import zstandard
            data = pickle.dumps(gptq_quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = zstandard.ZstdCompressor(level=22).compress(data)
            with open(args_cli.save, "wb") as f:
                f.write(compressed)
            log.info(f"Saved GPTQ artifact: {args_cli.save} ({len(compressed):,} bytes)")
        except ImportError:
            with open(args_cli.save, "wb") as f:
                pickle.dump(gptq_quant_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.info(f"Saved GPTQ artifact (uncompressed): {args_cli.save}")


if __name__ == "__main__":
    main()
