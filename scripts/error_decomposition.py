#!/usr/bin/env python3
"""
error_decomposition.py — Decompose quantization error into actionable categories.

Experiments:
  A: Metric vs Topological error (MLP activation sign flips)
  B: Cascade error source attribution (attn vs MLP vs interaction, per layer)
  C: Per-weight-matrix error attribution
  D: Rounding vs Saturation error (pure weight analysis)

Usage:
    NUM_LAYERS=11 MLP_MULT=4.5 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \
    BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=128 LOGIT_SOFTCAP=30.0 \
    QUANT_ATTN_BITS=5 QUANT_MLP_BITS=5 \
    .venv/bin/python3 scripts/error_decomposition.py logs/wd50_11L_5x_best.npz \
        --n-seqs 16 --seq-len 1024 \
        --log-file logs/error_decomposition_int5.txt
"""
from __future__ import annotations

import argparse
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
    _QUANT_FN,
)
from scripts.correction_mse_standalone import build_model


# =============================================================================
# Helpers
# =============================================================================

def _np32(arr) -> np.ndarray:
    return np.array(arr, dtype=np.float32)


def rmse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def forward_collect_preskip(model, tokens):
    """Collect hidden states entering (pre) and leaving (post) each block.

    pre_block[i] = h entering block i (after any skip connection, before block call)
    post_block[i] = h after block i

    Returns: (pre_block, post_block, x0, encoder_outputs)
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
    pre_block = []
    post_block = []

    for i in range(n_layers):
        if i >= n_enc:
            dec_j = i - n_enc
            if dec_j < n_skip:
                enc_j = n_enc - 1 - dec_j
                if encoder_outputs[enc_j] is not None:
                    h = h + model.skip_weights[dec_j].astype(h.dtype)[None, None, :] * encoder_outputs[enc_j]
        pre_block.append(h)
        h = model.blocks[i](h, x0)
        mx.eval(h)
        post_block.append(h)
        if i < n_enc:
            encoder_outputs[i] = h

    return pre_block, post_block, x0, encoder_outputs


# =============================================================================
# Block decomposition primitive
# =============================================================================

def block_forward_decomposed(block, h_in, x0):
    """Step-by-step block forward with all intermediate states exposed.

    Returns dict with keys:
        mixed, attn_normed, attn_out, post_attn,
        mlp_normed, pre_act, post_act, mlp_out, h_out
    For swiglu: pre_act = gate*up product (fused); post_act == pre_act.
    """
    slope = block.lrelu_slope
    mix = block.resid_mix.astype(h_in.dtype)
    mixed = mix[0][None, None, :] * h_in + mix[1][None, None, :] * x0

    attn_normed = rms_norm(mixed)
    attn_out = block.attn(attn_normed)
    post_attn = mixed + block.attn_scale.astype(mixed.dtype)[None, None, :] * attn_out

    mlp_normed = rms_norm(post_attn)
    act = block.mlp.act

    if act == "swiglu":
        pre_act = nn.silu(block.mlp.gate(mlp_normed)) * block.mlp.fc(mlp_normed)
        post_act = pre_act
    else:
        pre_act = block.mlp.fc(mlp_normed)
        if act == "lrelu2":
            h = nn.leaky_relu(pre_act, negative_slope=slope)
        elif act == "sugar":
            soft = nn.leaky_relu(pre_act, negative_slope=slope)
            hard = nn.relu(pre_act)
            h = soft + mx.stop_gradient(hard - soft)
        else:  # relu2
            h = nn.relu(pre_act)
        post_act = h * h

    mlp_out = block.mlp.proj(post_act)
    h_out = post_attn + block.mlp_scale.astype(post_attn.dtype)[None, None, :] * mlp_out

    return {
        'mixed': mixed, 'attn_normed': attn_normed, 'attn_out': attn_out,
        'post_attn': post_attn, 'mlp_normed': mlp_normed,
        'pre_act': pre_act, 'post_act': post_act, 'mlp_out': mlp_out, 'h_out': h_out,
    }


# =============================================================================
# Experiment A: Metric vs Topological Error
# =============================================================================

def run_metric_vs_topological(model_f, model_q, val_tokens, n_seqs=16, seq_len=1024, log_fn=print):
    """Decompose MLP error into sign-agree (metric) vs sign-flip (topological) parts.

    Uses float hidden states as input to both blocks to isolate within-block error.
    Skips swiglu blocks (no sign-flip concept applies).

    Verification: frac_metric + frac_topo ≈ 1.0 per layer.
    """
    n_layers = len(model_f.blocks)
    log_fn(f"\n=== Experiment A: Metric vs Topological ({n_seqs}×{seq_len}) ===")

    flip_rate          = np.zeros(n_layers)
    rmse_metric        = np.zeros(n_layers)
    rmse_topo          = np.zeros(n_layers)
    rmse_total_pa      = np.zeros(n_layers)
    frac_metric_mse    = np.zeros(n_layers)
    frac_topo_mse      = np.zeros(n_layers)
    oracle_metric_rmse = np.zeros(n_layers)
    oracle_topo_rmse   = np.zeros(n_layers)
    h_out_rmse_base    = np.zeros(n_layers)
    n_valid            = np.zeros(n_layers)

    for s in range(n_seqs):
        tokens = val_tokens[s * seq_len: (s + 1) * seq_len]
        if len(tokens) < seq_len:
            break

        pre_f, _, x0_f, _ = forward_collect_preskip(model_f, tokens)

        for i in range(n_layers):
            act = model_f.blocks[i].mlp.act
            if act == "swiglu":
                continue

            h_in = pre_f[i]
            f = block_forward_decomposed(model_f.blocks[i], h_in, x0_f)
            q = block_forward_decomposed(model_q.blocks[i], h_in, x0_f)
            mx.eval(f['pre_act'], f['post_act'], f['h_out'],
                    q['pre_act'], q['post_act'], q['h_out'])

            pre_fn = _np32(f['pre_act']).reshape(-1, f['pre_act'].shape[-1])
            pre_qn = _np32(q['pre_act']).reshape(-1, q['pre_act'].shape[-1])
            pa_fn  = _np32(f['post_act']).reshape(-1, f['post_act'].shape[-1])
            pa_qn  = _np32(q['post_act']).reshape(-1, q['post_act'].shape[-1])

            agree = (pre_fn > 0) == (pre_qn > 0)
            flip = ~agree
            n_total = agree.size
            n_agree = int(agree.sum())
            n_flip  = int(flip.sum())

            flip_rate[i] += flip.mean()

            diff_pa = pa_fn - pa_qn
            total_mse = (diff_pa ** 2).mean()
            rmse_total_pa[i] += float(np.sqrt(total_mse))
            h_out_rmse_base[i] += rmse_np(_np32(f['h_out']), _np32(q['h_out']))

            if n_agree > 0:
                rmse_metric[i] += float(np.sqrt((diff_pa[agree] ** 2).mean()))
                frac_metric_mse[i] += (diff_pa[agree] ** 2).mean() * (n_agree / n_total) / max(total_mse, 1e-30)
            if n_flip > 0:
                rmse_topo[i] += float(np.sqrt((diff_pa[flip] ** 2).mean()))
                frac_topo_mse[i] += (diff_pa[flip] ** 2).mean() * (n_flip / n_total) / max(total_mse, 1e-30)

            # Oracle corrections: inject float post_act at each subset, re-run proj
            pa_shape = f['post_act'].shape
            pa_om = pa_qn.copy(); pa_om[agree] = pa_fn[agree]  # fix agree neurons
            pa_ot = pa_qn.copy(); pa_ot[flip]  = pa_fn[flip]   # fix flip neurons

            for pa_corr, out_arr in [(pa_om, oracle_metric_rmse), (pa_ot, oracle_topo_rmse)]:
                pa_mx = mx.array(pa_corr.reshape(pa_shape)).astype(COMPUTE_DTYPE)
                mlp_out_corr = model_f.blocks[i].mlp.proj(pa_mx)
                h_corr = f['post_attn'] + model_f.blocks[i].mlp_scale.astype(
                    f['post_attn'].dtype)[None, None, :] * mlp_out_corr
                mx.eval(h_corr)
                # h_corr uses float proj; compare to float h_out (upper bound)
                out_arr[i] += rmse_np(_np32(h_corr), _np32(f['h_out']))

            n_valid[i] += 1

    for i in range(n_layers):
        if n_valid[i] > 0:
            c = n_valid[i]
            for arr in (flip_rate, rmse_metric, rmse_topo, rmse_total_pa,
                        frac_metric_mse, frac_topo_mse,
                        oracle_metric_rmse, oracle_topo_rmse, h_out_rmse_base):
                arr[i] /= c

    log_fn(f"\n{'Lyr':>4}  {'FlipRate':>9}  {'RMSEmet':>8}  {'RMSEtopo':>9}  "
           f"{'FracMet':>8}  {'FracTopo':>9}  {'OrcMet':>9}  {'OrcTopo':>9}  {'hRMSEbase':>10}")
    log_fn("-" * 100)
    for i in range(n_layers):
        if n_valid[i] == 0:
            log_fn(f"{i:>4d}  (swiglu — skipped)")
            continue
        log_fn(f"{i:>4d}  {flip_rate[i]:>9.4f}  {rmse_metric[i]:>8.6f}  {rmse_topo[i]:>9.6f}  "
               f"{frac_metric_mse[i]:>8.4f}  {frac_topo_mse[i]:>9.4f}  "
               f"{oracle_metric_rmse[i]:>9.6f}  {oracle_topo_rmse[i]:>9.6f}  "
               f"{h_out_rmse_base[i]:>10.6f}")

    valid = n_valid > 0
    if valid.any():
        top_flip = int(np.argmax(np.where(valid, flip_rate, 0)))
        top_topo = int(np.argmax(np.where(valid, rmse_topo * frac_topo_mse, 0)))
        log_fn(f"\nHighest flip rate: L{top_flip} ({flip_rate[top_flip]:.4f})")
        log_fn(f"Highest topo MSE contribution: L{top_topo} "
               f"(rmse_topo={rmse_topo[top_topo]:.6f}, frac={frac_topo_mse[top_topo]:.4f})")
        log_fn(f"\n[Verification] frac_metric+frac_topo per layer (should ≈ 1.0):")
        for i in range(n_layers):
            if n_valid[i] > 0:
                log_fn(f"  L{i}: {frac_metric_mse[i]+frac_topo_mse[i]:.4f}")

    return {
        'flip_rate': flip_rate, 'rmse_metric': rmse_metric, 'rmse_topo': rmse_topo,
        'rmse_total_pa': rmse_total_pa, 'frac_metric_mse': frac_metric_mse,
        'frac_topo_mse': frac_topo_mse,
        'oracle_metric_rmse': oracle_metric_rmse, 'oracle_topo_rmse': oracle_topo_rmse,
        'h_out_rmse_base': h_out_rmse_base,
    }


# =============================================================================
# Experiment B: Cascade Error Source Attribution
# =============================================================================

def run_cascade_attribution(model_f, model_q, val_tokens, n_seqs=16, seq_len=1024, log_fn=print):
    """Per-block decomposition of quantization error into attn, MLP, and interaction.

    For each block i with h_float_in (from float stream) and h_quant_in (quant stream):
      D = float_block(h_float_in)          — float baseline
      F = quant_block(h_quant_in)          — full quant output
      A = float_attn + quant_MLP (float input) — isolates MLP quant effect
      B = quant_attn + float_MLP (quant input) — isolates attn quant + upstream error

    Decomposition (vector deltas):
      δ_total = h_F - h_D
      δ_attn  = h_B - h_D
      δ_mlp   = h_A - h_D
      δ_inter = δ_total - δ_attn - δ_mlp

    Verification: δ_attn + δ_mlp + δ_inter == δ_total (exact by construction).
    """
    n_layers = len(model_f.blocks)
    log_fn(f"\n=== Experiment B: Cascade Attribution ({n_seqs}×{seq_len}) ===")

    rmse_total = np.zeros(n_layers)
    rmse_attn  = np.zeros(n_layers)
    rmse_mlp   = np.zeros(n_layers)
    rmse_inter = np.zeros(n_layers)
    count = 0

    for s in range(n_seqs):
        tokens = val_tokens[s * seq_len: (s + 1) * seq_len]
        if len(tokens) < seq_len:
            break

        pre_f, _, x0_f, _ = forward_collect_preskip(model_f, tokens)
        pre_q, _, x0_q, _ = forward_collect_preskip(model_q, tokens)

        for i in range(n_layers):
            h_fi = pre_f[i]
            h_qi = pre_q[i]
            slope = model_f.blocks[i].lrelu_slope

            # D: float block, float input
            d = block_forward_decomposed(model_f.blocks[i], h_fi, x0_f)
            h_D = d['h_out']

            # F: quant block, quant input
            fv = block_forward_decomposed(model_q.blocks[i], h_qi, x0_q)
            h_F = fv['h_out']

            # A: float attn + quant MLP, float input
            mix_f = model_f.blocks[i].resid_mix.astype(h_fi.dtype)
            mixed_fA = mix_f[0][None, None, :] * h_fi + mix_f[1][None, None, :] * x0_f
            attn_out_fA = model_f.blocks[i].attn(rms_norm(mixed_fA))
            post_attn_A = mixed_fA + model_f.blocks[i].attn_scale.astype(
                mixed_fA.dtype)[None, None, :] * attn_out_fA
            mlp_out_A = model_q.blocks[i].mlp(rms_norm(post_attn_A), slope=slope)
            h_A = post_attn_A + model_q.blocks[i].mlp_scale.astype(
                post_attn_A.dtype)[None, None, :] * mlp_out_A

            # B: quant attn + float MLP, quant input
            mix_q = model_q.blocks[i].resid_mix.astype(h_qi.dtype)
            mixed_qB = mix_q[0][None, None, :] * h_qi + mix_q[1][None, None, :] * x0_q
            attn_out_qB = model_q.blocks[i].attn(rms_norm(mixed_qB))
            post_attn_B = mixed_qB + model_q.blocks[i].attn_scale.astype(
                mixed_qB.dtype)[None, None, :] * attn_out_qB
            mlp_out_B = model_f.blocks[i].mlp(rms_norm(post_attn_B), slope=slope)
            h_B = post_attn_B + model_f.blocks[i].mlp_scale.astype(
                post_attn_B.dtype)[None, None, :] * mlp_out_B

            mx.eval(h_D, h_F, h_A, h_B)

            hD = _np32(h_D).ravel()
            hF = _np32(h_F).ravel()
            hA = _np32(h_A).ravel()
            hB = _np32(h_B).ravel()

            dT     = hF - hD
            dAttn  = hB - hD
            dMlp   = hA - hD
            dInter = dT - dAttn - dMlp

            rmse_total[i] += float(np.sqrt((dT     ** 2).mean()))
            rmse_attn[i]  += float(np.sqrt((dAttn  ** 2).mean()))
            rmse_mlp[i]   += float(np.sqrt((dMlp   ** 2).mean()))
            rmse_inter[i] += float(np.sqrt((dInter ** 2).mean()))

        count += 1
        log_fn(f"  seq {s}: done")

    if count > 0:
        for arr in (rmse_total, rmse_attn, rmse_mlp, rmse_inter):
            arr /= count

    log_fn(f"\n{'Lyr':>4}  {'RMSE_tot':>9}  {'RMSE_attn':>10}  {'RMSE_mlp':>9}  "
           f"{'RMSE_inter':>11}  {'attn%':>7}  {'mlp%':>7}  {'inter%':>8}")
    log_fn("-" * 85)
    for i in range(n_layers):
        tot = max(rmse_total[i], 1e-12)
        log_fn(f"{i:>4d}  {rmse_total[i]:>9.6f}  {rmse_attn[i]:>10.6f}  {rmse_mlp[i]:>9.6f}  "
               f"{rmse_inter[i]:>11.6f}  "
               f"{rmse_attn[i]/tot:>7.3f}  {rmse_mlp[i]/tot:>7.3f}  {rmse_inter[i]/tot:>8.3f}")

    top_mlp   = int(np.argmax(rmse_mlp))
    top_attn  = int(np.argmax(rmse_attn))
    top_inter = int(np.argmax(rmse_inter))
    log_fn(f"\nMLP dominates:   L{top_mlp}  (rmse_mlp={rmse_mlp[top_mlp]:.6f})")
    log_fn(f"Attn dominates:  L{top_attn}  (rmse_attn={rmse_attn[top_attn]:.6f})")
    log_fn(f"Interaction:     L{top_inter} (rmse_inter={rmse_inter[top_inter]:.6f})")

    return {
        'rmse_total': rmse_total, 'rmse_attn': rmse_attn,
        'rmse_mlp': rmse_mlp, 'rmse_inter': rmse_inter,
    }


# =============================================================================
# Experiment C: Per-Weight-Matrix Error Attribution
# =============================================================================

def run_per_weight_attribution(model_f, model_q, val_tokens, n_seqs=16, seq_len=1024, log_fn=print):
    """Measure h_out RMSE when each weight matrix is individually quantized.

    For each block, swaps one matrix at a time to its dequantized-quant version,
    runs block_forward_decomposed with float hidden state input, and measures
    RMSE vs the all-float baseline.

    interaction = RMSE_all_quant - sum(RMSE_individual)
    If interaction > 0: errors are superadditive (non-linear).
    If interaction < 0: errors partially cancel.
    """
    n_layers = len(model_f.blocks)
    log_fn(f"\n=== Experiment C: Per-Weight Attribution ({n_seqs}×{seq_len}) ===")

    base_mat_names = ['c_q', 'c_k', 'c_v', 'attn_proj', 'mlp_fc', 'mlp_proj']

    rmse_by_mat   = {m: np.zeros(n_layers) for m in base_mat_names + ['mlp_gate']}
    rmse_all_quant = np.zeros(n_layers)
    count = 0

    def get_weight(block, name):
        if name == 'c_q':        return block.attn.c_q.weight
        if name == 'c_k':        return block.attn.c_k.weight
        if name == 'c_v':        return block.attn.c_v.weight
        if name == 'attn_proj':  return block.attn.proj.weight
        if name == 'mlp_fc':     return block.mlp.fc.weight
        if name == 'mlp_proj':   return block.mlp.proj.weight
        if name == 'mlp_gate':   return block.mlp.gate.weight
        raise ValueError(f"Unknown matrix: {name}")

    def set_weight(block, name, w):
        if name == 'c_q':        block.attn.c_q.weight = w
        elif name == 'c_k':      block.attn.c_k.weight = w
        elif name == 'c_v':      block.attn.c_v.weight = w
        elif name == 'attn_proj': block.attn.proj.weight = w
        elif name == 'mlp_fc':   block.mlp.fc.weight = w
        elif name == 'mlp_proj': block.mlp.proj.weight = w
        elif name == 'mlp_gate': block.mlp.gate.weight = w

    for s in range(n_seqs):
        tokens = val_tokens[s * seq_len: (s + 1) * seq_len]
        if len(tokens) < seq_len:
            break

        pre_f, _, x0_f, _ = forward_collect_preskip(model_f, tokens)

        for i in range(n_layers):
            h_in = pre_f[i]
            act = model_f.blocks[i].mlp.act
            mats = base_mat_names + (['mlp_gate'] if act == "swiglu" else [])

            # Baseline: all float
            f_base = block_forward_decomposed(model_f.blocks[i], h_in, x0_f)
            # All quant (use quant block, float input)
            f_allq = block_forward_decomposed(model_q.blocks[i], h_in, x0_f)
            mx.eval(f_base['h_out'], f_allq['h_out'])

            h_base = _np32(f_base['h_out'])
            rmse_all_quant[i] += rmse_np(_np32(f_allq['h_out']), h_base)

            for mat in mats:
                w_f = get_weight(model_f.blocks[i], mat)
                w_q = get_weight(model_q.blocks[i], mat)

                # Swap float model's matrix to quant version
                set_weight(model_f.blocks[i], mat, w_q)
                f_swap = block_forward_decomposed(model_f.blocks[i], h_in, x0_f)
                mx.eval(f_swap['h_out'])
                rmse_by_mat[mat][i] += rmse_np(_np32(f_swap['h_out']), h_base)
                # Restore
                set_weight(model_f.blocks[i], mat, w_f)

        count += 1
        log_fn(f"  seq {s}: done")

    if count > 0:
        rmse_all_quant /= count
        for arr in rmse_by_mat.values():
            arr /= count

    # Print table
    log_fn(f"\n{'Lyr':>4}  {'ALL_q':>9}  {'c_q':>9}  {'c_k':>9}  {'c_v':>9}  "
           f"{'a_proj':>9}  {'m_fc':>9}  {'m_proj':>9}  {'sum_indiv':>10}  {'interact':>9}")
    log_fn("-" * 100)
    for i in range(n_layers):
        act = model_f.blocks[i].mlp.act
        mats = base_mat_names + (['mlp_gate'] if act == "swiglu" else [])
        sum_indiv = sum(rmse_by_mat[m][i] for m in mats)
        interact = rmse_all_quant[i] - sum_indiv
        vals = "  ".join(f"{rmse_by_mat[m][i]:>9.6f}" for m in base_mat_names)
        log_fn(f"{i:>4d}  {rmse_all_quant[i]:>9.6f}  {vals}  {sum_indiv:>10.6f}  {interact:>9.6f}")

    log_fn("\nTop contributor per layer:")
    for i in range(n_layers):
        act = model_f.blocks[i].mlp.act
        mats = base_mat_names + (['mlp_gate'] if act == "swiglu" else [])
        top_mat = max(mats, key=lambda m: rmse_by_mat[m][i])
        log_fn(f"  L{i}: {top_mat} ({rmse_by_mat[top_mat][i]:.6f})")

    return {'rmse_all_quant': rmse_all_quant, 'rmse_by_mat': rmse_by_mat}


# =============================================================================
# Experiment D: Rounding vs Saturation Error
# =============================================================================

def run_round_vs_saturation(model_f, hparams, log_fn=print):
    """Decompose weight quantization error into rounding vs saturation components.

    For int5 (scale = max_abs / 15), saturation is expected to be near-zero.
    This confirms that all error is rounding-grid noise.

    Verification: frac_saturated ≈ 0 for int5.
    """
    log_fn(f"\n=== Experiment D: Rounding vs Saturation ===")

    cat_bits_env = {"attn": hparams.quant_attn_bits, "mlp": hparams.quant_mlp_bits}
    quant_bits_str = os.environ.get("QUANT_BITS", "")
    if quant_bits_str:
        cat_bits_env = {}
        for part in quant_bits_str.split(","):
            k, v = part.strip().rsplit(":", 1)
            cat_bits_env[k.strip()] = int(v.strip())

    def get_range(bits):
        return {2: (-2, 1), 3: (-4, 3), 4: (-8, 7), 5: (-16, 15), 6: (-32, 31)}.get(bits, (-16, 15))

    def analyze(w_mx, cat):
        base_cat = cat.split(".")[0]
        bits = cat_bits_env.get(cat) or cat_bits_env.get(base_cat, 5)
        lo, hi = get_range(bits)

        w = np.array(w_mx, dtype=np.float32)
        if w.ndim != 2:
            return None

        row_max = np.abs(w).max(axis=1)
        scale = np.maximum(row_max / float(hi), 1e-12).astype(np.float32)

        w_scaled = w / scale[:, None]
        w_round_units = np.round(w_scaled)
        w_quant_units = np.clip(w_round_units, lo, hi)

        w_noclip = w_round_units * scale[:, None]   # rounded, not clipped
        w_quant  = w_quant_units * scale[:, None]   # rounded + clipped (actual quant)

        frac_sat  = float((w_quant_units != w_round_units).mean())
        rmse_round = float(np.sqrt(((w_noclip - w) ** 2).mean()))
        rmse_sat   = float(np.sqrt(((w_quant - w_noclip) ** 2).mean()))
        rmse_total = float(np.sqrt(((w_quant - w) ** 2).mean()))
        return {'bits': bits, 'frac_sat': frac_sat,
                'rmse_round': rmse_round, 'rmse_sat': rmse_sat, 'rmse_total': rmse_total}

    mat_info = [
        ('c_q',      lambda b: b.attn.c_q.weight,  'attn'),
        ('c_k',      lambda b: b.attn.c_k.weight,  'attn'),
        ('c_v',      lambda b: b.attn.c_v.weight,  'attn'),
        ('attn_proj',lambda b: b.attn.proj.weight, 'attn'),
        ('mlp_fc',   lambda b: b.mlp.fc.weight,    'mlp'),
        ('mlp_proj', lambda b: b.mlp.proj.weight,  'mlp'),
    ]

    log_fn(f"\n{'Lyr':>4}  {'Matrix':>10}  {'bits':>4}  {'frac_sat':>9}  "
           f"{'RMSE_round':>11}  {'RMSE_sat':>10}  {'RMSE_total':>11}")
    log_fn("-" * 76)

    all_frac_sat = []
    n_layers = len(model_f.blocks)

    for i in range(n_layers):
        for mat_name, getter, cat in mat_info:
            if mat_name == 'mlp_gate' and model_f.blocks[i].mlp.act != "swiglu":
                continue
            res = analyze(getter(model_f.blocks[i]), cat)
            if res is None:
                continue
            all_frac_sat.append(res['frac_sat'])
            log_fn(f"{i:>4d}  {mat_name:>10}  {res['bits']:>4d}  {res['frac_sat']:>9.6f}  "
                   f"{res['rmse_round']:>11.8f}  {res['rmse_sat']:>10.8f}  {res['rmse_total']:>11.8f}")
        log_fn("")  # blank line between layers

    avg_sat = float(np.mean(all_frac_sat)) if all_frac_sat else 0.0
    log_fn(f"Average frac_saturated: {avg_sat:.6f}")
    if avg_sat < 0.001:
        log_fn("  → Saturation negligible. All error is rounding noise (expected for int5).")
    else:
        log_fn(f"  → Non-trivial saturation ({avg_sat:.4f}). Consider better scale selection.")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Quantization error decomposition (A/B/C/D)")
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument("--experiments", type=str, default="A,B,C,D",
                        help="Comma-separated subset, e.g. --experiments A,B")
    parser.add_argument("--n-seqs", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    hparams = Hyperparameters()
    experiments = set(x.strip().upper() for x in args.experiments.split(","))

    if args.log_file is None:
        bits = hparams.quant_attn_bits
        args.log_file = f"logs/error_decomposition_int{bits}.txt"
    os.makedirs(os.path.dirname(args.log_file) if os.path.dirname(args.log_file) else ".", exist_ok=True)

    log_lines = []
    def log_fn(msg):
        print(msg)
        log_lines.append(msg)

    log_fn(f"Config: {hparams.num_layers}L/{hparams.model_dim}d  MLP {hparams.mlp_mult}x  "
           f"act={hparams.mlp_act}  quant=a{hparams.quant_attn_bits}m{hparams.quant_mlp_bits}")
    log_fn(f"Checkpoint: {args.checkpoint}")
    log_fn(f"n_seqs={args.n_seqs}  seq_len={args.seq_len}")
    log_fn(f"Experiments: {sorted(experiments)}")

    # Float model
    log_fn("\nLoading checkpoint...")
    flat = dict(mx.load(args.checkpoint))
    model_float = build_model(hparams)
    model_float.update(tree_unflatten(list(flat.items())))
    mx.eval(model_float.parameters())

    # Quantized model (only needed for A/B/C)
    model_quant = None
    if experiments & {"A", "B", "C"}:
        log_fn("Quantizing model...")
        cat_bits = {"attn": hparams.quant_attn_bits, "mlp": hparams.quant_mlp_bits}
        quant_bits_str = os.environ.get("QUANT_BITS", "")
        if quant_bits_str:
            cat_bits = {}
            for part in quant_bits_str.split(","):
                k, v = part.strip().rsplit(":", 1)
                cat_bits[k.strip()] = int(v.strip())
            log_fn(f"Per-layer quant: {cat_bits}")
        quant_obj, _ = quantize_state_dict_int8(flat, cat_bits=cat_bits)
        quant_flat = dequantize_state_dict_int8(quant_obj)
        model_quant = build_model(hparams)
        model_quant.update(tree_unflatten(list(quant_flat.items())))
        mx.eval(model_quant.parameters())

    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    log_fn(f"Val tokens: {len(val_tokens):,}")

    t0 = time.time()

    if "A" in experiments:
        run_metric_vs_topological(model_float, model_quant, val_tokens,
                                   n_seqs=args.n_seqs, seq_len=args.seq_len, log_fn=log_fn)

    if "B" in experiments:
        run_cascade_attribution(model_float, model_quant, val_tokens,
                                 n_seqs=args.n_seqs, seq_len=args.seq_len, log_fn=log_fn)

    if "C" in experiments:
        run_per_weight_attribution(model_float, model_quant, val_tokens,
                                    n_seqs=args.n_seqs, seq_len=args.seq_len, log_fn=log_fn)

    if "D" in experiments:
        run_round_vs_saturation(model_float, hparams, log_fn=log_fn)

    elapsed = time.time() - t0
    log_fn(f"\nTotal elapsed: {elapsed/60:.1f} min")

    with open(args.log_file, "w") as lf:
        lf.write("\n".join(log_lines) + "\n")
    print(f"Logged to {args.log_file}")


if __name__ == "__main__":
    main()
