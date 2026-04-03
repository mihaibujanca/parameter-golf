#!/usr/bin/env python3
"""
correction_sensitivity.py — Per-matrix correction sensitivity analysis.

For each quantized weight matrix (c_q, c_k, c_v, attn.proj, mlp.fc, mlp.proj
× N layers), measure the BPB improvement from applying the exact correction
c = -input @ E.T where E = W_quant - W_float, stored at int4.

This tells us which corrections are worth their storage cost and which can be
dropped. Outputs a ranked table of matrices by BPB impact.

Usage:
    NUM_LAYERS=11 MLP_MULT=5.0 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \
    QUANT_ATTN_BITS=4 QUANT_MLP_BITS=4 \
    .venv/bin/python3 scripts/correction_sensitivity.py logs/11L_5x_xsa_ema_best.npz \
        --log-file logs/correction_sensitivity_int4.txt

    # MLP-only quantization:
    QUANT_ATTN_BITS=8 QUANT_MLP_BITS=4 \
    .venv/bin/python3 scripts/correction_sensitivity.py logs/11L_5x_xsa_ema_best.npz
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gpt_mlx import (
    GPT, Hyperparameters, load_validation_tokens,
    quantize_state_dict_int8, dequantize_state_dict_int8,
    build_sentencepiece_luts, eval_val,
)

log = logging.getLogger("correction_sensitivity")

WEIGHT_NAMES_PER_LAYER = [
    "attn.c_q.weight", "attn.c_k.weight", "attn.c_v.weight",
    "attn.proj.weight", "mlp.fc.weight", "mlp.proj.weight",
]


# ============================================================================
# Model construction
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
        num_encoder_layers=hparams.num_encoder_layers,
    )


# ============================================================================
# Compute and quantize E matrices
# ============================================================================

def compute_corrections(flat_float, flat_quant, n_layers, corr_bits=4):
    """Compute E = W_quant - W_float for each weight matrix, quantize to corr_bits."""
    qmax = {2: 1, 3: 3, 4: 7, 5: 15, 6: 31, 8: 127}[corr_bits]
    corrections = {}
    sizes = {}

    for i in range(n_layers):
        for wname in WEIGHT_NAMES_PER_LAYER:
            key = f"blocks.{i}.{wname}"
            if key not in flat_float or key not in flat_quant:
                continue
            Wf = np.array(flat_float[key]).astype(np.float32)
            Wq = np.array(flat_quant[key]).astype(np.float32)
            E = Wq - Wf
            # Quantize E per-row
            row_max = np.abs(E).max(axis=1)
            scale = np.maximum(row_max / qmax, 1e-12).astype(np.float32)
            E_q = np.clip(np.round(E / scale[:, None]), -qmax - 1, qmax)
            E_dequant = (E_q * scale[:, None]).astype(np.float32)
            corrections[key] = mx.array(E_dequant)
            sizes[key] = E.shape[0] * E.shape[1] + E.shape[0] * 2  # int8 + fp16 scales

    mx.eval(list(corrections.values()))
    return corrections, sizes


# ============================================================================
# Eval with corrections via weight modification
# ============================================================================

def eval_with_corrections(model, flat_quant, active_corrections, hparams,
                          val_tokens, luts):
    """Apply corrections as weight modifications, recompile, eval via standard eval_val.

    Correction math: output - input @ E.T  ≡  (W_quant - E) @ input.
    So we set W = W_quant - E_dequant for active corrections, then use the
    standard model.loss / eval_val path. This guarantees BPB matches the
    training script exactly.
    """
    updates = dict(flat_quant)
    for key, E_dequant in active_corrections.items():
        if key in updates:
            updates[key] = updates[key] - E_dequant
    model.update(tree_unflatten(list(updates.items())))
    mx.eval(model.state)
    compiled_loss = mx.compile(
        lambda x, y: model.loss(x, y),
        inputs=model.state, outputs=model.state,
    )
    _, bpb = eval_val(hparams, compiled_loss, val_tokens, *luts)
    return bpb


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Per-matrix correction sensitivity")
    parser.add_argument("checkpoint", help="Path to .npz float checkpoint")
    parser.add_argument("--corr-bits", type=int, default=4,
                        help="Bit-width for correction matrices (default: 4)")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--n-eval-seqs", type=int, default=32,
                        help="Number of sequences for eval (0=full val split)")
    args_cli = parser.parse_args()

    hparams = Hyperparameters()

    # Logging
    global log
    log = logging.getLogger("correction_sensitivity")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if args_cli.log_file is None:
        args_cli.log_file = (f"logs/correction_sensitivity"
                             f"_a{hparams.quant_attn_bits}m{hparams.quant_mlp_bits}.txt")
    os.makedirs(os.path.dirname(args_cli.log_file), exist_ok=True)
    fh = logging.FileHandler(args_cli.log_file, mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.info(f"Logging to {args_cli.log_file}")

    cat_bits = {"attn": hparams.quant_attn_bits, "mlp": hparams.quant_mlp_bits}
    log.info(f"Config: {hparams.num_layers}L/{hparams.model_dim}d, MLP {hparams.mlp_mult}x, "
             f"act={hparams.mlp_act}, quant={cat_bits}, corr_bits={args_cli.corr_bits}")

    # Load float weights
    log.info(f"Loading checkpoint: {args_cli.checkpoint}")
    flat_float = dict(mx.load(args_cli.checkpoint))

    # Quantize
    model_q = build_model(hparams)
    model_q.update(tree_unflatten(list(flat_float.items())))
    mx.eval(model_q.parameters())
    fq = {k: v for k, v in tree_flatten(model_q.state)}
    qo, _ = quantize_state_dict_int8(fq, cat_bits=cat_bits)
    flat_quant = dequantize_state_dict_int8(qo)
    model_q.update(tree_unflatten(list(flat_quant.items())))
    mx.eval(model_q.parameters())

    # Compute correction matrices
    log.info(f"Computing int{args_cli.corr_bits} correction matrices...")
    def _to_np(d):
        out = {}
        for k, v in d.items():
            try:
                out[k] = np.array(v.astype(mx.float32)) if isinstance(v, mx.array) else np.array(v, dtype=np.float32)
            except Exception:
                pass  # skip non-convertible (e.g. integer arrays)
        return out

    corrections, sizes = compute_corrections(
        _to_np(flat_float), _to_np(flat_quant),
        hparams.num_layers, corr_bits=args_cli.corr_bits,
    )
    log.info(f"  {len(corrections)} matrices, "
             f"total raw: {sum(sizes.values()) / 1024 / 1024:.1f} MB")

    # Load eval data
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    sp = spm.SentencePieceProcessor(model_file=hparams.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, hparams.vocab_size)

    n_seqs = args_cli.n_eval_seqs
    if n_seqs > 0:
        hparams.val_max_tokens = n_seqs * hparams.train_seq_len
    log.info(f"Eval: {n_seqs} seqs ({hparams.val_max_tokens} tokens)"
             if n_seqs > 0 else "Eval: full val split")

    luts = (base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)

    def _eval(active):
        return eval_with_corrections(model_q, flat_quant, active, hparams,
                                     val_tokens, luts)

    # Baselines
    log.info("\n=== Baselines ===")
    t0 = time.time()
    bpb_none = _eval({})
    log.info(f"No corrections:  BPB={bpb_none:.4f}")
    bpb_all = _eval(corrections)
    log.info(f"All corrections: BPB={bpb_all:.4f}")
    total_gap = bpb_none - bpb_all
    log.info(f"Total recovery:  {total_gap * 1000:.1f} mBPB")
    log.info(f"Baselines: {time.time() - t0:.0f}s")

    # Per-matrix sensitivity: correct ONE matrix at a time
    log.info(f"\n=== Per-matrix sensitivity (one at a time) ===")
    log.info(f"{'Matrix':<40s} {'BPB':>7s} {'Δ mBPB':>8s} {'% gap':>7s} {'raw KB':>7s}")
    log.info("-" * 73)

    results = []
    t0 = time.time()
    for i in range(hparams.num_layers):
        for wname in WEIGHT_NAMES_PER_LAYER:
            key = f"blocks.{i}.{wname}"
            if key not in corrections:
                continue
            bpb = _eval({key: corrections[key]})
            delta = (bpb_none - bpb) * 1000
            pct = delta / (total_gap * 1000) * 100 if total_gap > 0 else 0
            raw_kb = sizes[key] / 1024
            short = f"L{i}.{wname.replace('.weight', '')}"
            log.info(f"{short:<40s} {bpb:7.4f} {delta:+8.2f} {pct:6.1f}% {raw_kb:7.1f}")
            results.append((key, short, bpb, delta, pct, raw_kb))
    log.info(f"Per-matrix sweep: {time.time() - t0:.0f}s")

    # Ranked table
    results.sort(key=lambda x: -x[3])
    log.info(f"\n=== Ranked by BPB impact ===")
    log.info(f"{'#':>3s} {'Matrix':<40s} {'Δ mBPB':>8s} {'cum mBPB':>9s} "
             f"{'cum KB':>8s} {'cum %':>7s}")
    log.info("-" * 80)
    cum_delta = 0.0
    cum_kb = 0.0
    for rank, (key, short, bpb, delta, pct, raw_kb) in enumerate(results):
        cum_delta += delta
        cum_kb += raw_kb
        cum_pct = cum_delta / (total_gap * 1000) * 100 if total_gap > 0 else 0
        log.info(f"{rank + 1:>3d} {short:<40s} {delta:+8.2f} {cum_delta:9.2f} "
                 f"{cum_kb:8.0f} {cum_pct:6.1f}%")

    # Summary by category
    log.info(f"\n=== Summary by weight type ===")
    categories = {}
    for key, short, bpb, delta, pct, raw_kb in results:
        # Extract category: attn.c_q, attn.c_k, attn.c_v, attn.proj, mlp.fc, mlp.proj
        parts = key.split(".")
        cat = f"{parts[2]}.{parts[3]}"  # e.g. "attn.c_q"
        if cat not in categories:
            categories[cat] = {"delta_sum": 0.0, "kb_sum": 0.0, "count": 0}
        categories[cat]["delta_sum"] += delta
        categories[cat]["kb_sum"] += raw_kb
        categories[cat]["count"] += 1

    log.info(f"{'Category':<15s} {'total mBPB':>11s} {'total KB':>9s} {'avg mBPB':>9s}")
    log.info("-" * 48)
    for cat in sorted(categories, key=lambda c: -categories[c]["delta_sum"]):
        d = categories[cat]
        log.info(f"{cat:<15s} {d['delta_sum']:+11.2f} {d['kb_sum']:9.0f} "
                 f"{d['delta_sum'] / d['count']:+9.2f}")

    # Greedy selection: pick corrections in order until budget is hit
    log.info(f"\n=== Greedy budget sweep (ranked order) ===")
    log.info(f"{'Budget MB':>9s} {'# corr':>7s} {'cum mBPB':>9s} {'BPB':>7s}")
    log.info("-" * 38)
    cum_delta = 0.0
    cum_kb = 0.0
    last_printed = 0
    for rank, (key, short, bpb, delta, pct, raw_kb) in enumerate(results):
        cum_delta += delta
        cum_kb += raw_kb
        cum_mb = cum_kb / 1024
        # Print at budget milestones
        if cum_mb >= last_printed + 0.5 or rank == len(results) - 1:
            est_bpb = bpb_none - cum_delta / 1000
            log.info(f"{cum_mb:9.1f} {rank + 1:>7d} {cum_delta:+9.2f} {est_bpb:7.4f}")
            last_printed = int(cum_mb * 2) / 2


if __name__ == "__main__":
    main()
