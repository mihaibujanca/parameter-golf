#!/usr/bin/env python3
"""
float_polish.py — Gradient polishing: short post-training fine-tuning with
fake quantization so weights move to positions that quantize better.

Inspired by gradient polishing from quant_aware_training: unfreeze weight
matrices, run ~500 Adam steps on global CE with quantization noise injected
via straight-through estimator (STE). Weights shift to grid-friendly positions,
reducing the quant gap when you do the final PTQ.

Usage:
    NUM_LAYERS=11 MLP_MULT=4.5 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \\
    QUANT_ATTN_BITS=5 QUANT_MLP_BITS=5 \\
    .venv/bin/python3 scripts/float_polish.py logs/warmdown_11L_45x_best.npz \\
        --steps 500 --lr 1e-4

    # Eval-only (measure quant gap without polishing):
    ... --eval-only
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gpt_mlx import (
    GPT, COMPUTE_DTYPE, Hyperparameters, load_validation_tokens,
    quantize_state_dict_int8, dequantize_state_dict_int8,
    _classify_param, CONTROL_TENSOR_NAME_PATTERNS, FP16_KEEP_NAME_PATTERNS,
    INT8_KEEP_FLOAT_MAX_NUMEL,
)
from scripts.eval_commons import (
    quick_ce as _quick_ce,
    load_train_tokens as _load_train_tokens,
)

log = logging.getLogger("float_polish")

# ============================================================================
# Fake quantization with STE
# ============================================================================

def _fake_quant_per_row(w: mx.array, qmax: int) -> mx.array:
    """Fake-quantize a 2D weight matrix: quantize→dequantize with STE.

    Forward: returns the quantized-then-dequantized value (grid-snapped).
    Backward: straight-through (gradient passes through as if identity).
    """
    row_max = mx.abs(w).max(axis=1, keepdims=True)
    scale = mx.maximum(row_max / qmax, 1e-12)
    q = mx.clip(mx.round(w / scale), -qmax - 1, qmax)
    w_fake = q * scale
    # STE: forward uses w_fake, backward uses gradient of w
    return w + mx.stop_gradient(w_fake - w)


_BITS_TO_QMAX = {2: 1, 3: 3, 4: 7, 5: 15, 6: 31, 8: 127}


def build_param_bits_map(model, cat_bits: dict[str, int],
                         filter_pattern: str | None = None) -> dict[str, int]:
    """Map each 2D weight parameter name to its quantization bit-width.

    filter_pattern: if set, only include params whose name matches this regex.
        e.g. "mlp.proj" for MLP output projections only,
             "mlp" for all MLP weights, None for all quantized weights.
    """
    param_bits = {}
    for name, p in tree_flatten(model.parameters()):
        if p.ndim != 2:
            continue
        if any(pat in name for pat in FP16_KEEP_NAME_PATTERNS):
            continue
        if any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
            continue
        if int(p.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            continue
        if filter_pattern and not re.search(filter_pattern, name):
            continue
        cat = _classify_param(name)
        bits = cat_bits.get(cat)
        if bits is None:
            base_cat = cat.split(".")[0]
            bits = cat_bits.get(base_cat, 8)
        param_bits[name] = bits
    return param_bits


# ============================================================================
# Core polish loop
# ============================================================================

def gradient_polish(model, train_tokens, param_bits: dict[str, int],
                    n_steps=500, lr=1e-4, seq_len=1024, batch_seqs=4,
                    log_every=50):
    """Gradient polish: fine-tune weights with fake quantization on global CE.

    All weight matrices get fake-quantized in the forward pass (STE), so
    gradients push weights toward grid-friendly positions.
    """
    # Freeze everything, then unfreeze only the weight matrices that get quantized
    model.freeze()
    for name in param_bits:
        keys = name.split(".")
        obj = model
        for k in keys[:-1]:
            if k.isdigit():
                obj = obj[int(k)]
            else:
                obj = getattr(obj, k)
        obj.unfreeze(keys=[keys[-1]])

    trainable = dict(tree_flatten(model.trainable_parameters()))
    n_trainable = sum(int(v.size) for v in trainable.values())
    log.info(f"Trainable: {n_trainable:,} params ({len(trainable)} tensors)")

    # Pre-resolve parameter references for fake-quant
    _fq_targets = []  # list of (parent_obj, attr_name, qmax)
    for name, bits in param_bits.items():
        qmax = _BITS_TO_QMAX[bits]
        keys = name.split(".")
        obj = model
        for k in keys[:-1]:
            obj = obj[int(k)] if k.isdigit() else getattr(obj, k)
        _fq_targets.append((obj, keys[-1], qmax))

    def loss_with_fake_quant(x, y):
        for obj, attr, qmax in _fq_targets:
            w = getattr(obj, attr)
            setattr(obj, attr, w + mx.stop_gradient(_fake_quant_per_row(w, qmax) - w))
        return model.loss(x, y)

    loss_and_grad = mx.compile(
        nn.value_and_grad(model, loss_with_fake_quant),
        inputs=model.state,
        outputs=model.state,
    )

    # Snapshot for delta check
    init_snapshot = {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(model.trainable_parameters())}

    # Cosine decay
    def get_lr(step):
        if n_steps <= 1:
            return lr
        return lr * 0.5 * (1.0 + math.cos(math.pi * step / (n_steps - 1)))

    optimizer = optim.Adam(learning_rate=lr)

    # Training loop
    history = []
    max_start = len(train_tokens) - (batch_seqs * seq_len + 1)
    if max_start <= 0:
        raise ValueError(f"Not enough tokens ({len(train_tokens)}) for batch={batch_seqs}×{seq_len}")

    t0 = time.time()
    for step in range(n_steps):
        optimizer.learning_rate = mx.array(get_lr(step))

        start = np.random.randint(0, max_start)
        n_tok = batch_seqs * seq_len + 1
        chunk = train_tokens[start:start + n_tok]
        x = mx.array(chunk[:batch_seqs * seq_len].reshape(batch_seqs, seq_len))
        y = mx.array(chunk[1:batch_seqs * seq_len + 1].reshape(batch_seqs, seq_len))

        loss, grads = loss_and_grad(x, y)

        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.trainable_parameters()))
        updated = optimizer.apply_gradients(flat_grads, flat_params)
        model.update(tree_unflatten(list(updated.items())))
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        history.append((step, loss_val))

        if step % log_every == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            log.info(f"  step {step:>4d}/{n_steps}  loss={loss_val:.6f}  "
                     f"lr={get_lr(step):.2e}  elapsed={elapsed:.1f}s")

    # Delta check
    max_delta = 0.0
    max_delta_name = "none"
    for k, v in tree_flatten(model.trainable_parameters()):
        if k in init_snapshot:
            delta = float(np.max(np.abs(np.array(v.astype(mx.float32)) - init_snapshot[k])))
            if delta > max_delta:
                max_delta = delta
                max_delta_name = k
    log.info(f"Polish done in {time.time() - t0:.1f}s  max_param_delta={max_delta:.6f} ({max_delta_name})")
    return history


# ============================================================================
# Eval + save
# ============================================================================

def quick_eval(model, hparams, n_seqs=32, seq_len=1024):
    """Quick CE evaluation on val tokens. NB: returns CE in nats, NOT BPB."""
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    avg = _quick_ce(model, val_tokens, n_seqs, seq_len)
    bpt = avg / math.log(2)  # NB: bits-per-token, not BPB
    log.info(f"  val_loss={avg:.6f} ({bpt:.4f} bits/tok, {n_seqs} seqs)")
    return avg


def quantize_and_eval(model, cat_bits, hparams, n_seqs=32, seq_len=1024):
    """Quantize→dequantize roundtrip, then eval. Returns (val_loss, quant_obj)."""
    flat = {k: v for k, v in tree_flatten(model.state)}
    quant_obj, stats = quantize_state_dict_int8(flat, cat_bits=cat_bits)
    quant_flat = dequantize_state_dict_int8(quant_obj)
    model.update(tree_unflatten(list(quant_flat.items())))
    mx.eval(model.parameters())
    val_loss = quick_eval(model, hparams, n_seqs, seq_len)
    return val_loss, quant_obj


# Re-export for downstream scripts (proper_bpb_eval.py, correction_ce.py)
def load_train_tokens(hparams, max_tokens=1_000_000):
    return _load_train_tokens(hparams, max_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-seqs", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save", action="store_true",
                        help="Save quantized artifact (pickle+zstd)")
    parser.add_argument("--save-float", type=str, default=None,
                        help="Save polished float weights as .npz (for downstream pipeline)")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--n-eval-seqs", type=int, default=32)
    parser.add_argument("--filter", type=str, default=None,
                        help="Regex to filter which weight matrices to polish. "
                             "e.g. 'mlp\\.proj' for MLP output projections only.")
    args_cli = parser.parse_args()

    hparams = Hyperparameters()

    # Logging
    global log
    log = logging.getLogger("float_polish")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if args_cli.log_file is None:
        args_cli.log_file = f"logs/gradient_polish_int{hparams.quant_attn_bits}_{args_cli.steps}s.txt"
    os.makedirs(os.path.dirname(args_cli.log_file), exist_ok=True)
    fh = logging.FileHandler(args_cli.log_file, mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.info(f"Logging to {args_cli.log_file}")

    per_layer = None
    if hparams.mlp_mult_per_layer:
        per_layer = [float(x) for x in hparams.mlp_mult_per_layer.split(",")]

    def build_model():
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

    cat_bits = {"attn": hparams.quant_attn_bits, "mlp": hparams.quant_mlp_bits}
    quant_bits_str = os.environ.get("QUANT_BITS", "")
    if quant_bits_str:
        cat_bits = {}
        for part in quant_bits_str.split(","):
            k, v = part.strip().rsplit(":", 1)
            cat_bits[k.strip()] = int(v.strip())

    # Load float checkpoint
    log.info(f"Loading checkpoint: {args_cli.checkpoint}")
    log.info(f"Config: {hparams.num_layers}L/{hparams.model_dim}d, MLP {hparams.mlp_mult}x, "
             f"act={hparams.mlp_act}, quant=a{hparams.quant_attn_bits}m{hparams.quant_mlp_bits}")

    model = build_model()
    flat = dict(mx.load(args_cli.checkpoint))
    model.update(tree_unflatten(list(flat.items())))
    mx.eval(model.parameters())

    # Build param→bits map
    param_bits = build_param_bits_map(model, cat_bits, filter_pattern=args_cli.filter)
    log.info(f"Fake-quant params: {len(param_bits)} weight matrices")
    total_polished = sum(
        int(dict(tree_flatten(model.parameters()))[k].size) for k in param_bits
    )
    log.info(f"Total polished params: {total_polished:,}")

    # Baseline: quantize→eval
    log.info("\n=== Baseline (quantize→dequantize→eval) ===")
    # Work on a copy for baseline eval
    baseline_model = build_model()
    baseline_model.update(tree_unflatten(list(flat.items())))
    mx.eval(baseline_model.parameters())
    baseline_loss, _ = quantize_and_eval(baseline_model, cat_bits, hparams,
                                         args_cli.n_eval_seqs, args_cli.seq_len)
    del baseline_model

    # Pre-quant reference
    log.info("\n=== Pre-quant (float model) ===")
    prequant_loss = quick_eval(model, hparams, args_cli.n_eval_seqs, args_cli.seq_len)
    quant_gap = baseline_loss - prequant_loss
    log.info(f"Quant gap: {quant_gap:.6f} CE ({quant_gap / math.log(2):.6f} bits)")

    if args_cli.eval_only:
        return

    # Load train data
    max_tokens = min(args_cli.batch_seqs * args_cli.seq_len * (args_cli.steps + 10), 2_000_000)
    train_tokens = load_train_tokens(hparams, max_tokens=max_tokens)
    log.info(f"Train tokens: {len(train_tokens):,}")

    # Gradient polish (on float weights with fake quant)
    log.info(f"\n=== Gradient polish: {args_cli.steps} steps, lr={args_cli.lr}, "
             f"batch={args_cli.batch_seqs}×{args_cli.seq_len} ===")
    history = gradient_polish(
        model, train_tokens, param_bits,
        n_steps=args_cli.steps, lr=args_cli.lr,
        seq_len=args_cli.seq_len, batch_seqs=args_cli.batch_seqs,
    )

    # Save polished float weights if requested
    if args_cli.save_float:
        flat_polished = dict(tree_flatten(model.state))
        mx.savez(args_cli.save_float, **flat_polished)
        log.info(f"Saved polished float weights: {args_cli.save_float}")

    # Eval after polish: quantize the polished weights and measure
    log.info("\n=== After polish (quantize→dequantize→eval) ===")
    polished_loss, quant_obj = quantize_and_eval(
        model, cat_bits, hparams, args_cli.n_eval_seqs, args_cli.seq_len)

    improvement = baseline_loss - polished_loss
    recovery_pct = 100.0 * improvement / quant_gap if quant_gap > 0 else 0.0
    log.info(f"\nBaseline quant loss: {baseline_loss:.6f}")
    log.info(f"Polished quant loss: {polished_loss:.6f}")
    log.info(f"Improvement: {improvement:.6f} CE ({improvement / math.log(2):.6f} bits)")
    log.info(f"Quant gap recovery: {recovery_pct:.1f}%")

    if args_cli.save:
        quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
        if _COMPRESSOR == "zstd":
            blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
        else:
            blob = zlib.compress(quant_raw, level=9)
        stem = Path(args_cli.checkpoint).stem
        qbits = f"a{hparams.quant_attn_bits}m{hparams.quant_mlp_bits}"
        out = Path("logs") / f"{stem}_polished.{qbits}.pt{_COMPRESSOR[0]}"
        out.write_bytes(blob)
        log.info(f"Saved: {out} ({len(blob):,} bytes)")


if __name__ == "__main__":
    main()
