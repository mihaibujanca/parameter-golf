#!/usr/bin/env python3
"""Proper BPB evaluation of polished models using sentencepiece byte counting."""
from __future__ import annotations
import math, os, sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import sentencepiece as spm

from train_gpt_mlx import (
    GPT, Hyperparameters, load_validation_tokens,
    quantize_state_dict_int8, dequantize_state_dict_int8,
    build_sentencepiece_luts, _classify_param,
    FP16_KEEP_NAME_PATTERNS, CONTROL_TENSOR_NAME_PATTERNS, INT8_KEEP_FLOAT_MAX_NUMEL,
)
from scripts.float_polish import (
    gradient_polish, build_param_bits_map, load_train_tokens, _BITS_TO_QMAX,
)


def proper_bpb_eval(model, hparams, val_tokens, sp_luts, max_tokens=1_048_576):
    """Non-sliding-window BPB eval matching train_gpt_mlx.py methodology."""
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = sp_luts
    seq_len = hparams.train_seq_len
    batch_seqs = max(65536 // seq_len, 1)

    total_seqs = (val_tokens.size - 1) // seq_len
    if max_tokens > 0:
        total_seqs = min(total_seqs, max_tokens // seq_len)

    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0

    compiled_loss = mx.compile(model.loss)

    for batch_start in range(0, total_seqs, batch_seqs):
        batch_end = min(batch_start + batch_seqs, total_seqs)
        raw_start = batch_start * seq_len
        raw_end = batch_end * seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, seq_len)
        y_np = chunk[1:].reshape(-1, seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        batch_loss = compiled_loss(x, y).astype(mx.float32)
        mx.eval(batch_loss)
        total_loss_sum += float(batch_loss.item()) * chunk_token_count

        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())

    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


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


def load_checkpoint(model, ckpt_path):
    flat = dict(mx.load(ckpt_path))
    model.update(tree_unflatten(list(flat.items())))
    mx.eval(model.parameters())
    return flat


def quantize_dequantize(model, flat_weights, cat_bits):
    """Quantize→dequantize roundtrip on model weights."""
    quant_obj, stats = quantize_state_dict_int8(flat_weights, cat_bits=cat_bits)
    quant_flat = dequantize_state_dict_int8(quant_obj)
    model.update(tree_unflatten(list(quant_flat.items())))
    mx.eval(model.parameters())
    return quant_obj


def parse_quant_bits(bits_str):
    cat_bits = {}
    for part in bits_str.split(","):
        k, v = part.strip().rsplit(":", 1)
        cat_bits[k.strip()] = int(v.strip())
    return cat_bits


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("proper_bpb")

    # Also log to file
    fh = logging.FileHandler("logs/proper_bpb_eval.txt", mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)

    hparams = Hyperparameters()
    ckpt = "logs/warmdown_11L_45x_best.npz"
    max_tokens = 1_048_576

    log.info(f"Model: {hparams.num_layers}L/{hparams.model_dim}d, MLP {hparams.mlp_mult}x")
    log.info(f"Checkpoint: {ckpt}")
    log.info(f"Val tokens: {max_tokens:,}")

    # Load sentencepiece and build LUTs
    sp = spm.SentencePieceProcessor(model_file=hparams.tokenizer_path)
    sp_luts = build_sentencepiece_luts(sp, hparams.vocab_size)

    # Load val tokens
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    log.info(f"Val tokens loaded: {val_tokens.size:,}")

    # ---- 1. Float reference ----
    log.info("\n" + "=" * 60)
    log.info("FLOAT REFERENCE (no quantization)")
    log.info("=" * 60)
    model = build_model(hparams)
    float_weights = load_checkpoint(model, ckpt)
    t0 = time.time()
    loss, bpb = proper_bpb_eval(model, hparams, val_tokens, sp_luts, max_tokens)
    log.info(f"  val_loss={loss:.6f}  val_bpb={bpb:.4f}  ({time.time()-t0:.1f}s)")

    # Config definitions
    BITS_A = "attn.0:5,mlp.0:5,attn.4:5,mlp.4:5,attn.9:5,mlp.9:5,attn.1:4,mlp.1:4,attn.2:4,mlp.2:4,attn.3:4,mlp.3:4,attn.5:3,mlp.5:3,attn.6:3,mlp.6:3,attn.7:3,mlp.7:3,attn.8:3,mlp.8:3,attn.10:3,mlp.10:3,attn:4,mlp:4"
    BITS_B = "attn.0:5,mlp.0:5,attn.4:4,mlp.4:4,attn.9:4,mlp.9:4,attn.1:3,mlp.1:3,attn.2:3,mlp.2:3,attn.3:3,mlp.3:3,attn.5:3,mlp.5:3,attn.6:3,mlp.6:3,attn.7:3,mlp.7:3,attn.8:3,mlp.8:3,attn.10:2,mlp.10:2,attn:3,mlp:3"

    configs = [
        ("uniform int6", {"attn": 6, "mlp": 6}),
        ("uniform int4", {"attn": 4, "mlp": 4}),
        ("uniform int3", {"attn": 3, "mlp": 3}),
        ("Config A", parse_quant_bits(BITS_A)),
        ("Config B", parse_quant_bits(BITS_B)),
    ]

    # ---- 2. Unpolished baselines for each config ----
    log.info("\n" + "=" * 60)
    log.info("UNPOLISHED BASELINES (quantize → dequantize → eval)")
    log.info("=" * 60)
    for name, cat_bits in configs:
        model = build_model(hparams)
        load_checkpoint(model, ckpt)
        flat = {k: v for k, v in tree_flatten(model.state)}
        quantize_dequantize(model, flat, cat_bits)
        t0 = time.time()
        loss, bpb = proper_bpb_eval(model, hparams, val_tokens, sp_luts, max_tokens)
        log.info(f"  {name:20s}  val_loss={loss:.6f}  val_bpb={bpb:.4f}  ({time.time()-t0:.1f}s)")
        del model

    # ---- 3. Polished configs ----
    polish_configs = [
        ("Config A polished (lr=3e-5, 500s)", BITS_A, 3e-5, 500),
        ("Config B polished (lr=1e-4, 1000s)", BITS_B, 1e-4, 1000),
        ("Config B polished (lr=3e-5, 1000s)", BITS_B, 3e-5, 1000),
    ]

    log.info("\n" + "=" * 60)
    log.info("POLISHED MODELS (polish → quantize → dequantize → eval)")
    log.info("=" * 60)

    train_tokens = load_train_tokens(hparams, max_tokens=2_000_000)

    for name, bits_str, lr, steps in polish_configs:
        log.info(f"\n--- {name} ---")
        cat_bits = parse_quant_bits(bits_str)

        # Fresh model from checkpoint
        model = build_model(hparams)
        load_checkpoint(model, ckpt)

        # Build param→bits map and run polish
        param_bits = build_param_bits_map(model, cat_bits)
        log.info(f"  Polishing {len(param_bits)} matrices, {steps} steps, lr={lr}")
        gradient_polish(model, train_tokens, param_bits,
                       n_steps=steps, lr=lr, seq_len=1024, batch_seqs=4, log_every=100)

        # Quantize→dequantize polished weights
        flat = {k: v for k, v in tree_flatten(model.state)}
        quant_obj = quantize_dequantize(model, flat, cat_bits)

        # Proper BPB eval
        t0 = time.time()
        loss, bpb = proper_bpb_eval(model, hparams, val_tokens, sp_luts, max_tokens)
        log.info(f"  {name:45s}  val_loss={loss:.6f}  val_bpb={bpb:.4f}  ({time.time()-t0:.1f}s)")

        # Measure artifact size
        import pickle, zstandard
        quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
        blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
        artifact_mb = len(blob) / (1024 * 1024)
        score = bpb * artifact_mb
        log.info(f"  artifact={artifact_mb:.2f} MB  score={score:.4f}")
        del model

    log.info("\n" + "=" * 60)
    log.info("DONE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
