#!/usr/bin/env python3
"""
build_artifact.py — End-to-end artifact builder: quantize + train corrections + bundle + compress.

Produces a single compressed artifact containing quantized model weights + correction nets.
Measures the final artifact size and validates it fits in the 16MB budget.

Usage:
    NUM_LAYERS=11 MLP_MULT=4.5 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \
    QUANT_BITS="attn:4,mlp:5,mlp.1:4,mlp.6:4,mlp.9:4,mlp.10:4" \
    .venv/bin/python3 scripts/build_artifact.py logs/warmdown_11L_45x_best.npz \
        --correction-layers 0,5,9 --compressor brotli

    # Without correction:
    .venv/bin/python3 scripts/build_artifact.py logs/checkpoint.npz --no-correction

    # zstd instead of brotli:
    .venv/bin/python3 scripts/build_artifact.py logs/checkpoint.npz --compressor zstd
"""
from __future__ import annotations

import argparse
import glob
import io
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger("build_artifact")


def main():
    parser = argparse.ArgumentParser(description="Build compressed submission artifact")
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument("--correction-layers", type=str, default="0,5,9",
                        help="Comma-separated layer indices for PTQ correction")
    parser.add_argument("--no-correction", action="store_true",
                        help="Skip PTQ correction, just quantize and compress")
    parser.add_argument("--hidden-size", type=int, default=0,
                        help="Correction net hidden size (0=linear)")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--n-train-seqs", type=int, default=64)
    parser.add_argument("--n-eval-seqs", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-corrections", type=str, default=None,
                        help="Save trained correction weights to this .npz path")
    parser.add_argument("--load-corrections", type=str, default=None,
                        help="Load pre-trained correction weights instead of training")
    parser.add_argument("--compressor", choices=["brotli", "zstd"], default="brotli")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: auto-generated)")
    parser.add_argument("--budget", type=int, default=16_000_000,
                        help="Artifact size budget in bytes")
    args = parser.parse_args()

    # Logging
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from train_gpt_mlx import (
        GPT, COMPUTE_DTYPE, Hyperparameters, load_validation_tokens,
        quantize_state_dict_int8, dequantize_state_dict_int8, load_data_shard, rms_norm,
    )

    hparams = Hyperparameters()
    per_layer = None
    if hparams.mlp_mult_per_layer:
        per_layer = [int(x) for x in hparams.mlp_mult_per_layer.split(",")]

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

    # Parse quant config
    quant_bits_str = os.environ.get("QUANT_BITS", "")
    if quant_bits_str:
        cat_bits = {}
        for part in quant_bits_str.split(","):
            k, v = part.strip().rsplit(":", 1)
            cat_bits[k.strip()] = int(v.strip())
    else:
        cat_bits = {"attn": hparams.quant_attn_bits, "mlp": hparams.quant_mlp_bits}

    correction_layers = [int(x) for x in args.correction_layers.split(",")] if not args.no_correction else []

    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"Config: {hparams.num_layers}L/{hparams.model_dim}d, MLP {hparams.mlp_mult}x, act={hparams.mlp_act}")
    log.info(f"Quant: {cat_bits}")
    log.info(f"Correction: {'disabled' if args.no_correction else f'layers={correction_layers} hidden={args.hidden_size}'}")
    log.info(f"Compressor: {args.compressor}")

    # Load and quantize
    flat = dict(mx.load(args.checkpoint))
    log.info(f"Loaded {sum(v.size for v in flat.values()):,} params")

    log.info("Quantizing...")
    qobj, _ = quantize_state_dict_int8(flat, cat_bits=cat_bits)

    # Train or load corrections
    correction_arrays = {}
    if correction_layers:
        from scripts.ptq_correction import CorrectionNet, train_corrections, eval_corrected_bpb

        if args.load_corrections:
            # Load pre-trained corrections
            log.info(f"Loading corrections from {args.load_corrections}")
            saved = dict(np.load(args.load_corrections))
            for k, v in saved.items():
                correction_arrays[k] = v
            log.info(f"Loaded {len(saved)} correction arrays")
        else:
            # Train corrections from scratch
            model_float = build_model()
            model_float.update(tree_unflatten(list(flat.items())))
            mx.eval(model_float.parameters())

            model_quant = build_model()
            qflat = dequantize_state_dict_int8(qobj)
            model_quant.update(tree_unflatten(list(qflat.items())))
            mx.eval(model_quant.parameters())

            files = sorted(glob.glob(hparams.train_files))
            tokens_list = []
            total = 0
            max_tokens = args.n_train_seqs * args.seq_len + 1024
            for f in files:
                shard = load_data_shard(Path(f))
                tokens_list.append(shard)
                total += len(shard)
                if total >= max_tokens:
                    break
            train_tokens = np.concatenate(tokens_list)[:max_tokens]
            val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)

            log.info(f"Training corrections ({args.n_epochs} epochs, {args.n_train_seqs} seqs)...")
            corrections = train_corrections(
                model_float, model_quant, train_tokens, val_tokens, hparams,
                correction_layers, hidden_size=args.hidden_size, n_epochs=args.n_epochs,
                lr=args.lr, seq_len=args.seq_len, n_train_seqs=args.n_train_seqs,
                n_eval_seqs=args.n_eval_seqs,
            )

            # Eval corrected model
            log.info("Evaluating corrected model...")
            eval_corrected_bpb(model_quant, corrections, correction_layers, model_float,
                               hparams, val_tokens, n_seqs=args.n_eval_seqs, seq_len=args.seq_len)

            # Serialize correction weights (quantized to int8 + fp16 for compression)
            for layer_idx, net in corrections.items():
                for name, param in tree_flatten(net.parameters()):
                    arr = np.array(param).astype(np.float32)
                    if arr.ndim == 2:  # weight matrix → int8 + per-row scale
                        row_max = np.abs(arr).max(axis=1)
                        scale = np.maximum(row_max / 127.0, 1e-12).astype(np.float16)
                        q = np.clip(np.round(arr / scale.astype(np.float32)[:, None]), -128, 127).astype(np.int8)
                        correction_arrays[f"correction.{layer_idx}.{name}.q"] = q
                        correction_arrays[f"correction.{layer_idx}.{name}.s"] = scale
                    else:  # bias → fp16
                        correction_arrays[f"correction.{layer_idx}.{name}"] = arr.astype(np.float16)
            correction_arrays["__correction_layers__"] = np.array(correction_layers)

            # Save corrections if requested
            if args.save_corrections:
                np.savez(args.save_corrections, **correction_arrays)
                log.info(f"Saved corrections to {args.save_corrections}")

        total_corr_params = sum(v.size for v in correction_arrays.values() if v.dtype != np.int64)
        log.info(f"Correction params: {total_corr_params:,}")

    # Bundle into single artifact
    artifact = dict(qobj)
    artifact.update(correction_arrays)

    buf = io.BytesIO()
    np.savez(buf, **{k: np.array(v) if hasattr(v, '__array__') else v for k, v in artifact.items()})
    raw = buf.getvalue()

    # Compress
    if args.compressor == "brotli":
        import brotli
        blob = brotli.compress(raw, quality=11)
    else:
        import zstandard
        blob = zstandard.ZstdCompressor(level=22).compress(raw)

    # Output
    if args.output is None:
        stem = Path(args.checkpoint).stem.replace("_best", "").replace("_mlx_model", "")
        args.output = f"logs/{stem}_artifact.{args.compressor[:2]}"

    with open(args.output, "wb") as f:
        f.write(blob)

    fits = len(blob) <= args.budget
    log.info("")
    log.info("=== Artifact Summary ===")
    log.info(f"Raw:        {len(raw)/1e6:.2f} MB")
    log.info(f"Compressed: {len(blob)/1e6:.2f} MB ({args.compressor})")
    log.info(f"Budget:     {args.budget/1e6:.2f} MB")
    log.info(f"Margin:     {(args.budget - len(blob))/1e6:+.2f} MB")
    log.info(f"Fits:       {'YES' if fits else 'NO'}")
    log.info(f"Saved:      {args.output}")


if __name__ == "__main__":
    main()
