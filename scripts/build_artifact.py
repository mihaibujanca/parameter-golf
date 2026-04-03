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
    parser.add_argument("--no-permute", action="store_true",
                        help="Skip weight permutation (for comparison)")
    parser.add_argument("--compressor", choices=["brotli", "zstd", "zpaq"], default="zpaq")
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
    import sentencepiece as spm
    from train_gpt_mlx import (
        COMPUTE_DTYPE, Hyperparameters, load_validation_tokens,
        quantize_state_dict_int8, dequantize_state_dict_int8, load_data_shard, rms_norm,
        build_sentencepiece_luts, eval_val,
    )
    from scripts.eval_commons import build_model as _build_model

    hparams = Hyperparameters()

    def build_model():
        return _build_model(hparams)

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

    # Load val tokens and byte LUTs for eval_val
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    sp = spm.SentencePieceProcessor(model_file=hparams.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, hparams.vocab_size)

    def measure_bpb(model, label):
        compiled_loss = mx.compile(
            lambda x, y: model.loss(x, y),
            inputs=model.state, outputs=model.state,
        )
        val_loss, val_bpb = eval_val(
            hparams, compiled_loss, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        log.info(f"  {label}: val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")
        return val_bpb

    # Step 1: Float baseline
    log.info("Measuring float baseline...")
    model_float = build_model()
    model_float.update(tree_unflatten(list(flat.items())))
    mx.eval(model_float.parameters())
    float_bpb = measure_bpb(model_float, "float")

    # Step 2: Quantize
    log.info("Quantizing...")
    qobj, _ = quantize_state_dict_int8(flat, cat_bits=cat_bits)

    # Step 3: Measure quant roundtrip BPB
    log.info("Measuring quant roundtrip...")
    qflat = dequantize_state_dict_int8(qobj)
    model_quant = build_model()
    model_quant.update(tree_unflatten(list(qflat.items())))
    mx.eval(model_quant.parameters())
    quant_bpb = measure_bpb(model_quant, "quant")
    log.info(f"  quant_gap: {quant_bpb - float_bpb:+.4f}")
    del model_quant, qflat

    # Weight permutation for better compression (lossless)
    if not args.no_permute:
        from scripts.weight_permutation import permute_mlp_qobj
        log.info("Permuting MLP weights for compression...")
        permute_mlp_qobj(qobj)

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
        log.info(f"Correction params (f32): {total_corr_params:,}")

        # Quantize correction weights to int8 for compression (skip if already int8)
        already_quantized = any(k.endswith('.q') and v.dtype == np.int8
                                for k, v in correction_arrays.items())
        if already_quantized:
            log.info("Corrections already int8-quantized, skipping re-quantization")
        else:
            corr_quantized = {}
            for k, v in correction_arrays.items():
                if k.startswith("__"):
                    corr_quantized[k] = v
                    continue
                arr = v.astype(np.float32) if v.dtype != np.float32 else v
                if arr.ndim == 2:  # weight matrix → int8 + per-row scale
                    row_max = np.abs(arr).max(axis=1)
                    scale = np.maximum(row_max / 127.0, 1e-12).astype(np.float16)
                    q = np.clip(np.round(arr / scale.astype(np.float32)[:, None]), -128, 127).astype(np.int8)
                    corr_quantized[k + ".q"] = q
                    corr_quantized[k + ".s"] = scale
                else:  # bias → fp16
                    corr_quantized[k] = arr.astype(np.float16)
            correction_arrays = corr_quantized
            log.info("Correction quantized to int8")

    # Bundle into single artifact
    artifact = dict(qobj)
    artifact.update(correction_arrays)

    buf = io.BytesIO()
    np.savez(buf, **{k: np.array(v) if hasattr(v, '__array__') else v for k, v in artifact.items()})
    raw = buf.getvalue()

    # Compress
    if args.compressor == "zpaq":
        import subprocess, tempfile
        raw_path = args.output or f"logs/{Path(args.checkpoint).stem}_artifact"
        raw_path = raw_path.rstrip(".zpaq") + ".raw"
        with open(raw_path, "wb") as f:
            f.write(raw)
        zpaq_path = raw_path.replace(".raw", ".zpaq")
        # Remove stale zpaq archive if it exists (zpaq appends by default)
        if os.path.exists(zpaq_path):
            os.remove(zpaq_path)
        subprocess.run(["/opt/homebrew/bin/zpaq", "a", zpaq_path, raw_path, "-method", "5"],
                       check=True)
        with open(zpaq_path, "rb") as f:
            blob = f.read()
        os.remove(raw_path)
    elif args.compressor == "brotli":
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

    artifact_mb = len(blob) / 1e6
    fits = len(blob) <= args.budget
    log.info("")
    log.info("=== Artifact Summary ===")
    log.info(f"Raw:        {len(raw)/1e6:.2f} MB")
    log.info(f"Compressed: {artifact_mb:.2f} MB ({args.compressor})")
    log.info(f"Budget:     {args.budget/1e6:.2f} MB")
    log.info(f"Margin:     {(args.budget - len(blob))/1e6:+.2f} MB")
    log.info(f"Fits:       {'YES' if fits else 'NO'}")
    log.info(f"Saved:      {args.output}")
    log.info("")
    log.info("=== BPB Tracking ===")
    log.info(f"Float BPB:  {float_bpb:.4f}")
    log.info(f"Quant BPB:  {quant_bpb:.4f}  (gap: {quant_bpb - float_bpb:+.4f})")
    log.info(f"Score:      {quant_bpb * artifact_mb:.4f}  (BPB x MB)")


if __name__ == "__main__":
    main()
