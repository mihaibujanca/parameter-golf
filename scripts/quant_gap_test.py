#!/usr/bin/env python3
"""
quant_gap_test.py — Measure quantization gap for different bit-widths and rotation strategies.

Loads a trained checkpoint, quantizes with various configs, and evaluates each
to measure post-quant BPB degradation. No training, eval-only.

Usage:
    .venv/bin/python3 quant_gap_test.py logs/overnight_long_best.npz
    .venv/bin/python3 quant_gap_test.py logs/overnight_long_best.npz --configs int6,int4,int4h
    .venv/bin/python3 quant_gap_test.py logs/overnight_long_best.npz --val-max-tokens 262144  # faster
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Import from the training script
from train_gpt_mlx import (
    GPT,
    COMPUTE_DTYPE,
    Hyperparameters,
    build_sentencepiece_luts,
    dequantize_state_dict_int8,
    eval_val,
    load_validation_tokens,
    quantize_float_array,
    quantize_int6_per_row,
    quantize_state_dict_int8,
    validate_dataset_tokenizer_pair,
    _classify_param,
    _np_float32,
    keep_float_array,
    FP16_KEEP_NAME_PATTERNS,
    CONTROL_TENSOR_NAME_PATTERNS,
    INT8_KEEP_FLOAT_MAX_NUMEL,
    INT8_KEEP_FLOAT_FP32_NAME_PATTERNS,
    MX_DTYPE_FROM_NAME,
)

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
import zlib


# ==============================================================================
# HADAMARD ROTATION
# ==============================================================================

def hadamard_matrix(dim: int) -> np.ndarray:
    """Construct normalized Hadamard matrix for power-of-2 dimensions.

    Properties: H @ H^T = I (orthogonal), H = H^T (symmetric), so H^{-1} = H.
    """
    assert dim > 0 and (dim & (dim - 1)) == 0, f"dim must be power of 2, got {dim}"
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < dim:
        H = np.kron(H, np.array([[1, 1], [1, -1]], dtype=np.float32))
    return H / np.sqrt(dim)


def block_hadamard_rotate(W: np.ndarray, block_size: int = 0) -> np.ndarray:
    """Apply block-diagonal Hadamard rotation along columns of W.

    If block_size=0, uses the largest power-of-2 that divides cols.
    For dim=512 with MLP 3x, cols can be 512 (power of 2) or 1536 (3x512).
    """
    rows, cols = W.shape
    if block_size <= 0:
        # Largest power-of-2 factor of cols (isolate lowest set bit)
        block_size = cols & -cols
    assert cols % block_size == 0, f"cols={cols} not divisible by block_size={block_size}"
    H = hadamard_matrix(block_size)
    out = np.empty_like(W)
    for i in range(0, cols, block_size):
        out[:, i:i + block_size] = W[:, i:i + block_size] @ H
    return out


# ==============================================================================
# INT4 QUANTIZATION
# ==============================================================================

def quantize_int4_per_row(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to int4 range [-8, 7] stored as int8. Per-row fp16 scales."""
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        row_max = np.abs(f32).max(axis=1)
        scale = np.maximum(row_max / 7.0, 1e-12).astype(np.float16)
        q = np.clip(np.round(f32 / scale.astype(np.float32)[:, None]), -8, 7).astype(np.int8)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale)
    amax = float(np.abs(f32).max()) if f32.size else 0.0
    scale = np.array(max(amax / 7.0, 1e-12), dtype=np.float16)
    q = np.clip(np.round(f32 / float(scale)), -8, 7).astype(np.int8)
    return np.ascontiguousarray(q), scale


def quantize_int5_per_row(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to int5 range [-16, 15] stored as int8. Per-row fp16 scales."""
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        row_max = np.abs(f32).max(axis=1)
        scale = np.maximum(row_max / 15.0, 1e-12).astype(np.float16)
        q = np.clip(np.round(f32 / scale.astype(np.float32)[:, None]), -16, 15).astype(np.int8)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale)
    amax = float(np.abs(f32).max()) if f32.size else 0.0
    scale = np.array(max(amax / 15.0, 1e-12), dtype=np.float16)
    q = np.clip(np.round(f32 / float(scale)), -16, 15).astype(np.int8)
    return np.ascontiguousarray(q), scale


# ==============================================================================
# CONFIGURABLE QUANTIZATION WITH ROTATION
# ==============================================================================

QUANT_FN = {
    4: quantize_int4_per_row,
    5: quantize_int5_per_row,
    6: quantize_int6_per_row,
    8: quantize_float_array,
}


def quantize_configurable(
    flat_state: dict[str, mx.array],
    attn_bits: int = 6,
    mlp_bits: int = 6,
    hadamard: bool = False,
    per_layer_bits: dict[str, int] | None = None,
) -> tuple[dict[str, object], dict[str, int]]:
    """Quantize with configurable bit-widths and optional Hadamard rotation.

    per_layer_bits: optional overrides like {"mlp.0": 6, "attn.3": 5}.
    Falls back to attn_bits/mlp_bits for layers not in the dict.

    Returns (quant_obj, stats) in the same format as quantize_state_dict_int8().
    """
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)

        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        # Embeddings kept in fp16
        if any(pattern in name for pattern in FP16_KEEP_NAME_PATTERNS):
            kept = np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=np.float16))
            passthrough[name] = kept
            passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        # Small float tensors kept directly
        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            if any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
                passthrough[name] = np.ascontiguousarray(_np_float32(arr))
            else:
                kept = keep_float_array(name, arr, passthrough_orig_dtypes)
                passthrough[name] = kept
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        stats["num_float_tensors"] += 1
        cat = _classify_param(name)

        # Determine bit-width for this param (per-layer override, then category)
        bits = (per_layer_bits or {}).get(cat)
        if bits is None:
            base_cat = cat.split(".")[0]
            if base_cat == "attn":
                bits = attn_bits
            elif base_cat == "mlp":
                bits = mlp_bits
            else:
                bits = 8

        quant_fn = QUANT_FN[bits]

        # Hadamard rotation for low-bit quantization
        rotated = False
        if hadamard and bits <= 5 and arr.ndim == 2:
            f32 = _np_float32(arr)
            f32_rotated = block_hadamard_rotate(f32)
            q, s = quant_fn(mx.array(f32_rotated))
            rotated = True
        else:
            q, s = quant_fn(arr)

        scheme = f"int{bits}_per_row"
        if rotated:
            scheme = f"int{bits}_hadamard_per_row"

        if s.ndim > 0:
            qmeta[name] = {"scheme": scheme, "axis": 0}
        else:
            qmeta[name] = {"scheme": scheme}

        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_configurable(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    """Dequantize with Hadamard rotation reversal for rotated weights."""
    out: dict[str, mx.array] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})

    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        meta = qmeta.get(name, {})

        if meta.get("scheme", "").endswith("per_row") or scale.ndim > 0:
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)

        # Reverse Hadamard rotation: H is self-inverse (H @ H = I)
        if "hadamard" in meta.get("scheme", "") and out_arr.ndim == 2:
            out_arr = block_hadamard_rotate(out_arr)

        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])

    for name, arr in quant_obj["passthrough"].items():
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)

    return out


# ==============================================================================
# ARTIFACT SIZE ESTIMATION
# ==============================================================================

def estimate_compressed_size(quant_obj: dict, stats: dict) -> int:
    """Estimate zstd-22 compressed artifact size by actually compressing."""
    raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    if _COMPRESSOR == "zstd":
        blob = zstandard.ZstdCompressor(level=22).compress(raw)
    else:
        blob = zlib.compress(raw, level=9)
    return len(blob)


# ==============================================================================
# MAIN
# ==============================================================================

CONFIGS = {
    "fp32":  {"label": "No quantization (fp32 baseline)", "attn_bits": None, "mlp_bits": None, "hadamard": False},
    "int8":  {"label": "int8 per-row",                    "attn_bits": 8,    "mlp_bits": 8,    "hadamard": False},
    "int6":  {"label": "int6 per-row (current SOTA)",     "attn_bits": 6,    "mlp_bits": 6,    "hadamard": False},
    "int5":  {"label": "int5 per-row",                    "attn_bits": 5,    "mlp_bits": 5,    "hadamard": False},
    "int4":  {"label": "int4 per-row (no rotation)",      "attn_bits": 4,    "mlp_bits": 4,    "hadamard": False},
    "int3":  {"label": "int3 per-row",                    "attn_bits": 3,    "mlp_bits": 3,    "hadamard": False},
    "int2":  {"label": "int2 per-row",                    "attn_bits": 2,    "mlp_bits": 2,    "hadamard": False},
    "int4h": {"label": "int4 per-row + Hadamard rotation","attn_bits": 4,    "mlp_bits": 4,    "hadamard": True},
    "int5h": {"label": "int5 per-row + Hadamard rotation","attn_bits": 5,    "mlp_bits": 5,    "hadamard": True},
    # Mixed configs
    "i6a4m":  {"label": "int6-attn / int4-MLP",           "attn_bits": 6,    "mlp_bits": 4,    "hadamard": False},
    "i6a4mh": {"label": "int6-attn / int4-MLP + Hadamard","attn_bits": 6,    "mlp_bits": 4,    "hadamard": True},
    "i6a5m":  {"label": "int6-attn / int5-MLP",           "attn_bits": 6,    "mlp_bits": 5,    "hadamard": False},
    # Per-layer mixed precision: single MLP layer upgraded, rest int4
    **{f"i6a4m_L{i}m{b}": {"label": f"a6/m4 + L{i} MLP int{b}", "attn_bits": 6, "mlp_bits": 4, "hadamard": False,
                             "per_layer_bits": {f"mlp.{i}": b}}
       for b in [5, 6] for i in range(20)},
}


def main():
    parser = argparse.ArgumentParser(description="Measure quantization gap for different configs")
    parser.add_argument("checkpoint", type=str, help="Path to .npz checkpoint")
    parser.add_argument("--configs", type=str, default="fp32,int6,int4,int4h",
                        help="Comma-separated config names to test (default: fp32,int6,int4,int4h)")
    parser.add_argument("--val-max-tokens", type=int, default=0,
                        help="Cap val tokens for faster eval (0=full split)")
    parser.add_argument("--val-batch-size", type=int, default=65536,
                        help="Val batch size")
    parser.add_argument("--list-configs", action="store_true", help="List available configs and exit")
    args_cli = parser.parse_args()

    if args_cli.list_configs:
        for k, v in CONFIGS.items():
            print(f"  {k:10s}  {v['label']}")
        return

    config_names = [c.strip() for c in args_cli.configs.split(",")]
    for c in config_names:
        if c not in CONFIGS:
            print(f"Unknown config: {c}. Available: {', '.join(CONFIGS.keys())}")
            sys.exit(1)

    # Use default hyperparameters for model construction + eval
    hparams = Hyperparameters()
    if args_cli.val_max_tokens > 0:
        hparams.val_max_tokens = args_cli.val_max_tokens
    hparams.val_batch_size = args_cli.val_batch_size

    # Load tokenizer + val data
    sp = spm.SentencePieceProcessor(model_file=hparams.tokenizer_path)
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, hparams.vocab_size
    )

    # Build model and load checkpoint
    per_layer = [float(x) for x in hparams.mlp_mult_per_layer else None
    model = GPT(
        vocab_size=hparams.vocab_size,
        num_layers=hparams.num_layers,
        dim=hparams.model_dim,
        num_heads=hparams.num_heads,
        num_kv_heads=hparams.num_kv_heads,
        mlp_mult=hparams.mlp_mult,
        logit_chunk_tokens=hparams.logit_chunk_tokens,
        logit_softcap=hparams.logit_softcap,
        rope_base=hparams.rope_base,
        tied_embed_init_std=hparams.tied_embed_init_std,
        qk_gain_init=hparams.qk_gain_init,
        mlp_act=hparams.mlp_act,
        mlp_mult_per_layer=per_layer,
        bigram_vocab_size=hparams.bigram_vocab_size,
        bigram_dim=hparams.bigram_dim,
        logit_temp=hparams.logit_temp,
        lrelu_slope=hparams.lrelu_slope,
        xsa_last_n=hparams.xsa_last_n,
        rope_dims=hparams.rope_dims,
    )

    ckpt_path = Path(args_cli.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt_state = dict(mx.load(str(ckpt_path)))
    model.update(tree_unflatten(list(ckpt_state.items())))
    mx.eval(model.state)

    # Save original state for re-loading between configs
    original_state = {k: v for k, v in tree_flatten(model.state)}

    # Compile eval function
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)

    n_val_tokens = val_tokens.size - 1
    if hparams.val_max_tokens > 0:
        n_val_tokens = min(n_val_tokens, hparams.val_max_tokens)
    print(f"Model: {hparams.num_layers}L/{hparams.model_dim}d, "
          f"params: {sum(p.size for _, p in tree_flatten(model.state)):,}")
    print(f"Evaluating on {n_val_tokens:,} val tokens")
    print(f"Configs to test: {config_names}")
    print()

    # Run each config
    results = []
    for cname in config_names:
        cfg = CONFIGS[cname]
        print(f"{'='*60}")
        print(f"Config: {cname} — {cfg['label']}")

        # Restore original weights
        model.update(tree_unflatten(list(original_state.items())))
        mx.eval(model.state)

        if cfg["attn_bits"] is None:
            # fp32 baseline — no quantization
            artifact_bytes = 0
        else:
            flat_state = {k: v for k, v in tree_flatten(model.state)}
            quant_obj, quant_stats = quantize_configurable(
                flat_state,
                attn_bits=cfg["attn_bits"],
                mlp_bits=cfg["mlp_bits"],
                hadamard=cfg["hadamard"],
                per_layer_bits=cfg.get("per_layer_bits"),
            )

            # Estimate compressed artifact size
            t0 = time.perf_counter()
            artifact_bytes = estimate_compressed_size(quant_obj, quant_stats)
            compress_ms = 1000 * (time.perf_counter() - t0)
            artifact_mb = artifact_bytes / (1024 * 1024)
            print(f"  Artifact: {artifact_mb:.2f} MB  (compression took {compress_ms:.0f}ms)")

            # Dequantize and load back
            quant_flat = dequantize_configurable(quant_obj)
            model.update(tree_unflatten(list(quant_flat.items())))
            mx.eval(model.state)

        # Recompile after weight update
        compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)

        t0 = time.perf_counter()
        val_loss, val_bpb = eval_val(
            hparams, compiled_loss, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        eval_ms = 1000 * (time.perf_counter() - t0)

        if cfg["attn_bits"] is not None:
            score = val_bpb * artifact_mb
            print(f"  val_bpb: {val_bpb:.4f}  val_loss: {val_loss:.4f}  "
                  f"score: {score:.4f}  eval: {eval_ms:.0f}ms")
        else:
            artifact_mb = 0.0
            score = 0.0
            print(f"  val_bpb: {val_bpb:.4f}  val_loss: {val_loss:.4f}  "
                  f"(fp32 reference)  eval: {eval_ms:.0f}ms")

        results.append({
            "config": cname,
            "label": cfg["label"],
            "val_bpb": val_bpb,
            "val_loss": val_loss,
            "artifact_mb": artifact_mb,
            "score": score,
        })
        print()

    # Summary table
    fp32_bpb = next((r["val_bpb"] for r in results if r["config"] == "fp32"), None)
    print(f"{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<12s} {'BPB':>8s} {'Gap':>8s} {'MB':>8s} {'Score':>8s}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        gap = f"+{r['val_bpb'] - fp32_bpb:.4f}" if fp32_bpb is not None and r["config"] != "fp32" else "—"
        mb = f"{r['artifact_mb']:.2f}" if r["artifact_mb"] > 0 else "—"
        score = f"{r['score']:.4f}" if r["score"] > 0 else "—"
        print(f"{r['config']:<12s} {r['val_bpb']:>8.4f} {gap:>8s} {mb:>8s} {score:>8s}")


if __name__ == "__main__":
    main()
