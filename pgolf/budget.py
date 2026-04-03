#!/usr/bin/env python3
"""
pgolf/budget.py — Artifact size predictor for Parameter Golf.

Answers the question: "Will my model fit in 16MB?" before you burn GPU time.

Usage
-----
    python -m pgolf.budget                          # default 9L config
    python -m pgolf.budget --layers 11 --wd 0.04   # 11L with high WD
    python -m pgolf.budget --sweep                  # find biggest model that fits
    python -m pgolf.budget --checkpoint path.npz    # exact size from trained model

Theory
------
Artifact size comes from three parts:
  1. Quantized weights: int6 (or int5) values stored in int8 containers,
     one fp16 per-row scale per matrix row.
  2. FP16 passthrough tensors: small tensors + tied embedding when fp16 keep.
  3. Overhead: torch serialization, code, metadata (~55 KB).

Compression depends on bit-width and weight magnitude:
  - int8 + zstd-22: ~1.20x (values use all 8 bits, little structure)
  - int6 + zstd-22: ~1.51-1.67x (top 2 bits always zero OR always one)
  - int5 + zstd-22: ~1.88x (top 3 bits all-zero or all-one)
  - fp16 passthrough + zstd-22: ~1.05x (pseudo-random bits)

Higher weight decay pushes weights toward zero, increasing the fraction of
"leading-zero" bytes, which bumps zstd ratio further. Calibrated on:
  - PR #162 (WD=0.02, 9L int6+zstd): 21.8M params → 15.92 MB
  - PR #179 (WD=0.038, 11L int6+zstd): 26.5M params → 15.17 MB
  - PR #180 (WD=0.04, 10L int5-MLP/int6-attn+zstd): 24.7M params → 15.21 MB
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Optional

MB = 1024 * 1024
BUDGET_BYTES = 16 * MB  # 16 MB hard limit


# ---------------------------------------------------------------------------
# Compression ratio table: (quant_bits, weight_decay, compressor) → ratio
# Empirical fits from PR data and local M4/Mac Studio measurements.
# The ratio applies to the *raw payload* (int8-container bytes + fp16 scale
# bytes). Higher WD → smaller weights → more zero high bits → better
# compression.
#
# Compressor options: "zstd" (zstd-22), "brotli" (brotli-11), "zpaq" (method 5).
# Weight permutation sorts MLP neurons by quant scale before compression,
# improving MLP compressibility by ~3% (lossless).
#
# Calibration data (2026-04-02, int4+permute+brotli across 8 models + 640d):
#   13L/3x 31.8M: brotli 11.47, zpaq 11.23, perm saves 0.63 MB
#   11L/4x 32.9M: brotli 11.02, zpaq 10.78, perm saves 0.83 MB
#   11L/5x 38.6M: brotli 12.58, zpaq 12.29, perm saves 1.15 MB
#   11L/4.5x 35.7M: brotli 11.97, zpaq 11.69, perm saves 0.99 MB
#   13L/3x/640d 49.2M: brotli 18.96, zpaq 18.56, perm saves 0.54 MB
# zpaq saves ~2-3% over brotli consistently.
# ---------------------------------------------------------------------------

# int6 compression: PR #162 (WD=0.02) → 1.55x; PR #179 (WD=0.038) → 1.80x
_INT6_ZSTD_BASE = 1.43   # WD ≈ 0 (random weights)
_INT6_ZSTD_WD_SLOPE = 10.0  # additional ratio per unit WD

# int5 compression: PR #180 (WD=0.04) → 1.88x (H100); M4 80k → 2.08x, M4 5k → 2.01x
# Revised to match M4 actuals; H100 will compress slightly better.
_INT5_ZSTD_BASE = 1.70
_INT5_ZSTD_WD_SLOPE = 9.0

# int8 compression: baseline (WD ≈ 0) → 1.27x (from PR #60 log: payload 3.81x smaller after int8+zlib)
# int8+zlib ~1.27x. int8+zstd-22 is slightly better at ~1.30x.
_INT8_ZSTD_BASE = 1.27
_INT8_ZSTD_WD_SLOPE = 2.0

# int4: Calibrated on 8 M4 models + 640d (WD=0.04) with brotli+permute.
# 512d models: 2.84x-3.13x (avg 3.02x). 640d: 2.64x (~12% worse).
# Conservative base targets 640d. 512d predictions will over-estimate size by ~10%.
_INT4_BROTLI_BASE = 2.10
_INT4_BROTLI_WD_SLOPE = 13.5

# int3: Calibrated on 640d model (WD=0.04) with brotli+permute.
# int3-all → 3.97x brotli. int3a+int4m → 3.00x (mixed, not a pure int3 ratio).
_INT3_BROTLI_BASE = 3.00
_INT3_BROTLI_WD_SLOPE = 24.0

# fp16 passthrough: barely compresses
_FP16_ZSTD_RATIO = 1.05

TORCH_OVERHEAD_BYTES = 55_000  # torch npz pickling overhead, code + submission.json

# Weight permutation: lossless MLP neuron reordering by quant scale.
# Saves ~3% of MLP compressed size (measured 0.54-1.15 MB across models).
_PERMUTE_MLP_BONUS = 1.03

# zpaq-5 saves ~2.5% over brotli-11 consistently.
_ZPAQ_VS_BROTLI = 1.025
# zstd-22 is ~5% worse than brotli-11 on quantized weights.
_ZSTD_VS_BROTLI = 0.95


def _compression_ratio(bits: int, weight_decay: float, compressor: str = "brotli",
                        is_mlp: bool = False, permuted: bool = True) -> float:
    """Empirical compression ratio for quantized weight matrices.

    Base ratios are calibrated on brotli-11 with weight permutation at WD=0.04.
    """
    if bits == 8:
        ratio = _INT8_ZSTD_BASE + _INT8_ZSTD_WD_SLOPE * weight_decay
    elif bits == 6:
        ratio = _INT6_ZSTD_BASE + _INT6_ZSTD_WD_SLOPE * weight_decay
    elif bits == 5:
        ratio = _INT5_ZSTD_BASE + _INT5_ZSTD_WD_SLOPE * weight_decay
    elif bits == 4:
        ratio = _INT4_BROTLI_BASE + _INT4_BROTLI_WD_SLOPE * weight_decay
    elif bits == 3:
        ratio = _INT3_BROTLI_BASE + _INT3_BROTLI_WD_SLOPE * weight_decay
    else:
        raise ValueError(f"Unsupported quant bits: {bits}")
    # Weight permutation bonus (MLP only)
    if is_mlp and permuted:
        ratio *= _PERMUTE_MLP_BONUS
    # Compressor adjustment (base is brotli-11)
    if compressor == "zpaq":
        ratio *= _ZPAQ_VS_BROTLI
    elif compressor == "zstd":
        ratio *= _ZSTD_VS_BROTLI
    return ratio


def _zstd_ratio(bits: int, weight_decay: float) -> float:
    """Empirical zstd-22 compression ratio (legacy, prefer _compression_ratio)."""
    return _compression_ratio(bits, weight_decay, compressor="zstd", is_mlp=False, permuted=False)


# ---------------------------------------------------------------------------
# Model shape helpers
# ---------------------------------------------------------------------------

def _count_params(
    vocab_size: int,
    num_layers: int,
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    mlp_mult: float,
    mlp_act: str = "relu2",
) -> dict[str, int]:
    """Return parameter counts by component, excluding tied lm_head."""
    head_dim = dim // num_heads
    kv_dim = num_kv_heads * head_dim  # total dim for K and V

    # MLP hidden dim depends on activation
    if mlp_act == "swiglu":
        # SwiGLU: hidden = int(dim * mlp_mult * 2/3), plus gate proj
        mlp_hidden = int(dim * mlp_mult * 2 / 3)
        mlp_params_per_layer = dim * mlp_hidden + dim * mlp_hidden + mlp_hidden * dim
        # fc (gate proj), up_proj, down_proj — all same hidden dim in SwiGLU
    else:
        # ReLU²: hidden = dim * mlp_mult
        mlp_hidden = int(dim * mlp_mult)
        mlp_params_per_layer = dim * mlp_hidden + mlp_hidden * dim  # fc + proj

    attn_params_per_layer = (
        dim * dim          # c_q: [dim, dim]
        + dim * kv_dim     # c_k: [kv_dim, dim]  (out_features, in_features in PyTorch)
        + dim * kv_dim     # c_v
        + dim * dim        # c_proj: [dim, dim]
    )
    ln_params_per_layer = 2 * dim  # two LayerNorms per block (weight only, no bias)
    per_layer = attn_params_per_layer + mlp_params_per_layer + ln_params_per_layer

    return {
        "tok_emb": vocab_size * dim,      # tied with lm_head, stored once
        "attn_per_layer": attn_params_per_layer,
        "mlp_per_layer": mlp_params_per_layer,
        "ln_per_layer": ln_params_per_layer,
        "total_matrix_attn": attn_params_per_layer * num_layers,
        "total_matrix_mlp": mlp_params_per_layer * num_layers,
        "total_ln": ln_params_per_layer * num_layers,
        "total": vocab_size * dim + per_layer * num_layers,
        "total_large_matrix": attn_params_per_layer * num_layers + mlp_params_per_layer * num_layers,
    }


def _matrix_shapes(
    num_layers: int,
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    mlp_mult: float,
    mlp_act: str = "relu2",
) -> list[tuple[str, int, int]]:
    """Return (name, rows, cols) for every 2D matrix in the model."""
    head_dim = dim // num_heads
    kv_dim = num_kv_heads * head_dim

    if mlp_act == "swiglu":
        mlp_hidden = int(dim * mlp_mult * 2 / 3)
        mlp_shapes = [
            ("mlp.gate", mlp_hidden, dim),
            ("mlp.fc", mlp_hidden, dim),
            ("mlp.proj", dim, mlp_hidden),
        ]
    else:
        mlp_hidden = int(dim * mlp_mult)
        mlp_shapes = [
            ("mlp.fc", mlp_hidden, dim),
            ("mlp.proj", dim, mlp_hidden),
        ]

    shapes: list[tuple[str, int, int]] = []
    for layer in range(num_layers):
        prefix = f"blocks.{layer}"
        shapes += [
            (f"{prefix}.attn.c_q", dim, dim),
            (f"{prefix}.attn.c_k", kv_dim, dim),
            (f"{prefix}.attn.c_v", kv_dim, dim),
            (f"{prefix}.attn.c_proj", dim, dim),
        ]
        for mname, r, c in mlp_shapes:
            shapes.append((f"{prefix}.{mname}", r, c))
    return shapes


# ---------------------------------------------------------------------------
# Payload size estimators
# ---------------------------------------------------------------------------

def _quantized_payload_bytes(rows: int, cols: int, bits: int) -> int:
    """Raw bytes for a quantized 2D matrix: int8 containers + fp16 per-row scales."""
    return rows * cols + rows * 2  # 1 byte/value + 2 bytes/row (fp16 scale)


def estimate_artifact_bytes(
    vocab_size: int = 1024,
    num_layers: int = 9,
    dim: int = 512,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    mlp_mult: float = 3.0,
    mlp_act: str = "relu2",
    attn_bits: int = 6,
    mlp_bits: int = 6,
    embed_fp16: bool = True,
    weight_decay: float = 0.02,
    compression: str = "zpaq",
    permute: bool = True,
    small_tensor_threshold: int = 65_536,
) -> dict:
    """
    Estimate artifact size from architecture + quantization spec.

    Parameters
    ----------
    vocab_size, num_layers, dim, num_heads, num_kv_heads, mlp_mult, mlp_act
        Model architecture.
    attn_bits : int
        Quantization bits for attention weight matrices (3, 4, 5, 6, or 8).
    mlp_bits : int
        Quantization bits for MLP weight matrices (3, 4, 5, 6, or 8).
    embed_fp16 : bool
        If True, tok_emb is kept as fp16 passthrough (not quantized).
        The current SOTA recipe always does this.
    weight_decay : float
        Muon weight decay value. Higher → better compression.
    compression : str
        "zpaq" (default, zpaq method 5), "brotli" (brotli-11), or "zstd" (zstd-22).
    permute : bool
        If True (default), assume MLP weight permutation is applied.
    small_tensor_threshold : int
        Tensors with ≤ this many elements are kept in fp16 (not quantized).

    Returns
    -------
    dict with keys: total_bytes, total_mb, headroom_bytes, headroom_mb,
        breakdown (per component), fits (bool)
    """
    shapes = _matrix_shapes(num_layers, dim, num_heads, num_kv_heads, mlp_mult, mlp_act)
    params = _count_params(vocab_size, num_layers, dim, num_heads, num_kv_heads, mlp_mult, mlp_act)

    attn_ratio = _compression_ratio(attn_bits, weight_decay, compressor=compression,
                                     is_mlp=False, permuted=False)
    mlp_ratio = _compression_ratio(mlp_bits, weight_decay, compressor=compression,
                                    is_mlp=True, permuted=permute)
    fp16_ratio = _FP16_ZSTD_RATIO

    breakdown: dict[str, dict] = {}

    # Embedding
    embed_numel = vocab_size * dim
    if embed_fp16:
        embed_raw = embed_numel * 2  # fp16
        embed_compressed = int(embed_raw / fp16_ratio)
        breakdown["tok_emb_fp16"] = {
            "numel": embed_numel, "raw_bytes": embed_raw,
            "compressed_bytes": embed_compressed, "scheme": "fp16",
        }
    else:
        # Would be quantized (rows=vocab_size, cols=dim)
        embed_raw = _quantized_payload_bytes(vocab_size, dim, attn_bits)
        embed_compressed = int(embed_raw / _zstd_ratio(attn_bits, weight_decay))
        breakdown["tok_emb_quant"] = {
            "numel": embed_numel, "raw_bytes": embed_raw,
            "compressed_bytes": embed_compressed, "scheme": f"int{attn_bits}",
        }

    # Weight matrices
    total_attn_raw = 0
    total_mlp_raw = 0
    total_small_raw = 0

    for name, rows, cols in shapes:
        numel = rows * cols
        is_mlp = "mlp" in name
        is_small = numel <= small_tensor_threshold

        if is_small:
            raw = numel * 2  # fp16 passthrough
            total_small_raw += raw
        elif is_mlp:
            raw = _quantized_payload_bytes(rows, cols, mlp_bits)
            total_mlp_raw += raw
        else:
            raw = _quantized_payload_bytes(rows, cols, attn_bits)
            total_attn_raw += raw

    attn_compressed = int(total_attn_raw / attn_ratio)
    mlp_compressed = int(total_mlp_raw / mlp_ratio)
    small_compressed = int(total_small_raw / fp16_ratio)

    breakdown["attn_matrices"] = {
        "numel": params["total_matrix_attn"], "raw_bytes": total_attn_raw,
        "compressed_bytes": attn_compressed, "scheme": f"int{attn_bits}+{compression}",
    }
    breakdown["mlp_matrices"] = {
        "numel": params["total_matrix_mlp"], "raw_bytes": total_mlp_raw,
        "compressed_bytes": mlp_compressed,
        "scheme": f"int{mlp_bits}+{compression}{'+permute' if permute else ''}",
    }
    breakdown["small_tensors_fp16"] = {
        "numel": params["total_ln"], "raw_bytes": total_small_raw,
        "compressed_bytes": small_compressed, "scheme": "fp16",
    }
    breakdown["overhead"] = {
        "numel": 0, "raw_bytes": TORCH_OVERHEAD_BYTES,
        "compressed_bytes": TORCH_OVERHEAD_BYTES, "scheme": "fixed",
    }

    total_compressed = (
        embed_compressed + attn_compressed + mlp_compressed
        + small_compressed + TORCH_OVERHEAD_BYTES
    )

    return {
        "total_bytes": total_compressed,
        "total_mb": total_compressed / MB,
        "headroom_bytes": BUDGET_BYTES - total_compressed,
        "headroom_mb": (BUDGET_BYTES - total_compressed) / MB,
        "fits": total_compressed <= BUDGET_BYTES,
        "total_params": params["total"],
        "breakdown": breakdown,
        "config": {
            "num_layers": num_layers, "dim": dim, "vocab_size": vocab_size,
            "num_heads": num_heads, "num_kv_heads": num_kv_heads,
            "mlp_mult": mlp_mult, "mlp_act": mlp_act,
            "attn_bits": attn_bits, "mlp_bits": mlp_bits,
            "embed_fp16": embed_fp16, "weight_decay": weight_decay,
            "compression": compression, "permute": permute,
        },
    }


def sweep_max_layers(
    vocab_size: int = 1024,
    dim: int = 512,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    mlp_mult: float = 3.0,
    mlp_act: str = "relu2",
    attn_bits: int = 6,
    mlp_bits: int = 6,
    embed_fp16: bool = True,
    weight_decay: float = 0.02,
    compression: str = "zpaq",
    permute: bool = True,
    max_layers: int = 20,
) -> list[dict]:
    """Find how many layers fit in 16 MB for a given architecture spec."""
    results = []
    for n in range(1, max_layers + 1):
        r = estimate_artifact_bytes(
            vocab_size=vocab_size, num_layers=n, dim=dim,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            mlp_mult=mlp_mult, mlp_act=mlp_act, attn_bits=attn_bits,
            mlp_bits=mlp_bits, embed_fp16=embed_fp16,
            weight_decay=weight_decay, compression=compression,
            permute=permute,
        )
        results.append(r)
        if not r["fits"]:
            break
    return results


# ---------------------------------------------------------------------------
# Exact size from a trained checkpoint (requires torch)
# ---------------------------------------------------------------------------

def measure_checkpoint(
    path: str,
    compression: str = "zstd",
    zstd_level: int = 22,
    zlib_level: int = 6,
) -> dict:
    """
    Measure the exact compressed artifact size from a trained checkpoint.
    Loads the state dict, applies the same quantization pipeline, measures bytes.

    Requires: torch, zstandard (for zstd), or zlib (for zlib).
    """
    try:
        import torch
        import io, zlib as _zlib
    except ImportError:
        raise ImportError("torch is required for checkpoint measurement")

    # Load
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        state_dict = obj["model"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise ValueError(f"Unexpected checkpoint format at {path}")

    # Simulate serialization: pickle each tensor, measure, then compress
    import pickle
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    raw_bytes = buf.tell()

    buf.seek(0)
    raw_data = buf.read()

    if compression == "zstd":
        try:
            import zstandard as zstd
        except ImportError:
            raise ImportError("pip install zstandard for zstd compression measurement")
        cctx = zstd.ZstdCompressor(level=zstd_level)
        compressed = cctx.compress(raw_data)
    else:
        compressed = _zlib.compress(raw_data, zlib_level)

    compressed_bytes = len(compressed)
    return {
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "compressed_mb": compressed_bytes / MB,
        "headroom_bytes": BUDGET_BYTES - compressed_bytes,
        "headroom_mb": (BUDGET_BYTES - compressed_bytes) / MB,
        "fits": compressed_bytes <= BUDGET_BYTES,
        "compression_ratio": raw_bytes / compressed_bytes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_estimate(result: dict, verbose: bool = True) -> None:
    cfg = result["config"]
    fits_str = "✓ FITS" if result["fits"] else "✗ OVER BUDGET"
    headroom = result["headroom_mb"]
    sign = "+" if headroom >= 0 else ""

    print(f"\n{'='*55}")
    print(f"  {cfg['num_layers']}L {cfg['dim']}d  mlp×{cfg['mlp_mult']}  "
          f"int{cfg['attn_bits']}-attn/int{cfg['mlp_bits']}-mlp  "
          f"WD={cfg['weight_decay']}  {cfg['compression']}")
    print(f"{'='*55}")
    print(f"  Params:       {result['total_params']:>12,}")
    print(f"  Artifact:     {result['total_mb']:>10.2f} MB   {fits_str}")
    print(f"  Headroom:     {sign}{abs(headroom):>9.2f} MB")

    if verbose:
        print(f"\n  Breakdown:")
        for name, info in result["breakdown"].items():
            mb = info["compressed_bytes"] / MB
            print(f"    {name:<26} {mb:>6.2f} MB  [{info['scheme']}]")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Predict artifact size for Parameter Golf submission.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pgolf.budget                          # default 9L baseline config
  python -m pgolf.budget --layers 11 --wd 0.04   # 11L with high WD
  python -m pgolf.budget --layers 10 --mlp-bits 5 --wd 0.04  # int5 MLP
  python -m pgolf.budget --sweep --wd 0.038      # how many layers fit?
  python -m pgolf.budget --checkpoint model.pt   # exact size from checkpoint
        """,
    )
    p.add_argument("--layers", type=int, default=9)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--mlp-mult", type=float, default=3.0)
    p.add_argument("--mlp-act", choices=["relu2", "swiglu"], default="relu2")
    p.add_argument("--attn-bits", type=int, default=6, choices=[4, 5, 6, 8])
    p.add_argument("--mlp-bits", type=int, default=6, choices=[4, 5, 6, 8])
    p.add_argument("--no-embed-fp16", action="store_true",
                   help="Quantize embedding instead of keeping fp16 (not recommended)")
    p.add_argument("--wd", type=float, default=0.02,
                   help="Muon weight decay (affects compression ratio estimate). "
                        "Typical: 0.02, high: 0.038-0.04")
    p.add_argument("--compression", choices=["zstd", "zlib"], default="zstd")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep layer count to find max that fits")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to trained checkpoint for exact size measurement")
    p.add_argument("--json", action="store_true", help="Output as JSON")

    args = p.parse_args(argv)

    if args.checkpoint:
        result = measure_checkpoint(args.checkpoint, compression=args.compression)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            fits_str = "✓ FITS" if result["fits"] else "✗ OVER BUDGET"
            print(f"\nCheckpoint: {args.checkpoint}")
            print(f"Raw size:       {result['raw_bytes'] / MB:.2f} MB")
            print(f"Compressed:     {result['compressed_mb']:.2f} MB  {fits_str}")
            print(f"Headroom:       {result['headroom_mb']:+.2f} MB")
            print(f"Ratio:          {result['compression_ratio']:.2f}x")
        return

    common_kwargs = dict(
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        mlp_act=args.mlp_act,
        attn_bits=args.attn_bits,
        mlp_bits=args.mlp_bits,
        embed_fp16=not args.no_embed_fp16,
        weight_decay=args.wd,
        compression=args.compression,
    )

    if args.sweep:
        results = sweep_max_layers(max_layers=20, **common_kwargs)
        print(f"\nLayer sweep  (dim={args.dim}, mlp×{args.mlp_mult}, "
              f"int{args.attn_bits}-attn/int{args.mlp_bits}-mlp, WD={args.wd})")
        print(f"{'Layers':>7}  {'Params':>10}  {'MB':>7}  {'Headroom':>9}  Status")
        print("-" * 50)
        for r in results:
            n = r["config"]["num_layers"]
            fits = "✓" if r["fits"] else "✗"
            sign = "+" if r["headroom_mb"] >= 0 else ""
            print(f"{n:>7}  {r['total_params']:>10,}  {r['total_mb']:>7.2f}  "
                  f"{sign}{r['headroom_mb']:>8.2f}  {fits}")
        # Show comparison for common quant schemes
        print(f"\nScheme comparison at {args.layers}L:")
        for abits, mbits, label in [
            (8, 8, "int8-all"),
            (6, 6, "int6-all"),
            (6, 5, "int6-attn/int5-mlp"),
            (5, 5, "int5-all"),
        ]:
            r = estimate_artifact_bytes(num_layers=args.layers, attn_bits=abits, mlp_bits=mbits, **{k: v for k,v in common_kwargs.items() if k not in ('attn_bits', 'mlp_bits')})
            fits = "✓" if r["fits"] else "✗"
            print(f"  {label:<26} {r['total_mb']:.2f} MB  {fits}")
    else:
        result = estimate_artifact_bytes(num_layers=args.layers, **common_kwargs)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            _print_estimate(result, verbose=True)

            # Also show WD sensitivity
            print(f"\n  WD sensitivity (int{args.attn_bits}+zstd-22, {args.layers}L):")
            for wd in [0.0, 0.01, 0.02, 0.03, 0.038, 0.04, 0.05]:
                r = estimate_artifact_bytes(num_layers=args.layers, weight_decay=wd, **{k: v for k,v in common_kwargs.items() if k != 'weight_decay'})
                fits = "✓" if r["fits"] else "·"
                print(f"    WD={wd:.3f}  → {r['total_mb']:.2f} MB  {fits}")


if __name__ == "__main__":
    main()
