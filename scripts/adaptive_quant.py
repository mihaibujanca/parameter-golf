#!/usr/bin/env python3
"""
adaptive_quant.py — Per-layer adaptive quantization guided by FFN geometry.

1. Probes per-layer channel scores, dead neurons, sensitivity
2. Assigns per-layer bit-widths (int4/int5/int6) based on sensitivity
3. Optionally prunes dead channels and refits output projections
4. Measures BPB for each configuration

Usage:
    NUM_LAYERS=13 MLP_MULT=3 .venv/bin/python3 adaptive_quant.py logs/scaling5k_13L_mlp3x_mlx_model.npz
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from train_gpt_mlx import (
    GPT, COMPUTE_DTYPE, Hyperparameters,
    build_sentencepiece_luts, eval_val, load_validation_tokens,
    validate_dataset_tokenizer_pair, _np_float32,
)
from quant_gap_test import (
    quantize_configurable, dequantize_configurable, estimate_compressed_size,
)


# ==============================================================================
# FFN GEOMETRY PROBING (adapted from quant_aware_training)
# ==============================================================================

def probe_ffn_geometry(model: GPT, val_tokens: np.ndarray, seq_len: int = 1024,
                       max_tokens: int = 4096) -> list[dict]:
    """Probe per-layer MLP geometry: channel scores, dead neurons, sensitivity.

    Reimplements the key parts of quant_aware_training's probe_ffn_geometry
    directly on MLX, without the TeacherBundle/ProbeCorpus abstraction.
    """
    num_layers = len(model.blocks)
    # Collect activations per layer
    layer_stats = []

    # Run forward passes and collect MLP activations via manual probing
    n_seqs = min(max_tokens // seq_len, (val_tokens.size - 1) // seq_len)
    tokens = val_tokens[:n_seqs * seq_len + 1]

    for layer_idx in range(num_layers):
        block = model.blocks[layer_idx]
        mlp = block.mlp

        # Collect pre-activation (z) and post-activation (h) for this MLP
        all_z = []
        all_h = []

        for seq_start in range(0, n_seqs * seq_len, seq_len):
            x_np = tokens[seq_start:seq_start + seq_len].reshape(1, seq_len)
            x = mx.array(x_np, dtype=mx.int32)

            # Forward through embedding + all blocks up to this layer
            hidden = model.tok_emb(x).astype(COMPUTE_DTYPE)
            if model.bigram is not None:
                hidden = hidden + model.bigram(x)
            from train_gpt_mlx import rms_norm
            hidden = rms_norm(hidden)
            hidden = model.smear(hidden)
            x0 = hidden

            skips = []
            for i in range(layer_idx + 1):
                if i < model.num_encoder_layers:
                    hidden = model.blocks[i](hidden, x0)
                    skips.append(hidden)
                else:
                    dec_idx = i - model.num_encoder_layers
                    if dec_idx < len(skips) and skips:
                        skip = skips[-(dec_idx + 1)] if dec_idx < len(skips) else None
                        if skip is not None:
                            hidden = hidden + model.skip_weights[dec_idx].astype(hidden.dtype)[None, None, :] * skip
                    hidden = model.blocks[i](hidden, x0)

            # Now get the MLP input for this layer (after attention + residual)
            # We need the normalized input to the MLP
            mlp_input = block.mlp_norm(hidden)

            # Compute pre-activation z = fc(mlp_input)
            z = mlp.fc(mlp_input)
            mx.eval(z)
            z_np = np.array(z.astype(mx.float32), dtype=np.float32).reshape(-1, z.shape[-1])

            # Compute post-activation h = act(z)
            if mlp.act == "lrelu2":
                h_np = np.maximum(z_np, 0.5 * z_np) ** 2
            else:  # relu2
                h_np = np.maximum(z_np, 0) ** 2

            all_z.append(z_np)
            all_h.append(h_np)

            if sum(a.shape[0] for a in all_h) >= max_tokens:
                break

        z_all = np.concatenate(all_z, axis=0)
        h_all = np.concatenate(all_h, axis=0)

        # Channel statistics
        mean_hidden = h_all.mean(axis=0)
        std_hidden = h_all.std(axis=0) if h_all.shape[0] > 1 else np.zeros_like(mean_hidden)

        # Activation derivative for relu²: d/dz(relu(z)²) = 2*relu(z)
        if mlp.act == "lrelu2":
            deriv = np.where(z_all > 0, 2 * z_all, 2 * 0.5 * z_all * 0.5)  # 2*leaky_relu(z)*slope
        else:
            deriv = 2.0 * np.maximum(z_all, 0)
        mean_deriv = deriv.mean(axis=0)

        # Output projection norms
        proj_weight = np.array(mlp.proj.weight.astype(mx.float32), dtype=np.float32)  # [dim, hidden]
        output_norms = np.linalg.norm(proj_weight, axis=0)  # per-channel L2 norm

        # Dead neuron detection
        dead_mask = h_all.max(axis=0) <= 1e-6
        dead_frac = dead_mask.mean()

        # Channel importance scores (from quant_aware_training)
        channel_scores = std_hidden * output_norms * (np.abs(mean_hidden) + np.abs(mean_deriv) + 1e-6)

        # Quantization sensitivity: how much does quantizing this layer's MLP hurt?
        # Approximate via weight magnitude × activation magnitude
        fc_weight = np.array(mlp.fc.weight.astype(mx.float32), dtype=np.float32)
        fc_frob = np.linalg.norm(fc_weight)
        proj_frob = np.linalg.norm(proj_weight)
        act_rms = np.sqrt(np.mean(h_all ** 2))

        layer_stats.append({
            "layer": layer_idx,
            "hidden_dim": h_all.shape[1],
            "dead_frac": float(dead_frac),
            "dead_count": int(dead_mask.sum()),
            "alive_count": int((~dead_mask).sum()),
            "channel_scores": channel_scores,
            "mean_score": float(channel_scores.mean()),
            "max_score": float(channel_scores.max()),
            "fc_frob": float(fc_frob),
            "proj_frob": float(proj_frob),
            "act_rms": float(act_rms),
            "sensitivity": float(fc_frob * proj_frob * act_rms),  # rough proxy
        })

    return layer_stats


def assign_layer_bits(layer_stats: list[dict], budget_mb: float = 16.0,
                      default_bits: int = 5) -> dict[int, int]:
    """Assign per-layer MLP bit-widths based on sensitivity ranking.

    Strategy: layers with low sensitivity get int4, high sensitivity get int6,
    medium get int5. Optimize to fit within budget.
    """
    sensitivities = [(s["layer"], s["sensitivity"]) for s in layer_stats]
    sensitivities.sort(key=lambda x: x[1])

    n = len(sensitivities)
    # Bottom third: int4, middle third: int5, top third: int6
    layer_bits = {}
    for rank, (layer_idx, _sens) in enumerate(sensitivities):
        if rank < n // 3:
            layer_bits[layer_idx] = 4
        elif rank < 2 * n // 3:
            layer_bits[layer_idx] = 5
        else:
            layer_bits[layer_idx] = 6

    return layer_bits


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Adaptive per-layer quantization")
    parser.add_argument("checkpoint", type=str, help="Path to .npz checkpoint")
    parser.add_argument("--val-max-tokens", type=int, default=1048576)
    parser.add_argument("--val-batch-size", type=int, default=65536)
    parser.add_argument("--probe-tokens", type=int, default=4096,
                        help="Tokens to collect for FFN geometry probing")
    args_cli = parser.parse_args()

    hparams = Hyperparameters()
    hparams.val_max_tokens = args_cli.val_max_tokens
    hparams.val_batch_size = args_cli.val_batch_size

    sp = spm.SentencePieceProcessor(model_file=hparams.tokenizer_path)
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, hparams.vocab_size
    )

    per_layer = [int(x) for x in hparams.mlp_mult_per_layer.split(",") if x] if hparams.mlp_mult_per_layer else None
    model = GPT(
        vocab_size=hparams.vocab_size, num_layers=hparams.num_layers, dim=hparams.model_dim,
        num_heads=hparams.num_heads, num_kv_heads=hparams.num_kv_heads, mlp_mult=hparams.mlp_mult,
        logit_chunk_tokens=hparams.logit_chunk_tokens, logit_softcap=hparams.logit_softcap,
        rope_base=hparams.rope_base, tied_embed_init_std=hparams.tied_embed_init_std,
        qk_gain_init=hparams.qk_gain_init, mlp_act=hparams.mlp_act,
        mlp_mult_per_layer=per_layer, bigram_vocab_size=hparams.bigram_vocab_size,
        bigram_dim=hparams.bigram_dim,
        num_encoder_layers=hparams.num_encoder_layers,
    )

    print(f"Loading checkpoint: {args_cli.checkpoint}")
    ckpt_state = dict(mx.load(args_cli.checkpoint))
    model.update(tree_unflatten(list(ckpt_state.items())))
    mx.eval(model.state)
    original_state = {k: v for k, v in tree_flatten(model.state)}

    # ==========================================
    # Phase 1: Probe FFN geometry
    # ==========================================
    print(f"\n{'='*60}")
    print("Phase 1: FFN Geometry Probing")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    layer_stats = probe_ffn_geometry(model, val_tokens, max_tokens=args_cli.probe_tokens)
    probe_ms = 1000 * (time.perf_counter() - t0)
    print(f"Probing took {probe_ms:.0f}ms\n")

    print(f"{'Layer':<6s} {'Hidden':<7s} {'Dead%':>6s} {'Dead#':>6s} {'Sensitivity':>12s} {'MeanScore':>10s}")
    print("-" * 55)
    for s in layer_stats:
        print(f"L{s['layer']:<5d} {s['hidden_dim']:<7d} {s['dead_frac']*100:>5.1f}% {s['dead_count']:>5d} "
              f"{s['sensitivity']:>12.1f} {s['mean_score']:>10.4f}")

    # ==========================================
    # Phase 2: Assign per-layer bit-widths
    # ==========================================
    print(f"\n{'='*60}")
    print("Phase 2: Per-Layer Bit Allocation")
    print(f"{'='*60}")
    layer_bits = assign_layer_bits(layer_stats)
    print("\nAssigned bit-widths (by sensitivity ranking):")
    for layer_idx in range(len(layer_stats)):
        bits = layer_bits[layer_idx]
        sens = layer_stats[layer_idx]["sensitivity"]
        print(f"  Layer {layer_idx}: int{bits}  (sensitivity={sens:.1f})")

    # ==========================================
    # Phase 3: Compare quantization configs
    # ==========================================
    print(f"\n{'='*60}")
    print("Phase 3: Quantization Comparison")
    print(f"{'='*60}")

    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)

    configs = [
        ("uniform int6", 6, 6, None),
        ("uniform int5", 5, 5, None),
        ("uniform int4", 4, 4, None),
        ("adaptive (geometry)", None, None, layer_bits),
    ]

    results = []
    for label, attn_bits, mlp_bits, per_layer_bits in configs:
        model.update(tree_unflatten(list(original_state.items())))
        mx.eval(model.state)

        flat_state = {k: v for k, v in tree_flatten(model.state)}

        if per_layer_bits is not None:
            # Build cat_bits with per-layer MLP overrides
            # For adaptive: quantize each MLP layer at its assigned bits
            # Attention stays at int6 (most sensitive)
            cat_bits_for_quant = {"attn": 6}
            # We need to do per-layer quantization manually
            quant_obj, quant_stats = _quantize_adaptive(flat_state, per_layer_bits, attn_bits=6)
        else:
            quant_obj, quant_stats = quantize_configurable(
                flat_state, attn_bits=attn_bits, mlp_bits=mlp_bits, hadamard=False
            )

        artifact_bytes = estimate_compressed_size(quant_obj, quant_stats)
        artifact_mb = artifact_bytes / (1024 * 1024)

        quant_flat = dequantize_configurable(quant_obj)
        model.update(tree_unflatten(list(quant_flat.items())))
        mx.eval(model.state)

        compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)

        val_loss, val_bpb = eval_val(
            hparams, compiled_loss, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )

        bits_desc = label
        if per_layer_bits:
            counts = {}
            for b in per_layer_bits.values():
                counts[b] = counts.get(b, 0) + 1
            bits_desc += f" [{', '.join(f'{c}×int{b}' for b, c in sorted(counts.items()))}]"

        print(f"  {bits_desc:<45s}  BPB={val_bpb:.4f}  MB={artifact_mb:.2f}")
        results.append({"label": label, "bpb": val_bpb, "mb": artifact_mb})

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    fp32_bpb = None
    # Get fp32 baseline
    model.update(tree_unflatten(list(original_state.items())))
    mx.eval(model.state)
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    _, fp32_bpb = eval_val(hparams, compiled_loss, val_tokens,
                            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"  {'fp32 baseline':<45s}  BPB={fp32_bpb:.4f}")
    for r in results:
        gap = r["bpb"] - fp32_bpb
        print(f"  {r['label']:<45s}  BPB={r['bpb']:.4f}  gap={gap:+.4f}  MB={r['mb']:.2f}")


def _quantize_adaptive(flat_state: dict[str, mx.array], per_layer_mlp_bits: dict[int, int],
                        attn_bits: int = 6) -> tuple[dict, dict]:
    """Quantize with per-layer MLP bit-widths."""
    from train_gpt_mlx import _classify_param

    # Build a cat_bits dict that includes layer info
    # We need to inspect param names to determine layer index
    cat_bits = {}
    for name in flat_state:
        cat = _classify_param(name)
        if cat == "attn":
            cat_bits[name] = attn_bits
        elif cat == "mlp":
            # Extract layer index from name like "blocks.5.mlp.fc.weight"
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "blocks" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        cat_bits[name] = per_layer_mlp_bits.get(layer_idx, 5)
                    except ValueError:
                        cat_bits[name] = 5
                    break
            else:
                cat_bits[name] = 5
        else:
            cat_bits[name] = 8  # bigram, other

    # Use quantize_configurable but with per-param bit override
    # We need to modify it slightly... or just call quantize functions directly
    from quant_gap_test import (
        quantize_int4_per_row, quantize_int5_per_row,
        _np_float32, keep_float_array,
        FP16_KEEP_NAME_PATTERNS, CONTROL_TENSOR_NAME_PATTERNS,
        INT8_KEEP_FLOAT_MAX_NUMEL, MX_DTYPE_FROM_NAME,
        INT8_KEEP_FLOAT_FP32_NAME_PATTERNS,
    )
    from train_gpt_mlx import (
        quantize_float_array, quantize_int6_per_row,
    )

    QUANT_FN = {4: quantize_int4_per_row, 5: quantize_int5_per_row,
                6: quantize_int6_per_row, 8: quantize_float_array}

    quantized = {}
    scales = {}
    dtypes = {}
    passthrough = {}
    passthrough_orig_dtypes = {}
    qmeta = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int8_payload_bytes"), 0)

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
                kept = keep_float_array(name, arr, passthrough_orig_dtypes)
                passthrough[name] = kept
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        stats["num_float_tensors"] += 1
        bits = cat_bits.get(name, 5)
        quant_fn = QUANT_FN[bits]
        q, s = quant_fn(arr)
        if bits < 8:
            qmeta[name] = {"scheme": f"int{bits}_per_row", "axis": 0}
        elif s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)

    obj = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized, "scales": scales, "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


if __name__ == "__main__":
    main()
