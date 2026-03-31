#!/usr/bin/env python3
"""
quant_analysis.py — Quantization analysis pipeline: sensitivity, compound error, correction placement.

Three importable functions:
  - per_layer_sensitivity(): isolated per-layer int4 sensitivity
  - compound_error_profile(): per-layer DeltaErr under actual quant config
  - optimal_correction_layers(): pick top-K layers by DeltaErr for correction placement

Standalone usage:
    NUM_LAYERS=11 MLP_MULT=4.5 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \
    .venv/bin/python3 scripts/quant_analysis.py logs/warmdown_11L_45x_best.npz \
        --quant-bits "attn:4,mlp:5,mlp.1:4" --n-correction-layers 3

    # Save results for later use:
    ... --save-analysis logs/analysis_11L_45x.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def per_layer_sensitivity(
    build_model_fn,
    flat_state: dict,
    eval_bpb_fn,
    num_layers: int,
    target_bits: int = 4,
) -> dict[str, list[float]]:
    """Quantize one component at a time to target_bits, measure BPB impact.

    Returns: {'attn': [gap_per_layer], 'mlp': [gap_per_layer], 'both': [gap_per_layer]}
    """
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from train_gpt_mlx import quantize_state_dict_int8, dequantize_state_dict_int8

    # Float baseline
    model = build_model_fn()
    model.update(tree_unflatten(list(flat_state.items())))
    mx.eval(model.parameters())
    float_bpt = eval_bpb_fn(model)

    results = {'attn': [], 'mlp': [], 'both': [], 'float_bpt': float_bpt}

    for i in range(num_layers):
        for comp in ['attn', 'mlp', 'both']:
            if comp == 'both':
                cat_bits = {f'attn.{i}': target_bits, f'mlp.{i}': target_bits}
            else:
                cat_bits = {f'{comp}.{i}': target_bits}
            qobj, _ = quantize_state_dict_int8(flat_state, cat_bits=cat_bits)
            qflat = dequantize_state_dict_int8(qobj)
            m = build_model_fn()
            m.update(tree_unflatten(list(qflat.items())))
            mx.eval(m.parameters())
            gap = eval_bpb_fn(m) - float_bpt
            results[comp].append(gap)

    return results


def design_quant_allocation(
    sensitivity: dict[str, list[float]],
    num_layers: int,
    default_attn_bits: int = 4,
    default_mlp_bits: int = 5,
    demote_threshold: float = 0.002,
) -> dict[str, int]:
    """Design per-layer quant allocation from sensitivity data.

    Layers with isolated sensitivity below demote_threshold get demoted to int4.
    Returns cat_bits dict for quantize_state_dict_int8.
    """
    cat_bits = {'attn': default_attn_bits, 'mlp': default_mlp_bits}

    for i in range(num_layers):
        if sensitivity['mlp'][i] < demote_threshold:
            cat_bits[f'mlp.{i}'] = 4
        if sensitivity['attn'][i] < demote_threshold:
            cat_bits[f'attn.{i}'] = 4

    return cat_bits


def compound_error_profile(
    build_model_fn,
    flat_state: dict,
    cat_bits: dict[str, int],
    val_tokens: np.ndarray,
    num_layers: int,
    n_seqs: int = 8,
    seq_len: int = 1024,
) -> list[dict]:
    """Measure per-layer hidden state error under actual quant config.

    Returns list of dicts with 'layer', 'error_rmse', 'delta_err', 'float_norm'.
    """
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from train_gpt_mlx import (
        COMPUTE_DTYPE, quantize_state_dict_int8, dequantize_state_dict_int8, rms_norm,
    )

    model_float = build_model_fn()
    model_float.update(tree_unflatten(list(flat_state.items())))
    mx.eval(model_float.parameters())

    qobj, _ = quantize_state_dict_int8(flat_state, cat_bits=cat_bits)
    qflat = dequantize_state_dict_int8(qobj)
    model_quant = build_model_fn()
    model_quant.update(tree_unflatten(list(qflat.items())))
    mx.eval(model_quant.parameters())

    def forward_collect(model, tokens):
        x = mx.array(tokens[:seq_len][np.newaxis, :])
        tok_emb = model.tok_emb(x).astype(COMPUTE_DTYPE)
        if model.bigram is not None:
            tok_emb = tok_emb + model.bigram(x)
        x0 = model.smear(rms_norm(tok_emb))
        h = x0
        n_enc = model.num_encoder_layers
        encoder_outputs = []
        hidden = {}
        for i, block in enumerate(model.blocks):
            if i >= n_enc and (i - n_enc) < model.num_skip_weights:
                skip_idx = n_enc - 1 - (i - n_enc)
                w = model.skip_weights[i - n_enc]
                h = h + w.astype(h.dtype)[None, None, :] * encoder_outputs[skip_idx]
            mix = block.resid_mix.astype(h.dtype)
            h = mix[0][None, None, :] * h + mix[1][None, None, :] * x0
            attn_out = block.attn(block.attn_norm(h))
            h = h + block.attn_scale.astype(h.dtype)[None, None, :] * attn_out
            h = h + block.mlp_scale.astype(h.dtype)[None, None, :] * block.mlp(block.mlp_norm(h), slope=block.lrelu_slope)
            if i < n_enc:
                encoder_outputs.append(h)
            mx.eval(h)
            hidden[i] = h
        return hidden

    errors = {i: [] for i in range(num_layers)}
    norms = {i: [] for i in range(num_layers)}
    for s in range(n_seqs):
        tokens = val_tokens[s * seq_len: (s + 1) * seq_len]
        if len(tokens) < seq_len:
            break
        fh = forward_collect(model_float, tokens)
        qh = forward_collect(model_quant, tokens)
        for i in range(num_layers):
            err = np.array(qh[i]) - np.array(fh[i])
            errors[i].append(np.sqrt((err ** 2).mean()))
            norms[i].append(np.sqrt((np.array(fh[i]) ** 2).mean()))

    results = []
    prev_err = 0
    for i in range(num_layers):
        err = np.mean(errors[i])
        norm = np.mean(norms[i])
        delta = err - prev_err
        results.append({
            'layer': i,
            'error_rmse': float(err),
            'delta_err': float(delta),
            'float_norm': float(norm),
        })
        prev_err = err
    return results


def optimal_correction_layers(
    compound_profile: list[dict],
    n_corrections: int = 3,
) -> list[int]:
    """Pick top-N layers by DeltaErr for correction placement."""
    sorted_layers = sorted(compound_profile, key=lambda x: x['delta_err'], reverse=True)
    return sorted([l['layer'] for l in sorted_layers[:n_corrections]])


def print_sensitivity(sensitivity: dict, num_layers: int) -> None:
    print(f"\n{'Layer':>6s} {'attn':>8s} {'mlp':>8s} {'both':>8s}")
    for i in range(num_layers):
        print(f"{i:>6d} {sensitivity['attn'][i]:>+8.4f} {sensitivity['mlp'][i]:>+8.4f} {sensitivity['both'][i]:>+8.4f}")
    print(f"\nFloat baseline: {sensitivity['float_bpt']:.4f} bpt")


def print_compound_profile(profile: list[dict]) -> None:
    print(f"\n{'Layer':>6s} {'ErrorRMSE':>12s} {'DeltaErr':>12s} {'FloatNorm':>12s}")
    for p in profile:
        print(f"{p['layer']:>6d} {p['error_rmse']:>12.4f} {p['delta_err']:>+12.4f} {p['float_norm']:>12.4f}")


def main():
    parser = argparse.ArgumentParser(description="Quantization analysis pipeline")
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument("--quant-bits", type=str, default="",
                        help="Quant allocation (e.g. 'attn:4,mlp:5,mlp.10:4'). "
                             "If empty, runs sensitivity analysis first to determine allocation.")
    parser.add_argument("--demote-threshold", type=float, default=0.002,
                        help="Sensitivity threshold below which MLP layers get demoted to int4")
    parser.add_argument("--n-correction-layers", type=int, default=3)
    parser.add_argument("--n-eval-seqs", type=int, default=16)
    parser.add_argument("--n-profile-seqs", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--save-analysis", type=str, default=None,
                        help="Save analysis results to JSON")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip sensitivity analysis, require --quant-bits")
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from train_gpt_mlx import (
        GPT, COMPUTE_DTYPE, Hyperparameters, load_validation_tokens,
        quantize_state_dict_int8, dequantize_state_dict_int8, rms_norm,
    )

    hparams = Hyperparameters()

    def build_model():
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

    flat = dict(mx.load(args.checkpoint))
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)

    def eval_bpb(model, n_seqs=None):
        n = n_seqs or args.n_eval_seqs
        total_loss, total_tokens = 0.0, 0
        for s in range(n):
            tokens = val_tokens[s * args.seq_len: (s + 1) * args.seq_len + 1]
            if len(tokens) < args.seq_len + 1:
                break
            x_arr = mx.array(tokens[:args.seq_len][np.newaxis, :])
            h = model(x_arr)
            logits = model._apply_logit_processing(model.tok_emb.as_linear(h))
            logits_2d = logits.reshape(-1, logits.shape[-1])
            targets = mx.array(tokens[1:args.seq_len + 1].reshape(-1))
            loss = nn.losses.cross_entropy(logits_2d, targets, reduction='sum')
            mx.eval(loss)
            total_loss += loss.item()
            total_tokens += args.seq_len
        return total_loss / total_tokens / math.log(2)

    analysis = {'checkpoint': args.checkpoint, 'num_layers': hparams.num_layers}

    # Step 1: Sensitivity (or skip if quant-bits provided)
    if args.quant_bits and args.skip_sensitivity:
        cat_bits = {}
        for part in args.quant_bits.split(","):
            k, v = part.strip().rsplit(":", 1)
            cat_bits[k.strip()] = int(v.strip())
        print(f"Using provided quant allocation: {cat_bits}")
    else:
        print("=== Per-layer sensitivity ===")
        sensitivity = per_layer_sensitivity(build_model, flat, eval_bpb, hparams.num_layers)
        print_sensitivity(sensitivity, hparams.num_layers)
        analysis['sensitivity'] = sensitivity

        if args.quant_bits:
            cat_bits = {}
            for part in args.quant_bits.split(","):
                k, v = part.strip().rsplit(":", 1)
                cat_bits[k.strip()] = int(v.strip())
        else:
            cat_bits = design_quant_allocation(sensitivity, hparams.num_layers,
                                               demote_threshold=args.demote_threshold)
        print(f"\nQuant allocation: {cat_bits}")

    analysis['cat_bits'] = cat_bits

    # Step 2: Compound error profile
    print("\n=== Compound error profile ===")
    profile = compound_error_profile(build_model, flat, cat_bits, val_tokens,
                                     hparams.num_layers, n_seqs=args.n_profile_seqs,
                                     seq_len=args.seq_len)
    print_compound_profile(profile)
    analysis['compound_profile'] = profile

    # Step 3: Optimal correction layers
    correction_layers = optimal_correction_layers(profile, args.n_correction_layers)
    print(f"\nOptimal correction layers: {correction_layers}")
    analysis['correction_layers'] = correction_layers

    # Save
    if args.save_analysis:
        with open(args.save_analysis, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Saved analysis to {args.save_analysis}")


if __name__ == "__main__":
    main()
