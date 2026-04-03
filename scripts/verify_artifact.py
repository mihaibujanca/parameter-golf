#!/usr/bin/env python3
"""
verify_artifact.py — Load a compressed artifact and verify correctness.

Decompresses, dequantizes, reconstructs corrections, runs eval,
and compares against float model.

Uses eval_val() from train_gpt_mlx.py for BPB — same compiled loss, same byte
counting, same token set as training. Reports float BPB, quant BPB (no corrections),
and quant+correction comparison metrics (KL, Top-1, Cos).

Usage:
    NUM_LAYERS=11 MLP_MULT=4.5 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \
    .venv/bin/python3 scripts/verify_artifact.py logs/11L_45x_final.br \
        --float-checkpoint logs/warmdown_11L_45x_best.npz
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_artifact(artifact_path: str) -> dict:
    """Load and decompress artifact, return numpy dict."""
    raw_bytes = Path(artifact_path).read_bytes()
    if artifact_path.endswith('.br'):
        import brotli
        raw = brotli.decompress(raw_bytes)
    elif artifact_path.endswith('.zs') or artifact_path.endswith('.ptz'):
        import zstandard
        raw = zstandard.ZstdDecompressor().decompress(raw_bytes)
    else:
        raw = raw_bytes
    return dict(np.load(io.BytesIO(raw), allow_pickle=True))


def extract_qobj(artifact: dict) -> dict:
    """Separate quantized model weights from correction arrays."""
    qobj_keys = {'__quant_format__', 'quantized', 'scales', 'dtypes',
                 'passthrough', 'qmeta', 'passthrough_orig_dtypes'}
    qobj = {}
    for k in qobj_keys:
        if k in artifact:
            v = artifact[k]
            qobj[k] = v.item() if isinstance(v, np.ndarray) and v.dtype == object else v
    return qobj


def extract_corrections(artifact: dict, dim: int):
    """Reconstruct CorrectionNet instances from artifact arrays."""
    from scripts.ptq_correction import CorrectionNet
    import mlx.core as mx

    if '__correction_layers__' not in artifact:
        return [], {}

    correction_layers = [int(x) for x in artifact['__correction_layers__']]
    corrections = {}

    for li in correction_layers:
        net = CorrectionNet(dim, hidden=0)
        w_key = f'correction.{li}.linear.weight'
        b_key = f'correction.{li}.linear.bias'

        # Handle int8-quantized corrections (.q + .s) or raw float
        if f'{w_key}.q.q' in artifact:
            # Double-quantized (legacy bug): .q.q is the int8, .q.s is scale
            q = artifact[f'{w_key}.q.q'].astype(np.float32)
            s = artifact[f'{w_key}.q.s'].astype(np.float32)
            w = q * s[:, None]
            b = artifact[b_key].astype(np.float32)
        elif f'{w_key}.q' in artifact and artifact[f'{w_key}.q'].dtype == np.int8:
            q = artifact[f'{w_key}.q'].astype(np.float32)
            s = artifact[f'{w_key}.s'].astype(np.float32)
            w = q * s[:, None]
            b = artifact[b_key].astype(np.float32)
        elif w_key in artifact:
            w = artifact[w_key].astype(np.float32)
            b = artifact[b_key].astype(np.float32)
        else:
            raise KeyError(f"Cannot find correction weights for layer {li}: tried {w_key}, {w_key}.q, {w_key}.q.q")

        net.linear.weight = mx.array(w)
        net.linear.bias = mx.array(b)
        mx.eval(net.parameters())
        corrections[li] = net

    return correction_layers, corrections


def main():
    parser = argparse.ArgumentParser(description="Verify compressed artifact")
    parser.add_argument("artifact", help="Path to compressed artifact (.br or .zs)")
    parser.add_argument("--float-checkpoint", type=str, required=True,
                        help="Path to float checkpoint (for comparison)")
    parser.add_argument("--n-comparison-seqs", type=int, default=32,
                        help="Number of seqs for KL/Top-1/Cos comparison (default 32)")
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import sentencepiece as spm
    from mlx.utils import tree_flatten, tree_unflatten
    from train_gpt_mlx import (
        COMPUTE_DTYPE, Hyperparameters, load_validation_tokens,
        dequantize_state_dict_int8, rms_norm, build_sentencepiece_luts, eval_val,
    )
    from scripts.eval_commons import build_model as _build_model
    from scripts.ptq_correction import forward_with_hidden_collection, forward_corrected

    hparams = Hyperparameters()
    seq_len = hparams.train_seq_len

    def build_model():
        return _build_model(hparams)

    # Load val tokens and tokenizer byte LUTs (same as training)
    val_tokens = load_validation_tokens(hparams.val_files, seq_len)
    sp = spm.SentencePieceProcessor(model_file=hparams.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, hparams.vocab_size)

    # =========================================================================
    # Load models
    # =========================================================================

    # Float model
    print(f"Loading float model: {args.float_checkpoint}")
    model_float = build_model()
    flat = dict(mx.load(args.float_checkpoint))
    model_float.update(tree_unflatten(list(flat.items())))
    mx.eval(model_float.parameters())

    # Artifact model (quantized, dequantized back to float for eval)
    print(f"Loading artifact: {args.artifact} ({os.path.getsize(args.artifact)/1e6:.2f} MB)")
    artifact = load_artifact(args.artifact)
    qobj = extract_qobj(artifact)
    qflat = dequantize_state_dict_int8(qobj)
    model_quant = build_model()
    model_quant.update(tree_unflatten(list(qflat.items())))
    mx.eval(model_quant.parameters())

    # Corrections
    correction_layers, corrections = extract_corrections(artifact, hparams.model_dim)
    print(f"Corrections: {correction_layers if correction_layers else 'none'}")

    # =========================================================================
    # BPB via eval_val — identical to training script
    # =========================================================================

    def run_eval_val(model, label):
        compiled_loss = mx.compile(
            lambda x, y: model.loss(x, y),
            inputs=model.state, outputs=model.state,
        )
        print(f"Evaluating {label}...")
        val_loss, val_bpb = eval_val(
            hparams, compiled_loss, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            log_fn=lambda s: print(f"  {s}"),
        )
        print(f"  {label}: val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")
        return val_loss, val_bpb

    float_loss, float_bpb = run_eval_val(model_float, "float")
    quant_loss, quant_bpb = run_eval_val(model_quant, "quant (no corrections)")

    # =========================================================================
    # Comparison metrics (KL, Top-1, Cos) — float vs artifact w/ corrections
    # =========================================================================

    slope = hparams.lrelu_slope
    collect_set = set(correction_layers)
    all_kl, all_top1, all_cos = [], [], []

    if corrections:
        print(f"Comparing float vs corrected ({args.n_comparison_seqs} seqs)...")
        for s in range(args.n_comparison_seqs):
            tokens = val_tokens[s * seq_len: (s + 1) * seq_len + 1]
            if len(tokens) < seq_len + 1:
                break
            x = tokens[:seq_len]

            _, fh = forward_with_hidden_collection(model_float, x, slope, collect_set)
            f_logits, _ = forward_corrected(model_float, x, slope, {}, [], fh)
            logits, _ = forward_corrected(model_quant, x, slope, corrections, correction_layers, fh)
            mx.eval(f_logits, logits)

            fl = np.array(f_logits).reshape(-1, f_logits.shape[-1])
            ql = np.array(logits).reshape(-1, logits.shape[-1])

            fp = np.exp(fl - fl.max(-1, keepdims=True))
            fp /= fp.sum(-1, keepdims=True)
            qp = np.exp(ql - ql.max(-1, keepdims=True))
            qp /= qp.sum(-1, keepdims=True)

            kl = (fp * np.log(np.maximum(fp, 1e-12) / np.maximum(qp, 1e-12))).sum(-1)
            all_kl.extend(kl.tolist())
            all_top1.extend((fl.argmax(-1) == ql.argmax(-1)).tolist())

            fn = fl / (np.linalg.norm(fl, -1, keepdims=True) + 1e-8)
            qn = ql / (np.linalg.norm(ql, -1, keepdims=True) + 1e-8)
            all_cos.extend((fn * qn).sum(-1).tolist())

    # =========================================================================
    # Report
    # =========================================================================

    kl_arr = np.array(all_kl) if all_kl else np.array([0.0])
    artifact_mb = os.path.getsize(args.artifact) / 1e6
    fits = os.path.getsize(args.artifact) <= 16_000_000

    print()
    print("=== Artifact Verification ===")
    print(f"Artifact:     {args.artifact}")
    print(f"Size:         {artifact_mb:.2f} MB (budget: 16.00 MB)")
    print(f"Fits 16MB:    {'YES' if fits else 'NO'}")
    print()
    print(f"Float BPB:    {float_bpb:.4f}")
    print(f"Quant BPB:    {quant_bpb:.4f}  (gap: {quant_bpb - float_bpb:+.4f})")
    print(f"Score:        {quant_bpb * artifact_mb:.4f}  (BPB x MB)")
    if corrections:
        print()
        print(f"--- Correction comparison (float vs quant+corrections) ---")
        print(f"KL mean:      {kl_arr.mean():.6f}")
        print(f"KL p99:       {np.percentile(kl_arr, 99):.6f}")
        print(f"KL max:       {np.max(kl_arr):.6f}")
        print(f"Top-1 agree:  {np.mean(all_top1):.4f}")
        print(f"Cos min:      {np.min(all_cos):.6f}")


if __name__ == "__main__":
    main()
