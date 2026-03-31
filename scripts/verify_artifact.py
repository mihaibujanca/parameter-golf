#!/usr/bin/env python3
"""
verify_artifact.py — Load a compressed artifact and verify correctness.

Decompresses, dequantizes, reconstructs corrections, runs eval,
and compares against float model.

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
                        help="Path to float checkpoint (for correction eval)")
    parser.add_argument("--n-seqs", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from train_gpt_mlx import (
        GPT, COMPUTE_DTYPE, Hyperparameters, load_validation_tokens,
        dequantize_state_dict_int8, rms_norm,
    )
    from scripts.ptq_correction import forward_with_hidden_collection, forward_corrected

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

    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)

    # Load artifact
    print(f"Loading artifact: {args.artifact} ({os.path.getsize(args.artifact)/1e6:.2f} MB)")
    artifact = load_artifact(args.artifact)

    # Reconstruct quantized model
    qobj = extract_qobj(artifact)
    qflat = dequantize_state_dict_int8(qobj)
    model = build_model()
    model.update(tree_unflatten(list(qflat.items())))
    mx.eval(model.parameters())

    # Reconstruct corrections
    correction_layers, corrections = extract_corrections(artifact, hparams.model_dim)
    print(f"Corrections: {correction_layers if correction_layers else 'none'}")

    # Load float model
    print(f"Loading float model: {args.float_checkpoint}")
    flat = dict(mx.load(args.float_checkpoint))
    model_float = build_model()
    model_float.update(tree_unflatten(list(flat.items())))
    mx.eval(model_float.parameters())

    # Eval
    slope = hparams.lrelu_slope
    collect_set = set(correction_layers)
    total_loss, total_tokens = 0.0, 0
    all_kl, all_top1, all_cos = [], [], []

    print(f"Evaluating ({args.n_seqs} seqs)...")
    for s in range(args.n_seqs):
        tokens = val_tokens[s * args.seq_len: (s + 1) * args.seq_len + 1]
        if len(tokens) < args.seq_len + 1:
            break
        x, y = tokens[:args.seq_len], tokens[1:args.seq_len + 1]

        _, fh = forward_with_hidden_collection(model_float, x, slope, collect_set)
        f_logits, _ = forward_corrected(model_float, x, slope, {}, [], fh)
        if corrections:
            logits, _ = forward_corrected(model, x, slope, corrections, correction_layers, fh)
        else:
            logits, _ = forward_corrected(model, x, slope, {}, [], fh)
        mx.eval(f_logits, logits)

        fl = np.array(f_logits).reshape(-1, f_logits.shape[-1])
        ql = np.array(logits).reshape(-1, logits.shape[-1])
        tgt = y.reshape(-1)

        fp = np.exp(fl - fl.max(-1, keepdims=True))
        fp /= fp.sum(-1, keepdims=True)
        qp = np.exp(ql - ql.max(-1, keepdims=True))
        qp /= qp.sum(-1, keepdims=True)

        loss = -np.log(np.maximum(qp[np.arange(len(tgt)), tgt], 1e-12))
        total_loss += loss.sum()
        total_tokens += len(tgt)

        kl = (fp * np.log(np.maximum(fp, 1e-12) / np.maximum(qp, 1e-12))).sum(-1)
        all_kl.extend(kl.tolist())
        all_top1.extend((fl.argmax(-1) == ql.argmax(-1)).tolist())

        fn = fl / (np.linalg.norm(fl, -1, keepdims=True) + 1e-8)
        qn = ql / (np.linalg.norm(ql, -1, keepdims=True) + 1e-8)
        all_cos.extend((fn * qn).sum(-1).tolist())

    bpt = total_loss / total_tokens / math.log(2)
    kl_arr = np.array(all_kl)

    print()
    print("=== Artifact Verification ===")
    print(f"Artifact:   {args.artifact}")
    print(f"Size:       {os.path.getsize(args.artifact)/1e6:.2f} MB (budget: 16.00 MB)")
    print(f"BPB:        {bpt:.4f}")
    print(f"KL mean:    {kl_arr.mean():.6f}")
    print(f"KL p99:     {np.percentile(kl_arr, 99):.6f}")
    print(f"KL max:     {np.max(kl_arr):.6f}")
    print(f"Top-1:      {np.mean(all_top1):.4f}")
    print(f"Cos min:    {np.min(all_cos):.6f}")
    fits = os.path.getsize(args.artifact) <= 16_000_000
    print(f"Fits 16MB:  {'YES' if fits else 'NO'}")


if __name__ == "__main__":
    main()
