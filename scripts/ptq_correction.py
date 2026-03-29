#!/usr/bin/env python3
"""
ptq_correction.py — Post-training quantization error correction.

Trains tiny CorrectionNet modules that predict and compensate quantization
error at strategic layer boundaries. Zero changes to the main training loop.

IMPORTANT: Corrections for submission MUST be trained on train data (--data-split train).
Val-trained corrections are for reference/debugging ONLY.

Usage:
    # Quick reference (val, DO NOT submit):
    NUM_LAYERS=13 MLP_MULT=3 MLP_ACT=lrelu2 QUANT_ATTN_BITS=5 QUANT_MLP_BITS=5 \
    .venv/bin/python3 scripts/ptq_correction.py logs/overnight_13L_3x_lrelu2_best.npz \
        --correction-layers 4,7,10 --data-split val

    # For submission (train data):
    ... --data-split train

    # Int4 test:
    QUANT_ATTN_BITS=4 QUANT_MLP_BITS=4 ... --correction-layers 4,7,10
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gpt_mlx import (
    GPT, COMPUTE_DTYPE, Hyperparameters, load_validation_tokens,
    quantize_state_dict_int8, dequantize_state_dict_int8, load_data_shard,
    build_sentencepiece_luts, validate_dataset_tokenizer_pair, eval_val,
)


# =============================================================================
# CorrectionNet — small network to predict quantization error correction
# =============================================================================

class CorrectionNet(nn.Module):
    """Predicts correction from quantization error. Zero-initialized output."""

    def __init__(self, dim: int, hidden: int = 32):
        super().__init__()
        self.hidden = hidden
        if hidden > 0:
            self.w1 = nn.Linear(dim, hidden)
            self.w2 = nn.Linear(hidden, dim)
            # Zero-init output so correction starts as identity
            self.w2.weight = mx.zeros_like(self.w2.weight)
            self.w2.bias = mx.zeros_like(self.w2.bias)
        else:
            # Linear correction
            self.linear = nn.Linear(dim, dim)
            self.linear.weight = mx.zeros_like(self.linear.weight)
            self.linear.bias = mx.zeros_like(self.linear.bias)

    def __call__(self, error: mx.array) -> mx.array:
        if self.hidden > 0:
            return self.w2(nn.relu(self.w1(error)))
        return self.linear(error)


# =============================================================================
# Forward passes with activation collection
# =============================================================================

def forward_with_hidden_collection(model, x_tokens, slope, collect_at):
    """Forward pass collecting hidden states at specified layer indices.

    Returns: (logits, {layer_idx: hidden_state}) where hidden_state is post-block.
    """
    x = mx.array(x_tokens[np.newaxis, :])
    tok_emb = model.tok_emb(x)
    if model.bigram is not None:
        tok_emb = tok_emb + model.bigram(x)
    x0 = model.smear(tok_emb)
    h = x0

    collected = {}
    n_enc = model.num_encoder_layers
    n_skip = model.num_skip_weights
    encoder_outputs = []

    for i, block in enumerate(model.blocks):
        if i >= n_enc and (i - n_enc) < n_skip:
            skip_idx = n_enc - 1 - (i - n_enc)
            skip_h = encoder_outputs[skip_idx]
            w = model.skip_weights[i - n_enc]
            h = h + w.astype(h.dtype)[None, None, :] * skip_h

        mix = block.resid_mix.astype(h.dtype)
        h = mix[0][None, None, :] * h + mix[1][None, None, :] * x0
        attn_out = block.attn(block.attn_norm(h))
        h = h + block.attn_scale.astype(h.dtype)[None, None, :] * attn_out
        h = h + block.mlp_scale.astype(h.dtype)[None, None, :] * block.mlp(block.mlp_norm(h), slope=block.lrelu_slope)

        if i < n_enc:
            encoder_outputs.append(h)

        if i in collect_at:
            mx.eval(h)
            collected[i] = h

    logits = model._apply_logit_processing(model.tok_emb.as_linear(model.final_norm(h)))
    mx.eval(logits)
    return logits, collected


def forward_corrected(model, x_tokens, slope, corrections, correction_layers, float_hidden):
    """Forward pass with corrections applied at specified layers.

    At each correction layer, computes error = h_quant - h_float (from cache),
    passes through CorrectionNet, and adds correction to hidden state.

    Returns: (logits, corrected_hidden_at_correction_points)
    """
    x = mx.array(x_tokens[np.newaxis, :])
    tok_emb = model.tok_emb(x)
    if model.bigram is not None:
        tok_emb = tok_emb + model.bigram(x)
    x0 = model.smear(tok_emb)
    h = x0

    n_enc = model.num_encoder_layers
    n_skip = model.num_skip_weights
    encoder_outputs = []
    corrected_hidden = {}

    for i, block in enumerate(model.blocks):
        if i >= n_enc and (i - n_enc) < n_skip:
            skip_idx = n_enc - 1 - (i - n_enc)
            skip_h = encoder_outputs[skip_idx]
            w = model.skip_weights[i - n_enc]
            h = h + w.astype(h.dtype)[None, None, :] * skip_h

        mix = block.resid_mix.astype(h.dtype)
        h = mix[0][None, None, :] * h + mix[1][None, None, :] * x0
        attn_out = block.attn(block.attn_norm(h))
        h = h + block.attn_scale.astype(h.dtype)[None, None, :] * attn_out
        h = h + block.mlp_scale.astype(h.dtype)[None, None, :] * block.mlp(block.mlp_norm(h), slope=block.lrelu_slope)

        if i < n_enc:
            encoder_outputs.append(h)

        # Apply correction at this layer
        if i in correction_layers and i in corrections:
            error = h - float_hidden[i]
            correction = corrections[i](error)
            h = h + correction
            corrected_hidden[i] = h

    logits = model._apply_logit_processing(model.tok_emb.as_linear(model.final_norm(h)))
    return logits, corrected_hidden


# =============================================================================
# Training
# =============================================================================

def load_train_tokens(hparams, max_tokens=1_000_000):
    """Load a subset of train tokens for correction training."""
    pattern = hparams.train_files
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No train files: {pattern}")
    tokens = []
    total = 0
    for f in files:
        shard = load_data_shard(Path(f))
        tokens.append(shard)
        total += len(shard)
        if total >= max_tokens:
            break
    return np.concatenate(tokens)[:max_tokens]


def train_corrections(
    model_float, model_quant, train_tokens, val_tokens, hparams,
    correction_layers, hidden_size=32, n_epochs=200, lr=1e-3,
    seq_len=1024, n_train_seqs=64, n_eval_seqs=8,
):
    """Train correction networks on train data, evaluate on val."""
    dim = hparams.model_dim
    slope = hparams.lrelu_slope
    collect_set = set(correction_layers)

    # Build correction nets as a list (MLX-friendly)
    corrections_list = [CorrectionNet(dim, hidden_size) for _ in correction_layers]
    corrections = dict(zip(correction_layers, corrections_list))

    # Flatten all correction params into a single list for mx.value_and_grad
    def get_all_params():
        params = []
        for net in corrections_list:
            params.extend(v for _, v in tree_flatten(net.parameters()))
        return params

    def set_all_params(params):
        idx = 0
        for net in corrections_list:
            flat = list(tree_flatten(net.parameters()))
            new_items = []
            for k, _ in flat:
                new_items.append((k, params[idx]))
                idx += 1
            net.update(tree_unflatten(new_items))

    mx.eval(get_all_params())

    optimizer = optim.Adam(learning_rate=lr)
    opt_state_initialized = False

    # Pre-cache float hidden states for train sequences
    print(f"Caching float hidden states ({n_train_seqs} seqs)...")
    float_cache = []
    float_logits_cache = []
    for s in range(n_train_seqs):
        tokens = train_tokens[s * seq_len : (s + 1) * seq_len]
        if len(tokens) < seq_len:
            break
        logits, hidden = forward_with_hidden_collection(model_float, tokens, slope, collect_set)
        float_cache.append({k: mx.stop_gradient(v) for k, v in hidden.items()})
        float_logits_cache.append(mx.stop_gradient(logits))

    actual_train_seqs = len(float_cache)
    print(f"Cached {actual_train_seqs} sequences")

    def loss_fn(params, seq_idx):
        """Compute loss for one sequence.

        Loss = intermediate MSE (match float hidden at correction points)
             + output KL (match float logit distribution)
        """
        set_all_params(params)
        tokens = train_tokens[seq_idx * seq_len : (seq_idx + 1) * seq_len]
        float_hidden = float_cache[seq_idx]
        float_logits = float_logits_cache[seq_idx]

        logits, corrected_hidden = forward_corrected(
            model_quant, tokens, slope, corrections, correction_layers, float_hidden)

        # Intermediate MSE: each correction net should make its output match float
        mse_loss = mx.array(0.0)
        for i in correction_layers:
            if i in corrected_hidden:
                target = mx.stop_gradient(float_hidden[i])
                diff = corrected_hidden[i] - target
                mse_loss = mse_loss + (diff * diff).mean()
        mse_loss = mse_loss / len(correction_layers)

        # Output KL: match float logit distribution
        T = 2.0
        f_probs = mx.stop_gradient(mx.softmax(float_logits / T, axis=-1))
        q_log_probs = logits / T - mx.logsumexp(logits / T, axis=-1, keepdims=True)
        kl = -(f_probs * q_log_probs).sum(axis=-1).mean()

        return mse_loss + 0.1 * kl

    grad_fn = mx.value_and_grad(loss_fn)

    print(f"\nTraining corrections: layers={correction_layers}, hidden={hidden_size}, "
          f"epochs={n_epochs}, lr={lr}")
    t0 = time.time()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for s in range(actual_train_seqs):
            params = get_all_params()
            loss, grads = grad_fn(params, s)
            # Manual Adam update
            if not opt_state_initialized:
                opt_state = [(mx.zeros_like(g), mx.zeros_like(g)) for g in grads]
                opt_step = 0
                opt_state_initialized = True
            opt_step += 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            for j, (g, (m, v)) in enumerate(zip(grads, opt_state)):
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                m_hat = m / (1 - b1 ** opt_step)
                v_hat = v / (1 - b2 ** opt_step)
                params[j] = params[j] - lr * m_hat / (mx.sqrt(v_hat) + eps)
                opt_state[j] = (m, v)
            set_all_params(params)
            mx.eval(get_all_params())
            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg = epoch_loss / actual_train_seqs
            elapsed = time.time() - t0
            print(f"  epoch {epoch+1:>4d}/{n_epochs}  loss={avg:.6f}  elapsed={elapsed:.1f}s")

    print(f"Training done in {time.time() - t0:.1f}s")
    return corrections


# =============================================================================
# Evaluation
# =============================================================================

def eval_corrected_bpb(model_quant, corrections, correction_layers, model_float,
                       hparams, val_tokens, n_seqs=32, seq_len=1024):
    """Evaluate corrected model BPB on val data."""
    slope = hparams.lrelu_slope
    collect_set = set(correction_layers)

    total_loss = 0.0
    total_tokens = 0

    for s in range(n_seqs):
        tokens = val_tokens[s * seq_len : (s + 1) * seq_len + 1]
        if len(tokens) < seq_len + 1:
            break
        x = tokens[:seq_len]
        y = tokens[1:seq_len + 1]

        # Get float hidden for correction
        _, float_hidden = forward_with_hidden_collection(model_float, x, slope, collect_set)

        # Corrected forward
        logits, _ = forward_corrected(model_quant, x, slope, corrections, correction_layers, float_hidden)
        mx.eval(logits)

        # Cross-entropy loss
        logits_2d = logits.reshape(-1, logits.shape[-1])
        targets = mx.array(y.reshape(-1))
        loss = nn.losses.cross_entropy(logits_2d, targets, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += len(y)

    val_loss = total_loss / total_tokens
    # Convert to BPB using the tokenizer's bytes-per-token ratio
    # Approximate: for sp1024, ~1.06 bytes per token on average
    # But for proper BPB we need the byte counts. Use rough estimate.
    bits_per_token = val_loss / math.log(2)
    print(f"  Corrected val_loss={val_loss:.4f} bits_per_token={bits_per_token:.4f}")
    return val_loss


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument("--correction-layers", type=str, default="4,7,10",
                        help="Comma-separated layer indices for correction")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-train-seqs", type=int, default=64)
    parser.add_argument("--n-eval-seqs", type=int, default=32)
    parser.add_argument("--data-split", choices=["train", "val"], default="train",
                        help="Data split for correction training. "
                             "MUST be 'train' for anything going into submission. "
                             "'val' is for reference/debugging ONLY.")
    parser.add_argument("--seq-len", type=int, default=1024)
    args_cli = parser.parse_args()

    correction_layers = [int(x) for x in args_cli.correction_layers.split(",")]
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
        )

    # Load float model
    print(f"Loading checkpoint: {args_cli.checkpoint}")
    print(f"Config: {hparams.num_layers}L/{hparams.model_dim}d, MLP {hparams.mlp_mult}x, "
          f"act={hparams.mlp_act}, quant=a{hparams.quant_attn_bits}m{hparams.quant_mlp_bits}")
    print(f"Correction layers: {correction_layers}, hidden={args_cli.hidden_size}")
    print(f"Data split for training: {args_cli.data_split}"
          f"{' *** REFERENCE ONLY - DO NOT SUBMIT ***' if args_cli.data_split == 'val' else ''}")

    model_float = build_model()
    flat = dict(mx.load(args_cli.checkpoint))
    model_float.update(tree_unflatten(list(flat.items())))
    mx.eval(model_float.parameters())

    # Build quantized model
    print("Quantizing model...")
    model_quant = build_model()
    cat_bits = {"attn": hparams.quant_attn_bits, "mlp": hparams.quant_mlp_bits}
    quant_obj, quant_stats = quantize_state_dict_int8(flat, cat_bits=cat_bits)
    quant_flat = dequantize_state_dict_int8(quant_obj)
    model_quant.update(tree_unflatten(list(quant_flat.items())))
    mx.eval(model_quant.parameters())

    # Load data
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    if args_cli.data_split == "train":
        train_tokens = load_train_tokens(hparams,
                                         max_tokens=args_cli.n_train_seqs * args_cli.seq_len + 1024)
        print(f"Train tokens: {len(train_tokens):,}")
    else:
        train_tokens = val_tokens
        print(f"*** Using val tokens for training (REFERENCE ONLY) ***")
    print(f"Val tokens: {len(val_tokens):,}")

    # Baseline: uncorrected quant eval
    print("\n=== Baseline (no correction) ===")
    eval_corrected_bpb(model_quant, {}, [], model_float, hparams,
                       val_tokens, n_seqs=args_cli.n_eval_seqs, seq_len=args_cli.seq_len)

    # Oracle correction: at correction points, replace quant hidden with float hidden
    print("\n=== Oracle correction (upper bound) ===")
    oracle_corrections = {}  # Empty corrections but we'll hack the forward
    # For oracle, just run float model — that gives us the upper bound
    eval_corrected_bpb(model_float, {}, [], model_float, hparams,
                       val_tokens, n_seqs=args_cli.n_eval_seqs, seq_len=args_cli.seq_len)

    # Train corrections
    print(f"\n=== Training corrections ({args_cli.data_split} split) ===")
    corrections = train_corrections(
        model_float, model_quant, train_tokens, val_tokens, hparams,
        correction_layers, hidden_size=args_cli.hidden_size,
        n_epochs=args_cli.n_epochs, lr=args_cli.lr,
        seq_len=args_cli.seq_len, n_train_seqs=args_cli.n_train_seqs,
        n_eval_seqs=args_cli.n_eval_seqs,
    )

    # Evaluate corrected model
    print("\n=== Corrected model ===")
    eval_corrected_bpb(model_quant, corrections, correction_layers, model_float, hparams,
                       val_tokens, n_seqs=args_cli.n_eval_seqs, seq_len=args_cli.seq_len)

    # Report correction overhead
    total_correction_params = 0
    for i, net in corrections.items():
        n = sum(v.size for _, v in tree_flatten(net.parameters()))
        total_correction_params += n
        print(f"  Correction L{i}: {n:,} params")
    print(f"  Total correction params: {total_correction_params:,} "
          f"({100*total_correction_params/(hparams.model_dim * hparams.model_dim * hparams.num_layers):.2f}% of model)")
    # Estimate compressed size
    est_bytes = total_correction_params * 1  # int8 = 1 byte/param
    print(f"  Estimated artifact overhead: ~{est_bytes/1024:.1f} KB (int8)")


if __name__ == "__main__":
    main()
