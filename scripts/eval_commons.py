"""
eval_commons.py — Shared model building, evaluation, and data loading utilities.

Centralizes the ~15 inline copies of build_model(), quick_eval(), and
forward_collect_hidden() scattered across scripts/.

Key functions:
    build_model(hparams)         — Factory: Hyperparameters → GPT
    eval_bpb(model, hparams)     — Proper BPB eval using sentencepiece byte LUTs
    quick_ce(model, val_tokens)  — Fast CE (nats). Intermediate monitoring only.
    quick_bpt(model, val_tokens) — quick_ce / ln(2). Bits-per-token. NOT BPB.
    forward_collect_hidden(model, tokens)  — Hidden states at every layer boundary
    load_train_tokens(hparams, max_tokens) — Load subset of training data
"""
from __future__ import annotations

import glob as globmod
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train_gpt_mlx import (
    GPT, COMPUTE_DTYPE, Hyperparameters,
    load_validation_tokens, load_data_shard, rms_norm,
    build_sentencepiece_luts, eval_val,
)
import sentencepiece as spm


# =============================================================================
# Model construction
# =============================================================================

def build_model(hparams: Hyperparameters) -> GPT:
    """Build a GPT model from Hyperparameters. Replaces ~15 inline copies."""
    per_layer = None
    if hparams.mlp_mult_per_layer:
        per_layer = [float(x) for x in hparams.mlp_mult_per_layer.split(",")]
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
        num_encoder_layers=hparams.num_encoder_layers,
    )


# =============================================================================
# Evaluation
# =============================================================================

def eval_bpb(model: GPT, hparams: Hyperparameters) -> tuple[float, float]:
    """Proper BPB evaluation using sentencepiece byte LUTs.

    Returns (val_loss, val_bpb) where val_bpb is the competition metric.
    Uses the same eval path as training (eval_val with compiled_loss).
    """
    sp = spm.SentencePieceProcessor(model_file=hparams.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, hparams.vocab_size)
    )
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    compiled_loss = model.loss
    return eval_val(
        hparams, compiled_loss, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )


def quick_ce(model: GPT, val_tokens: np.ndarray,
             n_seqs: int = 32, seq_len: int = 1024) -> float:
    """Quick cross-entropy evaluation in nats. For intermediate monitoring only.

    NOT BPB — this is raw CE. Use eval_bpb() for the competition metric.
    """
    total_loss = 0.0
    total_tokens = 0
    for s in range(n_seqs):
        tokens = val_tokens[s * seq_len: (s + 1) * seq_len + 1]
        if len(tokens) < seq_len + 1:
            break
        x = mx.array(tokens[:seq_len].reshape(1, seq_len))
        y = mx.array(tokens[1:seq_len + 1].reshape(1, seq_len))
        loss = model.loss(x, y)
        mx.eval(loss)
        total_loss += loss.item() * seq_len
        total_tokens += seq_len
    return total_loss / total_tokens


def quick_bpt(model: GPT, val_tokens: np.ndarray,
              n_seqs: int = 32, seq_len: int = 1024) -> float:
    """Quick bits-per-token evaluation. For intermediate monitoring only.

    NOT BPB. This is CE / ln(2) = bits per token, not bits per byte.
    Use eval_bpb() for the competition metric.
    """
    return quick_ce(model, val_tokens, n_seqs, seq_len) / math.log(2)


# =============================================================================
# Hidden state collection
# =============================================================================

def forward_collect_hidden(model: GPT, tokens: np.ndarray) -> list[mx.array]:
    """Full forward collecting hidden states at every layer boundary.

    Returns: list[mx.array] of length n_layers, each (1, T, D) post-block.
    """
    n_enc = model.num_encoder_layers
    n_skip = model.num_skip_weights
    n_layers = len(model.blocks)

    x = mx.array(tokens[np.newaxis, :])
    tok_emb = model.tok_emb(x).astype(COMPUTE_DTYPE)
    if model.bigram is not None:
        tok_emb = tok_emb + model.bigram(x)
    x0 = model.smear(rms_norm(tok_emb))
    h = x0

    encoder_outputs = [None] * n_enc
    hidden = []

    for i in range(n_layers):
        if i >= n_enc:
            dec_j = i - n_enc
            if dec_j < n_skip:
                enc_j = n_enc - 1 - dec_j
                h = h + model.skip_weights[dec_j].astype(h.dtype)[None, None, :] * encoder_outputs[enc_j]
        h = model.blocks[i](h, x0)
        mx.eval(h)
        hidden.append(h)
        if i < n_enc:
            encoder_outputs[i] = h

    return hidden


# =============================================================================
# Data loading
# =============================================================================

def load_train_tokens(hparams: Hyperparameters, max_tokens: int = 1_000_000) -> np.ndarray:
    """Load a subset of training tokens for calibration / correction training."""
    files = sorted(globmod.glob(hparams.train_files))
    if not files:
        raise FileNotFoundError(f"No train files: {hparams.train_files}")
    tokens = []
    total = 0
    for f in files:
        shard = load_data_shard(Path(f))
        tokens.append(shard)
        total += len(shard)
        if total >= max_tokens:
            break
    return np.concatenate(tokens)[:max_tokens]
