#!/usr/bin/env python3
"""
Ternary (1.58-bit) training script for parameter-golf on MLX/M4 Pro.

Separate from train_gpt_mlx.py because ternary requires fundamentally different:
- Linear layers (STE quantize to {-1,0,+1} every forward pass)
- Serialization (base-3 packing + LZMA instead of int8 + zstd)
- Training regime (no WD, no SWA/EMA, wider model)

Based on PR #920 (1.1539 BPB, 74M params, ~16MB artifact).
"""
from __future__ import annotations

import atexit
import glob
import json
import lzma
import math
import os
import pickle
import signal
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Reuse data loading, eval, optimizer components from the main script
from train_gpt_mlx import (
    COMPUTE_DTYPE,
    TokenLoader,
    TokenStream,
    load_data_shard,
    load_validation_tokens,
    build_sentencepiece_luts,
    validate_dataset_tokenizer_pair,
    eval_val,
    token_chunks,
    accumulate_flat_grads,
    rms_norm,
    zeropower_newtonschulz5,
    RMSNormNoWeight,
    clip_grad_tree,
)

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    out_dir: str = os.environ.get("OUT_DIR", "logs")
    seed: int = int(os.environ.get("SEED", 42))
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 1024))

    # Model — wider than standard (768d vs 512d), fewer layers
    num_layers: int = int(os.environ.get("NUM_LAYERS", 10))
    model_dim: int = int(os.environ.get("MODEL_DIM", 768))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 4))
    rope_base: float = float(os.environ.get("ROPE_BASE", 5000.0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Ternary quantization
    bitnet_group_size: int = int(os.environ.get("BITNET_GROUP_SIZE", 128))
    # Minimum numel for ternary (smaller tensors stay fp16)
    ternary_min_numel: int = int(os.environ.get("TERNARY_MIN_NUMEL", 65536))

    # Training
    iterations: int = int(os.environ.get("ITERATIONS", 20000))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 16384))
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    max_wallclock_seconds: int = int(os.environ.get("MAX_WALLCLOCK_SECONDS", 600))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 256))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 4256))

    # Optimizer — no weight decay for ternary
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.02))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-10))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 3))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    # Z-loss coefficient (prevents logit explosion)
    z_loss_coeff: float = float(os.environ.get("Z_LOSS_COEFF", 1e-4))

    # Validation
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 200))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524288))
    val_max_tokens: int = int(os.environ.get("VAL_MAX_TOKENS", 0))
    val_sliding_stride: int = int(os.environ.get("VAL_SLIDING_STRIDE", 0))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    loss_skip_threshold: float = float(os.environ.get("LOSS_SKIP_THRESHOLD", 0.0))
    checkpoint_every: int = int(os.environ.get("CHECKPOINT_EVERY", 0))

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        total = self.iterations
        warmdown_start = total - self.warmdown_iters
        if step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)
        if step >= warmdown_start:
            t = (step - warmdown_start) / max(self.warmdown_iters, 1)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * t)))
        return 1.0


# ==============================================================================
# TERNARY LINEAR LAYER
# ==============================================================================

class TernaryLinear(nn.Module):
    """Linear layer with STE ternary quantization in every forward pass.

    Weights are stored as fp32 (latent), quantized to {-1,0,+1} per group
    with absmean scaling during forward. Gradients flow through fp32 via STE.
    """
    def __init__(self, in_dim: int, out_dim: int, group_size: int = 128):
        super().__init__()
        # Orthogonal init scaled by 1/sqrt(in_dim)
        scale = 1.0 / math.sqrt(in_dim)
        self.weight = mx.random.normal((out_dim, in_dim)) * scale
        self.group_size = group_size
        # Pad cols to multiple of group_size
        self._pad = (group_size - in_dim % group_size) % group_size

    def __call__(self, x: mx.array) -> mx.array:
        w = self.weight
        g = self.group_size
        # Pad if needed
        if self._pad > 0:
            w = mx.pad(w, [(0, 0), (0, self._pad)])
        w_g = w.reshape(-1, g)
        scale = mx.maximum(mx.mean(mx.abs(w_g), axis=-1, keepdims=True), 1e-8)
        q = mx.clip(mx.round(w_g / scale), -1, 1)
        w_ternary = (q * scale).reshape(w.shape)
        if self._pad > 0:
            w_ternary = w_ternary[:, :self.weight.shape[1]]
        # STE: forward uses ternary, backward uses fp32
        w_out = self.weight + mx.stop_gradient(w_ternary - self.weight)
        return x @ w_out.astype(x.dtype).T


# ==============================================================================
# MODEL COMPONENTS
# ==============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, group_size: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_proj = TernaryLinear(dim, dim, group_size)
        self.k_proj = TernaryLinear(dim, self.head_dim * num_kv_heads, group_size)
        self.v_proj = TernaryLinear(dim, self.head_dim * num_kv_heads, group_size)
        self.o_proj = TernaryLinear(dim, dim, group_size)
        self.rope = nn.RoPE(self.head_dim, base=rope_base)
        self.qk_gain = mx.zeros((num_heads,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(q)
        k = self.rope(k)
        # GQA: repeat k,v heads
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, rep, axis=1)
            v = mx.repeat(v, rep, axis=1)
        # QK gain (per-head learnable scale)
        gain = mx.exp(self.qk_gain).astype(q.dtype)
        scale = gain / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale[:, None, None]
        # Causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(attn.dtype)
        attn = attn + mask
        attn = mx.softmax(attn, axis=-1).astype(v.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, group_size: int = 128):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = TernaryLinear(dim, hidden, group_size)
        self.proj = TernaryLinear(hidden, dim, group_size)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.fc(x)
        h = nn.relu(h)
        return self.proj(h * h)  # relu²


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, group_size: int = 128):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, group_size)
        self.mlp = MLP(dim, 4, group_size)  # Always 4x for ternary
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((
            np.ones((dim,), dtype=np.float32),
            np.zeros((dim,), dtype=np.float32))))

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, dim: int,
                 num_heads: int, num_kv_heads: int, logit_softcap: float,
                 rope_base: float, group_size: int = 128):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.tok_emb = nn.Embedding(vocab_size, dim)
        # U-Net structure
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, rope_base, group_size)
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()
        # Init embedding
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * 0.005
        ).astype(COMPUTE_DTYPE)

    def _apply_logit_processing(self, logits: mx.array) -> mx.array:
        cap = self.logit_softcap
        return cap * mx.tanh(logits / cap)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x0 = self.tok_emb(input_ids)
        h = x0
        encoder_outputs = []
        n_enc = self.num_encoder_layers
        n_skip = self.num_skip_weights

        for i, block in enumerate(self.blocks):
            # U-Net skip connections (decoder side)
            if i >= n_enc and (i - n_enc) < n_skip:
                skip_idx = n_enc - 1 - (i - n_enc)
                skip_h = encoder_outputs[skip_idx]
                w = self.skip_weights[i - n_enc]
                h = h + w.astype(h.dtype)[None, None, :] * skip_h
            h = block(h, x0)
            if i < n_enc:
                encoder_outputs.append(h)

        return self.final_norm(h)

    def loss(self, input_ids: mx.array, target_ids: mx.array,
             z_loss_coeff: float = 1e-4) -> mx.array:
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self._apply_logit_processing(x @ self.tok_emb.weight.astype(x.dtype).T)
        ce = nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")
        # Z-loss: stabilize logit magnitudes
        if z_loss_coeff > 0:
            lse = mx.logsumexp(logits.astype(mx.float32), axis=-1)
            ce = ce + z_loss_coeff * (lse * lse).mean()
        return ce


# ==============================================================================
# BASE-3 PACKING / UNPACKING
# ==============================================================================

def pack_ternary(q_flat: np.ndarray) -> tuple[bytes, int]:
    """Pack ternary {-1,0,+1} array into base-3 bytes (5 trits/byte)."""
    n = len(q_flat)
    f = (q_flat.astype(np.int8) + 1).astype(np.uint8)  # shift to {0,1,2}
    pad = (5 - n % 5) % 5
    if pad:
        f = np.concatenate([f, np.zeros(pad, dtype=np.uint8)])
    g = f.reshape(-1, 5)
    packed = (g[:, 0] + g[:, 1] * 3 + g[:, 2] * 9 + g[:, 3] * 27 + g[:, 4] * 81).astype(np.uint8)
    return packed.tobytes(), n


def unpack_ternary(data: bytes, n: int) -> np.ndarray:
    """Unpack base-3 bytes back to ternary {-1,0,+1}."""
    v = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    t = np.zeros((len(v), 5), dtype=np.int8)
    for i in range(5):
        t[:, i] = v % 3
        v //= 3
    return (t.reshape(-1)[:n] - 1).astype(np.int8)


def pack_ternary_bitmask(q_flat: np.ndarray) -> tuple[bytes, int]:
    """Bitmask packing: store nonzero mask + sign bits separately."""
    n = len(q_flat)
    f = q_flat.astype(np.int8)
    nz = (f != 0)
    mask_bytes = np.packbits(nz).tobytes()
    sign_bytes = np.packbits(f[nz] > 0).tobytes() if nz.any() else b""
    return mask_bytes + sign_bytes, n


def unpack_ternary_bitmask(data: bytes, n: int) -> np.ndarray:
    """Unpack bitmask-packed ternary."""
    mask_bytes_len = (n + 7) // 8
    mask = np.unpackbits(np.frombuffer(data[:mask_bytes_len], dtype=np.uint8))[:n].astype(bool)
    nnz = mask.sum()
    signs = np.unpackbits(np.frombuffer(data[mask_bytes_len:], dtype=np.uint8))[:nnz]
    result = np.zeros(n, dtype=np.int8)
    result[mask] = np.where(signs, 1, -1).astype(np.int8)
    return result


# ==============================================================================
# TERNARY SERIALIZATION
# ==============================================================================

def quantize_ternary_state_dict(
    flat_state: dict[str, mx.array],
    group_size: int = 128,
    ternary_min_numel: int = 65536,
) -> tuple[dict, dict]:
    """Quantize model state dict for ternary serialization."""
    result = {}
    stats = {"ternary_params": 0, "fp16_params": 0, "total_bytes_raw": 0}

    for name, arr in flat_state.items():
        f32 = np.array(arr.astype(mx.float32))

        # Ternary candidate: 2D weight matrix with enough elements
        if f32.ndim == 2 and f32.size >= ternary_min_numel:
            rows, cols = f32.shape
            # Pad cols to group_size
            pad = (group_size - cols % group_size) % group_size
            if pad:
                f32_padded = np.pad(f32, [(0, 0), (0, pad)])
            else:
                f32_padded = f32
            padded_cols = f32_padded.shape[1]

            # Per-group quantization
            w_g = f32_padded.reshape(-1, group_size)
            scale = np.maximum(np.abs(w_g).mean(axis=-1, keepdims=True), 1e-8).astype(np.float16)
            q = np.clip(np.round(w_g / scale.astype(np.float32)), -1, 1).astype(np.int8)
            q_flat = q.reshape(-1)

            # Try both packing methods
            packed_b3, n_b3 = pack_ternary(q_flat)
            packed_bm, n_bm = pack_ternary_bitmask(q_flat)
            if len(packed_b3) <= len(packed_bm):
                packed, method = packed_b3, "base3"
            else:
                packed, method = packed_bm, "bitmask"

            result[name] = {
                "type": "ternary",
                "method": method,
                "packed": packed,
                "scale": scale.squeeze(-1),  # (n_groups,) fp16
                "shape": list(f32.shape),
                "padded_cols": padded_cols,
                "group_size": group_size,
                "n_trits": len(q_flat),
            }
            stats["ternary_params"] += f32.size
            stats["total_bytes_raw"] += len(packed) + scale.squeeze(-1).nbytes
        else:
            # Store as fp16
            fp16 = f32.astype(np.float16)
            result[name] = {
                "type": "fp16",
                "data": fp16,
                "shape": list(f32.shape),
            }
            stats["fp16_params"] += f32.size
            stats["total_bytes_raw"] += fp16.nbytes

    return result, stats


def dequantize_ternary_state_dict(quant_obj: dict) -> dict[str, mx.array]:
    """Dequantize ternary state dict back to mx.array."""
    result = {}
    for name, entry in quant_obj.items():
        if entry["type"] == "ternary":
            if entry["method"] == "base3":
                q = unpack_ternary(entry["packed"], entry["n_trits"])
            else:
                q = unpack_ternary_bitmask(entry["packed"], entry["n_trits"])
            q = q.astype(np.float32).reshape(-1, entry["group_size"])
            scale = entry["scale"].astype(np.float32)[:, np.newaxis]
            # Shrinkage correction: scale / mean(|q|)
            q_absmean = np.maximum(np.abs(q).mean(axis=-1, keepdims=True), 1e-8)
            w = (q * (scale / q_absmean)).reshape(-1, entry["padded_cols"])
            shape = entry["shape"]
            w = w[:shape[0], :shape[1]]
            result[name] = mx.array(w).astype(COMPUTE_DTYPE)
        elif entry["type"] == "fp16":
            result[name] = mx.array(entry["data"].astype(np.float32)).astype(COMPUTE_DTYPE)
    return result


# ==============================================================================
# OPTIMIZER (reuses Muon from main script, just different defaults)
# ==============================================================================

class Muon:
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array],
             step: int, lr_mul: float) -> dict[str, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            # No weight decay for ternary
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    def __init__(self, params: dict[str, mx.array], args: Hyperparameters):
        self.args = args
        embed_keys = [k for k in params if "tok_emb" in k]
        matrix_keys = [k for k in params if params[k].ndim == 2 and k not in embed_keys]
        scalar_keys = [k for k in params if params[k].ndim < 2 and k not in embed_keys]
        self.embed_keys = embed_keys
        self.scalar_keys = scalar_keys
        self.muon = Muon(matrix_keys, params, args)
        # Adam state for embed + scalar
        self._adam_state = {}
        self._adam_step = 0

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array],
             step: int, lr_mul: float) -> dict[str, mx.array]:
        out = dict(params)
        # Muon for matrix params
        muon_out = self.muon.step(params, grads, step, lr_mul)
        out.update(muon_out)
        # Adam for embed + scalar
        self._adam_step += 1
        for k in self.embed_keys + self.scalar_keys:
            if k not in grads:
                continue
            lr = (self.args.tied_embed_lr if k in self.embed_keys else self.args.scalar_lr) * lr_mul
            g = grads[k]
            if k not in self._adam_state:
                self._adam_state[k] = (mx.zeros_like(g), mx.zeros_like(g))
            m, v = self._adam_state[k]
            m = self.args.beta1 * m + (1 - self.args.beta1) * g
            v = self.args.beta2 * v + (1 - self.args.beta2) * g * g
            m_hat = m / (1 - self.args.beta1 ** self._adam_step)
            v_hat = v / (1 - self.args.beta2 ** self._adam_step)
            out[k] = params[k] - lr * m_hat / (mx.sqrt(v_hat) + self.args.adam_eps)
            self._adam_state[k] = (m, v)
        return out


# ==============================================================================
# LOGGING
# ==============================================================================

_log_file = None
_metrics_file = None

def log(msg: str) -> None:
    line = f"{msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    if _log_file:
        _log_file.write(line)
        _log_file.flush()

def jlog(d: dict) -> None:
    if _metrics_file:
        _metrics_file.write(json.dumps(d) + "\n")
        _metrics_file.flush()


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad):
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)
    return loss_value, tree_unflatten(list(grad_accum.items()))


def compute_churn(model, prev_ternary_signs, group_size):
    """Track fraction of ternary weights that flipped since last step."""
    if not prev_ternary_signs:
        return 0.0, {}
    total_flipped = 0
    total_weights = 0
    new_signs = {}
    for k, v in tree_flatten(model.parameters()):
        arr = v
        if arr.ndim != 2 or arr.size < 65536:
            continue
        f32 = np.array(arr.astype(mx.float32))
        pad = (group_size - f32.shape[1] % group_size) % group_size
        if pad:
            f32 = np.pad(f32, [(0, 0), (0, pad)])
        w_g = f32.reshape(-1, group_size)
        scale = np.maximum(np.abs(w_g).mean(axis=-1, keepdims=True), 1e-8)
        q = np.clip(np.round(w_g / scale), -1, 1).astype(np.int8).reshape(-1)
        new_signs[k] = q
        if k in prev_ternary_signs:
            total_flipped += (q != prev_ternary_signs[k]).sum()
            total_weights += len(q)
    return total_flipped / max(total_weights, 1), new_signs


def main() -> None:
    global _log_file, _metrics_file
    args = Hyperparameters()
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log_file = open(out_dir / f"{args.run_id}.txt", "w")
    _metrics_file = open(out_dir / f"{args.run_id}_metrics.jsonl", "w")
    atexit.register(lambda: _log_file and _log_file.close())
    atexit.register(lambda: _metrics_file and _metrics_file.close())

    log(f"ternary_training run_id:{args.run_id}")
    log(f"config: {args.num_layers}L/{args.model_dim}d, MLP 4x, group_size={args.bitnet_group_size}")

    # Validate tokenizer
    kind, vocab_sz, _ = validate_dataset_tokenizer_pair(args.data_path, args.tokenizer_path)
    sp_model = spm.SentencePieceProcessor(args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp_model, args.vocab_size)

    # Data
    train_loader = TokenLoader(args.train_files, log_fn=log)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    log(f"val_tokens:{len(val_tokens):,}")

    # Model
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        group_size=args.bitnet_group_size,
    )
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    log(f"params:{n_params:,}")

    # Optimizer
    flat_params = dict(tree_flatten(model.parameters()))
    optimizer = SplitOptimizers(flat_params, args)

    # Compile loss+grad
    def loss_fn(x, y):
        return model.loss(x, y, z_loss_coeff=args.z_loss_coeff)
    compiled_loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Compile eval loss
    def eval_loss_fn(x, y):
        logits = model(x)
        logits = model._apply_logit_processing(
            logits.reshape(-1, model.tok_emb.weight.shape[1]) @ model.tok_emb.weight.astype(logits.dtype).T
        )
        return nn.losses.cross_entropy(logits.astype(mx.float32), y.reshape(-1), reduction="mean")
    compiled_loss = mx.compile(eval_loss_fn)
    compiled_sliding_loss = None  # No sliding window for ternary smoke tests

    # Training
    best_val_bpb = float("inf")
    best_ckpt_path = None
    prev_ternary_signs = {}
    t0 = time.perf_counter()
    train_time_ms = 0.0
    max_wallclock_ms = args.max_wallclock_seconds * 1000.0 if args.max_wallclock_seconds > 0 else None
    stop_after_step = None
    step = 0

    for step in range(1, args.iterations + 1):
        if stop_after_step is not None and step > stop_after_step:
            break
        last_step = (step == args.iterations)
        lr_mul = args.lr_mul(step, train_time_ms)

        # Forward + backward
        for _ in range(args.grad_accum_steps):
            loss_value, grads_tree = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)

        if args.grad_clip_norm > 0:
            grads_tree = clip_grad_tree(grads_tree, args.grad_clip_norm)

        # Optimizer step
        flat_grads = dict(tree_flatten(grads_tree))
        flat_params = dict(tree_flatten(model.parameters()))
        new_params = optimizer.step(flat_params, flat_grads, step, lr_mul)
        model.update(tree_unflatten(list(new_params.items())))
        mx.eval(model.parameters())

        train_loss_value = loss_value.item()
        step_ms = 1000.0 * (time.perf_counter() - t0)
        train_time_ms = step_ms
        tok_s = args.train_batch_tokens / (step_ms / max(step, 1) / 1000.0)

        # Validation
        if (args.val_loss_every > 0 and step % args.val_loss_every == 0) or last_step:
            t_val = time.perf_counter()
            val_loss, val_bpb = eval_val(
                args, compiled_loss, val_tokens, base_bytes_lut,
                has_leading_space_lut, is_boundary_token_lut,
                log_fn=log, compiled_sliding_loss=compiled_sliding_loss,
            )
            log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms")
            jlog({"step": step, "val_loss": val_loss, "val_bpb": val_bpb, "train_time_ms": train_time_ms})
            if val_bpb < best_val_bpb and step > 0:
                best_val_bpb = val_bpb
                best_ckpt_path = out_dir / f"{args.run_id}_best.npz"
                flat_best = {k: v for k, v in tree_flatten(model.state)}
                mx.savez(str(best_ckpt_path), **flat_best)
                log(f"best_ckpt:step={step} val_bpb={val_bpb:.4f} saved:{best_ckpt_path}")

        # Logging
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            approx_train_time_ms = train_time_ms
            log(f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}")
            jlog({"step": step, "train_loss": train_loss_value, "lr_mul": lr_mul,
                  "step_ms": step_ms / max(step, 1), "tok_s": tok_s, "train_time_ms": approx_train_time_ms})

        if max_wallclock_ms is not None and stop_after_step is None and train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # ==============================================================================
    # FINAL SERIALIZATION + ROUNDTRIP EVAL
    # ==============================================================================

    # Pre-quant eval
    pre_val_loss, pre_val_bpb = eval_val(
        args, compiled_loss, val_tokens, base_bytes_lut,
        has_leading_space_lut, is_boundary_token_lut,
        log_fn=log, compiled_sliding_loss=compiled_sliding_loss,
    )
    log(f"pre_quant val_loss:{pre_val_loss:.4f} val_bpb:{pre_val_bpb:.4f}")

    # Save raw model
    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    # Ternary quantize + serialize
    quant_obj, quant_stats = quantize_ternary_state_dict(
        flat_state, group_size=args.bitnet_group_size,
        ternary_min_numel=args.ternary_min_numel)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = lzma.compress(quant_raw, preset=9)
    quant_path = out_dir / f"{args.run_id}_mlx_model.ternary.ptl"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    log(f"ternary_artifact:{quant_file_bytes} bytes ({quant_file_bytes/1e6:.2f} MB) "
        f"ternary_params:{quant_stats['ternary_params']:,} fp16_params:{quant_stats['fp16_params']:,}")

    # Roundtrip eval
    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = dequantize_ternary_state_dict(pickle.loads(lzma.decompress(quant_blob_disk)))
    model.update(tree_unflatten(list(quant_flat.items())))
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_loss, val_tokens, base_bytes_lut,
        has_leading_space_lut, is_boundary_token_lut,
        log_fn=log, compiled_sliding_loss=compiled_sliding_loss,
    )
    log(f"ternary_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")
    log(f"quant_gap_bpb:{q_val_bpb - pre_val_bpb:.6f}")

    # Summary
    summary = {
        "run_id": args.run_id,
        "val_loss_pre_quant": pre_val_loss,
        "val_bpb_pre_quant": pre_val_bpb,
        "val_loss": q_val_loss,
        "val_bpb": q_val_bpb,
        "quant_gap_bpb": q_val_bpb - pre_val_bpb,
        "quant_file_bytes": quant_file_bytes,
        "model_file_bytes": int(out_path.stat().st_size),
        "n_params": n_params,
        "train_time_ms": train_time_ms,
        "steps": step,
        "compressor": "lzma",
        "config": {k: getattr(args, k) for k in Hyperparameters.__annotations__},
    }
    summary_path = out_dir / f"{args.run_id}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"summary:{summary_path}")


if __name__ == "__main__":
    main()
