#!/usr/bin/env python3
"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""
from __future__ import annotations

import atexit
import glob
import json
import math
import os
import pickle
import signal
import sys
import time
import uuid
import zlib
from collections.abc import Callable

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# ==============================================================================
# SHARD FORMAT + COMPUTE DTYPE
# ==============================================================================

COMPUTE_DTYPE = mx.bfloat16

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap
class Hyperparameters:
    # Data / tokenizer.
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop. These defaults now mirror train_gpt.py on a single process.
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    # Validation always uses the full fineweb_val split.
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    # Cap the number of val tokens evaluated (0 = full split). Useful for fast smoke tests.
    val_max_tokens: int = int(os.environ.get("VAL_MAX_TOKENS", 0))
    # Sliding window eval stride. 0 = disabled (non-overlapping windows).
    # stride=64 gives each scored token up to (seq_len-stride) tokens of extra left context.
    val_sliding_stride: int = int(os.environ.get("VAL_SLIDING_STRIDE", "0"))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))
    # Chunk each logical MLX microbatch into smaller sub-batches to reduce peak
    # memory pressure without changing the effective optimizer batch.
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    # Force MLX to materialize the graph after every sub-batch, preventing lazy
    # graph buildup across accumulation steps. Keeps peak memory low on 16GB machines.
    # Disable on 32GB+ unified memory for better throughput (MLX_EAGER_EVAL=0).
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model (defaults match the current baseline setup).
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    # Per-layer MLP mult override, comma-separated (e.g. "3,3,3,3,3,3,3,2,2" for 9 layers).
    # Empty string = use uniform mlp_mult for all layers.
    mlp_mult_per_layer: str = os.environ.get("MLP_MULT_PER_LAYER", "")
    mlp_act: str = os.environ.get("MLP_ACT", "relu2")  # "relu2", "lrelu2", or "swiglu"
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    logit_temp: float = float(os.environ.get("LOGIT_TEMP", 1.0))  # Temperature scaling: T<1 sharpens, T>1 smooths
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Optimizer. We keep the same per-group defaults as train_gpt.py.
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-10))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    normuon: bool = bool(int(os.environ.get("NORMUON", "0")))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    muon_wd: float = float(os.environ.get("MUON_WD", 0.04))
    adam_wd: float = float(os.environ.get("ADAM_WD", 0.01))
    cautious_wd: bool = bool(int(os.environ.get("CAUTIOUS_WD", "1")))

    # BigramHash embedding
    bigram_vocab_size: int = int(os.environ.get("BIGRAM_VOCAB_SIZE", 4096))
    bigram_dim: int = int(os.environ.get("BIGRAM_DIM", 128))

    # Stochastic Weight Averaging
    swa_enabled: bool = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_start_frac: float = float(os.environ.get("SWA_START_FRAC", 0.5))
    swa_every: int = int(os.environ.get("SWA_EVERY", 50))

    # Bad batch detection: skip optimizer step if loss > threshold * EMA(loss)
    loss_skip_threshold: float = float(os.environ.get("LOSS_SKIP_THRESHOLD", 1.5))

    # Periodic checkpoint saving (0 = disabled, only best-val checkpoint).
    checkpoint_every: int = int(os.environ.get("CHECKPOINT_EVERY", 0))

    # Post-training quantization bit-widths (4, 5, 6, or 8).
    quant_attn_bits: int = int(os.environ.get("QUANT_ATTN_BITS", "6"))
    quant_mlp_bits: int = int(os.environ.get("QUANT_MLP_BITS", "6"))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

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
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear.gate,bigram.scale",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    # Background on Muon: https://kellerjordan.github.io/posts/muon/
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


# ==============================================================================
# TOKEN STREAMING / BATCHING
# ==============================================================================


class TokenStream:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(
                    f"WARNING: starting epoch:{self.epoch} "
                    f"dataset:{self.dataset_name} train_shards:{len(self.files)}"
                )
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    # Byte-fallback tokens in sentencepiece sp1024: IDs 3-258 are raw byte encodings.
    # Normal English text has <2% byte tokens; garbled/binary content has 40-60%+.
    BYTE_TOKEN_LO = 3
    BYTE_TOKEN_HI = 258
    MAX_BYTE_FRAC = float(os.environ.get("DATA_MAX_BYTE_FRAC", 0.2))

    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
        vocab_size: int = 1024,
    ):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)
        self.log_fn = log_fn
        self.skipped_seqs = 0

    def _is_garbage(self, chunk: np.ndarray) -> bool:
        byte_count = int(np.sum((chunk >= self.BYTE_TOKEN_LO) & (chunk <= self.BYTE_TOKEN_HI)))
        return byte_count > self.MAX_BYTE_FRAC * len(chunk)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        num_seqs = usable // seq_len
        seqs_x: list[np.ndarray] = []
        seqs_y: list[np.ndarray] = []
        while len(seqs_x) < num_seqs:
            chunk = self.stream.take(seq_len + 1)
            if self._is_garbage(chunk):
                self.skipped_seqs += 1
                if self.log_fn and self.skipped_seqs <= 20:
                    byte_count = int(np.sum((chunk >= self.BYTE_TOKEN_LO) & (chunk <= self.BYTE_TOKEN_HI)))
                    self.log_fn(f"data_filter:skip seq byte_tokens={byte_count}/{len(chunk)} total_skipped={self.skipped_seqs}")
                continue
            seqs_x.append(chunk[:-1])
            seqs_y.append(chunk[1:])
        x = np.stack(seqs_x).reshape(-1, seq_len)
        y = np.stack(seqs_y).reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


# ==============================================================================
# MODEL BLOCKS
# ==============================================================================

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    # MLX module wrapper around the functional RMSNorm helper so it composes nicely in blocks.
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    # - separate q/k/v projections
    # - RMSNorm on q and k before attention
    # - RoPE on q and k
    # - causal masked SDPA
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # Baseline MLP uses relu^2 instead of GELU/SiLU. It is cheap and works well in this setup.
    def __init__(self, dim: int, mlp_mult: int, act: str = "relu2"):
        super().__init__()
        self.act = act
        if act == "swiglu":
            # SwiGLU: 3 matrices (gate, up, down). At matched params with relu2's
            # 2-matrix MLP, hidden = dim * mlp_mult * 2 / 3 (rounded).
            hidden = int(dim * mlp_mult * 2 / 3)
            self.gate = CastedLinear(dim, hidden)
            self.fc = CastedLinear(dim, hidden)
            self.proj = CastedLinear(hidden, dim)
        else:
            hidden = dim * mlp_mult
            self.fc = CastedLinear(dim, hidden)
            self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        if self.act == "swiglu":
            return self.proj(nn.silu(self.gate(x)) * self.fc(x))
        h = self.fc(x)
        if self.act == "lrelu2":
            h = nn.leaky_relu(h, negative_slope=0.5)
        else:
            h = nn.relu(h)
        return self.proj(h * h)


class SmearGate(nn.Module):
    """Blend each token's embedding with the previous token's embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        g = mx.sigmoid(self.gate.astype(x.dtype))[None, None, :]
        x_prev = mx.concatenate([mx.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    """Hash consecutive token pairs into a learned embedding table."""
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        self.embed.weight = (mx.random.normal(self.embed.weight.shape) * 0.01).astype(self.embed.weight.dtype)
        self.proj = CastedLinear(bigram_dim, model_dim) if bigram_dim != model_dim else None
        if self.proj is not None:
            self.proj.weight = mx.zeros_like(self.proj.weight)
        self.scale = mx.array(0.05, dtype=mx.float32)

    def __call__(self, token_ids: mx.array) -> mx.array:
        t = token_ids.astype(mx.int32)
        mod = self.bigram_vocab_size - 1
        hashed = mx.concatenate([mx.full(t[:, :1].shape, mod, dtype=mx.int32),
                                  (36313 * t[:, 1:]) ^ (27191 * t[:, :-1])], axis=1) % mod
        h = self.embed(hashed)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.astype(h.dtype)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, mlp_act: str = "relu2"):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult, act=mlp_act)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 logit_chunk_tokens: int, logit_softcap: float, rope_base: float, tied_embed_init_std: float,
                 qk_gain_init: float, mlp_act: str = "relu2", mlp_mult_per_layer: list[int] | None = None,
                 bigram_vocab_size: int = 0, bigram_dim: int = 128, logit_temp: float = 1.0):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.logit_temp = logit_temp
        self.num_layers = num_layers

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            Block(dim, num_heads, num_kv_heads,
                  mlp_mult_per_layer[i] if mlp_mult_per_layer else mlp_mult,
                  rope_base, qk_gain_init, mlp_act=mlp_act)
            for i in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()
        self._init_weights(dim, num_layers, tied_embed_init_std)

    def _init_weights(self, dim: int, num_layers: int, tied_embed_init_std: float) -> None:
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)
        proj_scale = 1.0 / math.sqrt(2 * num_layers)
        for b in self.blocks:
            for mod in [b.attn.c_q, b.attn.c_k, b.attn.c_v, b.attn.proj, b.mlp.fc, b.mlp.proj]:
                w = mod.weight
                if w.ndim == 2 and min(w.shape) >= 64:
                    # Orthogonal init via numpy QR
                    a = np.random.randn(w.shape[0], w.shape[1]).astype(np.float32)
                    q, _ = np.linalg.qr(a if a.shape[0] >= a.shape[1] else a.T)
                    q = q[:w.shape[0], :w.shape[1]] if a.shape[0] >= a.shape[1] else q[:w.shape[1], :w.shape[0]].T
                    mod.weight = mx.array(q, dtype=w.dtype)
            # Output projections scaled by 1/sqrt(2*num_layers)
            b.attn.proj.weight = (b.attn.proj.weight * proj_scale).astype(b.attn.proj.weight.dtype)
            b.mlp.proj.weight = (b.mlp.proj.weight * proj_scale).astype(b.mlp.proj.weight.dtype)
            if hasattr(b.mlp, 'gate'):  # SwiGLU gate projection
                w = b.mlp.gate.weight
                if w.ndim == 2 and min(w.shape) >= 64:
                    a = np.random.randn(w.shape[0], w.shape[1]).astype(np.float32)
                    q, _ = np.linalg.qr(a if a.shape[0] >= a.shape[1] else a.T)
                    q = q[:w.shape[0], :w.shape[1]] if a.shape[0] >= a.shape[1] else q[:w.shape[1], :w.shape[0]].T
                    b.mlp.gate.weight = mx.array(q, dtype=w.dtype)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = rms_norm(x)
        x = self.smear(x)
        x0 = x
        skips: list[mx.array] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def _apply_logit_processing(self, logits: mx.array) -> mx.array:
        logits = self.softcap(logits)
        if self.logit_temp != 1.0:
            logits = logits / self.logit_temp
        return logits

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        # Cross-entropy over flattened tokens. We keep optional logit chunking because it is a useful
        # memory knob on Macs, but the common path is chunk_tokens=0 (single matmul + CE).
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits = self._apply_logit_processing(x @ self.tok_emb.weight.astype(x.dtype).T)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits = self._apply_logit_processing(x[s:e] @ self.tok_emb.weight.astype(x.dtype).T)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)

    def loss_last_n(self, input_ids: mx.array, target_ids: mx.array, n: int) -> mx.array:
        """Like loss() but scores only the last n tokens of each sequence. Used for sliding window eval."""
        x = self(input_ids)[:, -n:, :].reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids[:, -n:].reshape(-1)
        logits = self._apply_logit_processing(x @ self.tok_emb.weight.astype(x.dtype).T)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

# ==============================================================================
# OPTIMIZERS (MUON + ADAM SPLIT)
# ==============================================================================
class Muon:
    # Muon applies SGD-momentum to matrix gradients, then orthogonalizes the result before the
    # parameter update.
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], step: int, lr_mul: float) -> dict[str, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            if self.args.normuon:
                # NorMuon: per-row normalization of the update
                row_norms = mx.sqrt(mx.sum(g_ortho * g_ortho, axis=-1, keepdims=True) + 1e-8)
                g_ortho = g_ortho / row_norms
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            if self.args.muon_wd > 0:
                if self.args.cautious_wd:
                    # Cautious WD: only decay where gradient and param agree in sign
                    mask = (g * p > 0).astype(p.dtype)
                    p = p * (1.0 - lr * self.args.muon_wd * mask)
                else:
                    p = p * (1.0 - lr * self.args.muon_wd)
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    # - embeddings: Adam with the tied-embedding LR
    # - block matrices (2D): Muon
    # - block scalars + skip weights: Adam
    # This preserves the high-level optimization behavior even though MLX internals differ.
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_keys = ["tok_emb.weight"]
        # BigramHash embed.weight uses the same Adam group as tok_emb
        if "bigram.embed.weight" in params:
            self.embed_keys.append("bigram.embed.weight")
        self.matrix_keys = [
            k
            for k, p in params.items()
            if p.ndim == 2 and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
               and k not in self.embed_keys
        ]
        self.scalar_keys = [
            k
            for k, p in params.items()
            if k not in self.embed_keys and k not in self.matrix_keys
        ]

        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.AdamW(
            learning_rate=args.tied_embed_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            weight_decay=args.adam_wd,
        )
        self.adam_scalar = optim.AdamW(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            weight_decay=args.adam_wd,
        )

    def step(self, model: GPT, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        embed_grads = {k: grads[k] for k in self.embed_keys if k in grads}
        embed_params = {k: params[k] for k in self.embed_keys if k in params}
        updated.update(self.adam_embed.apply_gradients(embed_grads, embed_params))

        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys if k in grads}
        scalar_params = {k: params[k] for k in self.scalar_keys if k in params}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))

# ==============================================================================
# QUANTIZATION (INT8 + ZLIB)
# ==============================================================================
# Mixed quantization: int6 for MLP+attention weights, int8 for the rest.
# tok_emb kept in fp16 (quantization errors compound through both input and output paths).
# - per-row scaling for 2D tensors
# - per-tensor scaling for other floats
# - fp16 passthrough for small float tensors and embeddings
# - exact passthrough for non-floats

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_int6_per_row(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to int6 range [-32, 31] stored as int8. Lower precision but better compression."""
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        row_max = np.abs(f32).max(axis=1)
        scale = np.maximum(row_max / 31.0, 1e-12).astype(np.float16)
        q = np.clip(np.round(f32 / scale.astype(np.float32)[:, None]), -32, 31).astype(np.int8)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale)
    amax = float(np.abs(f32).max()) if f32.size else 0.0
    scale = np.array(max(amax / 31.0, 1e-12), dtype=np.float16)
    q = np.clip(np.round(f32 / float(scale)), -32, 31).astype(np.int8)
    return np.ascontiguousarray(q), scale


def quantize_int5_per_row(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to int5 range [-16, 15] stored as int8."""
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


def quantize_int4_per_row(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to int4 range [-8, 7] stored as int8."""
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


_QUANT_FN = {4: quantize_int4_per_row, 5: quantize_int5_per_row, 6: quantize_int6_per_row, 8: quantize_float_array}


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name:
        return "attn"
    if "bigram." in name:
        return "bigram"
    return "other"

# Name patterns for tensors kept in fp16 (not quantized)
FP16_KEEP_NAME_PATTERNS = ("tok_emb",)


def quantize_state_dict_int8(flat_state: dict[str, mx.array], int6_cats: set[str] = frozenset({"mlp", "attn"}),
                              cat_bits: dict[str, int] | None = None) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
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

        # Embeddings kept in fp16 (quantization errors compound through tied input/output)
        if any(pattern in name for pattern in FP16_KEEP_NAME_PATTERNS):
            kept = np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=np.float16))
            passthrough[name] = kept
            passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        # Small float tensors kept directly (control tensors, scales, etc.)
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
        bits = (cat_bits or {}).get(cat)
        if bits is None:
            bits = 6 if cat in int6_cats else 8
        quant_fn = _QUANT_FN[bits]
        q, s = quant_fn(arr)
        if bits < 8:
            qmeta[name] = {"scheme": f"int{bits}_per_row", "axis": 0}
        elif s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
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


def dequantize_state_dict_int8(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            # Broadcast the saved row scale back across trailing dimensions.
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    return out


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    # The shard directory and tokenizer are coupled: val_bpb is only meaningful if we
    # decode bytes with the exact tokenizer that produced the shards. The manifest
    # lets the training script fail fast on accidental dataset/tokenizer mismatches.
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(file) for file in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader: TokenLoader,
    compiled_loss_and_grad,
) -> tuple[mx.array, dict]:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)  # materialize each chunk to cap peak memory
    return loss_value, tree_unflatten(list(grad_accum.items()))


def eval_val(
    args: Hyperparameters,
    compiled_loss,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
    compiled_sliding_loss=None,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    stride = args.val_sliding_stride if compiled_sliding_loss is not None else 0
    seq_len = args.train_seq_len

    if stride > 0:
        # Sliding window: windows of length seq_len stepped by stride.
        # Only the last `stride` tokens of each window are scored, giving each token
        # up to (seq_len - stride) tokens of left context beyond what non-sliding sees.
        val_len = val_tokens.size
        if args.val_max_tokens > 0:
            val_len = min(val_len, args.val_max_tokens + seq_len)
        # Number of complete windows (need seq_len+1 tokens each for input+target)
        total_windows = max((val_len - seq_len - 1) // stride + 1, 1)
        val_batch_seqs = max(args.val_batch_size // args.grad_accum_steps // seq_len, 1)
        total_batches = max((total_windows + val_batch_seqs - 1) // val_batch_seqs, 1)
        total_loss_sum = 0.0
        total_tokens = 0.0
        total_bytes = 0.0
        for batch_idx, win_start in enumerate(range(0, total_windows, val_batch_seqs), start=1):
            win_end = min(win_start + val_batch_seqs, total_windows)
            x_np = np.stack([val_tokens[w * stride: w * stride + seq_len] for w in range(win_start, win_end)])
            y_np = np.stack([val_tokens[w * stride + 1: w * stride + seq_len + 1] for w in range(win_start, win_end)])
            x = mx.array(x_np, dtype=mx.int32)
            y = mx.array(y_np, dtype=mx.int32)
            scored_tokens = float(x_np.shape[0] * stride)
            batch_loss = compiled_sliding_loss(x, y).astype(mx.float32)
            mx.eval(batch_loss)
            total_loss_sum += float(batch_loss.item()) * scored_tokens
            prev_scored = x_np[:, -stride:].reshape(-1)
            tgt_scored = y_np[:, -stride:].reshape(-1)
            bytes_np = base_bytes_lut[tgt_scored].astype(np.int16, copy=True)
            bytes_np += (
                has_leading_space_lut[tgt_scored] & ~is_boundary_token_lut[prev_scored]
            ).astype(np.int16, copy=False)
            total_tokens += scored_tokens
            total_bytes += float(bytes_np.astype(np.float64).sum())
            if log_fn is not None and total_batches > 1 and (
                batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
            ):
                log_fn(f"val_progress:{batch_idx}/{total_batches}")
    else:
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        if val_batch_tokens < seq_len:
            raise ValueError(
                "VAL_BATCH_SIZE must provide at least one sequence; "
                f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                f"TRAIN_SEQ_LEN={seq_len}"
            )
        val_batch_seqs = val_batch_tokens // seq_len
        total_seqs = (val_tokens.size - 1) // seq_len
        if args.val_max_tokens > 0:
            total_seqs = min(total_seqs, args.val_max_tokens // seq_len)
        total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
        total_loss_sum = 0.0
        total_tokens = 0.0
        total_bytes = 0.0
        for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
            batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            chunk = val_tokens[raw_start:raw_end]
            x_np = chunk[:-1].reshape(-1, seq_len)
            y_np = chunk[1:].reshape(-1, seq_len)
            x = mx.array(x_np, dtype=mx.int32)
            y = mx.array(y_np, dtype=mx.int32)
            chunk_token_count = float(y.size)
            batch_loss = compiled_loss(x, y).astype(mx.float32)
            mx.eval(batch_loss)
            total_loss_sum += float(batch_loss.item()) * chunk_token_count
            prev_ids = x_np.reshape(-1)
            tgt_ids = y_np.reshape(-1)
            bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
            bytes_np += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).astype(np.int16, copy=False)
            total_tokens += chunk_token_count
            total_bytes += float(bytes_np.astype(np.float64).sum())
            if log_fn is not None and total_batches > 1 and (
                batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
            ):
                log_fn(f"val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb

# -----------------------------
# TRAINING
# -----------------------------

def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        total_sq += float(np.sum(np.square(_np_float32(grad)), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


def main() -> None:
    # ==============================================================================
    # TOKENIZER + VALIDATION METRIC SETUP
    # ==============================================================================
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg, flush=True)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    metrics_file = out_dir / f"{args.run_id}_metrics.jsonl"
    def jlog(d: dict) -> None:
        with metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d) + "\n")

    _exit_reason = "unknown"

    def _on_exit():
        log(f"exit:reason={_exit_reason} pid={os.getpid()}")

    def _on_signal(signum, frame):
        nonlocal _exit_reason
        _exit_reason = f"signal={signal.Signals(signum).name}"
        log(f"exit:caught {_exit_reason}")
        sys.exit(128 + signum)

    atexit.register(_on_exit)
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _on_signal)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)
    log(f"run_id:{args.run_id} pid:{os.getpid()}")

    if not args.tie_embeddings:
        raise NotImplementedError("train_gpt_mlx.py only supports tied embeddings")
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )

    # ==============================================================================
    # TRAINING SETUP
    # ==============================================================================
    mx.random.seed(args.seed)

    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name, vocab_size=args.vocab_size)

    # ==============================================================================
    # MODEL + OPTIMIZER SETUP
    # ==============================================================================
    per_layer = [int(x) for x in args.mlp_mult_per_layer.split(",") if x] if args.mlp_mult_per_layer else None
    if per_layer and len(per_layer) != args.num_layers:
        raise ValueError(f"MLP_MULT_PER_LAYER has {len(per_layer)} entries but NUM_LAYERS={args.num_layers}")
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
        mlp_act=args.mlp_act,
        mlp_mult_per_layer=per_layer,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        logit_temp=args.logit_temp,
    )
    resume_ckpt = os.environ.get("RESUME_CHECKPOINT", "")
    if resume_ckpt:
        ckpt_state = dict(mx.load(resume_ckpt))
        model.update(tree_unflatten(list(ckpt_state.items())))
        log(f"resumed_from:{resume_ckpt} keys:{len(ckpt_state)}")

    opt = SplitOptimizers(model, args)

    # ==============================================================================
    # COMPILED TRAIN / EVAL FUNCTIONS (MLX)
    # ==============================================================================
    # The crucial MLX detail is capture scope: this model contains non-trainable arrays too (for example
    # inside RoPE modules), so compiling only against trainable parameters throws "uncaptured inputs".
    # Compiling the model-bound functions and capturing the full model state fixes that while still
    # returning gradients only for trainable parameters via nn.value_and_grad(...).
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state,
        outputs=model.state,
    )
    compiled_sliding_loss = None
    if args.val_sliding_stride > 0:
        _stride = args.val_sliding_stride
        compiled_sliding_loss = mx.compile(
            lambda x, y: model.loss_last_n(x, y, _stride),
            inputs=model.state,
            outputs=model.state,
        )

    # Print config once so logs are self-describing.
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    log(f"mlx_version:{mx.__version__}")
    log(f"train_loader:shards pattern={args.train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    if expected_train_files is None:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif actual_train_files < expected_train_files:
        log(
            f"WARNING: train_loader:subset dataset:{dataset_name} "
            f"train_shards:{actual_train_files}/{expected_train_files} "
            f"new epochs will arrive sooner than the full dataset"
        )
    else:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log(f"tokenizer_path:{args.tokenizer_path}")
    log(
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
        f"val_batch_size:{args.val_batch_size} val_max_tokens:{args.val_max_tokens or 'full'} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log(f"compute_dtype:{COMPUTE_DTYPE} compile:True")
    log(
        f"dtypes tok_emb:{model.tok_emb.weight.dtype} "
        f"linear_weight:{model.blocks[0].attn.c_q.weight.dtype} "
        f"skip_weights:{model.skip_weights.dtype}"
    )
    log(f"bigram:{args.bigram_vocab_size}x{args.bigram_dim} swa:{args.swa_enabled} muon_wd:{args.muon_wd} compressor:{_COMPRESSOR}")

    # ==============================================================================
    # TRAINING LOOP
    # ==============================================================================
    if args.warmup_steps > 0:
        # Warmup should only prime MLX compile/allocation paths. Updating parameters here forces us
        # to snapshot and restore model/optimizer state, which is expensive on unified-memory Macs.
        # Instead we run the real train shapes, force the loss/grads to materialize, and then reset
        # the loader so measured training still starts from the true init and token window.
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                accum = accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

        # Prime the standalone eval graph once too. It is compiled separately from value_and_grad.
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        if val_batch_tokens < args.train_seq_len:
            raise ValueError(
                "VAL_BATCH_SIZE must provide at least one sequence; "
                f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                f"TRAIN_SEQ_LEN={args.train_seq_len}"
            )
        warm_val_seqs = min(val_batch_tokens // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
        warm_chunk = val_tokens[: warm_val_seqs * args.train_seq_len + 1]
        x_val = mx.array(warm_chunk[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
        y_val = mx.array(warm_chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
        warm_val_loss = compiled_loss(x_val, y_val)
        mx.eval(warm_val_loss)
        mx.synchronize()

        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name, vocab_size=args.vocab_size)

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    best_val_bpb = float("inf")
    best_ckpt_path: Path | None = None
    swa_state: dict[str, np.ndarray] | None = None
    swa_count = 0
    loss_ema = 0.0
    loss_skip_threshold = args.loss_skip_threshold
    skipped_steps = 0
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            # Validation always scans the same fixed full validation split.
            val_loss, val_bpb = eval_val(
                args,
                compiled_loss,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log_fn=log,
                compiled_sliding_loss=compiled_sliding_loss,
            )
            if step % 25 == 0 or last_step:
                log(
                    f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                )
            jlog({"step": step, "val_loss": val_loss, "val_bpb": val_bpb, "train_time_ms": train_time_ms})
            if val_bpb < best_val_bpb and step > 0:
                best_val_bpb = val_bpb
                best_ckpt_path = out_dir / f"{args.run_id}_best.npz"
                flat_best = {k: v for k, v in tree_flatten(model.state)}
                mx.savez(str(best_ckpt_path), **flat_best)
                log(f"best_ckpt:step={step} val_bpb={val_bpb:.4f} saved:{best_ckpt_path}")
            t0 = time.perf_counter()
        # Periodic checkpoint (independent of val).
        if args.checkpoint_every > 0 and step > 0 and step % args.checkpoint_every == 0 and not last_step:
            ckpt_path = out_dir / f"{args.run_id}_step{step}.npz"
            flat_ckpt = {k: v for k, v in tree_flatten(model.state)}
            mx.savez(str(ckpt_path), **flat_ckpt)
            log(f"periodic_ckpt:step={step} saved:{ckpt_path}")
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale
            if args.mlx_eager_eval:
                mx.eval(train_loss, accum)  # materialize each microbatch to cap peak memory

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())

        # Skip optimizer step if loss is anomalously high (bad batch detection)
        if loss_ema > 0 and train_loss_value > loss_skip_threshold * loss_ema:
            skipped_steps += 1
            log(f"skip_bad_batch:step={step} loss={train_loss_value:.4f} ema={loss_ema:.4f} threshold={loss_skip_threshold * loss_ema:.4f}")
        else:
            opt.step(model, grads, step=step, lr_mul=lr_mul)
        loss_ema = 0.99 * loss_ema + 0.01 * train_loss_value if loss_ema > 0 else train_loss_value
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1

        # SWA: accumulate weight snapshots during warmdown
        scale = args.lr_mul(step, approx_train_time_ms)
        if args.swa_enabled and scale < args.swa_start_frac and step % args.swa_every == 0:
            current_flat = {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(model.state)}
            if swa_state is None:
                swa_state = {k: v.copy() for k, v in current_flat.items()}
                swa_count = 1
                log(f"swa:start step:{step}")
            else:
                for k in swa_state:
                    swa_state[k] += current_flat[k]
                swa_count += 1

        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}"
            )
            jlog({"step": step, "train_loss": train_loss_value, "lr_mul": lr_mul,
                  "step_ms": step_ms, "tok_s": tok_s, "train_time_ms": approx_train_time_ms})
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # Apply SWA if collected
    if args.swa_enabled and swa_state is not None and swa_count > 1:
        log(f"swa:applying averaged {swa_count} checkpoints")
        current_state = dict(tree_flatten(model.state))
        swa_flat = {
            k: mx.array(v / swa_count).astype(current_state[k].dtype)
            for k, v in swa_state.items()
        }
        model.update(tree_unflatten(list(swa_flat.items())))

    # ==============================================================================
    # FINAL SERIALIZATION + QUANTIZED ROUNDTRIP EVAL
    # ==============================================================================
    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    cat_bits = {"attn": args.quant_attn_bits, "mlp": args.quant_mlp_bits}
    quant_obj, quant_stats = quantize_state_dict_int8(flat_state, cat_bits=cat_bits)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    if _COMPRESSOR == "zstd":
        cctx = zstandard.ZstdCompressor(level=22)
        quant_blob = cctx.compress(quant_raw)
    else:
        quant_blob = zlib.compress(quant_raw, level=9)
    quant_serialized_bytes = len(quant_raw)
    qbits_label = f"a{args.quant_attn_bits}m{args.quant_mlp_bits}" if args.quant_attn_bits != args.quant_mlp_bits else f"int{args.quant_attn_bits}"
    quant_path = out_dir / f"{args.run_id}_mlx_model.{qbits_label}.pt{_COMPRESSOR[0]}"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    log(
        f"serialized_model_{_COMPRESSOR}:{quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x)"
    )

    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    if _COMPRESSOR == "zstd":
        dctx = zstandard.ZstdDecompressor()
        quant_flat = dequantize_state_dict_int8(pickle.loads(dctx.decompress(quant_blob_disk)))
    else:
        quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob_disk)))
    model.update(tree_unflatten(list(quant_flat.items())))
    q_t0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        compiled_loss,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_fn=log,
        compiled_sliding_loss=compiled_sliding_loss,
    )
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    log(f"final_quant_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")
    log(f"final_quant_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Structured run summary for post-run analysis
    summary = {
        "run_id": args.run_id,
        "val_loss": q_val_loss,
        "val_bpb": q_val_bpb,
        "quant_file_bytes": quant_file_bytes,
        "score": q_val_bpb * quant_file_bytes / (1024 * 1024),
        "model_file_bytes": int(out_path.stat().st_size),
        "n_params": n_params,
        "train_time_ms": train_time_ms,
        "steps": step,
        "compressor": _COMPRESSOR,
        "config": {k: getattr(args, k) for k in Hyperparameters.__annotations__},
    }
    summary_path = out_dir / f"{args.run_id}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"summary:{summary_path} score:{summary['score']:.4f}")

    _exit_reason = "clean"


if __name__ == "__main__":
    main()
