#!/usr/bin/env python3
"""
correction_ce.py — Train correction layers on actual CE task loss (not MSE error recovery).

Two modes:
  frozen: Freeze quantized base model, train only correction layers on CE.
  joint:  Train correction layers + base weights (with fake-quant STE) on CE.

Correction layer placement is determined automatically via compound error
attribution (marginal KL impact), or overridden with --correction-layers.

Usage:
    # Auto-select correction layers, frozen mode:
    NUM_LAYERS=11 MLP_MULT=4.5 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \\
    QUANT_ATTN_BITS=5 QUANT_MLP_BITS=5 \\
    .venv/bin/python3 scripts/correction_ce.py logs/checkpoint.npz --mode frozen

    # Joint mode with explicit layers:
    ... --mode joint --correction-layers 0,5,8

    # Eval-only (measure baseline, no training):
    ... --mode frozen --steps 0
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import re
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
    quantize_state_dict_int8, dequantize_state_dict_int8, rms_norm,
    _classify_param, CONTROL_TENSOR_NAME_PATTERNS, FP16_KEEP_NAME_PATTERNS,
    INT8_KEEP_FLOAT_MAX_NUMEL, load_data_shard,
)
from scripts.error_attribution import (
    compute_logit_kl_impact, print_kl_impact_table, forward_collect_all,
)
from scripts.float_polish import (
    _fake_quant_per_row, _BITS_TO_QMAX, build_param_bits_map,
)

log = logging.getLogger("correction_ce")


# =============================================================================
# Correction net (same as ptq_correction.py but takes h, not error)
# =============================================================================

class CorrectionNet(nn.Module):
    """Additive correction: h_out = h + correction(h). Zero-init output → identity start."""

    def __init__(self, dim: int, hidden: int = 0):
        super().__init__()
        self.hidden = hidden
        if hidden > 0:
            self.w1 = nn.Linear(dim, hidden)
            self.w2 = nn.Linear(hidden, dim)
            self.w2.weight = mx.zeros_like(self.w2.weight)
            self.w2.bias = mx.zeros_like(self.w2.bias)
        else:
            self.linear = nn.Linear(dim, dim)
            self.linear.weight = mx.zeros_like(self.linear.weight)
            self.linear.bias = mx.zeros_like(self.linear.bias)

    def __call__(self, h: mx.array) -> mx.array:
        if self.hidden > 0:
            return self.w2(nn.relu(self.w1(h))).astype(h.dtype)
        return self.linear(h).astype(h.dtype)


# =============================================================================
# Attach corrections to model
# =============================================================================

def attach_corrections(model, correction_layers: list[int], dim: int, hidden: int = 0):
    """Create CorrectionNets and attach them as model sub-modules.

    Stored as model.correction_0, model.correction_1, ... so MLX discovers them
    in the parameter tree. model._correction_layer_indices maps correction index
    to block index.
    """
    model._correction_layer_indices = correction_layers
    model._n_corrections = len(correction_layers)
    for idx, layer in enumerate(correction_layers):
        setattr(model, f"correction_{idx}", CorrectionNet(dim, hidden))
    mx.eval(model.parameters())


def get_correction_map(model) -> dict[int, CorrectionNet]:
    """Return {block_index: CorrectionNet} mapping."""
    return {
        layer: getattr(model, f"correction_{idx}")
        for idx, layer in enumerate(model._correction_layer_indices)
    }


# =============================================================================
# Forward with corrections (mirrors GPT.__call__ lines 583-599)
# =============================================================================

def forward_ce_with_corrections(model, input_ids: mx.array) -> mx.array:
    """GPT forward pass with correction layers injected. Returns final hidden."""
    correction_map = get_correction_map(model)

    x = model.tok_emb(input_ids).astype(COMPUTE_DTYPE)
    if model.bigram is not None:
        x = x + model.bigram(input_ids)
    x = rms_norm(x)
    x = model.smear(x)
    x0 = x
    skips: list[mx.array] = []

    for i in range(model.num_encoder_layers):
        x = model.blocks[i](x, x0)
        if i in correction_map:
            x = x + correction_map[i](x)
        skips.append(x)
    for i in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
        block_idx = model.num_encoder_layers + i
        x = model.blocks[block_idx](x, x0)
        if block_idx in correction_map:
            x = x + correction_map[block_idx](x)
    return model.final_norm(x)


def loss_ce_corrected(model, input_ids: mx.array, target_ids: mx.array) -> mx.array:
    """CE loss using corrected forward pass."""
    h = forward_ce_with_corrections(model, input_ids)
    h = h.reshape(-1, model.tok_emb.weight.shape[1])
    logits = model._apply_logit_processing(h @ model.tok_emb.weight.astype(h.dtype).T)
    return nn.losses.cross_entropy(logits.astype(mx.float32), target_ids.reshape(-1), reduction="mean")


# =============================================================================
# Fake-quant wrapper for joint mode
# =============================================================================

def make_loss_fn_with_fake_quant(model, param_bits: dict[str, int]):
    """Wrap loss_ce_corrected with STE fake-quant on base weights."""

    def loss_fn(x, y):
        originals = {}
        for name, bits in param_bits.items():
            qmax = _BITS_TO_QMAX[bits]
            keys = name.split(".")
            obj = model
            for k in keys[:-1]:
                if k.isdigit():
                    obj = obj[int(k)]
                else:
                    obj = getattr(obj, k)
            param_name = keys[-1]
            w = getattr(obj, param_name)
            originals[name] = w
            setattr(obj, param_name, _fake_quant_per_row(w, qmax))

        loss = loss_ce_corrected(model, x, y)

        for name, w in originals.items():
            keys = name.split(".")
            obj = model
            for k in keys[:-1]:
                if k.isdigit():
                    obj = obj[int(k)]
                else:
                    obj = getattr(obj, k)
            setattr(obj, keys[-1], w)

        return loss

    return loss_fn


# =============================================================================
# Sensitivity analysis for correction placement
# =============================================================================

def select_correction_layers(model_float, model_quant, val_tokens, n_corrections: int,
                              n_seqs: int = 4, seq_len: int = 512) -> list[int]:
    """Run compound error attribution and return top-N layers by marginal KL."""
    import sys
    log.info(f"\n=== Sensitivity analysis ({n_seqs} seqs × {seq_len} tokens) ===")
    sys.stdout.flush()
    res = compute_logit_kl_impact(model_float, model_quant, val_tokens, n_seqs=n_seqs, seq_len=seq_len)
    print_kl_impact_table(res)
    sys.stdout.flush()

    top_layers = [int(x) for x in np.argsort(res["marginal_kl"])[::-1][:n_corrections]]
    top_layers.sort()
    log.info(f"\nSelected correction layers (top-{n_corrections} by marginal KL): {top_layers}")
    return top_layers


# =============================================================================
# Data loading
# =============================================================================

def load_train_tokens(hparams, max_tokens=1_000_000):
    """Load a subset of train tokens for correction training."""
    import glob
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


# =============================================================================
# Training loop
# =============================================================================

def train_corrections_ce(model, train_tokens, hparams, mode: str,
                          param_bits: dict[str, int] | None = None,
                          n_steps: int = 500, lr: float = 3e-4,
                          seq_len: int = 1024, batch_seqs: int = 4,
                          log_every: int = 50):
    """Train correction layers (and optionally base weights) on CE loss."""

    # Freeze everything, then selectively unfreeze
    model.freeze()

    # Always unfreeze corrections
    for idx in range(model._n_corrections):
        getattr(model, f"correction_{idx}").unfreeze()

    if mode == "joint":
        assert param_bits is not None
        for name in param_bits:
            keys = name.split(".")
            obj = model
            for k in keys[:-1]:
                if k.isdigit():
                    obj = obj[int(k)]
                else:
                    obj = getattr(obj, k)
            obj.unfreeze(keys=[keys[-1]])

    trainable = dict(tree_flatten(model.trainable_parameters()))
    n_trainable = sum(int(v.size) for v in trainable.values())
    n_correction = sum(int(v.size) for k, v in trainable.items() if "correction_" in k)
    log.info(f"Trainable: {n_trainable:,} params ({len(trainable)} tensors)")
    log.info(f"  Correction params: {n_correction:,}")
    if mode == "joint":
        log.info(f"  Base model params: {n_trainable - n_correction:,}")

    # Build loss function
    if mode == "joint":
        loss_fn = make_loss_fn_with_fake_quant(model, param_bits)
    else:
        loss_fn = lambda x, y: loss_ce_corrected(model, x, y)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Cosine decay LR
    def get_lr(step):
        if n_steps <= 1:
            return lr
        return lr * 0.5 * (1.0 + math.cos(math.pi * step / (n_steps - 1)))

    optimizer = optim.Adam(learning_rate=lr)

    max_start = len(train_tokens) - (batch_seqs * seq_len + 1)
    if max_start <= 0:
        raise ValueError(f"Not enough tokens ({len(train_tokens)}) for batch={batch_seqs}×{seq_len}")

    history = []
    best_loss = float("inf")
    best_state = None
    t0 = time.time()

    for step in range(n_steps):
        optimizer.learning_rate = mx.array(get_lr(step))

        start = np.random.randint(0, max_start)
        n_tok = batch_seqs * seq_len + 1
        chunk = train_tokens[start:start + n_tok]
        x = mx.array(chunk[:batch_seqs * seq_len].reshape(batch_seqs, seq_len))
        y = mx.array(chunk[1:batch_seqs * seq_len + 1].reshape(batch_seqs, seq_len))

        loss, grads = loss_and_grad(x, y)

        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.trainable_parameters()))
        updated = optimizer.apply_gradients(flat_grads, flat_params)
        model.update(tree_unflatten(list(updated.items())))
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        history.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
            mx.eval(list(best_state.values()))

        if step % log_every == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            log.info(f"  step {step:>4d}/{n_steps}  loss={loss_val:.6f}  "
                     f"lr={get_lr(step):.2e}  best={best_loss:.6f}  elapsed={elapsed:.1f}s")

    # Restore best checkpoint
    if best_state is not None:
        model.update(tree_unflatten(list(best_state.items())))
        mx.eval(model.parameters())
        log.info(f"Restored best checkpoint (loss={best_loss:.6f})")

    log.info(f"Training done in {time.time() - t0:.1f}s  best_loss={best_loss:.6f}")
    return history


# =============================================================================
# Evaluation
# =============================================================================

def quick_eval_corrected(model, hparams, val_tokens, n_seqs: int = 32,
                          seq_len: int = 1024) -> float:
    """Quick CE evaluation on val tokens using corrected forward."""
    total_loss = 0.0
    total_tokens = 0
    for s in range(n_seqs):
        tokens = val_tokens[s * seq_len : (s + 1) * seq_len + 1]
        if len(tokens) < seq_len + 1:
            break
        x = mx.array(tokens[:seq_len][np.newaxis, :])
        y = mx.array(tokens[1:seq_len + 1][np.newaxis, :])

        h = forward_ce_with_corrections(model, x)
        h = h.reshape(-1, model.tok_emb.weight.shape[1])
        logits = model._apply_logit_processing(h @ model.tok_emb.weight.astype(h.dtype).T)
        loss = nn.losses.cross_entropy(logits.astype(mx.float32), y.reshape(-1), reduction="sum")
        mx.eval(loss)

        total_loss += loss.item()
        total_tokens += seq_len

    val_loss = total_loss / total_tokens
    bpt = val_loss / math.log(2)
    log.info(f"  val_loss={val_loss:.4f}  bits_per_token={bpt:.4f}  ({total_tokens:,} tokens)")
    return val_loss


# =============================================================================
# Model builder (same pattern as ptq_correction.py)
# =============================================================================

def build_model(hparams):
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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CE-trained correction layers")
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument("--mode", choices=["frozen", "joint"], required=True)
    parser.add_argument("--correction-layers", type=str, default=None,
                        help="Comma-separated layer indices (overrides auto-selection)")
    parser.add_argument("--n-corrections", type=int, default=3,
                        help="Number of correction layers to auto-select")
    parser.add_argument("--hidden-size", type=int, default=0,
                        help="CorrectionNet bottleneck (0=linear)")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-seqs", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-eval-seqs", type=int, default=32)
    parser.add_argument("--n-sensitivity-seqs", type=int, default=4)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--save", type=str, default=None,
                        help="Save correction weights to this .npz path")
    parser.add_argument("--filter", type=str, default=None,
                        help="(joint mode) regex filter for base weights to unfreeze")
    args = parser.parse_args()

    hparams = Hyperparameters()

    # Logging setup
    global log
    log = logging.getLogger("correction_ce")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if args.log_file is None:
        h_tag = f"h{args.hidden_size}" if args.hidden_size > 0 else "linear"
        bits = hparams.quant_attn_bits
        args.log_file = f"logs/correction_ce_{args.mode}_{h_tag}_int{bits}_{args.steps}s.txt"

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    fh = logging.FileHandler(args.log_file, mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.info(f"Logging to {args.log_file}")

    log.info(f"Config: {hparams.num_layers}L/{hparams.model_dim}d, MLP {hparams.mlp_mult}x, "
             f"act={hparams.mlp_act}, quant=a{hparams.quant_attn_bits}m{hparams.quant_mlp_bits}")
    log.info(f"Mode: {args.mode}, hidden={args.hidden_size}, steps={args.steps}, lr={args.lr}")

    # --- Build float model ---
    log.info(f"\nLoading checkpoint: {args.checkpoint}")
    model_float = build_model(hparams)
    flat = dict(mx.load(args.checkpoint))
    model_float.update(tree_unflatten(list(flat.items())))
    mx.eval(model_float.parameters())

    # --- Build quantized model ---
    log.info("Quantizing model...")
    cat_bits = {"attn": hparams.quant_attn_bits, "mlp": hparams.quant_mlp_bits}
    quant_bits_str = os.environ.get("QUANT_BITS", "")
    if quant_bits_str:
        cat_bits = {}
        for part in quant_bits_str.split(","):
            k, v = part.strip().rsplit(":", 1)
            cat_bits[k.strip()] = int(v.strip())
        log.info(f"Per-layer quant: {cat_bits}")
    quant_obj, quant_stats = quantize_state_dict_int8(flat, cat_bits=cat_bits)
    quant_flat = dequantize_state_dict_int8(quant_obj)

    model_quant = build_model(hparams)
    model_quant.update(tree_unflatten(list(quant_flat.items())))
    mx.eval(model_quant.parameters())

    # --- Load data ---
    val_tokens = load_validation_tokens(hparams.val_files, hparams.train_seq_len)
    log.info(f"Val tokens: {len(val_tokens):,}")

    # --- Sensitivity analysis / correction layer selection ---
    if args.correction_layers is not None:
        correction_layers = [int(x) for x in args.correction_layers.split(",")]
        log.info(f"Using specified correction layers: {correction_layers}")
    else:
        correction_layers = select_correction_layers(
            model_float, model_quant, val_tokens,
            n_corrections=args.n_corrections,
            n_seqs=args.n_sensitivity_seqs, seq_len=args.seq_len,
        )

    # --- Decide which model to train ---
    if args.mode == "frozen":
        # Train on dequantized (grid-snapped) weights, corrections only
        model = model_quant
        param_bits = None
    else:
        # Joint: train on float weights with fake-quant STE + corrections
        model = model_float
        param_bits = build_param_bits_map(model, cat_bits, filter_pattern=args.filter)
        # Exclude correction params from fake-quant
        param_bits = {k: v for k, v in param_bits.items() if "correction_" not in k}
        log.info(f"Joint mode: {len(param_bits)} base weight tensors with fake-quant")

    # --- Attach corrections ---
    attach_corrections(model, correction_layers, hparams.model_dim, args.hidden_size)

    # --- Baseline eval ---
    log.info("\n=== Baseline (zero-init corrections = identity) ===")
    quick_eval_corrected(model, hparams, val_tokens, n_seqs=args.n_eval_seqs, seq_len=args.seq_len)

    # --- Train ---
    if args.steps > 0:
        train_tokens = load_train_tokens(hparams, max_tokens=args.batch_seqs * args.seq_len * 200 + 2048)
        log.info(f"Train tokens: {len(train_tokens):,}")

        log.info(f"\n=== Training ({args.mode} mode, {args.steps} steps) ===")
        train_corrections_ce(
            model, train_tokens, hparams, mode=args.mode,
            param_bits=param_bits, n_steps=args.steps, lr=args.lr,
            seq_len=args.seq_len, batch_seqs=args.batch_seqs,
        )

        # --- Post-training eval ---
        log.info("\n=== After training ===")
        quick_eval_corrected(model, hparams, val_tokens, n_seqs=args.n_eval_seqs, seq_len=args.seq_len)

        # For joint mode: quantize base weights and re-eval
        if args.mode == "joint":
            log.info("\n=== After quantizing base weights ===")
            base_flat = dict(tree_flatten(model.parameters()))
            # Extract only base model params (not corrections)
            base_only = {k: v for k, v in base_flat.items() if "correction_" not in k}
            quant_obj2, _ = quantize_state_dict_int8(base_only, cat_bits=cat_bits)
            quant_flat2 = dequantize_state_dict_int8(quant_obj2)
            # Update only base params, keep correction params
            model.update(tree_unflatten(list(quant_flat2.items())))
            mx.eval(model.parameters())
            quick_eval_corrected(model, hparams, val_tokens, n_seqs=args.n_eval_seqs, seq_len=args.seq_len)

    # --- Report correction overhead ---
    log.info("\n=== Correction overhead ===")
    total_params = 0
    for idx, layer in enumerate(correction_layers):
        net = getattr(model, f"correction_{idx}")
        n = sum(v.size for _, v in tree_flatten(net.parameters()))
        total_params += n
        log.info(f"  L{layer}: {n:,} params")
    log.info(f"  Total: {total_params:,} params")
    est_bytes = total_params * 1  # int8 = 1 byte/param
    log.info(f"  Estimated artifact overhead: ~{est_bytes/1024:.1f} KB (int8)")

    # --- Save corrections ---
    if args.save:
        save_dict = {}
        for idx, layer in enumerate(correction_layers):
            net = getattr(model, f"correction_{idx}")
            for name, v in tree_flatten(net.parameters()):
                save_dict[f"correction.{layer}.{name}"] = np.array(v)
        save_dict["__correction_layers__"] = np.array(correction_layers)
        np.savez(args.save, **save_dict)
        log.info(f"Saved corrections to {args.save}")


if __name__ == "__main__":
    main()
