#!/usr/bin/env python3
"""
correction_mse_standalone.py — MSE-trained standalone correction layers.

Trains CorrectionNet modules to predict quantization error from h_quant alone.
Oracle (float model hidden states) used as training targets only — NOT at inference.

Training: MSE(correction(h_quant), h_float - h_quant) per layer
Eval: proper eval_val BPB with corrections injected into forward pass

Usage:
    NUM_LAYERS=11 MLP_MULT=4.5 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16 \
    QUANT_ATTN_BITS=5 QUANT_MLP_BITS=5 \
    .venv/bin/python3 scripts/correction_mse_standalone.py logs/wd50_11L_5x_best.npz \
        --steps 500 --batch-seqs 32

    # Explicit layers + bottleneck:
    ... --correction-layers 0,5,8 --hidden-size 32

    # Eval-only (measure baseline, no training):
    ... --steps 0
"""
from __future__ import annotations

import argparse
import logging
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
    quantize_state_dict_int8, dequantize_state_dict_int8, rms_norm,
    load_data_shard, build_sentencepiece_luts, eval_val,
)
from scripts.error_attribution import (
    compute_logit_kl_impact, print_kl_impact_table, forward_collect_all,
)

log = logging.getLogger("correction_mse")


# =============================================================================
# CorrectionNet — predicts correction from h_quant (no oracle input)
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
            # Full-rank linear correction
            self.linear = nn.Linear(dim, dim)
            self.linear.weight = mx.zeros_like(self.linear.weight)
            self.linear.bias = mx.zeros_like(self.linear.bias)

    def __call__(self, h: mx.array) -> mx.array:
        if self.hidden > 0:
            return self.w2(nn.relu(self.w1(h))).astype(h.dtype)
        return self.linear(h).astype(h.dtype)


# =============================================================================
# Forward passes
# =============================================================================

def forward_collect_hidden(model, tokens):
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


def forward_with_corrections(model, input_ids, correction_map):
    """GPT forward pass with correction layers injected. Returns final hidden.

    correction_map: {block_index: CorrectionNet}
    """
    x = model.tok_emb(input_ids).astype(COMPUTE_DTYPE)
    if model.bigram is not None:
        x = x + model.bigram(input_ids)
    x = rms_norm(x)
    x = model.smear(x)
    x0 = x
    skips = []

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


# =============================================================================
# Model / data helpers
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
# MSE training: per-layer hidden state matching
# =============================================================================

def train_corrections_mse(
    model_float, model_quant, corrections, correction_layers,
    train_tokens, hparams,
    n_steps=500, lr=3e-4, seq_len=1024, batch_seqs=32, log_every=50,
):
    """Train correction nets via per-layer MSE against float hidden states.

    For each training step:
    1. Sample a batch of sequences
    2. Run float model → collect hidden states at correction layers (targets)
    3. Run quant model → collect hidden states at correction layers (inputs)
    4. For each correction layer: loss += MSE(correction(h_quant), h_float - h_quant)
    """
    correction_set = set(correction_layers)

    # Collect all correction params
    all_params = []
    param_shapes = []  # (net_idx, param_key) for reconstructing
    for idx, layer in enumerate(correction_layers):
        net = corrections[layer]
        for key, val in tree_flatten(net.parameters()):
            all_params.append(val)
            param_shapes.append((idx, key))
    mx.eval(all_params)

    n_params = sum(int(p.size) for p in all_params)
    log.info(f"Correction params: {n_params:,} ({len(all_params)} tensors)")

    # Manual Adam state
    m_state = [mx.zeros_like(p) for p in all_params]
    v_state = [mx.zeros_like(p) for p in all_params]
    mx.eval(m_state + v_state)
    b1, b2, eps = 0.9, 0.999, 1e-8

    max_start = len(train_tokens) - (batch_seqs * seq_len + 1)
    if max_start <= 0:
        raise ValueError(f"Not enough tokens ({len(train_tokens)}) for batch={batch_seqs}×{seq_len}")

    def set_params(params):
        """Push flat param list back into correction nets."""
        idx = 0
        for net_idx, layer in enumerate(correction_layers):
            net = corrections[layer]
            flat = list(tree_flatten(net.parameters()))
            new_items = []
            for key, _ in flat:
                new_items.append((key, params[idx]))
                idx += 1
            net.update(tree_unflatten(new_items))

    def compute_mse_loss(params, float_hidden_batch, quant_hidden_batch):
        """MSE loss across all correction layers for one batch.

        float_hidden_batch: {layer: (B, T, D)}
        quant_hidden_batch: {layer: (B, T, D)}
        """
        set_params(params)
        total_loss = mx.array(0.0)
        for layer in correction_layers:
            h_q = quant_hidden_batch[layer]
            h_f = float_hidden_batch[layer]
            target = h_f - h_q  # what the correction should predict
            pred = corrections[layer](h_q)
            diff = pred - target.astype(pred.dtype)
            total_loss = total_loss + (diff * diff).mean()
        return total_loss / len(correction_layers)

    grad_fn = mx.value_and_grad(compute_mse_loss)

    # Cosine decay schedule
    def get_lr(step):
        if n_steps <= 1:
            return lr
        return lr * 0.5 * (1.0 + math.cos(math.pi * step / (n_steps - 1)))

    history = []
    best_loss = float("inf")
    best_params = None
    t0 = time.time()

    for step in range(n_steps):
        cur_lr = get_lr(step)

        # Sample batch
        start = np.random.randint(0, max_start)
        n_tok = batch_seqs * seq_len
        chunk = train_tokens[start:start + n_tok]
        batch = chunk.reshape(batch_seqs, seq_len)

        # Collect hidden states from both models (no grad through models)
        float_hidden_batch = {l: [] for l in correction_layers}
        quant_hidden_batch = {l: [] for l in correction_layers}

        for s in range(batch_seqs):
            tokens = batch[s]
            fh = forward_collect_hidden(model_float, tokens)
            qh = forward_collect_hidden(model_quant, tokens)
            for l in correction_layers:
                float_hidden_batch[l].append(mx.stop_gradient(fh[l]))
                quant_hidden_batch[l].append(mx.stop_gradient(qh[l]))

        # Stack into (B, T, D)
        for l in correction_layers:
            float_hidden_batch[l] = mx.concatenate(float_hidden_batch[l], axis=0)
            quant_hidden_batch[l] = mx.concatenate(quant_hidden_batch[l], axis=0)
        mx.eval(list(float_hidden_batch.values()) + list(quant_hidden_batch.values()))

        # Forward + backward through correction nets only
        loss, grads = grad_fn(all_params, float_hidden_batch, quant_hidden_batch)
        mx.eval(loss)

        # Adam update
        step_num = step + 1
        new_params = []
        new_m = []
        new_v = []
        for j, (p, g, m, v) in enumerate(zip(all_params, grads, m_state, v_state)):
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            m_hat = m / (1 - b1 ** step_num)
            v_hat = v / (1 - b2 ** step_num)
            p = p - cur_lr * m_hat / (mx.sqrt(v_hat) + eps)
            new_params.append(p)
            new_m.append(m)
            new_v.append(v)
        all_params = new_params
        m_state = new_m
        v_state = new_v
        set_params(all_params)
        mx.eval(all_params + m_state + v_state)

        loss_val = loss.item()
        history.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = [mx.array(p) for p in all_params]
            mx.eval(best_params)

        if step % log_every == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            log.info(f"  step {step:>4d}/{n_steps}  mse={loss_val:.8f}  "
                     f"lr={cur_lr:.2e}  best={best_loss:.8f}  elapsed={elapsed:.1f}s")

    # Restore best
    if best_params is not None:
        all_params = best_params
        set_params(all_params)
        mx.eval(all_params)
        log.info(f"Restored best checkpoint (mse={best_loss:.8f})")

    log.info(f"Training done in {time.time() - t0:.1f}s  best_mse={best_loss:.8f}")
    return history


# =============================================================================
# Proper BPB evaluation using eval_val
# =============================================================================

def eval_bpb_with_corrections(model_quant, corrections, correction_layers, hparams, val_tokens):
    """Evaluate BPB using the proper eval_val pipeline with corrections injected."""
    correction_map = {l: corrections[l] for l in correction_layers}

    # Build a loss function that uses corrected forward
    def corrected_loss(input_ids, target_ids):
        h = forward_with_corrections(model_quant, input_ids, correction_map)
        h = h.reshape(-1, model_quant.tok_emb.weight.shape[1])
        logits = model_quant._apply_logit_processing(h @ model_quant.tok_emb.weight.astype(h.dtype).T)
        return nn.losses.cross_entropy(logits.astype(mx.float32), target_ids.reshape(-1), reduction="mean")

    # Build sentencepiece LUTs for proper BPB
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=hparams.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, hparams.vocab_size)

    val_loss, val_bpb = eval_val(
        hparams, corrected_loss, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        log_fn=lambda msg: log.info(msg),
    )
    return val_loss, val_bpb


# =============================================================================
# Quick MSE eval on val data
# =============================================================================

def eval_mse_val(model_float, model_quant, corrections, correction_layers,
                 val_tokens, n_seqs=32, seq_len=1024):
    """Quick MSE evaluation: how well do corrections predict the error on val?"""
    total_mse = {l: 0.0 for l in correction_layers}
    total_baseline_mse = {l: 0.0 for l in correction_layers}
    count = 0

    for s in range(n_seqs):
        tokens = val_tokens[s * seq_len : (s + 1) * seq_len]
        if len(tokens) < seq_len:
            break

        fh = forward_collect_hidden(model_float, tokens)
        qh = forward_collect_hidden(model_quant, tokens)

        for l in correction_layers:
            h_q = qh[l]
            h_f = fh[l]
            error = h_f - h_q  # target
            pred = corrections[l](h_q)
            mx.eval(pred)

            # Corrected MSE: how much error remains after correction
            residual = error - pred.astype(error.dtype)
            total_mse[l] += float((residual * residual).mean().item())
            # Baseline MSE: error without correction
            total_baseline_mse[l] += float((error * error).mean().item())

        count += 1

    log.info(f"  MSE eval ({count} seqs):")
    for l in correction_layers:
        base = total_baseline_mse[l] / count
        corr = total_mse[l] / count
        recovery = 1.0 - corr / base if base > 0 else 0.0
        log.info(f"    L{l}: baseline_mse={base:.8f}  corrected_mse={corr:.8f}  "
                 f"recovery={recovery:.1%}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MSE-trained standalone correction layers")
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument("--correction-layers", type=str, default=None,
                        help="Comma-separated layer indices (overrides auto-selection)")
    parser.add_argument("--n-corrections", type=int, default=3,
                        help="Number of correction layers to auto-select")
    parser.add_argument("--hidden-size", type=int, default=0,
                        help="CorrectionNet bottleneck (0=full-rank linear)")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-seqs", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-sensitivity-seqs", type=int, default=4)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--save", type=str, default=None,
                        help="Save correction weights to this .npz path")
    args = parser.parse_args()

    hparams = Hyperparameters()

    # Logging setup
    global log
    log = logging.getLogger("correction_mse")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if args.log_file is None:
        h_tag = f"h{args.hidden_size}" if args.hidden_size > 0 else "linear"
        bits = hparams.quant_attn_bits
        args.log_file = f"logs/correction_mse_{h_tag}_int{bits}_{args.steps}s.txt"

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    fh = logging.FileHandler(args.log_file, mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.info(f"Logging to {args.log_file}")

    log.info(f"Config: {hparams.num_layers}L/{hparams.model_dim}d, MLP {hparams.mlp_mult}x, "
             f"act={hparams.mlp_act}, quant=a{hparams.quant_attn_bits}m{hparams.quant_mlp_bits}")
    log.info(f"hidden={args.hidden_size}, steps={args.steps}, lr={args.lr}, "
             f"batch_seqs={args.batch_seqs}, seq_len={args.seq_len}")

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

    # --- Correction layer selection ---
    if args.correction_layers is not None:
        correction_layers = [int(x) for x in args.correction_layers.split(",")]
        log.info(f"Using specified correction layers: {correction_layers}")
    else:
        correction_layers = []
        log.info(f"\n=== Sensitivity analysis ({args.n_sensitivity_seqs} seqs) ===")
        res = compute_logit_kl_impact(
            model_float, model_quant, val_tokens,
            n_seqs=args.n_sensitivity_seqs, seq_len=args.seq_len,
        )
        print_kl_impact_table(res)
        top = [int(x) for x in np.argsort(res["marginal_kl"])[::-1][:args.n_corrections]]
        correction_layers = sorted(top)
        log.info(f"Auto-selected correction layers (top-{args.n_corrections} by marginal KL): "
                 f"{correction_layers}")

    # --- Create correction nets ---
    corrections = {}
    for layer in correction_layers:
        corrections[layer] = CorrectionNet(hparams.model_dim, args.hidden_size)
    mx.eval([v for net in corrections.values() for _, v in tree_flatten(net.parameters())])

    # --- Baseline BPB (quant model, no corrections) ---
    log.info("\n=== Baseline BPB (quantized, no corrections) ===")
    base_loss, base_bpb = eval_bpb_with_corrections(
        model_quant, corrections, correction_layers, hparams, val_tokens)
    log.info(f"  val_loss={base_loss:.6f}  val_bpb={base_bpb:.6f}")

    # --- Float model BPB (upper bound) ---
    log.info("\n=== Float model BPB (upper bound) ===")
    float_loss, float_bpb = eval_bpb_with_corrections(
        model_float, {l: CorrectionNet(hparams.model_dim, 0) for l in correction_layers},
        correction_layers, hparams, val_tokens)
    log.info(f"  val_loss={float_loss:.6f}  val_bpb={float_bpb:.6f}")
    log.info(f"  Quant gap: {base_bpb - float_bpb:.6f} BPB")

    # --- Baseline MSE (uncorrected error magnitude) ---
    log.info("\n=== Baseline MSE (zero corrections) ===")
    eval_mse_val(model_float, model_quant, corrections, correction_layers,
                 val_tokens, n_seqs=32, seq_len=args.seq_len)

    # --- Train ---
    if args.steps > 0:
        train_tokens = load_train_tokens(
            hparams, max_tokens=args.batch_seqs * args.seq_len * 200 + 2048)
        log.info(f"Train tokens: {len(train_tokens):,}")

        log.info(f"\n=== Training MSE corrections ({args.steps} steps) ===")
        train_corrections_mse(
            model_float, model_quant, corrections, correction_layers,
            train_tokens, hparams,
            n_steps=args.steps, lr=args.lr,
            seq_len=args.seq_len, batch_seqs=args.batch_seqs,
        )

        # --- Post-training MSE eval ---
        log.info("\n=== Post-training MSE (val) ===")
        eval_mse_val(model_float, model_quant, corrections, correction_layers,
                     val_tokens, n_seqs=32, seq_len=args.seq_len)

        # --- Post-training BPB eval ---
        log.info("\n=== Post-training BPB (corrected quantized model) ===")
        corr_loss, corr_bpb = eval_bpb_with_corrections(
            model_quant, corrections, correction_layers, hparams, val_tokens)
        log.info(f"  val_loss={corr_loss:.6f}  val_bpb={corr_bpb:.6f}")
        log.info(f"  Recovery: {base_bpb - corr_bpb:.6f} BPB of {base_bpb - float_bpb:.6f} gap "
                 f"({(base_bpb - corr_bpb) / max(base_bpb - float_bpb, 1e-9):.1%})")

    # --- Correction overhead ---
    log.info("\n=== Correction overhead ===")
    total_params = 0
    for layer in correction_layers:
        net = corrections[layer]
        n = sum(int(v.size) for _, v in tree_flatten(net.parameters()))
        total_params += n
        log.info(f"  L{layer}: {n:,} params")
    log.info(f"  Total: {total_params:,} params")
    est_kb = total_params / 1024  # int8 = 1 byte/param
    log.info(f"  Estimated artifact overhead: ~{est_kb:.1f} KB (int8)")

    # Score impact estimate
    if args.steps > 0:
        log.info("\n=== Score impact estimate ===")
        log.info(f"  Base score:     {base_bpb:.6f} × S_MB")
        log.info(f"  Corrected:      {corr_bpb:.6f} × (S_MB + {est_kb/1024:.4f})")
        # Assume current artifact ~14 MB
        S = 14.0
        base_score = base_bpb * S
        corr_score = corr_bpb * (S + est_kb / 1024)
        log.info(f"  Base score (14MB):     {base_score:.4f}")
        log.info(f"  Corrected score (14MB): {corr_score:.4f}")
        log.info(f"  Delta: {corr_score - base_score:+.4f} ({'better' if corr_score < base_score else 'WORSE'})")

    # --- Save corrections ---
    if args.save:
        save_dict = {}
        for layer in correction_layers:
            net = corrections[layer]
            for name, v in tree_flatten(net.parameters()):
                save_dict[f"correction.{layer}.{name}"] = np.array(v)
        save_dict["__correction_layers__"] = np.array(correction_layers)
        save_dict["__hidden_size__"] = np.array([args.hidden_size])
        np.savez(args.save, **save_dict)
        log.info(f"Saved corrections to {args.save}")


if __name__ == "__main__":
    main()
