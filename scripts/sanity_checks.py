#!/usr/bin/env python3
"""
Training sanity checks: gradient flow, causal masking, single-batch overfit,
data exploration (sp1024 vs sp8192), and input-independence baseline.

Run: python sanity_checks.py
"""
from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from train_gpt_mlx import (
    COMPUTE_DTYPE,
    GPT,
    Hyperparameters,
    build_sentencepiece_luts,
    load_data_shard,
    load_validation_tokens,
    rms_norm,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def make_model(args: Hyperparameters, perturb_zero_init: bool = False) -> GPT:
    per_layer = [float(x) for x in hparams.mlp_mult_per_layer else None
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
    )
    if perturb_zero_init:
        # The baseline zero-inits attn.proj and mlp.proj weights, which blocks
        # gradient flow to upstream params and prevents inter-position information
        # flow. Perturb them slightly so sanity checks can verify the full graph.
        for b in model.blocks:
            b.attn.proj.weight = mx.random.normal(b.attn.proj.weight.shape) * 0.01
            b.mlp.proj.weight = mx.random.normal(b.mlp.proj.weight.shape) * 0.01
        mx.eval(model.state)
    return model


# ==========================================================================
# 1. Gradient Flow Verification
# ==========================================================================
def check_gradient_flow(args: Hyperparameters) -> bool:
    print("\n" + "=" * 60)
    print("CHECK 1: Gradient Flow Verification")
    print("=" * 60)

    mx.random.seed(args.seed)
    # Perturb zero-init proj weights so gradients can flow through the full graph.
    # Without this, zero proj weights block gradients to c_q/c_k/c_v/fc — that's
    # expected behavior of the zero-init residual pattern, not a bug.
    model = make_model(args, perturb_zero_init=True)

    x = mx.random.randint(0, args.vocab_size, (2, 64), dtype=mx.int32)
    y = mx.random.randint(0, args.vocab_size, (2, 64), dtype=mx.int32)

    loss, grads = nn.value_and_grad(model, lambda xi, yi: model.loss(xi, yi))(x, y)
    mx.eval(loss, grads)

    flat_grads = dict(tree_flatten(grads))
    all_ok = True
    zero_grads = []
    for name, g in sorted(flat_grads.items()):
        has_nonzero = bool(mx.any(g != 0).item())
        if not has_nonzero:
            all_ok = False
            zero_grads.append(name)
        grad_norm = float(mx.sqrt(mx.sum(g * g)).item())
        status = PASS if has_nonzero else FAIL
        print(f"  {status} {name:50s} shape={tuple(g.shape)}  |grad|={grad_norm:.6f}")

    print(f"\n  Loss = {float(loss.item()):.4f}")
    if all_ok:
        print(f"  Result: ALL PARAMS HAVE GRADIENT (with perturbed proj weights)")
    else:
        print(f"  Result: ZERO GRADIENT in: {', '.join(zero_grads)}")
    return all_ok


# ==========================================================================
# 2. Causal Masking Verification
# ==========================================================================
def check_causal_masking(args: Hyperparameters) -> bool:
    print("\n" + "=" * 60)
    print("CHECK 2: Causal Masking Verification")
    print("=" * 60)

    mx.random.seed(args.seed)
    # Perturb zero-init so attention actually flows between positions.
    model = make_model(args, perturb_zero_init=True)

    seq_len = 32
    base_seq = mx.random.randint(0, args.vocab_size, (1, seq_len), dtype=mx.int32)

    # Test 1: Changing position 0 should affect positions 1+
    seq_a = base_seq
    seq_b = mx.concatenate([mx.array([[42]], dtype=mx.int32), base_seq[:, 1:]], axis=1)

    hidden_a = model(seq_a)
    hidden_b = model(seq_b)
    mx.eval(hidden_a, hidden_b)

    diff_pos0_onward = mx.abs(hidden_a[:, 1:, :] - hidden_b[:, 1:, :])
    max_diff_forward = float(mx.max(diff_pos0_onward).item())
    forward_flow_ok = max_diff_forward > 1e-6

    print(f"  Forward flow: max diff at pos 1+ when pos 0 changed = {max_diff_forward:.6e}")
    print(f"  {PASS if forward_flow_ok else FAIL} Information flows forward from position 0")

    # Test 2: Changing position 5 must NOT affect positions 0-4
    seq_c = base_seq
    seq_d = mx.concatenate([
        base_seq[:, :5],
        mx.array([[99]], dtype=mx.int32),
        base_seq[:, 6:],
    ], axis=1)

    hidden_c = model(seq_c)
    hidden_d = model(seq_d)
    mx.eval(hidden_c, hidden_d)

    diff_before = mx.abs(hidden_c[:, :5, :] - hidden_d[:, :5, :])
    max_diff_before = float(mx.max(diff_before).item())
    causality_ok = max_diff_before < 1e-6

    print(f"  Causality: max diff at pos 0-4 when pos 5 changed = {max_diff_before:.6e}")
    print(f"  {PASS if causality_ok else FAIL} No future information leakage")

    # Also check that pos 5+ IS affected (change propagated forward)
    diff_after = mx.abs(hidden_c[:, 5:, :] - hidden_d[:, 5:, :])
    max_diff_after = float(mx.max(diff_after).item())
    print(f"  Sanity: max diff at pos 5+ = {max_diff_after:.6e} (should be > 0)")

    ok = forward_flow_ok and causality_ok
    print(f"\n  Result: {'CAUSAL MASKING CORRECT' if ok else 'CAUSAL MASKING BROKEN!'}")
    return ok


# ==========================================================================
# 3. Single-Batch Overfit Test
# ==========================================================================
def check_single_batch_overfit(args: Hyperparameters) -> bool:
    print("\n" + "=" * 60)
    print("CHECK 3: Single-Batch Overfit Test (200 steps)")
    print("=" * 60)

    mx.random.seed(args.seed)
    model = make_model(args)

    # Fixed batch: 4 sequences of length 64
    batch_size, seq_len = 4, 64
    x = mx.random.randint(0, args.vocab_size, (batch_size, seq_len), dtype=mx.int32)
    y = mx.random.randint(0, args.vocab_size, (batch_size, seq_len), dtype=mx.int32)

    # Adam with aggressive lr. SGD is too slow with zero-init proj weights —
    # Adam's per-param adaptive step sizes handle the mixed-scale initialization.
    import mlx.optimizers as optim
    optimizer = optim.Adam(learning_rate=3e-3)
    loss_and_grad = nn.value_and_grad(model, lambda xi, yi: model.loss(xi, yi))

    # Record gate weight norm if SwiGLU
    gate_key = "blocks.0.mlp.gate.weight"
    params_before = dict(tree_flatten(model.parameters()))
    gate_norm_before = None
    if gate_key in params_before:
        gate_norm_before = float(mx.sqrt(mx.sum(params_before[gate_key] ** 2)).item())

    losses = []
    t0 = time.time()
    for step in range(200):
        loss, grads = loss_and_grad(x, y)
        mx.eval(loss, grads)
        optimizer.update(model, grads)

        loss_val = float(loss.item())
        losses.append(loss_val)
        if step % 50 == 0 or step == 199:
            print(f"  step {step:3d}: loss = {loss_val:.6f}")

    elapsed = time.time() - t0
    final_loss = losses[-1]
    ok = final_loss < 0.01

    if gate_norm_before is not None:
        params_after = dict(tree_flatten(model.parameters()))
        gate_norm_after = float(mx.sqrt(mx.sum(params_after[gate_key] ** 2)).item())
        print(f"\n  SwiGLU gate weight |W|: {gate_norm_before:.4f} -> {gate_norm_after:.4f} "
              f"(delta = {gate_norm_after - gate_norm_before:.4f})")

    print(f"\n  Final loss: {final_loss:.6f}  (target < 0.01)")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  {PASS if ok else FAIL} {'Model memorized single batch' if ok else 'Model FAILED to memorize!'}")
    return ok


# ==========================================================================
# 4. Data Exploration (sp1024 vs sp8192)
# ==========================================================================
def check_data_exploration() -> None:
    print("\n" + "=" * 60)
    print("CHECK 4: Data Exploration — sp1024 vs sp8192")
    print("=" * 60)

    base = Path("./data")
    configs = [
        ("sp1024", base / "tokenizers/fineweb_1024_bpe.model", base / "datasets/fineweb10B_sp1024", 1024),
        ("sp8192", base / "tokenizers/fineweb_8192_bpe.model", base / "datasets/fineweb10B_sp8192", 8192),
    ]

    results = {}
    for name, tok_path, data_dir, vocab_size in configs:
        print(f"\n  --- {name} (vocab={vocab_size}) ---")
        sp = spm.SentencePieceProcessor(model_file=str(tok_path))

        # Load val shard
        val_files = sorted(data_dir.glob("fineweb_val_*.bin"))
        if not val_files:
            print(f"  SKIP: no val files found")
            continue
        tokens = load_data_shard(val_files[0])

        # --- Token frequency distribution ---
        counts = Counter(tokens.tolist())
        total_tokens = len(tokens)
        sorted_counts = sorted(counts.values(), reverse=True)
        top10_mass = sum(sorted_counts[:10]) / total_tokens
        top100_mass = sum(sorted_counts[:100]) / total_tokens
        used_ids = len(counts)

        print(f"  Total tokens in val shard: {total_tokens:,}")
        print(f"  Unique token IDs used: {used_ids}/{vocab_size} ({100*used_ids/vocab_size:.1f}%)")
        print(f"  Top-10 tokens cover: {100*top10_mass:.1f}% of tokens")
        print(f"  Top-100 tokens cover: {100*top100_mass:.1f}% of tokens")
        print(f"  Long-tail (used once): {sum(1 for c in counts.values() if c == 1)}")

        # --- Bytes per token ---
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, vocab_size)
        # Compute average bytes per token (approximation using base_bytes_lut)
        # For each token, bytes = base_bytes + 1 if it has a leading space and prev is not boundary
        # Simple approximation: use base_bytes + expected leading space contribution
        byte_counts = base_bytes_lut[tokens].astype(np.float64)
        # Add leading space bytes where applicable (approximate: assume most prev tokens are not boundary)
        for i in range(1, len(tokens)):
            if has_leading_space_lut[tokens[i]] and not is_boundary_token_lut[tokens[i - 1]]:
                byte_counts[i] += 1
        total_bytes = byte_counts.sum()
        bytes_per_token = total_bytes / total_tokens

        print(f"  Total decoded bytes: {int(total_bytes):,}")
        print(f"  Bytes per token: {bytes_per_token:.3f}")
        print(f"  Tokens per byte: {1/bytes_per_token:.3f}")

        # --- Document length distribution ---
        # Document boundaries: check for token_id that represents boundaries
        # In sentencepiece BPE for this setup, control token 0 is typically <unk>,
        # token 1 is <s> (BOS), token 2 is </s> (EOS).
        # Look for actual boundary markers
        boundary_positions = np.where(is_boundary_token_lut[tokens])[0]
        if len(boundary_positions) > 1:
            doc_lengths = np.diff(boundary_positions)
            print(f"  Document boundaries found: {len(boundary_positions):,}")
            print(f"  Document lengths (tokens): "
                  f"min={doc_lengths.min()}, median={int(np.median(doc_lengths))}, "
                  f"mean={doc_lengths.mean():.1f}, max={doc_lengths.max()}")
            # How many docs fit in seq_len=1024?
            pct_short = 100 * np.mean(doc_lengths < 1024)
            pct_very_short = 100 * np.mean(doc_lengths < 256)
            print(f"  Docs shorter than 1024 tokens: {pct_short:.1f}%")
            print(f"  Docs shorter than 256 tokens: {pct_very_short:.1f}%")
        else:
            print(f"  No document boundaries detected in val shard")

        results[name] = {
            "bytes_per_token": bytes_per_token,
            "used_ids": used_ids,
            "vocab_size": vocab_size,
            "total_tokens": total_tokens,
            "total_bytes": total_bytes,
        }

    # --- Comparison ---
    if len(results) == 2:
        r1, r2 = results["sp1024"], results["sp8192"]
        print(f"\n  --- Comparison ---")
        print(f"  {'Metric':<30s} {'sp1024':>10s} {'sp8192':>10s} {'ratio':>10s}")
        print(f"  {'-'*60}")
        print(f"  {'Bytes per token':<30s} {r1['bytes_per_token']:>10.3f} {r2['bytes_per_token']:>10.3f} "
              f"{r2['bytes_per_token']/r1['bytes_per_token']:>10.2f}x")
        print(f"  {'Tokens per byte':<30s} {1/r1['bytes_per_token']:>10.3f} {1/r2['bytes_per_token']:>10.3f} "
              f"{r1['bytes_per_token']/r2['bytes_per_token']:>10.2f}x")
        print(f"  {'Vocab utilization':<30s} {100*r1['used_ids']/r1['vocab_size']:>9.1f}% "
              f"{100*r2['used_ids']/r2['vocab_size']:>9.1f}%")
        print(f"  {'Tokens for same text':<30s} {r1['total_tokens']:>10,} {r2['total_tokens']:>10,} "
              f"{r1['total_tokens']/r2['total_tokens']:>10.2f}x")

        # BPB decomposition: BPB = bits_per_token * tokens_per_byte
        # At equal bits_per_token, sp8192 would win if it has higher bytes_per_token
        print(f"\n  BPB = bits_per_token × tokens_per_byte")
        print(f"  sp8192 compresses {r2['bytes_per_token']/r1['bytes_per_token']:.2f}x more bytes per token")
        print(f"  → At equal model quality, sp8192 needs {r1['bytes_per_token']/r2['bytes_per_token']:.2f}x the "
              f"bits_per_token to break even on BPB")


# ==========================================================================
# 5. Input-Independence Baseline
# ==========================================================================
def check_input_independence(args: Hyperparameters) -> bool:
    print("\n" + "=" * 60)
    print("CHECK 5: Input-Independence Baseline")
    print("=" * 60)

    mx.random.seed(args.seed)
    model = make_model(args)

    # Use a small subset of val tokens for speed
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    max_tokens = min(len(val_tokens), 32 * args.train_seq_len + 1)
    val_subset = val_tokens[:max_tokens]

    seq_len = args.train_seq_len
    n_seqs = (len(val_subset) - 1) // seq_len
    x_np = val_subset[:n_seqs * seq_len].reshape(n_seqs, seq_len)
    y_np = val_subset[1:n_seqs * seq_len + 1].reshape(n_seqs, seq_len)

    # Normal loss (untrained model)
    x = mx.array(x_np, dtype=mx.int32)
    y = mx.array(y_np, dtype=mx.int32)
    normal_loss = float(model.loss(x, y).item())
    mx.eval(model.state)

    # Shuffled loss: permute tokens within each sequence
    x_shuffled_np = x_np.copy()
    rng = np.random.RandomState(42)
    for i in range(n_seqs):
        rng.shuffle(x_shuffled_np[i])
    x_shuffled = mx.array(x_shuffled_np, dtype=mx.int32)
    shuffled_loss = float(model.loss(x_shuffled, y).item())
    mx.eval(model.state)

    # Theoretical random baseline: log(vocab_size)
    random_loss = math.log(args.vocab_size)

    normal_bpt = normal_loss / math.log(2)
    shuffled_bpt = shuffled_loss / math.log(2)
    random_bpt = random_loss / math.log(2)

    print(f"  Normal input loss:   {normal_loss:.4f} ({normal_bpt:.2f} bits/token)")
    print(f"  Shuffled input loss: {shuffled_loss:.4f} ({shuffled_bpt:.2f} bits/token)")
    print(f"  Random baseline:     {random_loss:.4f} ({random_bpt:.2f} bits/token)")
    print(f"  Gap (shuffled - normal): {shuffled_loss - normal_loss:.4f}")

    # For an untrained model, both should be close to random baseline.
    # The key insight: shuffled should be >= normal (destroying order can't help).
    ok = shuffled_loss >= normal_loss - 0.01  # small tolerance for numerical noise
    print(f"\n  Note: With an untrained model, both losses are near random baseline.")
    print(f"  After training, re-run to measure context utilization gap.")
    print(f"  {PASS if ok else FAIL} Shuffled loss >= normal loss (order matters)")
    return ok


# ==========================================================================
# Main
# ==========================================================================
def main():
    print("=" * 60)
    print("TRAINING SANITY CHECKS")
    print("=" * 60)

    args = Hyperparameters()
    results = {}

    # 1. Gradient flow (go/no-go)
    results["gradient_flow"] = check_gradient_flow(args)

    # 2. Causal masking (go/no-go)
    results["causal_masking"] = check_causal_masking(args)

    # 3. Single-batch overfit
    results["single_batch_overfit"] = check_single_batch_overfit(args)

    # 4. Data exploration (informational, no pass/fail)
    check_data_exploration()

    # 5. Input independence
    results["input_independence"] = check_input_independence(args)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status} {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print(f"\nAll checks passed.")
    else:
        print(f"\nSome checks FAILED — investigate before continuing.")


if __name__ == "__main__":
    main()
