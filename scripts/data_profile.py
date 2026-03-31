#!/usr/bin/env python3
"""
data_profile.py — Profile FineWeb dataset quality and characteristics.

Quick pass over all shards to measure:
- Token distribution (byte-fallback fraction, vocab utilization)
- Sequence quality (garbage fraction, repetition, entropy)
- Per-shard statistics (for identifying problematic shards)

Usage:
    python3 scripts/data_profile.py [--split train] [--max-shards 0] [--seq-len 1024]
"""
import argparse
import glob
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BYTE_TOKEN_LO = 3
BYTE_TOKEN_HI = 258


def load_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520:
        raise ValueError(f"Bad header: {path}")
    num_tokens = int(header[2])
    offset = 256 * np.dtype("<i4").itemsize
    return np.fromfile(path, dtype="<u2", count=num_tokens, offset=offset)


def profile_shard(tokens: np.ndarray, seq_len: int = 1024) -> dict:
    n_tokens = len(tokens)
    n_seqs = n_tokens // seq_len

    # Token-level stats
    byte_mask = (tokens >= BYTE_TOKEN_LO) & (tokens <= BYTE_TOKEN_HI)
    byte_frac = byte_mask.mean()
    unique_tokens = len(np.unique(tokens))

    # Per-sequence stats
    garbage_seqs = 0
    high_repeat_seqs = 0
    low_entropy_seqs = 0
    seq_byte_fracs = []
    seq_entropies = []

    for i in range(n_seqs):
        seq = tokens[i * seq_len: (i + 1) * seq_len]

        # Byte fraction
        bf = ((seq >= BYTE_TOKEN_LO) & (seq <= BYTE_TOKEN_HI)).mean()
        seq_byte_fracs.append(bf)
        if bf > 0.2:
            garbage_seqs += 1

        # Repetition: fraction of tokens equal to previous token
        repeat_frac = (seq[1:] == seq[:-1]).mean()
        if repeat_frac > 0.5:
            high_repeat_seqs += 1

        # Entropy (bits per token)
        counts = np.bincount(seq, minlength=1024)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * np.log2(probs)).sum()
        seq_entropies.append(entropy)
        if entropy < 3.0:  # very low entropy
            low_entropy_seqs += 1

    return {
        'n_tokens': n_tokens,
        'n_seqs': n_seqs,
        'byte_frac': float(byte_frac),
        'unique_tokens': int(unique_tokens),
        'garbage_seqs': garbage_seqs,
        'garbage_frac': garbage_seqs / max(n_seqs, 1),
        'high_repeat_seqs': high_repeat_seqs,
        'low_entropy_seqs': low_entropy_seqs,
        'byte_frac_p50': float(np.median(seq_byte_fracs)),
        'byte_frac_p90': float(np.percentile(seq_byte_fracs, 90)),
        'byte_frac_p99': float(np.percentile(seq_byte_fracs, 99)),
        'entropy_p10': float(np.percentile(seq_entropies, 10)),
        'entropy_p50': float(np.median(seq_entropies)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--max-shards", type=int, default=0, help="0 = all shards")
    parser.add_argument("--seq-len", type=int, default=1024)
    args = parser.parse_args()

    pattern = os.path.join(args.data_path, f"fineweb_{args.split}_*.bin")
    shards = sorted(glob.glob(pattern))
    if args.max_shards > 0:
        shards = shards[:args.max_shards]
    print(f"Profiling {len(shards)} {args.split} shards ({pattern})")

    total_tokens = 0
    total_seqs = 0
    total_garbage = 0
    total_repeat = 0
    total_low_ent = 0
    worst_shards = []

    for i, path in enumerate(shards):
        tokens = load_shard(Path(path))
        stats = profile_shard(tokens, args.seq_len)
        total_tokens += stats['n_tokens']
        total_seqs += stats['n_seqs']
        total_garbage += stats['garbage_seqs']
        total_repeat += stats['high_repeat_seqs']
        total_low_ent += stats['low_entropy_seqs']
        worst_shards.append((stats['garbage_frac'], path, stats))

        if (i + 1) % 10 == 0 or i == len(shards) - 1:
            print(f"  [{i+1}/{len(shards)}] {total_tokens/1e6:.0f}M tokens, "
                  f"{total_garbage}/{total_seqs} garbage ({100*total_garbage/max(total_seqs,1):.1f}%)")

    print(f"\n{'='*60}")
    print(f"SUMMARY ({args.split})")
    print(f"{'='*60}")
    print(f"Shards:           {len(shards)}")
    print(f"Total tokens:     {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"Total sequences:  {total_seqs:,}")
    print(f"Garbage (>20% byte): {total_garbage:,} ({100*total_garbage/max(total_seqs,1):.1f}%)")
    print(f"High repeat (>50%):  {total_repeat:,} ({100*total_repeat/max(total_seqs,1):.1f}%)")
    print(f"Low entropy (<3b):   {total_low_ent:,} ({100*total_low_ent/max(total_seqs,1):.1f}%)")

    # Worst shards
    worst_shards.sort(reverse=True)
    print(f"\nWorst shards by garbage fraction:")
    for frac, path, stats in worst_shards[:10]:
        print(f"  {frac*100:5.1f}% garbage  {stats['byte_frac']*100:5.1f}% byte  "
              f"entropy_p10={stats['entropy_p10']:.1f}  {os.path.basename(path)}")


if __name__ == "__main__":
    main()
