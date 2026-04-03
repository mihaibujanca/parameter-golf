#!/usr/bin/env python3
"""
data_profile.py — Profile FineWeb dataset quality and characteristics.

Quick pass over all shards to measure:
- Token distribution (byte-fallback fraction, vocab utilization)
- Sequence quality (garbage fraction, repetition, entropy)
- Per-shard statistics (for identifying problematic shards)

Outputs:
- Console summary with per-shard stats
- JSON shard manifest ({data_path}/shard_manifest.json) with per-shard
  quality metrics and skip recommendations. Training scripts can load this
  to skip bad shards entirely.

Usage:
    python3 scripts/data_profile.py [--split train] [--max-shards 0] [--seq-len 1024]
    python3 scripts/data_profile.py --split train --split val   # profile both
    python3 scripts/data_profile.py --garbage-threshold 0.5     # shards with >50% garbage seqs get skip=true
"""
import argparse
import glob
import json
import os
import sys
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


def profile_split(data_path: str, split: str, seq_len: int, max_shards: int,
                   garbage_threshold: float) -> tuple[dict, list[dict]]:
    """Profile one split, return (summary_dict, per_shard_list)."""
    pattern = os.path.join(data_path, f"fineweb_{split}_*.bin")
    shards = sorted(glob.glob(pattern))
    if max_shards > 0:
        shards = shards[:max_shards]
    print(f"Profiling {len(shards)} {split} shards ({pattern})")

    total_tokens = 0
    total_seqs = 0
    total_garbage = 0
    total_repeat = 0
    total_low_ent = 0
    shard_records: list[dict] = []

    for i, path in enumerate(shards):
        tokens = load_shard(Path(path))
        stats = profile_shard(tokens, seq_len)
        total_tokens += stats['n_tokens']
        total_seqs += stats['n_seqs']
        total_garbage += stats['garbage_seqs']
        total_repeat += stats['high_repeat_seqs']
        total_low_ent += stats['low_entropy_seqs']

        skip = stats['garbage_frac'] > garbage_threshold
        shard_records.append({
            'file': os.path.basename(path),
            'path': path,
            'split': split,
            'skip': skip,
            **stats,
        })

        if (i + 1) % 10 == 0 or i == len(shards) - 1:
            print(f"  [{i+1}/{len(shards)}] {total_tokens/1e6:.0f}M tokens, "
                  f"{total_garbage}/{total_seqs} garbage ({100*total_garbage/max(total_seqs,1):.1f}%)")

    n_skip = sum(1 for r in shard_records if r['skip'])
    skip_tokens = sum(r['n_tokens'] for r in shard_records if r['skip'])

    print(f"\n{'='*60}")
    print(f"SUMMARY ({split})")
    print(f"{'='*60}")
    print(f"Shards:           {len(shards)}")
    print(f"Total tokens:     {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"Total sequences:  {total_seqs:,}")
    print(f"Garbage (>20% byte): {total_garbage:,} ({100*total_garbage/max(total_seqs,1):.1f}%)")
    print(f"High repeat (>50%):  {total_repeat:,} ({100*total_repeat/max(total_seqs,1):.1f}%)")
    print(f"Low entropy (<3b):   {total_low_ent:,} ({100*total_low_ent/max(total_seqs,1):.1f}%)")
    print(f"Skip shards:      {n_skip}/{len(shards)} "
          f"({skip_tokens/1e9:.2f}B tokens, garbage_frac > {garbage_threshold:.0%})")

    # Worst shards
    sorted_shards = sorted(shard_records, key=lambda r: -r['garbage_frac'])
    print(f"\nWorst shards by garbage fraction:")
    for r in sorted_shards[:10]:
        tag = " [SKIP]" if r['skip'] else ""
        print(f"  {r['garbage_frac']*100:5.1f}% garbage  {r['byte_frac']*100:5.1f}% byte  "
              f"entropy_p10={r['entropy_p10']:.1f}  {r['file']}{tag}")

    summary = {
        'split': split,
        'n_shards': len(shards),
        'n_skip': n_skip,
        'total_tokens': total_tokens,
        'clean_tokens': total_tokens - skip_tokens,
        'total_seqs': total_seqs,
        'total_garbage_seqs': total_garbage,
        'garbage_threshold': garbage_threshold,
    }
    return summary, shard_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", default=None,
                        help="Split(s) to profile (repeatable). Default: train")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--max-shards", type=int, default=0, help="0 = all shards")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--garbage-threshold", type=float, default=0.5,
                        help="Shards with garbage_frac above this get skip=true (default 0.5)")
    parser.add_argument("--output", default="",
                        help="Output manifest path (default: {data_path}/shard_manifest.json)")
    args = parser.parse_args()

    splits = args.split or ["train"]
    all_records: list[dict] = []
    all_summaries: list[dict] = []

    for split in splits:
        summary, records = profile_split(
            args.data_path, split, args.seq_len, args.max_shards, args.garbage_threshold)
        all_summaries.append(summary)
        all_records.extend(records)

    # Write manifest
    out_path = args.output or os.path.join(args.data_path, "shard_manifest.json")
    manifest = {
        'garbage_threshold': args.garbage_threshold,
        'seq_len': args.seq_len,
        'summaries': all_summaries,
        'shards': [{k: v for k, v in r.items() if k != 'path'} for r in all_records],
    }
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {out_path}")

    # Print skip list for easy copy-paste
    skip_files = [r['file'] for r in all_records if r['skip']]
    if skip_files:
        print(f"\nSkip list ({len(skip_files)} shards):")
        for f in skip_files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
