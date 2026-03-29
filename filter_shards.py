#!/usr/bin/env python3
"""Scan training shards for garbage sequences (byte-fallback heavy regions) and write filtered copies.

Usage:
    python filter_shards.py [--input-dir DIR] [--output-dir DIR] [--seq-len 1024] [--max-unique-frac 0.9] [--dry-run]

The shard binary format:
  - 256 int32 header (magic=20240520, version=1, num_tokens, ...)
  - num_tokens uint16 values
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np


def load_shard(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (header_i32[256], tokens_u16[N])."""
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad header: {path}")
    num_tokens = int(header[2])
    offset = 256 * np.dtype("<i4").itemsize
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=offset)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read: {path}")
    return header, tokens


def write_shard(path: Path, header: np.ndarray, tokens: np.ndarray) -> None:
    """Write a shard with updated token count in header."""
    header = header.copy()
    header[2] = len(tokens)
    with open(path, "wb") as f:
        header.tofile(f)
        tokens.tofile(f)


# Sentencepiece sp1024: byte-fallback tokens are IDs 3-258
BYTE_TOKEN_LO = 3
BYTE_TOKEN_HI = 258


def is_garbage(chunk: np.ndarray, max_byte_frac: float) -> bool:
    """A sequence is garbage if byte-fallback tokens exceed max_byte_frac of its length."""
    byte_count = int(np.sum((chunk >= BYTE_TOKEN_LO) & (chunk <= BYTE_TOKEN_HI)))
    return byte_count > max_byte_frac * len(chunk)


def scan_shard(tokens: np.ndarray, seq_len: int, max_byte_frac: float) -> list[tuple[int, int, int]]:
    """Find garbage regions. Returns list of (start_idx, end_idx, byte_count)."""
    bad = []
    n = len(tokens)
    for start in range(0, n - seq_len, seq_len):
        chunk = tokens[start : start + seq_len]
        byte_count = int(np.sum((chunk >= BYTE_TOKEN_LO) & (chunk <= BYTE_TOKEN_HI)))
        if byte_count > max_byte_frac * seq_len:
            bad.append((start, start + seq_len, byte_count))
    return bad


def filter_shard(tokens: np.ndarray, seq_len: int, max_byte_frac: float) -> tuple[np.ndarray, int, int]:
    """Remove garbage sequences. Returns (filtered_tokens, total_seqs, removed_seqs)."""
    n = len(tokens)
    total_seqs = n // seq_len
    keep_chunks: list[np.ndarray] = []
    removed = 0

    for i in range(total_seqs):
        chunk = tokens[i * seq_len : (i + 1) * seq_len]
        if is_garbage(chunk, max_byte_frac):
            removed += 1
        else:
            keep_chunks.append(chunk)

    remainder = n - total_seqs * seq_len
    if remainder > 0:
        keep_chunks.append(tokens[total_seqs * seq_len :])

    filtered = np.concatenate(keep_chunks) if keep_chunks else np.array([], dtype=tokens.dtype)
    return filtered, total_seqs, removed


def main():
    parser = argparse.ArgumentParser(description="Filter garbage sequences from training shards")
    parser.add_argument("--input-dir", default="./data/datasets/fineweb10B_sp1024", help="Input shard directory")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: input-dir, overwrites)")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--max-byte-frac", type=float, default=0.1,
                        help="Skip sequences where byte-fallback tokens > this fraction")
    parser.add_argument("--dry-run", action="store_true", help="Scan only, don't write")
    parser.add_argument("--val", action="store_true", help="Also filter val shards")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    max_byte_frac = args.max_byte_frac

    patterns = [str(input_dir / "fineweb_train_*.bin")]
    if args.val:
        patterns.append(str(input_dir / "fineweb_val_*.bin"))

    total_removed = 0
    total_seqs = 0

    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"No files for {pattern}", file=sys.stderr)
            continue

        for path in files:
            path = Path(path)
            header, tokens = load_shard(path)
            original_tokens = len(tokens)

            if args.dry_run:
                bad_regions = scan_shard(tokens, args.seq_len, max_byte_frac)
                n_seqs = original_tokens // args.seq_len
                total_seqs += n_seqs
                total_removed += len(bad_regions)
                if bad_regions:
                    print(f"{path.name}: {len(bad_regions)}/{n_seqs} garbage sequences ({100*len(bad_regions)/n_seqs:.2f}%)")
                    for start, end, unique in bad_regions[:5]:
                        print(f"  offset {start:>10}-{end:>10}  byte_tokens={unique}/{args.seq_len}")
                    if len(bad_regions) > 5:
                        print(f"  ... and {len(bad_regions)-5} more")
                else:
                    print(f"{path.name}: clean ({n_seqs} sequences)")
            else:
                filtered, n_seqs, removed = filter_shard(tokens, args.seq_len, max_byte_frac)
                total_seqs += n_seqs
                total_removed += removed

                out_path = output_dir / path.name
                output_dir.mkdir(parents=True, exist_ok=True)
                write_shard(out_path, header, filtered)

                pct = 100 * removed / n_seqs if n_seqs > 0 else 0
                print(f"{path.name}: removed {removed}/{n_seqs} sequences ({pct:.2f}%), "
                      f"{original_tokens} -> {len(filtered)} tokens, saved to {out_path}")

    pct = 100 * total_removed / total_seqs if total_seqs > 0 else 0
    print(f"\nTotal: {total_removed}/{total_seqs} garbage sequences removed ({pct:.2f}%)")


if __name__ == "__main__":
    main()
