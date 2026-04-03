#!/usr/bin/env python3
"""
allocate_bits.py — Optimal per-layer bit allocation under a size budget.

Takes pre-computed sensitivity data (from quant_analysis.py or layer_sensitivity.py)
and solves a knapsack-like optimization: minimize total BPB penalty subject to the
artifact fitting in the budget. Optionally applies a polish recovery discount.

Input formats:
  1. quant_analysis.py JSON (--analysis): has sensitivity.{attn,mlp,both} at one target bitwidth
  2. layer_sensitivity.py logs at multiple bitwidths (--sensitivity-logs): parsed from text output

Usage:
    # From analysis JSON (sensitivity at int4 only):
    .venv/bin/python3 scripts/allocate_bits.py \
        --analysis logs/analysis_wd50_11L_5x.json \
        --budget-mb 15.5 --wd 0.04 --mlp-mult 4.5

    # With polish discount:
    .venv/bin/python3 scripts/allocate_bits.py \
        --analysis logs/analysis_wd50_11L_5x.json \
        --budget-mb 15.5 --wd 0.04 --mlp-mult 4.5 --polish

    # From multi-bitwidth sensitivity CSV:
    .venv/bin/python3 scripts/allocate_bits.py \
        --sensitivity-csv logs/sensitivity_sweep.csv \
        --budget-mb 15.5 --wd 0.04 --mlp-mult 4.5

    # Sweep budgets to see the BPB-vs-size tradeoff:
    .venv/bin/python3 scripts/allocate_bits.py \
        --analysis logs/analysis_wd50_11L_5x.json \
        --wd 0.04 --mlp-mult 4.5 --sweep
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pgolf.budget import _matrix_shapes, _quantized_payload_bytes, _zstd_ratio, _FP16_ZSTD_RATIO, TORCH_OVERHEAD_BYTES

MB = 1024 * 1024
BUDGET_BYTES_DEFAULT = 16 * MB

# Bit-widths we consider for allocation
CANDIDATE_BITS = [3, 4, 5, 6]


# =============================================================================
# Polish recovery model
# =============================================================================

def polish_recovery_fraction(quant_gap: float) -> float:
    """Predict fraction of quant gap recovered by gradient polish.

    Fitted from three data points:
      gap=0.040 → 39% recovery
      gap=0.088 → 52% recovery
      gap=0.596 → 73% recovery

    Model: recovery = a + b * (1 - exp(-c * gap))
    This saturates at (a+b) for large gaps and gives ~a for tiny gaps.
    """
    # Fitted parameters (least-squares on the 3 data points)
    a = 0.33
    b = 0.45
    c = 5.0
    return min(a + b * (1.0 - math.exp(-c * quant_gap)), 0.80)


def apply_polish_discount(bpb_cost: float) -> float:
    """Discount a BPB penalty by estimated polish recovery."""
    recovery = polish_recovery_fraction(bpb_cost)
    return bpb_cost * (1.0 - recovery)


# =============================================================================
# Size model: per-layer, per-component, per-bitwidth
# =============================================================================

def compute_layer_sizes(
    num_layers: int,
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    mlp_mult: float,
    mlp_act: str,
    weight_decay: float,
) -> dict[tuple[int, str, int], float]:
    """Compute compressed artifact bytes for each (layer, component, bits) combo.

    Returns dict: (layer_idx, 'attn'|'mlp', bits) → compressed_bytes
    """
    shapes = _matrix_shapes(num_layers, dim, num_heads, num_kv_heads, mlp_mult, mlp_act)
    sizes = {}

    for bits in CANDIDATE_BITS:
        ratio = _zstd_ratio(bits, weight_decay)
        for name, rows, cols in shapes:
            # Parse layer index and component
            parts = name.split(".")
            layer_idx = int(parts[1])
            comp = "mlp" if "mlp" in name else "attn"

            raw = _quantized_payload_bytes(rows, cols, bits)
            compressed = raw / ratio

            key = (layer_idx, comp, bits)
            sizes[key] = sizes.get(key, 0) + compressed

    return sizes


def compute_fixed_overhead(
    vocab_size: int,
    dim: int,
    num_layers: int,
    weight_decay: float,
) -> float:
    """Bytes for embedding (fp16) + small tensors + serialization overhead."""
    # Embedding: fp16 passthrough
    embed_raw = vocab_size * dim * 2
    embed_compressed = embed_raw / _FP16_ZSTD_RATIO

    # Small tensors: ~2*dim per layer (norms, scales, etc.) as fp16
    small_raw = num_layers * 2 * dim * 2 * 6  # rough: 6 small tensors/layer
    small_compressed = small_raw / _FP16_ZSTD_RATIO

    return embed_compressed + small_compressed + TORCH_OVERHEAD_BYTES


# =============================================================================
# Sensitivity data loading
# =============================================================================

def load_sensitivity_from_analysis(path: str) -> dict:
    """Load from quant_analysis.py JSON.

    If the JSON has 'sensitivity_by_bits' (multi-bitwidth), uses real measured data.
    Falls back to extrapolation from int4 only if single-bitwidth data is all that's available.
    """
    with open(path) as f:
        data = json.load(f)

    num_layers = data["num_layers"]
    costs = {}  # (layer, component, bits) → BPB cost

    if "sensitivity_by_bits" in data:
        # Multi-bitwidth: use real measured per-component, per-bitwidth data
        measured_bits = {}
        for bits_str, sens in data["sensitivity_by_bits"].items():
            bits = int(bits_str)
            measured_bits[bits] = sens
            for layer_i in range(num_layers):
                for comp in ["attn", "mlp"]:
                    costs[(layer_i, comp, bits)] = sens[comp][layer_i]

        # For any CANDIDATE_BITS not measured, interpolate from nearest measured
        for bits in CANDIDATE_BITS:
            if bits in measured_bits:
                continue
            # Find nearest measured bitwidths above and below
            measured = sorted(measured_bits.keys())
            below = [b for b in measured if b < bits]
            above = [b for b in measured if b > bits]
            if below and above:
                b_lo, b_hi = max(below), min(above)
                # Log-linear interpolation
                frac = (bits - b_lo) / (b_hi - b_lo)
                for layer_i in range(num_layers):
                    for comp in ["attn", "mlp"]:
                        lo = costs[(layer_i, comp, b_lo)]
                        hi = costs[(layer_i, comp, b_hi)]
                        if lo > 0 and hi > 0:
                            costs[(layer_i, comp, bits)] = math.exp(
                                math.log(lo) * (1 - frac) + math.log(hi) * frac)
                        else:
                            costs[(layer_i, comp, bits)] = lo * (1 - frac) + hi * frac
            elif below:
                # Extrapolate down (fewer bits = more error)
                b_ref = max(below)
                for layer_i in range(num_layers):
                    for comp in ["attn", "mlp"]:
                        costs[(layer_i, comp, bits)] = costs[(layer_i, comp, b_ref)] * 0.25
            elif above:
                # Extrapolate up (more bits = more error)
                b_ref = min(above)
                for layer_i in range(num_layers):
                    for comp in ["attn", "mlp"]:
                        costs[(layer_i, comp, bits)] = costs[(layer_i, comp, b_ref)] * 3.5

        float_bpb = list(measured_bits.values())[0].get("float_bpt", 0)
    else:
        # Single-bitwidth fallback: extrapolate from int4
        sensitivity = data["sensitivity"]
        BIT_SCALING = {3: 3.5, 4: 1.0, 5: 0.25, 6: 0.06}
        for layer_i in range(num_layers):
            for comp in ["attn", "mlp"]:
                base_cost = sensitivity[comp][layer_i]
                for bits in CANDIDATE_BITS:
                    costs[(layer_i, comp, bits)] = base_cost * BIT_SCALING[bits]
        float_bpb = sensitivity.get("float_bpt", 0)

    return {"costs": costs, "num_layers": num_layers,
            "float_bpb": float_bpb, "source": path}


def load_sensitivity_from_csv(path: str) -> dict:
    """Load multi-bitwidth sensitivity from CSV.

    Expected format:
        layer,component,bits,bpb_gap
        0,attn,3,0.0045
        0,attn,4,0.0031
        ...
    """
    costs = {}
    num_layers = 0
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(row["layer"])
            comp = row["component"]
            bits = int(row["bits"])
            gap = float(row["bpb_gap"])
            costs[(layer, comp, bits)] = gap
            num_layers = max(num_layers, layer + 1)

    return {"costs": costs, "num_layers": num_layers, "float_bpb": 0, "source": path}


# =============================================================================
# Solver: greedy knapsack
# =============================================================================

def solve_allocation(
    sensitivity: dict,
    layer_sizes: dict[tuple[int, str, int], float],
    fixed_bytes: float,
    budget_bytes: float,
    max_bits: int = 5,
    min_bits: int = 3,
    use_polish: bool = False,
) -> dict:
    """Find per-(layer, component) bit allocation minimizing BPB cost within budget.

    Strategy: start everything at max_bits (expensive but accurate), then greedily
    demote the cheapest-to-demote items until we fit in budget. "Cheapest" means
    lowest BPB-cost-per-byte-saved ratio.
    """
    costs = sensitivity["costs"]
    num_layers = sensitivity["num_layers"]

    # Initialize: everything at max_bits
    allocation = {}
    for layer_i in range(num_layers):
        for comp in ["attn", "mlp"]:
            allocation[(layer_i, comp)] = max_bits

    def total_size():
        s = fixed_bytes
        for (layer_i, comp), bits in allocation.items():
            s += layer_sizes.get((layer_i, comp, bits), 0)
        return s

    def total_bpb_cost():
        c = 0.0
        for (layer_i, comp), bits in allocation.items():
            raw = costs.get((layer_i, comp, bits), 0)
            c += apply_polish_discount(raw) if use_polish else raw
        return c

    # If already fits, we're done
    if total_size() <= budget_bytes:
        return _build_result(allocation, sensitivity, layer_sizes, fixed_bytes,
                             budget_bytes, use_polish)

    # Greedy demotion: repeatedly demote the item with lowest BPB-cost / byte-saved
    while total_size() > budget_bytes:
        best_item = None
        best_ratio = float("inf")

        for (layer_i, comp), current_bits in allocation.items():
            if current_bits <= min_bits:
                continue
            target_bits = current_bits - 1
            if target_bits < min_bits:
                continue

            # BPB cost of demotion
            current_cost = costs.get((layer_i, comp, current_bits), 0)
            target_cost = costs.get((layer_i, comp, target_bits), 0)
            delta_bpb = target_cost - current_cost
            if use_polish:
                delta_bpb = apply_polish_discount(target_cost) - apply_polish_discount(current_cost)

            # Bytes saved
            current_size = layer_sizes.get((layer_i, comp, current_bits), 0)
            target_size = layer_sizes.get((layer_i, comp, target_bits), 0)
            bytes_saved = current_size - target_size
            if bytes_saved <= 0:
                continue

            ratio = delta_bpb / bytes_saved
            if ratio < best_ratio:
                best_ratio = ratio
                best_item = (layer_i, comp, target_bits)

        if best_item is None:
            break  # Can't demote further

        layer_i, comp, target_bits = best_item
        allocation[(layer_i, comp)] = target_bits

    return _build_result(allocation, sensitivity, layer_sizes, fixed_bytes,
                         budget_bytes, use_polish)


def _build_result(allocation, sensitivity, layer_sizes, fixed_bytes,
                  budget_bytes, use_polish):
    costs = sensitivity["costs"]
    num_layers = sensitivity["num_layers"]

    total_size = fixed_bytes
    total_bpb = 0.0
    layer_details = []

    for layer_i in range(num_layers):
        for comp in ["attn", "mlp"]:
            bits = allocation[(layer_i, comp)]
            size = layer_sizes.get((layer_i, comp, bits), 0)
            raw_cost = costs.get((layer_i, comp, bits), 0)
            eff_cost = apply_polish_discount(raw_cost) if use_polish else raw_cost
            total_size += size
            total_bpb += eff_cost
            layer_details.append({
                "layer": layer_i, "component": comp, "bits": bits,
                "size_kb": size / 1024, "bpb_cost": raw_cost,
                "bpb_cost_polished": apply_polish_discount(raw_cost),
            })

    # Build QUANT_BITS string
    quant_bits = _allocation_to_quant_bits(allocation, num_layers)

    return {
        "allocation": dict(allocation),
        "quant_bits_str": quant_bits,
        "total_size_mb": total_size / MB,
        "budget_mb": budget_bytes / MB,
        "headroom_mb": (budget_bytes - total_size) / MB,
        "fits": total_size <= budget_bytes,
        "total_bpb_cost": total_bpb,
        "total_bpb_cost_raw": sum(
            costs.get((l, c, allocation[(l, c)]), 0)
            for l in range(num_layers) for c in ["attn", "mlp"]
        ),
        "polish_enabled": use_polish,
        "layer_details": layer_details,
    }


def _allocation_to_quant_bits(allocation: dict, num_layers: int) -> str:
    """Convert allocation dict to QUANT_BITS env var string.

    Uses defaults where possible: if all attn layers share the same bits,
    emit just 'attn:N'. Only emit per-layer overrides for exceptions.
    """
    from collections import Counter

    # Find most common bits for each component
    attn_bits = [allocation[(i, "attn")] for i in range(num_layers)]
    mlp_bits = [allocation[(i, "mlp")] for i in range(num_layers)]

    attn_default = Counter(attn_bits).most_common(1)[0][0]
    mlp_default = Counter(mlp_bits).most_common(1)[0][0]

    parts = [f"attn:{attn_default}", f"mlp:{mlp_default}"]

    for i in range(num_layers):
        if attn_bits[i] != attn_default:
            parts.append(f"attn.{i}:{attn_bits[i]}")
        if mlp_bits[i] != mlp_default:
            parts.append(f"mlp.{i}:{mlp_bits[i]}")

    return ",".join(parts)


# =============================================================================
# Display
# =============================================================================

def print_result(result: dict, sensitivity: dict):
    num_layers = sensitivity["num_layers"]
    alloc = result["allocation"]

    print(f"\n{'='*70}")
    print(f"  Bit Allocation  |  budget={result['budget_mb']:.1f} MB  "
          f"polish={'ON' if result['polish_enabled'] else 'OFF'}")
    print(f"{'='*70}")
    print(f"  {'Layer':>5s}  {'Attn':>5s}  {'MLP':>5s}  "
          f"{'Attn BPB':>9s}  {'MLP BPB':>9s}  {'Attn KB':>8s}  {'MLP KB':>8s}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}")

    details_by_key = {(d["layer"], d["component"]): d for d in result["layer_details"]}
    for i in range(num_layers):
        ad = details_by_key[(i, "attn")]
        md = details_by_key[(i, "mlp")]
        cost_key = "bpb_cost_polished" if result["polish_enabled"] else "bpb_cost"
        print(f"  L{i:>3d}  int{ad['bits']}  int{md['bits']}  "
              f"{ad[cost_key]:>+9.4f}  {md[cost_key]:>+9.4f}  "
              f"{ad['size_kb']:>8.1f}  {md['size_kb']:>8.1f}")

    print(f"\n  Total size:     {result['total_size_mb']:.2f} MB "
          f"({'FITS' if result['fits'] else 'OVER'})")
    print(f"  Headroom:       {result['headroom_mb']:+.2f} MB")
    bpb_label = "BPB cost (post-polish)" if result["polish_enabled"] else "BPB cost"
    print(f"  {bpb_label}: {result['total_bpb_cost']:+.4f}")
    if result["polish_enabled"]:
        print(f"  BPB cost (raw):       {result['total_bpb_cost_raw']:+.4f}")
    print(f"\n  QUANT_BITS=\"{result['quant_bits_str']}\"")


def print_sweep(results: list[dict]):
    print(f"\n{'Budget MB':>10s}  {'Size MB':>8s}  {'Fits':>5s}  "
          f"{'BPB cost':>9s}  {'BPB raw':>9s}  {'Unique bits':>11s}")
    print(f"{'-'*10}  {'-'*8}  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*11}")
    for r in results:
        bits_used = sorted(set(r["allocation"].values()))
        bits_str = ",".join(str(b) for b in bits_used)
        print(f"{r['budget_mb']:>10.1f}  {r['total_size_mb']:>8.2f}  "
              f"{'Y' if r['fits'] else 'N':>5s}  "
              f"{r['total_bpb_cost']:>+9.4f}  {r['total_bpb_cost_raw']:>+9.4f}  "
              f"{bits_str:>11s}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimal per-layer bit allocation for Parameter Golf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input sources
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--analysis", type=str,
                     help="Path to quant_analysis.py JSON output")
    src.add_argument("--sensitivity-csv", type=str,
                     help="Path to multi-bitwidth sensitivity CSV")

    # Architecture (must match the checkpoint)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=float, default=3.0)
    parser.add_argument("--mlp-act", type=str, default="relu2")
    parser.add_argument("--vocab-size", type=int, default=1024)

    # Budget & compression
    parser.add_argument("--budget-mb", type=float, default=15.5,
                        help="Target artifact size in MB (default: 15.5, leaves margin)")
    parser.add_argument("--wd", type=float, default=0.04,
                        help="Weight decay (affects compression ratio estimate)")

    # Allocation constraints
    parser.add_argument("--max-bits", type=int, default=5,
                        help="Maximum bits to assign (default: 5)")
    parser.add_argument("--min-bits", type=int, default=3,
                        help="Minimum bits to assign (default: 3)")

    # Polish
    parser.add_argument("--polish", action="store_true",
                        help="Apply polish recovery discount to BPB costs")

    # Modes
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep budgets from 10-16 MB to show BPB-vs-size tradeoff")
    parser.add_argument("--compare", action="store_true",
                        help="Show side-by-side: with and without polish discount")
    parser.add_argument("--json", action="store_true",
                        help="Output result as JSON")

    args = parser.parse_args()

    # Load sensitivity
    if args.analysis:
        sensitivity = load_sensitivity_from_analysis(args.analysis)
    else:
        sensitivity = load_sensitivity_from_csv(args.sensitivity_csv)

    num_layers = sensitivity["num_layers"]

    # Compute per-layer sizes
    layer_sizes = compute_layer_sizes(
        num_layers=num_layers, dim=args.dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        mlp_act=args.mlp_act, weight_decay=args.wd,
    )
    fixed_bytes = compute_fixed_overhead(
        vocab_size=args.vocab_size, dim=args.dim,
        num_layers=num_layers, weight_decay=args.wd,
    )

    if args.sweep:
        results = []
        for budget_mb in [10, 11, 12, 13, 13.5, 14, 14.5, 15, 15.5, 16]:
            r = solve_allocation(
                sensitivity, layer_sizes, fixed_bytes,
                budget_bytes=budget_mb * MB,
                max_bits=args.max_bits, min_bits=args.min_bits,
                use_polish=args.polish,
            )
            results.append(r)
        print_sweep(results)
        # Also show the best one that fits
        fitting = [r for r in results if r["fits"]]
        if fitting:
            best = min(fitting, key=lambda r: r["total_bpb_cost"])
            print(f"\n  Best fitting: {best['budget_mb']:.1f} MB, "
                  f"BPB cost={best['total_bpb_cost']:+.4f}")
            print(f"  QUANT_BITS=\"{best['quant_bits_str']}\"")
    elif args.compare:
        budget_bytes = int(args.budget_mb * MB)
        r_no = solve_allocation(
            sensitivity, layer_sizes, fixed_bytes, budget_bytes,
            max_bits=args.max_bits, min_bits=args.min_bits, use_polish=False,
        )
        r_yes = solve_allocation(
            sensitivity, layer_sizes, fixed_bytes, budget_bytes,
            max_bits=args.max_bits, min_bits=args.min_bits, use_polish=True,
        )
        print("\n--- Without polish ---")
        print_result(r_no, sensitivity)
        print("\n--- With polish discount ---")
        print_result(r_yes, sensitivity)
        if r_no["quant_bits_str"] != r_yes["quant_bits_str"]:
            print("\n  Polish changes the allocation! The discount makes aggressive "
                  "quant cheaper, so the solver picks different tradeoffs.")
        else:
            print("\n  Same allocation with and without polish.")
    else:
        budget_bytes = int(args.budget_mb * MB)
        result = solve_allocation(
            sensitivity, layer_sizes, fixed_bytes, budget_bytes,
            max_bits=args.max_bits, min_bits=args.min_bits,
            use_polish=args.polish,
        )
        if args.json:
            # Strip non-serializable keys
            out = {k: v for k, v in result.items() if k != "allocation"}
            out["allocation"] = {f"{l}.{c}": b for (l, c), b in result["allocation"].items()}
            print(json.dumps(out, indent=2))
        else:
            print_result(result, sensitivity)


if __name__ == "__main__":
    main()
