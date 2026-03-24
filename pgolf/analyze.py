#!/usr/bin/env python3
"""
pgolf/analyze.py — Post-run analysis for Parameter Golf experiments.

Loads structured metrics from train_gpt_mlx.py, computes trajectory stats,
generates plots, and maintains a results ledger for cross-run comparison.

Usage:
    python -m pgolf.analyze <run_id> [--dir logs] [--desc "description"]
    python -m pgolf.analyze <run_id> --compare <other_run_id>
    python -m pgolf.analyze --list                # show results.tsv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(run_id: str, log_dir: Path) -> list[dict]:
    path = log_dir / f"{run_id}_metrics.jsonl"
    if not path.exists():
        print(f"WARNING: {path} not found", file=sys.stderr)
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_summary(run_id: str, log_dir: Path) -> dict | None:
    path = log_dir / f"{run_id}_summary.json"
    if not path.exists():
        print(f"WARNING: {path} not found", file=sys.stderr)
        return None
    return json.load(path.open())

# ---------------------------------------------------------------------------
# Trajectory stats
# ---------------------------------------------------------------------------

def compute_trajectory_stats(metrics: list[dict]) -> dict:
    train = [m for m in metrics if "train_loss" in m]
    val = [m for m in metrics if "val_bpb" in m]
    if not train:
        return {}

    losses = [m["train_loss"] for m in train]
    stats: dict = {
        "num_steps": len(train),
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "min_loss_step": train[losses.index(min(losses))]["step"],
    }

    # Smoothed losses (EMA, alpha=0.1)
    smooth = []
    ema = losses[0]
    for l in losses:
        ema = 0.9 * ema + 0.1 * l
        smooth.append(ema)
    stats["final_smooth_loss"] = smooth[-1]

    # End-of-run slope (last 10% of steps)
    if len(smooth) > 20:
        tail_start = int(len(smooth) * 0.9)
        tail = smooth[tail_start:]
        slope = (tail[-1] - tail[0]) / len(tail) if len(tail) > 1 else 0
        stats["end_slope"] = slope
        stats["still_improving"] = slope < -1e-5

    # Val trajectory
    if val:
        bpbs = [m["val_bpb"] for m in val]
        stats["best_val_bpb"] = min(bpbs)
        stats["final_val_bpb"] = bpbs[-1]

    # Throughput
    tok_s = [m["tok_s"] for m in train if "tok_s" in m]
    if tok_s:
        # Skip first few steps (warmup artifacts)
        steady = tok_s[min(3, len(tok_s)):]
        if steady:
            stats["mean_tok_s"] = sum(steady) / len(steady)

    step_ms = [m["step_ms"] for m in train if "step_ms" in m]
    if step_ms:
        steady = step_ms[min(3, len(step_ms)):]
        if steady:
            stats["mean_step_ms"] = sum(steady) / len(steady)

    return stats

# ---------------------------------------------------------------------------
# Results TSV
# ---------------------------------------------------------------------------

RESULTS_HEADER = "run_id\tscore\tval_bpb\tquant_mb\tsteps\tdescription\n"

def load_results(log_dir: Path) -> list[dict]:
    path = log_dir / "results.tsv"
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= len(header):
            rows.append(dict(zip(header, parts)))
    return rows


def append_result(log_dir: Path, run_id: str, summary: dict, description: str) -> None:
    path = log_dir / "results.tsv"
    if not path.exists():
        path.write_text(RESULTS_HEADER)
    quant_mb = summary.get("quant_file_bytes", 0) / (1024 * 1024)
    score = summary.get("score", 0)
    val_bpb = summary.get("val_bpb", 0)
    steps = summary.get("steps", 0)
    with path.open("a") as f:
        f.write(f"{run_id}\t{score:.4f}\t{val_bpb:.6f}\t{quant_mb:.2f}\t{steps}\t{description}\n")


def find_best(results: list[dict]) -> tuple[dict, float] | None:
    best = None
    for r in results:
        try:
            s = float(r["score"])
            if s > 0 and (best is None or s < best[1]):
                best = (r, s)
        except (ValueError, KeyError):
            continue
    return best

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _extract(metrics: list[dict], key: str) -> tuple[list, list]:
    steps, vals = [], []
    for m in metrics:
        if key in m and m[key] is not None:
            steps.append(m.get("step", 0))
            vals.append(m[key])
    return steps, vals


def generate_plots(
    metrics: list[dict],
    run_id: str,
    log_dir: Path,
    prev_metrics: list[dict] | None = None,
    prev_label: str = "prev",
) -> list[str]:
    if not HAS_MPL or not metrics:
        if not HAS_MPL:
            print("  (install matplotlib for plots)")
        return []

    plot_dir = log_dir / f"{run_id}_plots"
    plot_dir.mkdir(exist_ok=True)
    generated = []

    def _save(fig, name):
        path = plot_dir / name
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(path))

    datasets = [("current", metrics)]
    if prev_metrics:
        datasets.append((prev_label, prev_metrics))

    # 1. Loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, data in datasets:
        train = [m for m in data if "train_loss" in m]
        if train:
            steps = [m["step"] for m in train]
            raw = [m["train_loss"] for m in train]
            ax.plot(steps, raw, alpha=0.2, linewidth=0.5)
            # Smoothed
            ema, smooth = raw[0], []
            for l in raw:
                ema = 0.9 * ema + 0.1 * l
                smooth.append(ema)
            ax.plot(steps, smooth, label=f"{label} (smooth)", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "loss_curve.png")

    # 2. Throughput
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for label, data in datasets:
        s, v = _extract(data, "tok_s")
        if v:
            ax1.plot(s, v, label=label, linewidth=1)
        s, v = _extract(data, "step_ms")
        if v:
            ax2.plot(s, v, label=label, linewidth=1)
    ax1.set_xlabel("Step"); ax1.set_ylabel("tok/s"); ax1.set_title("Throughput")
    ax2.set_xlabel("Step"); ax2.set_ylabel("ms"); ax2.set_title("Step Time")
    ax1.legend(); ax2.legend()
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "throughput.png")

    # 3. LR schedule
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, data in datasets:
        s, v = _extract(data, "lr_mul")
        if v:
            ax.plot(s, v, label=label, linewidth=1.2)
    ax.set_xlabel("Step"); ax.set_ylabel("lr_mul"); ax.set_title("LR Multiplier")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, "lr_schedule.png")

    return generated

# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_summary(
    run_id: str,
    summary: dict | None,
    traj: dict,
    results: list[dict],
) -> None:
    print(f"\n{'='*60}")
    print(f"  Run: {run_id}")
    print(f"{'='*60}")

    if summary:
        quant_mb = summary.get("quant_file_bytes", 0) / (1024 * 1024)
        print(f"  val_bpb:        {summary.get('val_bpb', 0):.6f}")
        print(f"  quant size:     {quant_mb:.2f} MB")
        print(f"  score:          {summary.get('score', 0):.4f}")
        print(f"  params:         {summary.get('n_params', 0):,}")
        print(f"  steps:          {summary.get('steps', 0)}")
        print(f"  train time:     {summary.get('train_time_ms', 0)/1000:.1f}s")
        print(f"  compressor:     {summary.get('compressor', '?')}")

    if traj:
        print(f"\n  --- Trajectory ---")
        print(f"  first loss:     {traj.get('first_loss', '?')}")
        print(f"  final loss:     {traj.get('final_loss', '?')}")
        print(f"  final smooth:   {traj.get('final_smooth_loss', '?')}")
        print(f"  min loss:       {traj.get('min_loss', '?')} (step {traj.get('min_loss_step', '?')})")
        if "end_slope" in traj:
            status = "still improving" if traj.get("still_improving") else "converged/plateau"
            print(f"  end slope:      {traj['end_slope']:.8f} ({status})")
        if "mean_tok_s" in traj:
            print(f"  mean tok/s:     {traj['mean_tok_s']:.0f}")
        if "mean_step_ms" in traj:
            print(f"  mean step_ms:   {traj['mean_step_ms']:.1f}")

    if results:
        best = find_best(results)
        print(f"\n  --- Cross-Run ({len(results)} runs) ---")
        if best:
            print(f"  best score:     {best[1]:.4f} (run: {best[0]['run_id']})")
            if summary:
                delta = summary.get("score", 0) - best[1]
                print(f"  this vs best:   {'+' if delta > 0 else ''}{delta:.4f}")
        recent = results[-5:]
        print(f"  recent:")
        for r in recent:
            print(f"    {r.get('run_id', '?'):20s}  score={r.get('score', '?'):>8s}  bpb={r.get('val_bpb', '?'):>10s}  {r.get('description', '')}")

    print()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parameter Golf post-run analysis")
    parser.add_argument("run_id", nargs="?", help="Run ID to analyze")
    parser.add_argument("--dir", default="logs", help="Log directory (default: logs)")
    parser.add_argument("--desc", default="", help="Description for results.tsv")
    parser.add_argument("--compare", default=None, help="Run ID to overlay in plots")
    parser.add_argument("--list", action="store_true", help="List all results")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-append", action="store_true", help="Don't append to results.tsv")
    args = parser.parse_args()

    log_dir = Path(args.dir)

    if args.list:
        results = load_results(log_dir)
        if not results:
            print("No results yet.")
            return
        # Header
        print(f"{'run_id':20s}  {'score':>8s}  {'val_bpb':>10s}  {'quant_mb':>8s}  {'steps':>6s}  description")
        print("-" * 80)
        for r in results:
            print(f"{r.get('run_id', '?'):20s}  {r.get('score', '?'):>8s}  {r.get('val_bpb', '?'):>10s}  {r.get('quant_mb', '?'):>8s}  {r.get('steps', '?'):>6s}  {r.get('description', '')}")
        best = find_best(results)
        if best:
            print(f"\nBest: {best[0]['run_id']} score={best[1]:.4f}")
        return

    if not args.run_id:
        parser.error("run_id is required (or use --list)")

    run_id = args.run_id
    print(f"analyze: {run_id}")

    # Load data
    metrics = load_metrics(run_id, log_dir)
    summary = load_summary(run_id, log_dir)
    traj = compute_trajectory_stats(metrics) if metrics else {}
    results = load_results(log_dir)

    # Append to results
    if summary and not args.no_append:
        # Check if already in results
        existing = [r for r in results if r.get("run_id") == run_id]
        if existing:
            print(f"  (already in results.tsv, skipping append)")
        else:
            desc = args.desc or f"params={summary.get('n_params', 0):,} steps={summary.get('steps', 0)}"
            append_result(log_dir, run_id, summary, desc)
            results = load_results(log_dir)  # reload
            print(f"  appended to results.tsv")

    # Plots
    if not args.no_plots:
        prev_metrics = None
        prev_label = "prev"
        if args.compare:
            prev_metrics = load_metrics(args.compare, log_dir)
            prev_label = args.compare
        elif results and len(results) >= 2:
            # Auto-compare with previous run
            prev_candidates = [r for r in results if r.get("run_id") != run_id]
            if prev_candidates:
                prev_id = prev_candidates[-1]["run_id"]
                prev_metrics = load_metrics(prev_id, log_dir)
                prev_label = prev_id
        plots = generate_plots(metrics, run_id, log_dir, prev_metrics, prev_label)
        if plots:
            print(f"  plots: {', '.join(plots)}")

    # Print summary
    print_summary(run_id, summary, traj, results)


if __name__ == "__main__":
    main()
