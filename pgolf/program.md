# Parameter Golf — Experiment Protocol

Autonomous experiment loop for minimizing **score = file_size × val_bpb**.

## Goal

Train the best language model that fits in a **16 MB** quantized artifact, evaluated by bits-per-byte on FineWeb validation. Lower score is better. You have far more leeway for architecture changes than in autoresearch — the constraint is the artifact size, not the code.

## What you can change

- **Architecture**: layers, width, heads, MLP structure, attention variants, skip connections, embeddings, activation functions, normalization, weight tying strategies
- **Quantization**: bit widths (int5/int6/int8), mixed precision per layer type, embedding handling
- **Compression**: compressor choice (zstd/zlib), weight decay (pushes weights toward zero → better compression)
- **Training dynamics**: optimizer hyperparameters, schedules, batch size, sequence length, warmdown
- **Tokenizer**: vocab size (1024 vs 8192), different tokenizer models

The only file you modify is `train_gpt_mlx.py` (hard cap: 1500 lines).

## Tools

| Tool | Purpose |
|------|---------|
| `python -m pgolf.analyze <run_id>` | Post-run: trajectory stats, plots, results tracking |
| `python -m pgolf.analyze --list` | Show all runs and best score |
| `python -m pgolf.budget` | Predict artifact size before training |
| `python -m pgolf.budget --sweep` | Find biggest model that fits in 16 MB |
| `analyze_checkpoint.py` | Deep diagnostics: SVD rank, dead neurons, quant sensitivity, logit lens |
| `analyze_quant_sensitivity.py` | Per-layer quantization degradation |
| `smoke.sh` | Fast sanity check (~2 min) |

## The Experiment Loop

**LOOP:**

### 0. Refresh context
Re-read this file. Context drifts — this takes 2 seconds and prevents wasted experiments.

### 1. Train
```bash
.venv/bin/python3 train_gpt_mlx.py
```
Produces in `logs/`: `{run_id}.txt` (text log), `{run_id}_metrics.jsonl` (per-step metrics), `{run_id}_summary.json` (final results), `{run_id}_mlx_model.int6.pt{z,s}` (quantized artifact).

### 2. Analyze
```bash
python -m pgolf.analyze <run_id> --desc "what you changed"
```
This computes trajectory stats (still improving? end slope?), generates plots (loss curve overlaid with previous run, throughput, LR schedule), appends to `logs/results.tsv`, and prints a summary with cross-run comparison.

### 3. Investigate
Look at the plots in `logs/{run_id}_plots/`. Then dig deeper:

```python
import json
history = [json.loads(l) for l in open(f"logs/{run_id}_metrics.jsonl")]
# Compare loss trajectories, find divergence points, check throughput
```

Things to look for:
- **Still improving at end?** → try longer warmdown or more steps
- **Loss plateau?** → architecture change needed, not just hyperparameters
- **Throughput drop?** → model too large for batch size, or inefficient op
- **Score regression despite better bpb?** → artifact grew; check quantization

### 4. Decide: keep or revert
Compare against the **all-time best score** (not just previous run):
- **New best**: keep the change, note what worked in your analysis
- **Worse**: revert `train_gpt_mlx.py` to the best version. Don't build on regressions.

### 5. Plan next change
Pick based on what the investigation told you. The three levers of score are:
1. **val_bpb** — architecture quality, training dynamics
2. **file_size** — quantization, compression, parameter efficiency
3. **Both** — weight decay (helps compression + regularization), architecture choices that are parameter-efficient

### 6. Implement and repeat
Modify `train_gpt_mlx.py`. Commit with a clear description. Go to step 1.

## Key metrics in results.tsv

| Column | Meaning |
|--------|---------|
| `score` | file_size_MB × val_bpb — **the competition metric** |
| `val_bpb` | Bits per byte on FineWeb validation |
| `quant_mb` | Quantized artifact size in MB |
| `steps` | Training steps completed |

## Rules

- **Compare against all-time best**, not just the previous run.
- **Use `pgolf.budget --sweep`** before trying a bigger model — know if it fits in 16 MB.
- **Don't grid search blindly.** Look at loss curves. Form hypotheses about *why*.
- **Simpler is better** at equal score. Removing complexity for same score is a win.
- **Log everything.** Always run `pgolf.analyze` after training. Write what you observed.
