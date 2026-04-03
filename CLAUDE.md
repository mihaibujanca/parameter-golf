# Parameter Golf

OpenAI competition: minimize BPB on FineWeb validation. Constraints: 16MB compressed artifact, 10-minute training on 8×H100. Current SOTA cluster is ~1.145 BPB.

## Metric: BPB (bits per byte), NOT bits per token

**Always report BPB, never bits-per-token.** BPB is the tokenizer-agnostic competition metric:
```
bits_per_token = val_loss / ln(2)
val_bpb = bits_per_token × (total_scored_tokens / total_scored_bytes)
```
The bytes/token ratio depends on the tokenizer (sp1024 ≈ 2.45 bytes/token). Raw `CE / ln(2)` gives bits-per-token (~3.95), NOT BPB (~1.61). These differ by ~2.45×.

**Always use `eval_val()` with `build_sentencepiece_luts()` for proper BPB.** Never compute BPB as `loss / ln(2)` — that's bits-per-token. Quick eval helpers that skip the sentencepiece LUTs produce the wrong metric.

## Project layout

- `train_gpt_mlx.py` — MLX training script for local iteration (M4 Pro). This is the main file you'll edit.
- `train_gpt.py` — CUDA/H100 training script (for final submissions). Mirror structure of MLX version.
- `records/` — Competition submission artifacts.
- `logs/` — Training run outputs and checkpoints.
- `data/datasets/` — Tokenized FineWeb shards (sp1024). `data/tokenizers/` has BPE models.
- `pgolf/budget.py` — Compression budget calculator.
- `pgolf/analyze.py` — Post-run analysis: trajectory stats, plots, cross-run tracking.
- `scripts/` — Tracked utility scripts. See "Scripts inventory" section below.
- `docs/` — Local research notes (gitignored): AGENDA.md, NEXT_STEPS.md, PR_DEEP_DIVE.md, etc.
- `local_scripts/` — Local one-off analysis scripts (gitignored): analyze_*.py, run_*.sh.

## Scripts inventory

**Quantization evaluation:**
- `quant_gap_test.py` — Multi-config BPB eval (int2–int8, mixed precision, hadamard). The main "how much does quantization hurt" tool.
- `quant_analysis.py` — Per-layer isolated sensitivity: quantize one layer to int4, rest float. For designing mixed-precision allocations.

**Error analysis:**
- `error_decomposition.py` — 4 experiments: sign flips (A), attn vs MLP attribution (B), per-weight-matrix (C), rounding vs saturation (D). Also provides `block_forward_decomposed()` and `forward_collect_preskip()` used by other scripts.
- `error_attribution.py` — Logit-space KL divergence impact per layer. Provides `forward_collect_all()`.
- `correctability_analysis.py` — Manifold distortion analysis: pre/post-nonlinearity affine R², ReLU flip rates, local vs propagated error decomposition, correctability scores.

**Correction approaches:**
- `correction_sensitivity.py` — Per-matrix BPB impact of applying exact E correction (E=W_quant-W_float) at int4. Measures which weight matrices are worth correcting.
- `float_polish.py` — STE gradient polish: fine-tune with fake quantization so weights land on grid-friendly positions. Proven: 33% recovery at int4.
- `gptq_quant.py` — GPTQ optimal rounding using calibration Hessian. Untested.
- `bias_correction.py` — Mean bias E[h_float-h_quant] per layer. Untested.

**Dead ends (kept for reference):**
- `correction_mse_standalone.py` — MSE-trained correction nets. Concluded: matches OLS ceiling, ~8% recovery. Also provides `build_model()` and `forward_collect_hidden()` used by other scripts.
- `correction_diagnostic.py` — 7-experiment diagnostic proving linear corrections cap at 8%. Concluded.
- `correction_ce.py` — CE-trained corrections. 2.4 mBPB at best, joint training overfits.
- `error_spectrum.py` — One-off analysis.
- `layer_sensitivity.py` — Superseded by `quant_analysis.py`.

## Running locally (M4 Pro 24GB)

**Python environment:** Always use `.venv/bin/python3` (not system python). The venv has mlx, sentencepiece, zstandard, etc.

**M4 Pro measured performance** (9L/512d default model, 2026-03-24):
- Step time: ~990ms/step at `TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2`
- Throughput: ~16.5k tok/s
- 10k steps ≈ 2h 45min, 80k steps ≈ 22 hours

**Baseline results** (default 9L/512d, 16K batch, M4 Pro):

| Run | Steps | val_bpb (post-quant) | quant MB | score | tokens seen |
|-----|-------|---------------------|----------|-------|-------------|
| baseline_10k | 10k | 1.5005 | 11.15 | 16.74 | 164M |
| overnight_long | 80k | 1.4428 | 10.41 | 15.02 | 1.3B |

Val BPB trajectory (overnight_long): 1.6531 @10k → 1.6393 @20k → 1.6272 @50k → 1.4242 @80k (big final drop from warmdown+SWA).
Still improving at 80k (end slope -0.008), but diminishing returns: 8x steps bought 0.058 BPB.
Checkpoints saved every 5k steps in `logs/overnight_long_step{N}.npz`. Resume with `RESUME_CHECKPOINT=logs/overnight_long_step{N}.npz`.

Note: M4 batch (16K tokens/step) is 32x smaller than H100 (524K). Same model on H100 sees 10.5B tokens in 20k steps vs 1.3B here.

**Standard M4 local config** (override H100-sized defaults):
```bash
TRAIN_BATCH_TOKENS=16384 \
GRAD_ACCUM_STEPS=2 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_BATCH_SIZE=65536 \
VAL_MAX_TOKENS=1048576 \
.venv/bin/python3 train_gpt_mlx.py
```

Key differences from defaults: `TRAIN_BATCH_TOKENS` (16384 vs 524288), `GRAD_ACCUM_STEPS` (2 vs 8), `MAX_WALLCLOCK_SECONDS` (0=disabled vs 600), `VAL_BATCH_SIZE` (65536 vs 524288), `VAL_MAX_TOKENS` (1M vs full split).

## Running on workhorse (Mac Studio M2 Max 32GB)

**Access:** `ssh mihai@workhorse`, project at `~/Projects/parameter-golf`.
**Python:** `.venv/bin/python3`, MLX 0.29.3. Same codebase as local (minus post-processing scripts not yet synced).

**MLX 0.29.3 vs local 0.31.1:** 0.31.1 has an intermittent segfault (`objc: Method cache corrupted`) triggered by repeated quantize/dequantize/model.update cycles on 640d models. Non-deterministic — same code sometimes passes, sometimes crashes. 0.29.3 is stable. Run all 640d quantization analysis and polish on workhorse until the local MLX is updated.

**Gradient polish memory limits (13L/640d/3x, measured 2026-04-03):**

| batch_seqs | Tokens/step | Peak Metal | Step time | Status |
|------------|------------|-----------|-----------|--------|
| 16 | 16K | 21.7 GB | 1.5s | OK — 10 GB headroom |
| 20 | 20K | 27.1 GB | 1.9s | OK — 5 GB headroom |
| 24 | 24K | 32.1 GB | 3.3s | At limit — swapping starts |
| 28 | 28K | 35.7 GB | 9.8s | Swap-bound, 6.5x slower |
| 32 | 32K | 40.7 GB | 30.2s | Swap-bound, 20x slower |

**Use `--batch-seqs 16` for 640d polish.** 500 steps ≈ 12.5 min. Do NOT use batch 24+ — exceeds 32GB and thrashes swap. The cliff is sharp: batch 20→24 doubles step time, 24→32 is 20x.

**zpaq:** Available at `/opt/homebrew/bin/zpaq`. Use `--compressor zpaq` in `build_artifact.py`.

## Workflows

**Smoke test:** Set low ITERATIONS, small VAL_MAX_TOKENS, short wallclock. Always compute eval overhead before running (sliding window stride=64 is 16x slower). See `smoke.sh` for a <2 min example.

**Launching experiments:** Print the full command with ALL env vars. Call out any that differ from `Hyperparameters` defaults. Flag defaults that could cap training (MAX_WALLCLOCK_SECONDS=600, MUON_MOMENTUM).

**Warmdown-only runs** (resume from checkpoint, LR decay to zero + SWA):
```bash
RESUME_CHECKPOINT=logs/<checkpoint>.npz \
ITERATIONS=<warmdown_steps> WARMDOWN_ITERS=<warmdown_steps> \
MAX_WALLCLOCK_SECONDS=0 \
MUON_MOMENTUM_WARMUP_STEPS=0 \
SWA_ENABLED=1 SWA_EVERY=50 \
<...other model/training env vars must match the original run...> \
.venv/bin/python3 train_gpt_mlx.py
```
Key points:
- `ITERATIONS=N WARMDOWN_ITERS=N` with `MAX_WALLCLOCK_SECONDS=0` gives step-based linear LR decay 1→0 over N steps.
- `MUON_MOMENTUM_WARMUP_STEPS=0` — optimizer state is NOT restored from checkpoint (only model weights are). Default momentum warmup (500 steps) wastes early warmdown steps ramping momentum; set to 0.
- SWA starts at `SWA_START_FRAC` of peak LR (default 0.5), so halfway through the run.
- **SWA currently hurts BPB on every tested model** (see below). The best checkpoint (pre-SWA) is always better. Artifact should be built from `_best.npz`, not the SWA state. Investigating why SWA damages quality is a TODO.
- All model architecture env vars (NUM_LAYERS, MLP_MULT, XSA_LAST_N, etc.) must match the original run exactly.
- The 20-step compilation warmup is just MLX graph priming — no weight updates, safe to keep.

**Warmdown-on-warmdown** (continuing from an already-warmed-down checkpoint):
Never resume a warmed-down checkpoint with full LR — the high-LR steps destroy what warmdown achieved. Instead, scale down base LRs to ~25% of originals:
```bash
RESUME_CHECKPOINT=logs/<warmed_down_checkpoint>.npz \
MATRIX_LR=0.01 SCALAR_LR=0.01 TIED_EMBED_LR=0.0125 \
ITERATIONS=<steps> WARMDOWN_ITERS=<steps> \
MAX_WALLCLOCK_SECONDS=0 MUON_MOMENTUM_WARMUP_STEPS=0 \
SWA_ENABLED=1 SWA_EVERY=50 \
<...model env vars...> \
.venv/bin/python3 train_gpt_mlx.py
```
This starts at 25% of the original peak LR and decays to zero — effectively continuing the decay from where the first warmdown left off.

**Time estimates:** Always measure step_avg from the first 10 steps of the actual config before quoting ETAs. Step times vary significantly across model configs (1.7-2.0s/step observed for 11-13L models). Do not extrapolate from one config to another.

## Structured logging & analysis

Every training run emits three files in `logs/`:
- `{run_id}.txt` — human-readable text log (unchanged)
- `{run_id}_metrics.jsonl` — per-step structured metrics (train_loss, lr_mul, step_ms, tok_s, val_loss, val_bpb)
- `{run_id}_summary.json` — final results: score, val_bpb, quant_file_bytes, n_params, full config snapshot

**Post-run analysis:**
```bash
python -m pgolf.analyze <run_id> --desc "what you changed"   # analyze + append to results.tsv
python -m pgolf.analyze --list                                 # cross-run leaderboard
python -m pgolf.analyze <run_id> --compare <other_id>          # overlay loss curves
```

This produces loss curve plots (with previous-run overlay), throughput/step-time charts, LR schedule, and trajectory stats (end slope, still-improving flag). Plots go to `logs/{run_id}_plots/`. Results accumulate in `logs/results.tsv`.

**Always run analyze after training.** The trajectory stats tell you whether the model was still improving at cutoff (need more steps/warmdown) or plateaued (need architecture change). The score breakdown (val_bpb × file_size_MB) catches regressions where better bpb is offset by a larger artifact.

**Key metric columns in results.tsv:** score (the competition metric), val_bpb, quant_mb, steps.

## Git

- Work happens on `main` with feature branches when needed.
- The working tree may have uncommitted changes from manual editing between sessions. Don't assume HEAD is the current state — check `git status` and `git diff` before doing anything destructive.
- When asked to separate "your changes" from existing state: your changes are what you did in this session. Everything else was already there. Use the file-history-snapshot or diff against HEAD to distinguish.

## Mixed precision quantization & PTQ correction methodology

**Two-stage analysis required for mixed precision + correction:**

1. **Per-layer isolated sensitivity** (one layer int4, rest float): identifies which layers are cheap/expensive to quantize. Use this to design the quant allocation (which layers get int4 vs int5).

2. **Compound error profile** (all layers at their assigned bit-widths simultaneously): measures actual per-layer hidden state error under the real quant config. Errors compound across layers — a layer that looks "free" in isolation can contribute significant error when upstream layers are also quantized. **Correction layer placement MUST be based on this profile, not on isolated sensitivity.**

The compound error profile is computed by running both float and quantized models forward, collecting hidden states at every layer boundary, and measuring per-layer RMSE + DeltaErr (error added by each layer). Place corrections at the layers with the highest DeltaErr.

**Validated example:** On SUGAR 11L/4x, isolated sensitivity suggested L0,2,4 for correction (36% recovery). Compound error profile identified L0,5,8 as the actual error concentration points (95% recovery). The difference is 60 percentage points of recovery quality.

**Never place corrections based only on isolated per-layer sensitivity.** Always run the compound error analysis on the actual quant config first.

**Never hardcode correction layers or quant allocations.** The sensitivity profile changes with architecture, training duration, and activation function. Always run `scripts/quant_analysis.py` on the actual checkpoint — it's cheap (<5 min) compared to the cost of wrong placement (36% vs 95% recovery).

## SWA damages quality (consistent finding, 2026-04-02)

SWA hurts BPB on every model tested. The best checkpoint (pre-SWA) always outperforms SWA averaging:

| Model | Best ckpt BPB | SWA BPB | SWA damage (mBPB) |
|-------|--------------|---------|-------------------|
| 13L/3x/640d | 1.3769 | 1.3572* | +20 |
| 11L/4x/640d | 1.3848 | 1.3654* | +19 |
| 13L/3x/512d | 1.4010 | 1.4135 | +12.5 |
| 11L/5x/512d | 1.4097 | 1.4391 | +29.4 |

\*SWA BPB appears lower but this is the SWA model evaluated directly — the _best.npz checkpoint (which the artifact is built from) has worse BPB because SWA weights aren't saved to it correctly, or the averaging degrades the weights. TODO: investigate why. For now, always build artifacts from `_best.npz` and treat SWA BPB as unreliable.

## Research context

Three levers: (A) compression — fit more params in 16MB, (B) architecture — more capacity per param, (C) training — extract more in 10 min. The field has converged on a consensus stack (int6 QAT, MLP 3x, 10-11 layers, SmearGate, BigramHash, SWA, zstd-22). Gains come from stacking proven techniques and finding new orthogonal improvements, not from reimplementing what's known.

Short-run results on M4 are directionally useful but not definitive. Quant sensitivity gaps widen with training. Don't harden findings from <500-step runs as permanent rules.

# Rules:

1500 lines rule for train_gpt.py and train_gpt_mlx.py are **SOFT_CONSTRAINTS**. This is for readability not for obsessing over.