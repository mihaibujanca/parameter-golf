# Logging & Evaluation Standards

## Run ID Naming

**Format:** `{layers}L_{dim}d_{mlp}x_{description}`

Examples:
- `13L_512d_3x_warmdown_from14k` — 13 layers, 512d, MLP 3x, warmdown from step 14k
- `13L_640d_322_7k_wd70` — 13 layers, 640d, tapered 3/2/2 MLP, 7k steps, warmdown 70%
- `11L_512d_5x_polish_int4` — 11 layers, 512d, MLP 5x, polished for int4

For short experiments / smoke tests, UUID run IDs are acceptable. For runs that produce checkpoints worth keeping, always set `RUN_ID` to a descriptive name.

**Non-standard suffixes:** `_best` (best val checkpoint), `_mlx_model` (final/SWA model), `_step{N}` (periodic checkpoint).

## Files Per Run

Every training run emits to `logs/`:

| File | Contents | Always? |
|------|----------|---------|
| `{run_id}.txt` | Human-readable training log | Yes |
| `{run_id}_metrics.jsonl` | Per-step structured metrics | Yes |
| `{run_id}_summary.json` | Final results + full config snapshot | Yes |
| `{run_id}_best.npz` | Best val checkpoint (use for artifacts) | Yes |
| `{run_id}_mlx_model.npz` | Final model (SWA if enabled) | Yes |
| `{run_id}_step{N}.npz` | Periodic checkpoint | If CHECKPOINT_EVERY > 0 |

Post-training pipeline adds to `logs/pipeline_{run_id}_{timestamp}/`:

| File | Contents |
|------|----------|
| `01_float_eval.txt` | Float BPB via eval_val |
| `02_sensitivity.txt` | Per-layer quant sensitivity |
| `03_polish.txt` | Gradient polish log (if --polish) |
| `04_build_artifact.txt` | Quantize + compress |
| `05_verify.txt` | Artifact verification |

## Log Header (training)

Every training log starts with these structured lines (key:value format, parseable):

```
run_id:{id}
resumed_from:{checkpoint_path}     (if resuming)
mlx_version:{version}
model_params:{n} vocab_size:{v} layers:{l} dim:{d} heads:{h} kv_heads:{kv} seq_len:{s}
iterations:{n} train_batch_tokens:{t} grad_accum_steps:{g} ...
optimizer:muon+adam ...
bigram:{vocab}x{dim} swa:{bool} muon_wd:{wd} compressor:{comp}
```

## Summary JSON Schema

The `_summary.json` is the authoritative record of a training run. Required fields:

```json
{
  "run_id": "string",
  "val_loss": 2.358,              // CE in nats (post-quant)
  "val_bpb": 1.413,               // BPB (post-quant) — THE competition metric
  "val_loss_pre_quant": 2.356,    // CE in nats (float)
  "val_bpb_pre_quant": 1.412,    // BPB (float)
  "quant_gap_bpb": 0.001,        // val_bpb - val_bpb_pre_quant
  "quant_file_bytes": 20495013,   // raw compressed artifact size
  "score": 27.63,                 // val_bpb × (quant_file_bytes / 1e6)
  "n_params": 31815273,
  "steps": 2000,
  "train_time_ms": 3695757.0,
  "compressor": "zstd",
  "config": { ... }               // full Hyperparameters snapshot
}
```

## Standardized Evaluation Report

The pipeline report tracks BPB and size through every transformation. Each lossy step reports its cost. Each lossless step reports its savings.

```
=== Evaluation Report ===
Checkpoint:   logs/13L_640d_322_7k_wd70_best.npz
Model:        13L/640d, MLP 3x (tapered 322) lrelu2, 49.2M params

--- SWA comparison ---
Best ckpt BPB:    1.3598
SWA ckpt BPB:     1.3719  (SWA damage: +0.0121)
→ Using: best checkpoint

--- Float baseline ---
val_loss:         2.2710
val_bpb:          1.3598

--- Quantization (lossy) ---
Config:           attn:int4, mlp:int5 (mixed, 5 layers int5)
Post-quant BPB:   1.3703  (gap: +0.0105)
Float params:     192.4 MB → raw quant: 32.6 MB  (saved 159.8 MB)

--- Gradient polish (lossy, optional) ---
Pre-polish BPB:   1.3703  (post-quant baseline)
Post-polish BPB:  1.3668  (recovered 0.0035 BPB, 33% of quant gap)

--- Corrections (lossy, optional) ---
Layers:           [0, 5, 9]
Pre-corr BPB:     1.3668  (post-polish)
Post-corr BPB:    1.3645  (recovered 0.0023 BPB, 22% of remaining gap)
Overhead:         0.12 MB (48K correction params)
[or: skipped]

--- Weight permutation (lossless) ---
Compressed w/o permute: 12.85 MB
Compressed w/ permute:  12.09 MB  (saved 0.76 MB)

--- Compression (lossless) ---
Raw payload:      32.62 MB
Compressed:       12.09 MB  (ratio 2.70x, zpaq method 5)

--- Final ---
Artifact:         12.09 MB  (budget: 16.00 MB, margin: +3.91 MB)
Artifact BPB:     1.3645
Score:            16.50  (1.3645 × 12.09)

--- Cumulative breakdown ---
Step              BPB       Δ BPB      Size (MB)   Δ Size
Float             1.3598    —          192.4       —
Quantization      1.3703    +0.0105    32.6        -159.8
Polish            1.3668    -0.0035    32.6        0.0
Permute+compress  1.3668    0.0000     12.09       -20.5
Corrections       1.3645    -0.0023    12.21       +0.12
```

### Principles

1. **Every lossy step reports:** BPB before, BPB after, gap or recovery percentage
2. **Every lossless step reports:** size before, size after, savings
3. **SWA is always compared** if both checkpoints exist — never silently used
4. **Cumulative breakdown table** at the end shows the full cost/benefit chain

### What NOT to report as BPB

- `val_loss / ln(2)` — this is **bits-per-token (BPT)**, ~2.45x larger than BPB for sp1024
- Any `quick_eval` / `quick_ce` output — these are CE in nats, for intermediate monitoring only
- Manual softmax + NLL calculations

## Verification

Verification is a **separate mandatory step** that proves the artifact is self-consistent.

```
=== Verification ===
Artifact:         logs/13L_640d_322_artifact.zpaq (12.09 MB)
Decompress:       OK (32.62 MB)
Dequantize:       OK (49.2M params)

Float BPB:        1.3598  (expected: 1.3598, Δ: 0.0000) ✓
Artifact BPB:     1.3645  (expected: 1.3645, Δ: 0.0000) ✓
Score:            16.50   (expected: 16.50)              ✓
Fits budget:      YES (12.09 MB ≤ 16.00 MB)             ✓
RESULT:           PASS
```

### Running verification

```bash
# Manual (always available):
.venv/bin/python3 scripts/verify_artifact.py <artifact> --float-checkpoint <ckpt>

# Before committing a new best result (mandatory):
.venv/bin/python3 scripts/verify_artifact.py <artifact> \
    --float-checkpoint <ckpt> \
    --expected-bpb <reported_bpb> --strict
```

The `--strict` flag fails with non-zero exit if artifact BPB differs from expected by more than 0.0005, or if the artifact exceeds the 16 MB budget. Use this as a gate before updating RESULTS.md.

## Finding Information in Logs

| Question | Where to look |
|----------|---------------|
| What BPB did run X achieve? | `logs/{run_id}_summary.json` → `val_bpb` |
| What arch config was used? | `logs/{run_id}_summary.json` → `config` |
| Is the model still improving? | `logs/{run_id}_metrics.jsonl` — check end slope |
| What checkpoint should I use? | `logs/{run_id}_best.npz` (never SWA/final) |
| How long did training take? | `logs/{run_id}_summary.json` → `train_time_ms` |
| What was the quant gap? | `logs/{run_id}_summary.json` → `quant_gap_bpb` |
| Per-step loss curve? | `logs/{run_id}_metrics.jsonl` |
| Full training output? | `logs/{run_id}.txt` |
| Cross-run comparison? | `python -m pgolf.analyze --list` or `logs/results.tsv` |
