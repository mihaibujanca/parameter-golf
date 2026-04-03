# Post-Training Pipeline

## Overview

Everything between "model finished training" and "submission artifact ready". Takes a trained checkpoint and produces a compressed artifact under 16MB.

## Pipeline Diagram

```
Trained checkpoint (.npz, float32, ~130MB)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. BASELINE EVAL                                        │
│    eval_val() on the float checkpoint                   │
│                                                         │
│    Establishes the ground truth BPB. This is the number │
│    we're trying to preserve through compression.        │
│    ▶ METRIC: float_bpb                                 │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 2. PER-LAYER SENSITIVITY ANALYSIS                       │
│    scripts/quant_analysis.py → per_layer_sensitivity()  │
│                                                         │
│    For each layer, quantize ONE component (attn or mlp) │
│    to int4 while rest stays float. Measure BPB impact.  │
│    → Ranks layers by quantization cost                  │
│    → Identifies cheap layers (demote to int4)           │
│       and expensive layers (keep at int5)               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 3. DESIGN QUANT ALLOCATION                              │
│    scripts/quant_analysis.py → design_quant_allocation() │
│                                                         │
│    Layers with sensitivity < threshold → int4           │
│    Sensitive layers → int5                              │
│    All attention → int4 (empirically cheap)             │
│    → QUANT_BITS string (e.g. "attn:4,mlp:5,mlp.1:4")  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 4. QUANTIZE + ROUNDTRIP EVAL                            │
│    train_gpt_mlx.py → quantize_state_dict_int8()        │
│                                                         │
│    Per-row quantization with GPTQ-lite clip search.     │
│    Each weight matrix row: find optimal clip percentile  │
│    minimizing reconstruction MSE. Store as int8 values   │
│    + float16 per-row scales.                            │
│    → qobj dict (quantized weights + scales + metadata)  │
│                                                         │
│    ▶ METRIC: quant_bpb (via eval_val on dequantized)   │
│    ▶ METRIC: quant_gap = quant_bpb - float_bpb         │
│                                                         │
│    ⚠ The quant gap MUST be measured on the same         │
│    checkpoint that goes into the artifact.               │
│    SWA and best-checkpoint have very different quant     │
│    sensitivity — don't mix them.                        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 5. WEIGHT PERMUTATION (lossless)                        │
│    scripts/weight_permutation.py → permute_mlp_qobj()   │
│                                                         │
│    Sort MLP hidden neurons by per-row scale.            │
│    Reorder fc.weight rows + proj.weight columns with    │
│    same permutation. Network is mathematically          │
│    identical. Groups similar-magnitude rows together     │
│    → ~1.1-1.5 MB compression savings.                  │
│    → Same qobj, reordered in place                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 6. BUILD ARTIFACT                                       │
│    scripts/build_artifact.py                             │
│                                                         │
│    Bundle quantized weights.                            │
│    Compress with zpaq (default) or brotli-11.           │
│    → Single compressed file (<16MB)                     │
│                                                         │
│    ▶ METRIC: artifact_size_mb                           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 7. VERIFY                                               │
│    scripts/verify_artifact.py                            │
│                                                         │
│    Load artifact, decompress, dequantize, reconstruct   │
│    model, run eval_val(). Compare against float model.  │
│                                                         │
│    ▶ METRIC: artifact_bpb (must match quant_bpb)       │
│    ▶ METRIC: score = artifact_bpb × artifact_size_mb   │
│    Comparison (float vs artifact):                      │
│    - KL p99 / KL max (sensitive tail metric)            │
│    - Top-1 agreement (prediction consistency)           │
│    - Logit cosine min (worst-case distortion)           │
│    - Artifact size ≤ 16,000,000 bytes                   │
└─────────────────────────────────────────────────────────┘
```

## Scripts

| Step | Script | Key function | Input | Output |
|------|--------|-------------|-------|--------|
| 1 | `train_gpt_mlx.py` | `eval_val()` | checkpoint | float_bpb |
| 2-3 | `scripts/quant_analysis.py` | `per_layer_sensitivity()`, `design_quant_allocation()` | checkpoint | QUANT_BITS string |
| 4 | `train_gpt_mlx.py` | `quantize_state_dict_int8()` | checkpoint + QUANT_BITS | qobj dict + quant_bpb |
| 5 | `scripts/weight_permutation.py` | `permute_mlp_qobj()` | qobj | qobj (mutated) |
| 6 | `scripts/build_artifact.py` | CLI | qobj | artifact (zpaq/br) |
| 7 | `scripts/verify_artifact.py` | CLI | artifact + float checkpoint | pass/fail |

## Automated Pipeline

Use `scripts/run_post_training_pipeline.sh` for reproducible runs:

```bash
# Config
export NUM_LAYERS=11 MLP_MULT=5.0 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16
CKPT=logs/warmdown_11L_5x_best.npz

# Fast tier (default, ~5-10 min): eval + sensitivity + quantize + compress + verify
./scripts/run_post_training_pipeline.sh $CKPT

# With gradient polish (~12-30 min extra):
./scripts/run_post_training_pipeline.sh $CKPT --polish

# Full pipeline (polish + corrections):
./scripts/run_post_training_pipeline.sh $CKPT --polish --corrections

# On workhorse (640d models):
./scripts/run_post_training_pipeline.sh $CKPT --polish --batch-seqs 16
```

All logs go to `logs/pipeline_<checkpoint>_<timestamp>/`.

### Pipeline Tiers

**Fast (default):** No training. Always runs.
1. Float eval (`eval_bpb`)
2. Sensitivity analysis (`quant_analysis.py`)
3. Quantize + weight permutation
4. Compress with zpaq
5. Verify artifact (`eval_bpb`)

**Expensive (opt-in):**
- `--polish`: Gradient polish (`float_polish.py`, ~500 steps)
- `--corrections`: PTQ corrections (`ptq_correction.py`)

### Manual Pipeline

```bash
# Steps 1-3: Sensitivity analysis → quant allocation
python3 scripts/quant_analysis.py $CKPT --save-analysis logs/analysis.json

# Steps 4-6: Build artifact (quantize + permute + compress with zpaq)
QUANT_BITS="attn:4,mlp:5,mlp.1:4,..." \
python3 scripts/build_artifact.py $CKPT \
    --no-correction \
    --compressor zpaq

# Step 7: Verify
python3 scripts/verify_artifact.py logs/artifact.zpaq \
    --float-checkpoint $CKPT
```

### Pre-SWA vs Post-SWA

SWA currently damages BPB on all tested models. Always build artifacts from `*_best.npz` (pre-SWA). If both checkpoints exist, the pipeline reports both BPB values for comparison.

## Metrics at Each Step

Every step that transforms the model should report BPB via `eval_val()` — the same function used during training. This catches regressions that comparison metrics (KL, Top-1) can miss.

| Metric | Source | What it tells you |
|--------|--------|-------------------|
| `float_bpb` | eval_val on float checkpoint | Ground truth, target to preserve |
| `quant_bpb` | eval_val on dequantized qobj | Cost of quantization |
| `quant_gap` | quant_bpb - float_bpb | How much quality we lose to compression |
| `artifact_bpb` | eval_val on model loaded from artifact | End-to-end verification (catches serialization bugs) |
| `score` | artifact_bpb × artifact_size_mb | Competition-comparable score |

## Critical Methodology Notes

**Always use eval_val() for BPB.** The training script's eval_val counts actual bytes per token using sentencepiece LUTs. Any other BPB calculation (manual softmax + NLL, bits_per_token without byte conversion) will give different numbers. All BPB metrics in the pipeline must use eval_val for comparability.

**Measure quant gap on the artifact checkpoint.** The training script reports quant roundtrip on the SWA/final model, but the artifact may be built from the best checkpoint. These have very different quant sensitivity (SWA smooths outliers, making weights more quantization-friendly). Always measure the gap on the same checkpoint that goes into the artifact.

**Isolated sensitivity ≠ compound error.** Per-layer sensitivity (step 2) tells you which layers are cheap to quantize. Compound error profiles tell you where errors concentrate under the actual quant config. These give different answers because errors propagate and compound across layers.

**Always verify end-to-end.** The artifact must be loaded, decompressed, dequantized, and evaluated. Any bug in serialization/deserialization silently corrupts quality. artifact_bpb should match quant_bpb — if it doesn't, there's a serialization bug.

## Known Limitations

**PTQ corrections require the float model at inference.** The current correction architecture (scripts/ptq_correction.py) computes `error = h_quant - h_float` using the float model's hidden states, then predicts a correction from that error. This means corrections cannot be used in a standalone artifact — the float model isn't available at deployment. The near-zero KL divergence reported by corrections is real but not achievable without the float oracle.

For corrections to be usable in submissions, they'd need to be rewritten to work from `h_quant` alone (e.g. a learned residual on the quantized hidden state, not on the quantization error).

## On H100

Steps 2-3 (sensitivity analysis) are pre-computed locally and hardcoded as constants in the CUDA `train_gpt.py`, OR run online within the 600s training budget (~20-30s overhead). Steps 4-6 run after training completes, within the training budget. Step 7 runs during the eval budget.
