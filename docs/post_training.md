# Post-Training Pipeline

## Overview

Everything between "model finished training" and "submission artifact ready". Takes a trained checkpoint and produces a compressed artifact under 16MB with near-float quality.

## Pipeline Diagram

```
Trained checkpoint (.npz, float32, ~130MB)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. PER-LAYER SENSITIVITY ANALYSIS                       │
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
│ 2. DESIGN QUANT ALLOCATION                              │
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
│ 3. QUANTIZE MODEL                                       │
│    train_gpt_mlx.py → quantize_state_dict_int8()        │
│                                                         │
│    Per-row quantization with GPTQ-lite clip search.     │
│    Each weight matrix row: find optimal clip percentile  │
│    minimizing reconstruction MSE. Store as int8 values   │
│    + float16 per-row scales.                            │
│    → qobj dict (quantized weights + scales + metadata)  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 4. WEIGHT PERMUTATION (lossless)                        │
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
│ 5. COMPOUND ERROR PROFILE                               │
│    scripts/quant_analysis.py → compound_error_profile()  │
│                                                         │
│    Run float + quantized models forward with ALL layers  │
│    at their assigned bit-widths simultaneously.          │
│    Measure per-layer DeltaErr (error added by each       │
│    layer under actual compound quantization).            │
│                                                         │
│    ⚠ NOT the same as isolated sensitivity (step 1).     │
│    Errors compound across layers. A layer that looks     │
│    "free" in isolation can be the biggest error source   │
│    when everything is quantized together.                │
│    → Correction layer placement (top-K by DeltaErr)     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 6. TRAIN PTQ CORRECTIONS                                │
│    scripts/ptq_correction.py → train_corrections()      │
│                                                         │
│    Small linear CorrectionNets at high-DeltaErr layers. │
│    Input: quantization error (h_quant - h_float).       │
│    Output: correction to add to h_quant.                │
│    Trained on train data, best-checkpoint tracking.      │
│    100 epochs, 64 sequences, ~10 min on M4.             │
│    → corrections dict + saved .npz                     │
│                                                         │
│    Correction weights are near-binary (values cluster    │
│    at -1 and 0) → compress to nearly nothing at int8.   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 7. BUILD ARTIFACT                                       │
│    scripts/build_artifact.py                             │
│                                                         │
│    Bundle quantized weights + int8 corrections.         │
│    Compress with brotli-11.                             │
│    → Single .br file (<16MB)                            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 8. VERIFY                                               │
│    scripts/verify_artifact.py                            │
│                                                         │
│    Load artifact, decompress, dequantize, reconstruct   │
│    corrections, run eval on val data. Compare against   │
│    float model.                                         │
│                                                         │
│    Metrics checked:                                     │
│    - BPB (must match float within noise)                │
│    - KL p99 / KL max (sensitive tail metric)            │
│    - Top-1 agreement (prediction consistency)           │
│    - Logit cosine min (worst-case distortion)           │
│    - Artifact size ≤ 16,000,000 bytes                   │
└─────────────────────────────────────────────────────────┘
```

## Scripts

| Step | Script | Key function | Input | Output |
|------|--------|-------------|-------|--------|
| 1-2 | `scripts/quant_analysis.py` | `per_layer_sensitivity()`, `design_quant_allocation()` | checkpoint | QUANT_BITS string |
| 3 | `train_gpt_mlx.py` | `quantize_state_dict_int8()` | checkpoint + QUANT_BITS | qobj dict |
| 4 | `scripts/weight_permutation.py` | `permute_mlp_qobj()` | qobj | qobj (mutated) |
| 5 | `scripts/quant_analysis.py` | `compound_error_profile()`, `optimal_correction_layers()` | checkpoint + qobj | correction layer list |
| 6 | `scripts/ptq_correction.py` | `train_corrections()` | float model + quant model + train data | corrections .npz |
| 7 | `scripts/build_artifact.py` | CLI | checkpoint + corrections + QUANT_BITS | artifact .br |
| 8 | `scripts/verify_artifact.py` | CLI | artifact + float checkpoint | pass/fail |

## Example: Full Pipeline Run

```bash
# Config
export NUM_LAYERS=11 MLP_MULT=5.0 MLP_ACT=lrelu2 XSA_LAST_N=11 ROPE_DIMS=16
CKPT=logs/warmdown_11L_5x_best.npz

# Steps 1-2: Sensitivity analysis → quant allocation
python3 scripts/quant_analysis.py $CKPT --save-analysis logs/analysis.json
# Outputs QUANT_BITS and correction layers

# Steps 3-7: Build artifact (quantize + permute + train corrections + compress)
QUANT_BITS="attn:4,mlp:5,mlp.1:4,..." \
python3 scripts/build_artifact.py $CKPT \
    --correction-layers 7,9,10 \
    --save-corrections logs/corrections.npz \
    --compressor brotli \
    --output logs/artifact.br

# Step 8: Verify
python3 scripts/verify_artifact.py logs/artifact.br \
    --float-checkpoint $CKPT
```

## Typical Results

| Component | Size contribution | Quality impact |
|-----------|------------------|----------------|
| Model weights (int4/int5 mixed) | 12-15 MB | +0.01-0.03 bpt gap |
| Weight permutation | -1.1 to -1.5 MB | None (lossless) |
| Corrections (int8) | <0.01 MB | Closes 90-100% of gap |
| Brotli vs zstd | -0.4 to -0.6 MB | None |
| **Total** | **13-15 MB** | **~0.000 gap from float** |

## Critical Methodology Notes

**Isolated sensitivity ≠ compound error.** Per-layer sensitivity (step 1) tells you which layers are cheap to quantize. Compound error profile (step 5) tells you where to place corrections. These give different answers because errors propagate and compound across layers. A layer that's "free" to quantize in isolation can be the biggest error contributor when all layers are quantized.

**Always verify end-to-end.** The artifact must be loaded, decompressed, dequantized, corrections applied, and evaluated. Any bug in serialization/deserialization silently corrupts quality. BPB alone can hide degradation — check KL p99 and Top-1 agreement as early warning metrics.

**Correction weights are near-binary.** On our models, the trained corrections cluster at -1 and 0. This means they compress to <0.01 MB at int8 with no quality loss. This is model-dependent — verify with `scripts/verify_artifact.py` when changing architectures.

## On H100

Steps 1-2 (sensitivity analysis) are pre-computed locally and hardcoded as constants in the CUDA `train_gpt.py`, OR run online within the 600s training budget (~20-30s overhead). Steps 3-7 run after training completes, within the training budget. Step 8 runs during the eval budget.
