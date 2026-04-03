# Results Tracker

All local (M4 Pro) training runs. All models use XSA (all layers), Partial RoPE 16/64, EMA 0.997, SWA every 50.

*Last updated: 2026-04-02*

## Leaderboard

Sorted by Best float BPB (pre-SWA `_best.npz` checkpoint). SWA hurts all runs — see analysis below.

| # | Run | Arch | Act | Params | Steps | Best float | SWA int6 | SWA dmg | Int6 MB | Pipeline MB | Fits 16MB |
|---|-----|------|-----|--------|-------|-----------|----------|---------|---------|-------------|-----------|
| 1 | warmdown_from14k | 13L/3x | lrelu2 | 31.8M | 14k+2k | **1.4010** | 1.4135 | +12.5 | 20.50 | 15.77* | YES |
| 2 | overnight_11L_4x_from4k | 11L/4x | lrelu2 | 32.9M | 4k+12k | **1.4080** | 1.4215 | +13.5 | 19.72 | — | — |
| 3 | wd70_11L_5x | 11L/5x | lrelu2 | 38.6M | 7k (70%wd) | **1.4097** | 1.4391 | +29.4 | 22.03 | ~13† | YES |
| 4 | warmdown_from8k | 13L/3x | lrelu2 | 31.8M | 8k+2k | 1.4151 | 1.4294 | +14.3 | 19.79 | — | — |
| 5 | wd50_11L_5x | 11L/5x | lrelu2 | 38.6M | 7k (50%wd) | 1.4174 | 1.4402 | +22.8 | 21.89 | building | YES |
| 6 | warmdown2_11L_4x | 11L/4x | lrelu2 | 32.9M | 5k+2k+2k | 1.4178 | 1.4222 | +4.4 | 19.60 | — | — |
| 7 | warmdown_sugar_11L_4x | 11L/4x | sugar | 32.9M | 5k+3k | 1.4268 | 1.4453 | +18.5 | 20.35 | 15.75* | YES |
| 8 | warmdown_11L_5x | 11L/5x | lrelu2 | 38.6M | 5k+2k | 1.4467 | 1.4640 | +17.3 | 21.98 | — | — |
| 9 | warmdown_11L_45x | 11L/4.5x | lrelu2 | 35.7M | 5k+2k | 1.4482 | 1.4655 | +17.3 | 20.76 | 13.65 | YES |
| 3p | wd70_11L_5x_polished | 11L/5x | lrelu2 | 38.6M | 7k (70%wd) | 1.4097 | 1.4213‡ | +3.3‡ | 22.03 | 13.63 | YES |

‡ Polished: gradient polish (500 steps, lr=1e-4) + mixed precision (attn int4-5, mlp int4-5, L10 int3) + zstd. Quant BPB measured on 1M val tokens. SWA dmg column shows quant gap in mBPB instead.

*Best float = val_bpb at best step before SWA. Int6 quant gap is ~0.5 mBPB on top of this. SWA dmg = mBPB increase from SWA averaging.*

*Pipeline MB = full post-training pipeline (mixed precision int4/int5 + weight permutation + int8 PTQ correction + brotli-11).*

\* Estimated from component measurements, not end-to-end verified artifact.

† Estimated from 11L/5x pre-warmdown artifact (13.12 MB verified). Post-warmdown checkpoint pending.

### SWA degrades all warmdown runs (2026-04-02)

SWA averages checkpoints starting at 50% of peak LR (`SWA_START_FRAC=0.5`). During warmdown, the model improves monotonically — early SWA checkpoints are much worse than the final model. Equal-weight averaging pulls the result toward the midpoint, not the bottom. Damage scales with warmdown length (more checkpoints = more early junk averaged in).

The `_best.npz` checkpoints (pre-SWA, last training step) are uniformly better than the SWA models. For submissions, use `_best.npz` and skip SWA, or fix SWA by lowering `SWA_START_FRAC` to 0.1-0.2.

## Key Findings

### Warmdown ratio 70% was best found. Improvements > architectural improvements so far.
The 0.025 BPB improvement from 29%→70% warmdown is larger than any architecture change tested.

### Compression pipeline savings
| Component | Savings | Quality impact |
|-----------|---------|---------------|
| int4 attn + int5 MLP (vs int6) | −5 to −7 MB | +0.01-0.02 bpt |
| Cherry-pick cheap layers to int4 | −1 to −2 MB | +0.002-0.005 bpt |
| Weight permutation | −1.1 to −1.5 MB | None (lossless) |
| Brotli vs zstd | −0.4 to −0.6 MB | None |
| **Total pipeline** | **−8 to −10 MB** | **~0.000 gap from float** |

## 50M Scaling Experiments (2026-04-02)

**Question:** At ~50M params with 7k steps and 70% warmdown, which scaling axis wins: depth (more layers), MLP width (higher mult), or model dimension (wider residual stream)?

### Results

| Run | Scaling axis | Arch | Params | Best float | SWA BPB | SWA dmg | Int6 BPB | Int6 MB | ms/step | Machine |
|-----|-------------|------|--------|-----------|---------|---------|----------|---------|---------|---------|
| 13L_640d_3x_7k_wd70 | **Model dim** | 13L/3x/640d | 49.2M | **1.3769** | 1.3572 | +20 | **1.3772** | 32.2 | 1600 | Mac Studio M2 Max |
| 20L_3x_7k_wd70 | Depth | 20L/3x/512d | 48.3M | 1.4274 | 1.3951 | +32 | 1.4280 | 29.4 | 2800 | M4 Pro |
| 14L_5x_7k_wd70 | MLP width | 14L/5x/512d | 48.9M | 1.4422 | 1.4119 | +30 | 1.4425 | 26.8 | 2270 | M4 Pro |

All three runs used identical training budget: 7k steps × 16384 tokens/step = 114.7M tokens, 70% warmdown ratio.

### Key findings

1. **Model dim scaling (640d) wins decisively.** 0.051 BPB better than depth, 0.065 better than MLP width (post-quant).
2. **Depth beats MLP width.** 20L/3x > 14L/5x by 0.015 BPB. MLP width scaling is the least effective axis.
3. **Wider models quantize better.** 640d quant gap is only +0.020 vs +0.030-0.033 for 512d models.
4. **Wider models are faster per step.** Fewer layers with wider matmuls parallelizes better on Apple Silicon.
5. **Nobody in the competition has tried d>512 seriously.** This is unexplored territory.

### Follow-up: Depth vs MLP width at 640d

| Run | Arch | Params | SWA BPB | Int6 BPB | ms/step |
|-----|------|--------|---------|----------|---------|
| 13L_640d_3x_7k_wd70 | **13L/3x/640d** | 49.2M | **1.3572** | **1.3772** | 1600 |
| 11L_640d_4x_7k_wd70 | 11L/4x/640d | 50.9M | 1.3654 | 1.3852 | 1532 |

**13L/3x wins by 0.008 BPB despite 1.7M fewer params.** Same finding as 512d: at 640d, depth (more layers at 3x) beats MLP width (fewer layers at 4x) at iso-params. The extra 2 layers of processing are worth more than wider MLP.

This confirms depth > MLP width holds across model dimensions. Both were run on Mac Studio M2 Max with identical training budget (7k steps, 70% warmdown, LR=0.02).

### 640d LR sweep (200 steps on workhorse)

| MATRIX_LR | EMBED_LR | val_bpb@200 | Notes |
|-----------|----------|-------------|-------|
| 0.04 | 0.05 | 2.1230 | Unstable — loss spikes to 8.48 at step 5 |
| 0.028 | 0.035 | 2.1055 | Moderate instability |
| 0.02 | 0.025 | **2.0783** | Clean descent, no spikes |

Default LRs (0.04) are too aggressive for 640d. Scaled down to 0.02 for all 640d runs.

### 640d compression pipeline (13L/3x/640d, no correction)

Sensitivity analysis run on workhorse (MLX 0.29.3). Note: local MLX 0.31.1 crashes on dequantize — version bug, not memory.

**Per-layer sensitivity (int4, isolated):**
- Cheapest MLP: L12 (+0.0008), L2 (+0.0012), L10 (+0.0013)
- Most expensive MLP: L4 (+0.0040), L0 (+0.0035), L1 (+0.0031)
- Most expensive attn: L7 (+0.0033), L4 (+0.0025)

**Compression sweep (all with weight permutation, no correction):**

| Config | Quant BPB | Gap | Brotli MB | zpaq MB | Fits? |
|--------|-----------|-----|-----------|---------|-------|
| Auto int4/int5 | 1.3671 | +0.010 | 20.57 | 20.18 | NO |
| All int4 | 1.3705 | +0.013 | 18.96 | 18.56 | NO |
| int3 attn + int4 MLP | 1.4033 | +0.046 | 16.73 | 16.25 | NO |
| int3 attn + int4 MLP + L12 MLP int3 | 1.4033 | +0.046 | 16.48 | **15.997** | **YES** |
| All int3 | 1.4814 | +0.124 | 12.63 | — | YES |

**Best fit: int3 attn + int4 MLP + L12 MLP int3 + zpaq = 15.997 MB (3.3 KB margin)**. BPB 1.4033, no correction applied. The +0.046 gap is recoverable — gradient polish or correction could claw back 30-90% of it.

## Notes
- BPB values are post-quant int6 (default). Pipeline BPB is lower (mixed precision + correction).
- Local M4 runs see ~80-230M tokens. H100 sees ~3.2B tokens in 600s. Absolute BPB will be much lower (~1.11-1.14 range).
- Relative ordering between configs should hold on H100 but is not guaranteed.
- Full results data: `logs/results_table.csv`
