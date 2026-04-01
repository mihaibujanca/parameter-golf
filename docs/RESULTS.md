# Results Tracker

All local (M4 Pro) training runs, sorted by post-quant BPB. All models use XSA (all layers), Partial RoPE 16/64, EMA 0.997, SWA every 50.

*Last updated: 2026-04-01*

## Leaderboard

| # | Run | Arch | Act | Params | Steps | BPB (int6) | Int6 MB | Pipeline MB | Fits 16MB |
|---|-----|------|-----|--------|-------|-----------|---------|-------------|-----------|
| 1 | warmdown_from14k | 13L/3x | lrelu2 | 31.8M | 14k+2k | **1.4135** | 20.50 | 15.77* | YES |
| 2 | overnight_11L_4x_from4k | 11L/4x | lrelu2 | 32.9M | 4k+12k | 1.4215 | 19.72 | — | — |
| 3 | warmdown2_11L_4x | 11L/4x | lrelu2 | 32.9M | 5k+2k+2k | 1.4222 | 19.60 | — | — |
| 4 | warmdown_from8k | 13L/3x | lrelu2 | 31.8M | 8k+2k | 1.4294 | 19.79 | — | — |
| 5 | wd70_11L_5x | 11L/5x | lrelu2 | 38.6M | 7k (70%wd) | **1.4391** | 22.03 | ~13† | YES |
| 6 | wd50_11L_5x | 11L/5x | lrelu2 | 38.6M | 7k (50%wd) | 1.4402 | 21.89 | building | YES |
| 7 | warmdown_sugar_11L_4x | 11L/4x | sugar | 32.9M | 5k+3k | 1.4453 | 20.35 | 15.75* | YES |
| 8 | warmdown_11L_5x | 11L/5x | lrelu2 | 38.6M | 5k+2k | 1.4640 | 21.98 | — | — |
| 9 | warmdown_11L_45x | 11L/4.5x | lrelu2 | 35.7M | 5k+2k | 1.4655 | 20.76 | 13.65 | YES |

*Pipeline MB = full post-training pipeline (mixed precision int4/int5 + weight permutation + int8 PTQ correction + brotli-11).*

\* Estimated from component measurements, not end-to-end verified artifact.

† Estimated from 11L/5x pre-warmdown artifact (13.12 MB verified). Post-warmdown checkpoint pending.

## Key Findings

### Warmdown ratio matters more than architecture
| Warmdown % | BPB (11L/5x, 7k total) |
|-----------|----------------------|
| 29% (5k+2k separate) | 1.4640 |
| 50% (SOTA ratio) | 1.4402 |
| 70% | 1.4391 |
| 86% | running |

The 0.025 BPB improvement from 29%→70% warmdown is larger than any architecture change tested.

### Compression pipeline savings
| Component | Savings | Quality impact |
|-----------|---------|---------------|
| int4 attn + int5 MLP (vs int6) | −5 to −7 MB | +0.01-0.02 bpt |
| Cherry-pick cheap layers to int4 | −1 to −2 MB | +0.002-0.005 bpt |
| Weight permutation | −1.1 to −1.5 MB | None (lossless) |
| Brotli vs zstd | −0.4 to −0.6 MB | None |
| PTQ correction (int8) | <0.01 MB added | Recovers 90-100% of gap |
| **Total pipeline** | **−8 to −10 MB** | **~0.000 gap from float** |

## Notes
- BPB values are post-quant int6 (default). Pipeline BPB is lower (mixed precision + correction).
- Local M4 runs see ~80-230M tokens. H100 sees ~3.2B tokens in 600s. Absolute BPB will be much lower (~1.11-1.14 range).
- Relative ordering between configs should hold on H100 but is not guaranteed.
- Full results data: `logs/results_table.csv`
