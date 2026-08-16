# ScaleRAG-TS — Final Held-Out Test Report (Phase 10)

**Single locked confirmation run on the reserved M5 test split `d_1914–d_1941`.**
The method, hyperparameters, gate, retrieval config, and pre-registered success
criteria were **frozen before** this run; nothing was tuned on the test results.
The test split is now **consumed** (`M5_TEST_CONSUMED.lock`) and further
test-driven tuning is blocked (rules 2, 9, 12).

## Provenance

| | |
|---|---|
| Eval window | `d_1914–d_1941` (last 28 public M5 days), forecast origin `d_1913` |
| Population | **Full M5 panel, 30,490 series** (canonical WRMSSE valid) |
| Frozen method | `ScaleRAG_gated` = mean/L2/category-filter/k=20 + scale restoration + LightGBM gate (seeds 42/43/44) |
| Gate-training origins | `val_m2 = d_1829`, `val_m1 = d_1857` — **unchanged from the frozen study** (historical-only) |
| Git commit | `d42d20e3bc4d6096f82349d16284b5b6da00346f` |
| Hardware | RTX 5070 Ti, CUDA 13.0, torch 2.13.0+cu130, Python 3.11.15 |
| Retriever | GPU batched-exact k-NN, **verified bit-identical** to the frozen numpy retriever on the 1,000-series validation subset (max point diff `0.0`) |

The GPU retriever is an arithmetic-only acceleration of the frozen exact k-NN (no
method change): it reproduces the frozen validation numbers bit-for-bit
(`scripts/verify_gpu_retrieval.py`), and made the full-panel run tractable
(retrieval **42 s/origin** vs an estimated ~4.8 h for the numpy path).

## Headline verdict (confirmed on untouched test data)

**All three pre-registered success criteria FAIL on the held-out test.** ScaleRAG
improves the frozen Chronos-2 backbone on **RMSSE** but does not beat the strongest
simple baseline by the pre-registered margin, does not win the official
**WRMSSE**, and its RMSSE gain does **not** carry over to absolute-error or
probabilistic metrics. The controlled-study framing stands, now on data never seen
during development.

## Full metrics (30,490 series, `d_1914–d_1941`)

Lower is better for every column. Best in **bold**.

| method | RMSSE | WRMSSE | MASE | WAPE | MAE | pinball |
|---|---|---|---|---|---|---|
| **ScaleRAG_gated** (proposed) | **0.7612** | 1.2231 | 1.020 | 0.698 | 1.008 | 0.298 |
| lightgbm | 0.7665 | **0.8663** | 1.077 | 0.740 | 1.067 | 0.534 |
| recent_mean | 0.7669 | 1.0876 | 1.074 | 0.745 | 1.075 | 0.538 |
| fusion_fixed0.5 | 0.7692 | 1.3809 | 0.991 | 0.691 | 0.997 | 0.296 |
| retrieval_scaleaware | 0.7795 | 1.2293 | 1.111 | 0.743 | 1.072 | 0.326 |
| chronos2_target (frozen) | 0.8054 | 1.9395 | **0.893** | **0.665** | **0.960** | **0.289** |
| seasonal_naive | 0.9972 | 0.8697 | 1.211 | 0.862 | 1.244 | 0.622 |

**No method dominates — the winner depends on the metric:**
- **RMSSE (squared, scaled):** ScaleRAG wins (0.7612), +5.49% over Chronos-2.
- **WRMSSE (official M5, dollar-weighted, hierarchical):** LightGBM (0.8663) and
  Seasonal-Naive (0.8697) win; ScaleRAG (1.2231) is mid-pack, Chronos-2 worst.
- **MASE / WAPE / MAE / pinball / coverage (absolute & probabilistic):** the
  **frozen Chronos-2 alone is best**. ScaleRAG's fusion trades typical-case and
  calibrated-distribution accuracy for squared-error accuracy.

This metric-dependent reordering is the central honest result: retrieval fusion
reduces large (squared) errors on intermittent series while slightly worsening
average-magnitude and probabilistic accuracy.

## Paired-bootstrap 95% CIs (per-series relative RMSSE improvement, 2,000 resamples)

| comparison | rel. improvement | 95% CI | excludes 0 |
|---|---|---|---|
| ScaleRAG vs **Chronos-2** | **+5.49%** | [+5.40%, +5.59%] | yes |
| ScaleRAG vs **strongest baseline (lightgbm)** | **+0.69%** | [+0.57%, +0.82%] | yes |
| retrieval_scaleaware vs Chronos-2 | +3.22% | [+3.03%, +3.40%] | yes |
| fusion_fixed0.5 vs Chronos-2 | +4.49% | [+4.40%, +4.58%] | yes |
| lightgbm vs Chronos-2 | +4.83% | [+4.67%, +5.00%] | yes |
| recent_mean vs Chronos-2 | +4.78% | [+4.62%, +4.94%] | yes |

The ScaleRAG–vs–strongest CI excludes zero (a real but small edge over LightGBM on
RMSSE), yet is an order of magnitude below the pre-registered **3%** bar.

## Pre-registered success criteria — verdict

| # | criterion | result on test | met? |
|---|---|---|---|
| C1 | ≥3% rel. RMSSE/WRMSSE over the strongest matched baseline, CI excl. 0 | RMSSE +0.69% (<3%); WRMSSE **worse** than lightgbm/seasonal | **NO** |
| C2 | ≥5% over target-only Chronos-2 on **both** M5 and Favorita | M5 test +5.49% ✓; Favorita +0.83% ✗ | **NO** |
| C3 | ≥7% on a predefined sparse/intermittent/low-volume/reduced-history slice over strongest | best such slice +0.93% (reduced-history); intermittent −0.03%, low-volume −0.30% | **NO** |

**0 / 3 criteria met** → no positive forecasting-improvement claim; the work is a
controlled study (as pre-registered).

## Pre-registered slices (ScaleRAG vs Chronos-2 / vs strongest, RMSSE)

| slice | n | vs Chronos-2 | vs strongest (lightgbm) |
|---|---|---|---|
| intermittent (zero-frac > 0.8) | 11,783 | **+5.97%** [+5.81, +6.13] | −0.03% [−0.17, +0.13] |
| low-volume (< median) | 15,241 | **+6.85%** [+6.71, +6.99] | −0.30% [−0.43, −0.17] |
| reduced-history (< 100 non-zero) | 1,778 | +3.14% [+2.63, +3.61] | +0.93% [+0.40, +1.51] |
| dense (zero-frac < 0.3) | 2,247 | +0.34% [+0.12, +0.56] | **+5.04%** [+4.28, +5.90] |

Retrieval augmentation helps the **backbone** most on sparse/low-volume series
(+6–7% vs Chronos-2) but there it merely *matches* the strong LightGBM/recent-mean
baselines. The one place ScaleRAG clearly beats the strongest baseline is the
**dense** slice (+5.04%) — the opposite of the sparse-data motivation, and still
short of C3's 7% on a *sparse* slice.

## Calibration / coverage (nominal 50 / 80 / 90%)

| method | cov50 | cov80 | cov90 | width80 |
|---|---|---|---|---|
| chronos2_target | 0.358 | **0.786** | **0.907** | 2.84 |
| retrieval_scaleaware | 0.525 | 0.728 | 0.805 | 2.04 |
| fusion_fixed0.5 | 0.307 | 0.715 | 0.848 | 2.44 |
| ScaleRAG_gated | 0.306 | 0.698 | 0.828 | 2.40 |

The frozen Chronos-2 is the best-calibrated forecaster; the gated fusion
**under-covers** (80% coverage 0.698 vs 0.786). No post-hoc calibration multiplier
was applied because none was fitted and frozen during the study (rule 5) — raw
coverage is reported. Point baselines (recent-mean, LightGBM, seasonal-naive) emit
degenerate zero-width intervals and are shown for point metrics only.

## Gate behaviour (test)

- Mean retrieval weight α = **0.622** (std 0.215); the gate routes **33.8%** of
  series toward Chronos-2 (α < 0.5).
- α correlates **+0.485** with intermittency (more zeros → trust retrieval more) and
  **−0.438** with Chronos uncertainty. The gate learned to lean on retrieval for
  intermittent/low-volume series and defer to the backbone elsewhere — consistent
  with the slice results and with validation.

## Validation ↔ test consistency (item 8 — method NOT changed)

| quantity | val (`d_1886–1913`, 30,490) | test (`d_1914–1941`, 30,490) |
|---|---|---|
| ScaleRAG RMSSE | 0.7371 | 0.7612 |
| ScaleRAG vs Chronos-2 | +5.08% [+4.97, +5.19] | +5.49% [+5.40, +5.59] |
| ScaleRAG vs strongest (lightgbm) | +0.92% [+0.78, +1.07] | +0.69% [+0.57, +0.82] |
| strongest baseline | lightgbm | lightgbm |
| gate α mean / corr(intermittency) | 0.615 / +0.44 | 0.622 / +0.48 |
| criteria met | 0 / 3 | 0 / 3 |

Test behaviour matches validation (test is if anything slightly *stronger* vs
Chronos-2 — no sign of validation over-fitting). Per the protocol, the frozen method
was **not** altered in response to the test results.

## Efficiency / footprint (full panel, one eval origin)

| metric | value |
|---|---|
| Chronos-2 inference | 35.7 s (30,490 series, batched) |
| Retrieval (GPU exact k-NN) | 42.2 s |
| GPU VRAM peak | 1.76 GiB (frozen Chronos-2) |
| Host RAM peak | 26.4 GiB (LightGBM full-panel feature build dominates) |
| Backbone parameters updated | **0** (Chronos-2 frozen) |
| Trainable component | LightGBM gate only (~9,000 ensemble leaf values, 3 seeds averaged) |

ScaleRAG adds only a lightweight non-gradient gate on top of a fully frozen
backbone; no backbone weights are trained or stored.

## What this run establishes

1. On **RMSSE**, scale-aware retrieval + gated fusion improves a frozen Chronos-2 by
   **+5.49%** on untouched M5 test data (tight CI), confirming the validation result.
2. That gain is **RMSSE-specific and regime-dependent**: it does not beat strong
   simple baselines by the pre-registered margin, does not win the official WRMSSE,
   and reverses on MAE/WAPE/MASE/pinball/coverage where the frozen backbone is best.
3. The pre-registered criteria fail 0/3 on held-out data — the study's negative
   framing is **confirmed, not softened**, on data never used in development.
