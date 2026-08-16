# ScaleRAG-TS — Ablation Report (M5, 1,000 series, val split)

All ablations vary the retrieval/gate configuration while **reusing a single
Chronos-2 computation**. RMSSE (lower better) and 80% interval coverage.
Reference: `ScaleRAG_gated` = mean/L2/category/k20 + restoration + learned gate,
RMSSE **0.7173**. Machine-readable: `reports/scalerag-matrix-m5-1000.json`.

## Retrieval design

| Ablation | RMSSE | cov80 | Takeaway |
|----------|------:|------:|----------|
| no normalization (raw) | 0.7576 | 0.849 | raw matching is weak |
| **normalization without restoration** | **2.7884** | 0.840 | **scale restoration is decisive** (3.8× worse without it) |
| mean scaling (+restore) | 0.7425 | 0.714 | best scale strategy |
| RMS scaling (+restore) | 0.7951 | 0.797 | worse than mean |
| no category filter | 0.7383 | 0.713 | ≈ (marginally better here; category filter not essential at 1k) |
| no seasonal (z-norm) | 0.8719 | 0.439 | z-norm much worse than mean |

**Headline:** scale restoration is the single most important component — without it
retrieval is catastrophic (2.79). Mean-scaling dominates z-norm/RMS.

## Top-k

| k | 1 | 3 | 5 | 10 | 20 |
|---|--|--|--|--|--|
| RMSSE | 0.9019 | 0.7976 | 0.7708 | 0.7511 | **0.7425** |
| cov80 | 0.278 | 0.393 | 0.547 | 0.674 | 0.714 |

Monotonic: larger k averages out intermittent noise and improves both accuracy and
calibration. k=20 best in range.

## Context length

| Context | 28 | 56 | 84 |
|---------|--|--|--|
| RMSSE | 0.7455 | 0.7425 | **0.7349** |
| cov80 | 0.789 | 0.714 | 0.651 |

Longer context slightly improves point accuracy but *worsens* coverage — a mild
accuracy/calibration trade-off within the retriever.

## Fusion & gate

| Variant | RMSSE | cov80 |
|---------|------:|------:|
| **learned gate** (proposed) | **0.7173** | 0.689 |
| fixed gate α=0.5 | 0.7252 | 0.709 |
| gate − uncertainty feature | 0.7182 | 0.691 |
| gate − reliability features | 0.7181 | 0.689 |
| gate − intermittency feature | 0.7171 | 0.688 |

- **The learned gate beats fixed fusion** (0.7173 vs 0.7252, ~1.1% relative) — the
  gate genuinely adds value over a constant blend.
- Individual gate features contribute little marginally (dropping any one leaves
  RMSSE 0.717–0.718); the gate is robust but no single feature dominates. Removing
  the intermittency feature is marginally best here — reported, not hidden.

## Interpretation

The value stack, in order of importance: **scale restoration ≫ mean-scaling ≫
top-k ≈ context ≈ learned gate > individual gate features**. The proposed system's
gains over target-only Chronos-2 come overwhelmingly from *scale-aware retrieval*;
the learned gate adds a small, real increment over fixed fusion.
