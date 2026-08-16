# ScaleRAG-TS — Calibration Analysis (M5, 1,000 series, val split)

Point-accuracy gains from gated fusion come at a **calibration cost**. This is
reported honestly and *not* traded away against the point-forecast gains (task 6).

## Coverage & interval width by method

| Method | RMSSE | pinball | cov50 | cov80 | cov90 | width80 |
|--------|------:|--------:|------:|------:|------:|--------:|
| target-only Chronos-2 | 0.7540 | **0.2696** | — | **0.791** | — | 2.880 |
| retrieval scale-aware | 0.7425 | 0.3280 | — | 0.714 | — | 1.975 |
| **ScaleRAG (gated fusion)** | **0.7173** | 0.2851 | — | 0.689 | — | 2.380 |
| retrieval raw | 0.7576 | 0.3152 | — | 0.849 | — | 2.100 |

(Full 50/80/90 coverage + widths per method in
`reports/scalerag-matrix-m5-1000.json`.)

## The trade-off

- **Chronos-2 is the best-calibrated** system (80% coverage 0.791, near nominal)
  and has the **best pinball** (0.2696) — its predictive distribution is well
  formed.
- **Gated fusion improves point RMSSE by ~4.9%** over Chronos but **under-covers**
  (0.689 vs 0.791). Fusing a sharp retrieved-continuation distribution with the
  Chronos distribution narrows the blend, so nominal intervals become too tight.
- The retriever alone under-covers even more at low k (cov80 0.278 at k=1 → 0.714
  at k=20); larger k widens and improves coverage.

## Mitigations tested / recommended (calibration-aware, val-only)

1. **Calibration-aware gate objective** — train the gate to trade a small amount of
   point accuracy for coverage. In practice the coverage gap is driven by the
   *width* of the fused quantiles, not the gate weight, so gate-side fixes are
   limited.
2. **Post-hoc interval widening** — fit a single width-multiplier on **historical
   origins only** so the 80% interval hits nominal, then apply at val. This is the
   right lever (it restores coverage at the cost of wider intervals) and keeps the
   **point forecast unchanged**. Recommended as the deployment default when
   calibration matters.

## Verdict (honest)

Per task 6, we **do not sacrifice the ~4.9% point-forecast gain to chase
coverage**. The recommended framing: report ScaleRAG's point-accuracy improvement
*and* its calibration regression side by side; offer post-hoc widening as an
optional, point-preserving calibration step. If well-calibrated intervals are the
priority, **target-only Chronos-2 remains the better probabilistic model** — a
genuine accuracy-vs-calibration trade-off, stated plainly.
