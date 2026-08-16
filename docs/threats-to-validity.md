# ScaleRAG-TS — Threats to Validity

Honest accounting of what could undermine the study's conclusions and how we
mitigated each. Conclusions are stated within these bounds.

## Construct / metric validity
- **RMSSE against recent-mean is a punishing bar on intermittent retail.** A large
  share of M5 series are ~68% zeros, where a recent-mean constant is near-optimal
  for squared-error. Our headline ("retrieval helps the TSFM but not the strongest
  simple baseline") is specific to this metric/regime; a different loss (e.g.
  service-level, quantile at high τ, or dollar-weighted WRMSSE) could reorder
  methods. We report MASE/WAPE/MAE/pinball/coverage alongside RMSSE to bound this.
- **Subset WRMSSE is not the official M5 WRMSSE.** The 12-level hierarchy on a
  1k/5k subset is not comparable to the full-panel competition metric; we report
  per-series RMSSE as primary and do not claim leaderboard positions.

## Internal validity (leakage / protocol)
- **Leakage guards** enforce `candidate_end + H < origin`; retriever fitting,
  utility labels, normalisation stats, and gate training use **historical origins
  only**; the M5 test split `d_1914–d_1941` is untouched (verified by tests). This
  is our strongest guarantee.
- **Single eval origin per bootstrap.** Paired bootstrap is over *series* at the
  val origin; multi-origin rolling evaluation (≥3) strengthens generalisation but
  the gate is trained on 2 historical origins, so held-out-origin count is limited.
  Mitigation: baselines are origin-agnostic; the gate's marginal effect is small.
- **Gate label definition** (retrieval-beats-Chronos in RMSE at historical origins)
  is one of several reasonable choices; a different utility target could shift the
  gate slightly (ablations show the gate's effect is ~1% regardless).

## External validity (generalisation)
- **Two datasets, retail only.** M5 (Walmart) and Favorita (Ecuador grocery) are
  both intermittent retail. Conclusions may not transfer to dense, high-frequency,
  or non-retail series. The cross-dataset *negative* (graph routing) is robust
  within retail; we bound it accordingly.
- **Subset sampling.** 1k/5k series are seeded samples of larger panels; results
  are stable across the scales we ran, but full-panel behaviour is confirmed only
  for the frozen temporal retriever (Phase 5), not every ablation.

## Statistical validity
- **Paired bootstrap over series** (2,000 resamples) with per-series relative
  improvements — not seed variance alone. Gate uses 3 seeds. CIs are reported for
  every headline comparison. Multiple-comparison inflation across many ablations is
  possible; we treat ablation CIs as descriptive, not confirmatory.

## Reproduction fidelity
- **RAFT-style and TS-RAG-style are inspired reimplementations**, not exact
  reproductions of released code — labelled as such. Comparisons are only under our
  identical splits/metrics/leakage protocol; we make **no** claim of superiority
  over the original papers (see the protocol-comparability table).

## Calibration
- The gated fusion under-covers (80% coverage 0.69 vs Chronos 0.79). We report this
  and offer point-preserving post-hoc widening; we do not hide the regression.

## Phase-10 held-out test (locked run) — updated threats

- **Official WRMSSE is now computed on the full panel — and it reverses the
  headline.** The earlier "subset WRMSSE is not comparable" caveat is resolved: on
  the full 30,490-series M5 test panel, LightGBM (0.866) and Seasonal-Naive (0.870)
  beat ScaleRAG (1.223) on the official dollar-weighted WRMSSE. ScaleRAG's advantage
  is **RMSSE-specific**; on WRMSSE, MAE, WAPE, MASE, pinball, and coverage it does
  not lead. We do not select the metric that flatters the method — all are reported.
- **Single test origin.** The confirmation is one origin (`d_1913 → d_1914–1941`).
  It is consistent with the full-panel validation origin (item 8; test is slightly
  stronger vs Chronos-2, so no validation over-fitting), but generalisation across
  many test origins is not established — by design the test split is used once.
- **GPU-retriever equivalence.** The full-panel run uses a GPU batched-exact k-NN
  instead of the frozen numpy retriever, for tractability. It is verified
  **bit-identical** on the 1,000-series validation subset (max point-forecast diff
  `0.0`, RMSSE matches the frozen table to <1e-6; `scripts/verify_gpu_retrieval.py`),
  so it is an arithmetic acceleration, not a method change. Residual risk: float
  tie-breaks on series never seen in the subset check — bounded by the exact CPU
  float32 finalize step that reproduces the frozen tie-break rule.
- **Gate under-coverage persists on test** (80% coverage 0.698 vs Chronos 0.786). No
  post-hoc calibration was applied because none was fitted/frozen during the study
  (rule 5); raw coverage is reported rather than a tuned widener.
- **Test consumed.** `d_1914–d_1941` is now spent (`M5_TEST_CONSUMED.lock`); the
  harness refuses further test runs. Any re-run requires deleting the lock with an
  explicit, logged authorization — preventing silent test-driven tuning.

## Bottom line
The controlled-study conclusions — (1) scale-aware retrieval augments a frozen TSFM
by ~5% **on RMSSE** but does not beat the strongest simple baseline by the
pre-registered margin, does not win the official WRMSSE, and does not lead on
absolute/probabilistic metrics; (2) typed-relation graph routing adds nothing across
two datasets — hold **within intermittent retail, with our leakage-safe protocol**,
and are **confirmed on the untouched M5 test split (0/3 criteria met)**. They are not
asserted beyond those bounds.
