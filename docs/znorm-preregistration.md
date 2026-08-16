# Pre-Registration: Location-Aware Scale Restoration (`znorm` vs `mean`)

**Status: WRITTEN AND FROZEN, NOT EXECUTED.** This document is committed *before*
any run. Nothing in it may be edited after the confirmation run begins.

## Why this exists

Two independent findings converge on one hypothesis:

1. `docs/affine-probe-report.md` — under an imposed affine transform, `znorm`
   restoration is exactly equivariant (error constant to 8.7e-19 across the whole
   `(a, b)` grid). `mean` restoration is exactly *scale*-equivariant but carries no
   location term, so a pure offset degrades it 291x and drops top-1 hit rate to
   0.580.
2. `docs/retrieval-forecasting-gap.md` — on ETTm2, the frozen `mean` restoration
   leaves **85.2%** of the restored retrieval's error as unrecovered magnitude.
   The shape floor is 0.0647 against a backbone at 0.1486, so the analogues are
   2.3x better than the backbone once optimally rescaled. The bottleneck is the
   magnitude correction, not analogue quality.

The literature supports the direction. Z-score with **both** mean and standard
deviation is the standard instance normalization (RevIN, TiRex, Timer-S1,
TimeFound), and RAID applies exactly that form to retrieved trajectories. No source
in the surveyed corpus uses a mean-only denominator for retrieval restoration.

**This is a hypothesis, not a result.** It has not been tested end to end.

## Why it cannot simply be run

The obvious move (swap `znorm` in, re-evaluate) is blocked:

- **M5 test** `d_1914-d_1941` is consumed (`M5_TEST_CONSUMED.lock`). Re-evaluating
  it for a changed configuration is test-driven tuning (rules 2, 9, 12).
- **ETTm2 test** was consumed once by the Phase-11A decision gate. Selecting a scale
  strategy after seeing its attribution and then re-scoring on it is the same
  violation.
- **ETTh1 / ETTm1 / Weather / Electricity** are Phase-11B, pre-registered and
  **blocked** because the gate was not passed. Opening them to rescue a negative is
  precisely what rule 12 forbids.

## The one clean split

**Favorita test: origin 972, window d_973-d_1000.**

Verified unconsumed as of 2026-07-26:

- Favorita has 1,000 days; `make_rolling_splits` gives `val_m2=888`, `val_m1=916`,
  `val=944`, `test=972`.
- Every recorded Favorita run (`reports/scalerag-favorita.json`,
  `reports/favorita-router.json`, Phase 9) used **origin 944** (val).
- No report in `reports/` has `eval_origin == 972`.
- No lock file exists for Favorita.

This split is spent by the run below. A `FAVORITA_TEST_CONSUMED.lock` is written
immediately afterwards, mirroring the M5 lock, and the split is never used again.

## Design

**Selection (already-used development splits, no new information consumed):**
ETTm2 validation and M5 validation. Choose between `mean` and `znorm` on validation
evidence alone. If validation does not favour `znorm`, the confirmation run is
**not** performed and this hypothesis is recorded as unsupported.

**Confirmation (single locked run):** Favorita test, origin 972.

**Frozen, not re-tuned.** Only the scale strategy varies: `mean` -> `znorm`.
`k=20`, `L=56`, `H=28`, `meta_filter=cat_id`, the gate hyperparameters and the gate
training origins all stay at their frozen values. Varying anything else would make
this a search rather than a test.

**Arms:** `chronos2_target`, `retrieval_mean`, `retrieval_znorm`,
`ScaleRAG_gated(mean)`, `ScaleRAG_gated(znorm)`, plus `recent_mean` and `lightgbm`
as the standing baselines.

## Pre-registered criteria

Declared now, in advance, and not revisable.

**H1 (mechanism, primary).** On Favorita test, the retrieval branch under `znorm`
has lower RMSSE than under `mean`, paired-bootstrap 95% CI on the relative
improvement excluding zero.

**H2 (residual magnitude).** The error decomposition of
`docs/retrieval-forecasting-gap.md`, recomputed on this run, shows the residual
scale error (restored minus shape floor) is a smaller fraction of the restored
retrieval error under `znorm` than under `mean`. Pre-registered bar: the shape
fraction rises above 30% (it is 14.8% on ETTm2 under `mean`).

**H3 (end to end).** `ScaleRAG_gated(znorm)` beats `chronos2_target` on Favorita
test RMSSE with a CI excluding zero.

**H4 (the honest bar).** `ScaleRAG_gated(znorm)` beats the strongest non-ScaleRAG
baseline by at least 3% RMSSE, the same bar the original pre-registration used.

### What each outcome means, decided in advance

| Outcome | Interpretation |
|---|---|
| H1 and H2 hold, H3/H4 fail | The mechanism claim strengthens; the end-to-end verdict is unchanged. This is the **expected** result given every prior finding, and it will be reported as such, not buried. |
| H1 fails | The affine-probe prediction does not transfer to real data. Reported as a falsification of the hypothesis, and the affine probe's scope is narrowed to synthetic data in the paper. |
| All four hold | `znorm` materially changes the verdict. Even then the frozen M5 results are **not** retro-fitted, because M5 test is consumed. It would be reported as a Favorita-only finding requiring fresh confirmation elsewhere. |

**A negative result here is publishable and will be published.** No outcome of this
run licenses re-opening M5 test or Phase 11B.

## Threats

- **One dataset, one origin, one horizon.** Favorita is a 5,000-series subset with
  a single 28-day test window. It cannot settle the question generally.
- **Regime confound.** Favorita is the *denser* of the two retail panels, the regime
  where `docs/regime-threshold-report.md` shows retrieval helping least. A location
  term may matter more on M5, which cannot be tested. This weakens H3/H4 a priori
  and is a reason to expect the expected outcome above.
- **Non-negativity.** Favorita sales are non-negative and the pipeline clips at 0,
  so the large negative offsets that most damage `mean` scaling do not arise. The
  affine probe's 291x degradation is an upper bound, not a prediction for this data.
- **`znorm` is not free of failure modes.** It divides by a window standard
  deviation, which is near zero for flat or all-zero contexts. The existing
  `scale_eps` guard and its invalid-scale counters must be reported, not silenced.

## Execution checklist

1. Run selection on ETTm2 val and M5 val. Record. **Stop here if `znorm` does not
   win on validation.**
2. Commit the selection result before touching Favorita test.
3. Single confirmation run on Favorita test, origin 972.
4. Write `FAVORITA_TEST_CONSUMED.lock` with commit, timestamp and report path.
5. Report all four hypotheses, including failures, in
   `docs/znorm-confirmation-report.md`.

Pre-registered 2026-07-26. Not executed.
