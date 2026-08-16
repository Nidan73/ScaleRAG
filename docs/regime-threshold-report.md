# When Does Retrieval Stop Helping? A Regime Band on M5

**The gap in the literature.** TS-RAG characterises retrieval effectiveness by
autocorrelation, noise ratio, volatility and stationarity, reporting the correlation
row `0.70 / −0.55 / −0.65 / −0.19` — but names **no value** of any property at which
retrieval ceases to pay. That leaves "retrieval sometimes helps" as the state of the
art. This estimates the crossing.

**The granularity gap is the reason this is possible.** TS-RAG computes each
property as an average *over a whole dataset* and then correlates across datasets:
its Table 13 is a linear Pearson coefficient over **four points** (ETTh1, ETTh2,
ETTm1, ETTm2). Four dataset-level averages and a linear coefficient cannot express a
non-monotone relationship even if one is present. The analysis below is per series
(n = 1,000) and rank-based, which is what makes the inverted U visible at all. The
finding is therefore not a contradiction of TS-RAG's result so much as a consequence
of measuring at a finer grain. Verified against the source, see
`docs/literature-verification.md`.

**Method.** Per series at the M5 **validation** origin (d_1886–d_1913), retrieval
utility is `U = RMSSE(Chronos-2) − RMSSE(scale-restored retrieval)`; `U > 0` means
retrieval beat the frozen backbone on that series. The boundary is where
`P(U > 0)` crosses one half, fitted by isotonic regression — monotone but not
parametric, so a misspecified curve cannot place the threshold in its own tails.

Validation split only; the consumed test split is never touched (rule 2), and the
script refuses to run if the eval origin reaches it. Nothing is selected or tuned
(rules 9, 12) — the frozen `scale=mean` / `k=20` / `cat_id` retrieval is imported
from `scripts/scalerag_eval.py` rather than re-derived. 1,000 series, 34.7 s.

- Code: `src/graphroute_ts/regime.py`, `scripts/regime_threshold_run.py`
- Results: `reports/regime-threshold/m5-val-regime-threshold-1000.json`


## Hardened across 50 forecast origins (2026-07-27)

The single-origin estimate below invited an obvious dismissal: that the shape was
a property of one timestamp rather than of sparsity. It is not. Re-run over **50
non-overlapping origins** (stride = H = 28, spanning `d_513`–`d_1885`, all
evaluation windows closing at or before the consumed test boundary), the inverted
U reproduces at every origin.

Bins are **fixed absolute intervals**, not quantile deciles, so a bin denotes the
same thing at every origin and the origins can be pooled. Quantile bins would
shift meaning as panel sparsity drifts, which is the "shifting sands" objection;
fixed bins risk thin strata instead, so no non-empty bin is dropped and the
minimum bin held ~52 series at any origin.

| Zero fraction | Win rate (mean ± SEM over 50 origins) | Mean ΔU |
|---|---|---|
| 0.0–0.1 | 0.220 ± 0.009 | -0.1049 ± 0.0038 |
| 0.1–0.2 | 0.341 ± 0.011 | -0.0322 ± 0.0029 |
| 0.2–0.3 | 0.406 ± 0.009 | -0.0191 ± 0.0025 |
| 0.3–0.4 | 0.514 ± 0.008 | -0.0014 ± 0.0023 |
| 0.4–0.5 | 0.636 ± 0.009 | +0.0302 ± 0.0028 |
| 0.5–0.6 | 0.764 ± 0.006 | +0.0577 ± 0.0026 |
| 0.6–0.7 | 0.794 ± 0.006 | +0.0583 ± 0.0026 |
| 0.7–0.8 | 0.770 ± 0.006 | +0.0408 ± 0.0019 |
| 0.8–0.9 | 0.700 ± 0.005 | +0.0112 ± 0.0018 |
| 0.9–1.0 | 0.472 ± 0.007 | -0.0956 ± 0.0036 |

**The non-monotonicity is statistically separated.** The peak
(0.6–0.7) has a 95% lower bound of
0.782, strictly above the 95%
upper bounds of both the densest bin (0.237)
and the sparsest (0.485).

**Band, refit per origin:** lower 0.327 ± 0.055
(found at 50/50 origins), upper 0.949 ±
0.021 (48/50). Runtime 465s.

The single-origin decile numbers below are superseded and retained only as the
record of the first pass. The paper reports the fixed-bin, 50-origin version and
does not report both, since presenting two stratifications of one profile invites
the charge of choosing the more favourable.

## Headline: the relationship is not monotone

Retrieval beats Chronos-2 on **62.3%** of series overall (mean ΔU = +0.0115 RMSSE).
The intermittency profile is an inverted U:

| Zero-fraction band | n | Win rate | Mean ΔU (RMSSE) |
|---|---|---|---|
| 0.00–0.14 | 99 | 0.19 | **−0.1027** |
| 0.14–0.30 | 99 | 0.37 | −0.0221 |
| 0.30–0.43 | 99 | 0.54 | +0.0192 |
| 0.43–0.54 | 94 | 0.69 | +0.0461 |
| 0.54–0.62 | 97 | **0.81** | **+0.0834** |
| 0.62–0.71 | 107 | **0.84** | +0.0636 |
| 0.71–0.79 | 99 | 0.83 | +0.0614 |
| 0.79–0.86 | 104 | 0.71 | +0.0241 |
| 0.86–0.93 | 101 | 0.73 | +0.0151 |
| 0.93–1.00 | 101 | 0.50 | **−0.0731** |

Retrieval fails at **both** extremes. On dense series the backbone is already good
and retrieval actively hurts; on near-empty series (>93% zeros) there is too little
signal for any analogue to match, and utility goes negative again.

**This corrects the paper's current claim.** "Retrieval utility tracks
intermittency" implies monotonicity. It does not hold at the sparse extreme, and a
method that keys on "more intermittent ⇒ trust retrieval more" is wrong in the top
decile. The honest statement is a **band**, not a threshold.

Estimated band on intermittency: **[0.359, 0.964]** — win rate **0.74 inside**
against **0.34 outside**. Every diagnostic is bounded above:

| Diagnostic | Spearman ρ vs U | Band lower | Band upper | Win in | Win out |
|---|---|---|---|---|---|
| `retr_nn_dist` | +0.287 | 45.30 | 466.67 | 0.74 | 0.32 |
| `retr_disagreement` | −0.246 | 0.0234 | 0.815 | 0.75 | 0.37 |
| `intermittency` | +0.161 | 0.359 | 0.964 | 0.74 | 0.34 |
| `log_volume` | −0.180 | 0.0306 | 0.744 | 0.76 | 0.36 |
| `chronos_uncertainty` | −0.169 | 0.0098 | 3.179 | 0.75 | 0.34 |
| `scale_spread` | +0.201 | 0.997 | 4.097 | 0.74 | 0.35 |

## The six gate features are close to one feature

The near-identical win rates above (0.74–0.76 in, 0.32–0.37 out) are a clue, and the
rank correlations among the diagnostics confirm it:

| | nn_dist | disagree | intermit | log_vol | chr_unc | scale_sp |
|---|---|---|---|---|---|---|
| `retr_nn_dist` | · | −0.58 | 0.65 | −0.63 | −0.58 | **0.85** |
| `retr_disagreement` | −0.58 | · | −0.79 | **0.88** | **0.88** | −0.59 |
| `intermittency` | 0.65 | −0.79 | · | **−0.97** | **−0.90** | 0.79 |
| `log_volume` | −0.63 | 0.88 | −0.97 | · | **0.94** | −0.73 |
| `chronos_uncertainty` | −0.58 | 0.88 | −0.90 | 0.94 | · | −0.68 |
| `scale_spread` | 0.85 | −0.59 | 0.79 | −0.73 | −0.68 | · |

Intermittency and log-volume are correlated at **−0.97**; log-volume and Chronos
uncertainty at **+0.94**. Essentially one latent axis — demand level — drives all
six. The frozen LightGBM gate therefore has roughly one to two effective dimensions,
not six, which is a plausible explanation for why the learned gate beats fixed
fusion by only ~1.1%.

It also explains the one counterintuitive sign. `retr_nn_dist` has the *strongest*
marginal correlation with utility (+0.287) and the sign says a **worse** nearest
match predicts retrieval doing **better** — which is backwards as a statement about
match quality. It is not one: `nn_dist` correlates 0.65 with intermittency and 0.85
with scale-spread, so it is acting as a proxy for sparsity rather than measuring
match quality. Do not read it as "distant neighbours help".

## Caveats

**The intervals are indicative, not calibrated.** Isotonic regression has
cube-root asymptotics at a point and the naive bootstrap is *inconsistent* for it,
so the percentile intervals are too narrow — on planted data with a sharp boundary
the interval can exclude the true value it was built around. The point estimates are
the result; a calibrated interval needs subsampling or a smoothed bootstrap. A unit
test pins this behaviour so a future fix registers as an intentional change.

**One origin, one subset.** 1,000 series at a single validation origin. The decile
profile is model-free and unlikely to be an artefact, but the exact band edges would
move with the subset. Multi-origin estimation is the obvious next step and is
legal on validation.

**Correlational, not causal.** These are observational regimes, not interventions.
The affine probe (`docs/affine-probe-report.md`) supplies the causal half.

## Reproduce

```bash
uv run python scripts/regime_threshold_run.py --subset 1000 --n-boot 500
```
