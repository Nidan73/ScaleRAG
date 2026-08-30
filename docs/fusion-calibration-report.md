# Referee item 10 — the fused predictive distribution, defined and diagnosed

**Date:** 2026-08-30 · **Branch:** `phase14-unit2` · **Script:**
`scripts/fusion_calibration_run.py` · **Data:** `reports/fusion-calibration/`
**Run:** M5 **validation**, 1,000 series × 20 origins (stride 7, origins
d_1885 down to d_1752), 19,986 series-origin pairs, seed 42, 236 s.
The consumed test split is refused by the harness (rule 2).

## What the referee asked

> The paper reports coverage and pinball from a fused predictive distribution it
> never defines.

Reading the code, the definition was not merely missing. It is two separate
constructions, and neither is a predictive law in the usual sense.

1. **`scalerag.fuse` blends quantile by quantile**,
   `q_fused(τ) = (1−w)·q_c(τ) + w·q_r(τ)`. That object is the **Wasserstein-2
   barycentre** of the two branches (Vincentization), not the mixture
   `(1−w)F_c + w F_r`. Its standard deviation is `(1−w)σ_c + w σ_r`, whereas the
   mixture's is `√((1−w)σ_c² + wσ_r² + w(1−w)(μ_c−μ_r)²)`. The location-disagreement
   term is absent from the barycentre, so blending two branches that disagree
   about *where* the forecast sits produces something narrower than either the
   mixture or, in general, the evidence warrants.
2. **The retrieval branch's quantiles are `np.quantile` over the k=20 restored
   neighbours.** That is retrieval *disagreement* — how much the analogues
   differ from one another — not forecast uncertainty. At k=20 the 0.05 and 0.95
   levels sit essentially on the 2nd and 19th order statistics, so the outer
   interval cannot reach past the neighbour sample range.

## The recorded runs already showed the symptom

No new run was needed to see that something is wrong with the operator. From
`reports/scalerag-heldout-{val,test}-30490.json`, nominal 80%:

| | Cov.80 val | Cov.80 test | Width.80 val |
|---|---|---|---|
| Chronos-2 backbone | 0.7906 | 0.7864 | 2.8317 |
| retrieval branch | 0.7393 | 0.7280 | 2.0543 |
| `fusion_fixed0.5` | **0.7218** | **0.7147** | 2.4408 |
| `ScaleRAG_gated` | 0.7062 | **0.6975** | 2.4148 |

Two things fall out. The fused width reproduces the plain quantile average of
the two branches (2.4430) to within the zero floor, which identifies the
deployed object as the barycentre. And the fused coverage lands **below both of
its own inputs** at the 50% and 80% levels, on validation and on test alike.
The 0.6975 in the test column is the `0.698` the manuscript prints in
`sec:calibration`. Validation tracks test to within 0.009 at every level, so the
whole diagnosis below runs on validation and no consumed split is reopened.

## The measurement

Hold the retrieved set, the weight (w = 0.5) and the scoring grid fixed, and
swap only the construction. Everything is scored on one population — never-launched
series excluded — and on the harness's own seven-level grid
`{0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95}`, because item 10 is about the numbers
the paper prints.

| construction | Cov.50 | Cov.80 | Cov.90 | Width.80 | miss > hi | miss < lo | pinball |
|---|---|---|---|---|---|---|---|
| backbone (unfloored) | 0.3554 | 0.7921 | 0.9120 | 2.7964 | 0.0980 | 0.1099 | 0.2739 |
| backbone | 0.3555 | 0.7921 | 0.9120 | 2.7879 | 0.0980 | 0.1099 | 0.2736 |
| retrieval, empirical | 0.5170 | 0.7211 | 0.7951 | 1.9452 | 0.1879 | 0.0911 | 0.3338 |
| retrieval, Gaussian | 0.3537 | 0.7118 | 0.8063 | 2.1886 | 0.1772 | 0.1110 | 0.3341 |
| **fused, barycentre (deployed)** | **0.2967** | **0.7091** | **0.8436** | 2.3665 | 0.1340 | 0.1568 | 0.2907 |
| fused, barycentre over Gaussian branch | 0.2604 | 0.6958 | 0.8480 | 2.4882 | 0.1294 | 0.1748 | 0.2923 |
| **fused, mixture** | **0.4182** | **0.7836** | **0.9395** | 2.6473 | 0.1218 | 0.0947 | 0.2833 |

Paired bootstrap over series, ΔCov.80 against the deployed construction:

| contrast | Δ | 95% CI |
|---|---|---|
| mixture − barycentre | **+0.0744** | [+0.0728, +0.0762] |
| Gaussian branch − barycentre | −0.0134 | [−0.0142, −0.0126] |
| backbone − barycentre | +0.0830 | [+0.0815, +0.0844] |

## What it says

**1. The combination rule is the cause, and it accounts for ~90% of the loss.**
The backbone-to-deployed gap in Cov.80 is 0.0830. Swapping the barycentre for
the mixture — same branches, same weight, same neighbours — recovers **0.0744
of it, 89.7%**, with a CI nowhere near zero. The mixture is also better on the
proper scoring rule (pinball 0.2833 against 0.2907), so this is not
coverage-chasing at the expense of sharpness.

**2. The empirical-quantile estimator is not the culprit, though the spread is
genuinely too narrow.** Replacing `np.quantile` with `mean ± z·sd` over the same
neighbour cloud makes coverage *worse* (−0.0134). It widens the interval
(Width.80 1.9452 → 2.1886) but imposes symmetry on a skewed count distribution,
so it buys a little above the bound (0.1879 → 0.1772) and loses more below
(0.0911 → 0.1110). Separately, the neighbour spread *is* too small in magnitude:
mean neighbour sd is 0.9360 against a mean absolute error of 1.0885, so
E|err|/sd = 1.1629 where a calibrated spread would give 0.798 — **the neighbour
cloud is 68.6% as wide as a calibrated one**. Both statements hold at once: the
scale is about a third short, and fixing the shape without fixing the operator
does not help.

**3. The loss is two-sided, not one-sided.** The motivating guess was that a
lower bound floored at zero cannot be violated by a non-negative actual, so the
whole story would be the upper tail. It is not: the barycentre misses *below*
the bound 15.68% of the time against the backbone's 10.99%, and above it 13.40%
against 9.80%. The lower bound is above zero often enough to be violated, and
the barycentre damages that side more than the upper one.

**4. The zero floor is coverage-neutral, but it does move the widths.** The
backbone's quantiles are scored unfloored by the harness while the fused ones
are floored inside `scalerag.fuse`. Flooring cannot lose coverage on
non-negative targets — Cov.80 is 0.7921 either way, to four decimals — but it
narrows Width.80 from 2.7964 to 2.7879. The paper's backbone and fused widths
are therefore measured under slightly different conventions; the difference is
0.3% and changes nothing, but it should be stated rather than discovered.

**5. What is *not* claimed.** The mixture is not uniformly better-behaved. At
the 90% level it over-covers (0.9395 against a nominal 0.90) and exceeds *both*
branches, so "the barycentre falls below both its inputs" is an observation at
the 50% and 80% levels here, not a theorem. The mixture is wider than the
barycentre at all three levels measured (+5.6%, +11.9%, +27.7%), which is what
the variance argument predicts, but that ordering is asserted only where it was
measured. And the backbone alone remains the best-calibrated and lowest-pinball
construction of the seven — fusion buys point accuracy and pays for it in
distributional quality. That was already the paper's finding; item 10 supplies
the mechanism.

## Prior art

Checked against the 101-source corpus (notebook `e32fce51`), phrased to force a
`NOT IN SOURCES` where warranted:

- **No source combines a retrieval forecast and a base-model forecast by
  averaging quantile functions level by level** (Vincentization / quantile
  averaging / W2 barycentre). `NOT IN SOURCES`.
- **No source combines them as a mixture of CDFs or densities either.** The
  nearest construct is Bimodal Mixture Augmentation in arXiv:2606.18367 §2,
  which mixes trajectory draws from a foundation model with a historical
  transition distribution — not a pooling of retrieval and backbone predictive
  laws.
- **No source discusses the choice between the two rules**, nor the narrowing of
  the barycentre under location disagreement. `NOT IN SOURCES`.
- **No source builds its predictive distribution from the empirical spread of
  the k retrieved neighbours**, nor notes that this measures retrieval
  disagreement rather than forecast uncertainty. `NOT IN SOURCES`.

Two comparators do report interval coverage, which is worth citing rather than
claiming novelty for the practice:

- **RAID (arXiv:2606.16925, Table 8)** reports nominal 80% coverage across nine
  datasets and attributes its own undercoverage on the sparse Job-SDF panel to
  "the well-known narrow-interval pathology of Gaussian-residual diffusion".
  **The location was wrong on first pass.** It is **Appendix C ("Extended
  Experimental Results"), page 16 — not Appendix E.6**, which is "Semantic
  Homophily". Flagged unverified here, then corrected on a second corpus query.
  **Still not checked against the PDF itself**; verify the section before it
  goes into the manuscript, because a reviewer opening the paper will catch a
  wrong appendix immediately. This is the failure mode referee item 1 was
  about.

Scope this the same way as the signed-divisor finding: a diagnostic about *our*
operator, not a law about the field. What is unclaimed elsewhere is the
observation that the combination rule, not the retrieval branch, is what costs
the coverage.

## Consequences for the manuscript

1. **`sec:calibration` must define the fused law.** State that the deployed
   object is the quantile-average (W2 barycentre) of the backbone and retrieval
   quantile functions, floored at zero, and that the retrieval branch's
   quantiles are the empirical quantiles of the k=20 restored neighbours.
2. **The 0.698 is largely an artifact of that choice, and the paper should say
   so.** Roughly 90% of the undercoverage relative to the backbone is the
   combination rule; a mixture over the same two branches recovers 0.0744 of the
   0.0830 gap and improves pinball as well.
3. **The retrieval branch's spread is 68.6% of a calibrated one** — report it as
   a second, smaller defect, and do not conflate it with the operator.
4. **Withdraw nothing about the headline.** RMSSE, WRMSSE and the point metrics
   are untouched; this is entirely about the predictive distribution.

**Reported, never selected from.** The frozen configuration stays the
quantile-average at w = 0.5 / gate (rules 9, 12). The mixture is documented as
the fix a future version should adopt, not retro-fitted to the recorded results.

## Defect found and fixed during the run

The first smoke run reported a 59.9% below-interval miss rate for the mixture,
which is impossible for an interval that is *wider* than the one it is compared
against. Cause: `_cdf_at` counted `v < x` rather than `v <= x`, so at a mass
point it returned the **lowest** level of a tie block instead of the highest. An
M5 series that is zero half the time has `q(0.05) = … = q(0.5) = 0`, so
`P(X ≤ 0)` is 0.5; the code called it 0.05, pushed the mixture's lower bound
above zero, and manufactured the miss rate. Fixed to the right-continuous
convention and pinned by eight unit tests in
`tests/unit/test_fusion_calibration.py`, including a brute-force inversion check
and the shifted-Gaussian case the ordering claim rests on.

A second defect was caught before it reached this report: the spread diagnostic
printed `sd / E|err| = 0.860` against a benchmark of `0.798`, but those two are
reciprocals of one another. Read naively it says the neighbour cloud is slightly
*wider* than calibrated; the truth is that it is 68.6% as wide. Both quantities
are now reported in the same direction with the share spelled out.
