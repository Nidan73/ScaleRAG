# Referee item 3: in-context conditioning against the convex blend

**Date:** 2026-08-29. **Split:** M5 **validation**, 5 non-overlapping origins,
1,000 series (4,997 scored), frozen Chronos-2, `L=56`, `H=28`, retrieval `k=20`.
**Script:** `scripts/m5_incontext_baseline_run.py`.
**Artifacts:** `reports/incontext-baseline/`.

## The question

> *"Pass the k retrieved neighbours to Chronos-2 as additional in-context series
> or covariates and compare that against your convex blend. ... it tests whether
> the frozen backbone can use the analogues better than an external weighted
> average can. If in-context conditioning wins, the fusion stage is the wrong
> design; if it loses, that is a genuinely interesting result and strengthens the
> paper considerably. Either way it belongs in Table V."*

The referee exempts Chronos-Bolt explicitly, so there is no ETTm2 arm: this is
M5 with Chronos-2, on validation origins, because the test split is consumed.

## Design

Neighbour contexts are passed as **extra variates** of a 2-D Chronos-2 input,
which its group attention cross-attends over. Row 0 is the target; rows 1..k are
the retrieved contexts, **left-NaN-padded to the target's own history length** so
the target's input is byte-identical to the target-only arm and the only
difference between arms is the added variates. The covariate route was rejected:
`chronos2/preprocess.py` raises on a covariate length mismatch, and the
referee's "or" permits the series route.

Two checks preceded the experiment. Adding neighbours **does** change the
target's forecast, and the change **depends on which** neighbours are added, so
the arm is not vacuous.

**Control.** Every k is run twice — once with the retrieved neighbours, once with
the same number of **random** series from the same legal candidate pool. Without
it a positive result cannot separate "the analogues are informative" from
"multivariate mode is just different," and the literature documents that
undiscriminated context expansion degrades accuracy.

## Result: in-context conditioning loses, decisively

RMSSE, pooled over 4,997 scored series, paired bootstrap (2,000 resamples):

| arm | RMSSE | vs backbone | vs blend |
|---|---:|---:|---:|
| **convex blend** ($w = 0.5$) | **0.7198** | **+3.16%** [+2.88, +3.45] | — |
| retrieval branch alone | 0.7400 | +0.44% [−0.15, +1.01] | −2.81% |
| in-context, retrieved, k=20 | 0.7401 | +0.43% [+0.35, +0.51] | **−2.82%** [−3.11, −2.52] |
| in-context, retrieved, k=5 | 0.7412 | +0.28% | −2.97% |
| in-context, random, k=5 | 0.7422 | +0.15% | −3.11% |
| in-context, random, k=20 | 0.7422 | +0.14% | −3.12% |
| in-context, retrieved, k=1 | 0.7427 | +0.08% | −3.18% |
| in-context, random, k=1 | 0.7427 | +0.08% | −3.18% |
| target-only backbone | 0.7433 | — | −3.27% |

The best in-context arm is **2.82% worse** than the convex blend
(CI [−3.11, −2.52], excludes zero). **The fusion stage is not the wrong design.**

## Three findings the table alone does not show

**1. Group attention does respond to analogue quality — it is not just extra
context.** Retrieved against random at matched k:

| k | retrieved vs random | 95% CI | excludes 0 |
|---:|---:|---|---|
| 1 | +0.000% | [−0.032, +0.030] | no |
| 5 | +0.134% | [+0.092, +0.177] | yes |
| 20 | +0.291% | [+0.231, +0.349] | yes |

At k=1 the model extracts nothing from analogue quality; the effect appears at
k=5 and roughly doubles by k=20. So the mechanism the referee pointed at is real.
It is simply small.

**2. Group attention adds nothing over the analogues by themselves.** In-context
at k=20 against the retrieval branch alone: **−0.014%**, CI [−0.588, +0.586] —
statistically indistinguishable. Feeding the analogues through the backbone
reproduces what the analogues already gave, and no more.

That is the sharp form of the answer. The blend beats both branches; the value
lies in *combining* them, and group attention does not perform that combination.
An external convex weight does it roughly **seven times** more effectively
(+3.16% against +0.43%).

**3. No degradation from the added context.** Random neighbours at k=20 still
help slightly (+0.14%) rather than hurting, so the "stochastic noise
accumulation" failure mode reported for very long contexts does not fire here.
At k=20 the added context is 1,120 positions, well under the ~3,000 at which
attention-entropy saturation is described.

## Status

Referee item 3 is answered, and it is the branch the referee called *"genuinely
interesting ... strengthens the paper considerably."* It belongs in Table V as
three rows (in-context k=1/5/20) plus the random control, with the retrieved-vs
-random contrast in the text.

Nothing was selected from this run. The frozen configuration is unchanged
(rules 9, 12).

## Reproduce

```fish
python scripts/m5_incontext_baseline_run.py --subset 1000 --origins 5 --resume
```

~6 minutes. Validation origins only; the harness refuses any evaluation window
reaching the consumed test split.
