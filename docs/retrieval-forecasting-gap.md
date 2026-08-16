# Why a +85.4% Retrieval Gain Does Not Become a Forecasting Gain

**The unexplained Phase-11A result.** Scale restoration improves the *retrieved
continuation* by **+85.4% MSE** (CI [85.2%, 85.6%], 7/7 channels), and the
*end-to-end* forecast still loses to the frozen Chronos-Bolt backbone by **0.85%**.
An aggregate number cannot say why, because it sums over four distinct failure
modes. This decomposition separates them.

**Method.** Post-hoc attribution over forecasts Phase 11A already stored
(`reports/phase11a/scalerag_native_ettm2_{split}_preds.npz`). Nothing is
recomputed: no model runs, no retrieval, no dataset is opened, no configuration is
selected. The frozen config (`scale=mean`, `k=20`, `weight=0.25`) is read from
`docs/scalerag-native-frozen-config.json` and used as-is. 80,199 windows × H=64.
Paired bootstrap, 2,000 resamples.

- Code: `src/graphroute_ts/error_decomposition.py`, `scripts/error_decomposition_run.py`
- Results: `reports/error-decomposition/ettm2-{test,val}-error-decomposition.json`

**The oracle floor.** `E_shape` is the MSE after the *least-squares* affine
correction `α·r + β` fitted to the realised future. Being the minimiser over all
affine maps, it is a genuine lower bound — no scale-restoration rule can beat it
for this retrieved set. Whatever survives is the analogue having the wrong shape.
(Moment matching, forcing the prediction onto the truth's mean and standard
deviation, is *not* that minimiser and gives no bound; an earlier draft of this
analysis used it and overstated the shape share by ~2×.)

## Results

MSE, frozen configuration:

| Stage | ETTm2 test | ETTm2 val |
|---|---|---|
| Raw retrieval (donor coordinates) | 2.9989 | 1.4785 |
| Restored retrieval (shipped) | 0.4376 | 0.2354 |
| **Shape floor** (oracle affine correction) | **0.0647** | **0.0541** |
| Backbone (frozen Chronos-Bolt) | 0.1486 | 0.1060 |
| Fused, `w=0.25` (shipped) | 0.1498 | 0.1110 |

| Attribution | test | val |
|---|---|---|
| Scale error removed by restoration | 2.5613 | 1.2431 |
| Scale error still remaining | 0.3730 | 0.1813 |
| Shape share of restored error | **14.8%** | **23.0%** |
| Restored retrieval ÷ backbone | 2.95× | 2.22× |
| Windows where retrieval beats backbone | 23.4% | 25.7% |
| Optimal fusion weight (diagnostic only) | 0.118 | 0.057 |

Paired-bootstrap contrasts, ETTm2 test (all CIs exclude 0):

| Contrast | ΔMSE | CI95 |
|---|---|---|
| Restoration gain (raw − restored) | +2.5613 | [+2.5224, +2.5994] |
| Residual scale error (restored − shape floor) | +0.3730 | [+0.3671, +0.3792] |
| Retrieval deficit (restored − backbone) | +0.2891 | [+0.2832, +0.2950] |
| **Fusion effect (fused − backbone)** | **+0.0013** | **[+0.0005, +0.0021]** |

## The answer

**The analogues are excellent. The magnitude correction is the bottleneck.**

The shape floor is **0.0647** against a backbone at **0.1486** — optimally
rescaled, the retrieved continuations would be **2.3× more accurate than
Chronos-Bolt** (2.0× on val). Retrieval is finding genuinely better futures than
the foundation model produces. That is the opposite of the intuitive reading of
the Phase-11A negative.

What fails is the magnitude correction. Restoration removes 2.56 of the 2.93 MSE of
scale error (87% of it), yet the remaining **0.373 is 5.8× the shape floor** and
**2.5× the backbone's entire error**. Only 14.8% of the restored retrieval's error
is shape; **85.2% is magnitude the mechanism did not recover**.

That is the whole gap. The retrieval branch reaches the fusion step 2.95× worse
than the backbone, so any meaningful weight on it hurts: the optimal weight is
0.118 against a frozen 0.25, and the shipped blend is significantly *worse* than
the backbone alone (+0.0013, CI excludes 0). A +85.4% improvement on the retrieval
branch is consistent with an end-to-end loss precisely because the branch was 20×
worse before restoration and is still 3× worse after it.

**Why the magnitude correction is incomplete** is answered mechanically by
`docs/affine-probe-report.md`: `scale="mean"` carries a scale term but **no
location term**, so it is exactly scale-equivariant and breaks under an offset.
ETTm2 is train-normalised and therefore roughly zero-centred — the regime where a
mean denominator is both degenerate and unable to represent the location shift.
Phase 11A's own code already flags near-zero `mean` denominators on zero-centred
normalised data as a counted failure case.

## What this does not license

**It does not license switching to `znorm`.** That is the obvious implied fix, and
this analysis deliberately stops short of it. The stored arrays contain only
`raw|res × {mean, rms} × k∈{5,10,20}` — no `znorm` variant exists, so testing the
hypothesis needs a new run, and selecting a scale strategy after seeing test-split
attribution is exactly the test-driven tuning rules 9 and 12 forbid. It would need
fresh pre-registration on a split not yet used for selection.

**It does not license adopting the optimal fusion weight.** 0.118 is reported as a
diagnostic. Fitting it here would be selection on an evaluation split.

**It does not overturn any recorded result.** ScaleRAG still meets 0/3
pre-registered criteria on the consumed M5 test split, still loses WRMSSE to
LightGBM and seasonal-naive, and is still −0.85% MSE against frozen Chronos-Bolt on
ETTm2. This explains the mechanism behind that negative; it does not soften it.

## Reproduce

```bash
uv run python scripts/error_decomposition_run.py --split test
uv run python scripts/error_decomposition_run.py --split val
```

Requires the Phase-11A prediction archives to be present in `reports/phase11a/`.
