# Controlled Synthetic Affine Probe

**Purpose.** Answer the reviewer critique that `y_restored = y*·σ_q + μ_q` is
"simply denormalizing a normalized signal" with a causal experiment rather than an
argument, and separate two claims the observational M5/ETTm2 studies cannot
separate:

| Property | Stage | Measured by |
|---|---|---|
| **Invariance** | retrieval | top-1 hit rate against the known donor motif |
| **Equivariance** | reconstruction | scale-free error against the known continuation |

On real data the affine relationship between a query and its analogue is never
known, so neither stage can be scored on its own. Here it is *imposed*: a query is
built as an exact affine image `x' = a·x + b` of a known donor motif, making both
the correct analogue and the correct continuation `a·y + b` available in closed
form.

**Scope.** Synthetic data only. Touches no M5 or Favorita split, opens none of the
blocked Phase-11B datasets, and makes **no end-to-end accuracy claim**. It tests
whether the mechanism does what it says, not whether it wins.

- Code: `src/graphroute_ts/affine_probe.py`, `scripts/affine_probe_run.py`
- Tests: `tests/unit/test_affine_probe.py`, `tests/leakage/test_affine_probe_leakage.py`
- Results: `reports/affine-probe/affine-probe-results.json` (regenerable)

## Design

12 motifs × 4 rows = 48 series. Each motif is a harmonic mixture, centred and
unit-scaled, then instantiated on four rows at **different magnitudes** (amplitude
log-uniform over 4 orders of magnitude, plus a per-row offset). Observation noise
scales with row amplitude, holding SNR constant across scales.

The magnitude heterogeneity is essential and was not obvious. An earlier version
unit-scaled every row; that makes `‖c‖` constant across candidates, so the L2
ranking `‖q‖² − 2q·c + ‖c‖²` collapses to shape correlation and **raw retrieval
becomes accidentally scale-invariant** — scoring 1.00 hit rate at every `(a, b)`
and hiding the exact failure the probe exists to expose. `make_motif_panel`
documents this, and a unit test guards against the regression.

Retrieval runs against the untransformed pool, and the query's own row stays in it
as it would in deployment, so `a=1, b=0` is a clean control. Candidates satisfy
`t_r + H < origin` (rule 3), re-asserted after selection.

5 seeds, 200 queries, k=5, noise 0.02, context 48, horizon 12. Chance hit rate =
1/12 = 0.083. CIs are paired bootstrap, 2,000 resamples, paired by seed.

## Results

### Retrieval invariance

| Condition | (1, 0) control | a=2, b=0 | a=10, b=0 | a=100, b=200 |
|---|---|---|---|---|
| `raw` | 1.000 | 0.343 | 0.249 | **0.176** |
| `znorm` | 1.000 | 1.000 | 1.000 | **1.000** |

Raw retrieval is exact in the control and collapses toward chance once the query's
magnitude moves away from its own row's. A pure location shift is enough: at
`a=1, b=200` it already falls to 0.274.

**znorm − raw top-1 hit rate, pooled over all transformed cells: +0.744, CI95
[+0.702, +0.784].** CI excludes 0.

### Reconstruction equivariance

Scale-free error (`nmse`; 1.0 ≈ predicting the mean):

| Condition | a=1, b=0 | a=10, b=0 | a=100, b=200 | spread over full grid |
|---|---|---|---|---|
| `znorm` (no restore) | 0.748 | 1.65 | 1.87 | — |
| `znorm+restore` | 0.000438 | 0.000438 | 0.000438 | **8.7 × 10⁻¹⁹** |

Two things follow.

**Invariance alone is worthless.** `znorm` without restoration retrieves the
correct motif *every time* (hit rate 1.000) and still forecasts no better than the
series mean (nmse 0.75–1.87). Finding the right analogue and using it are separate
problems; the retrieved future is in the donor's coordinate system.

**Restoration is exactly affine-equivariant.** With restoration the error is
constant to 8.7 × 10⁻¹⁹ across the whole 12-cell `(a, b)` grid — float noise. The
residual 0.000438 is the observation-noise floor, not method error.

**znorm+restore − znorm, nmse: −1.508, CI95 [−1.71, −1.272].** CI excludes 0.

### The frozen M5 configuration is NOT affine-equivariant

The shipped `ScaleRAG_gated` config uses `scale="mean"`, which carries a scale term
but **no location term**. The probe measures the consequence:

| Spread of nmse | `znorm+restore` | `mean+restore` | `rms+restore` |
|---|---|---|---|
| across `a`, at `b=0` | 8.7e-19 | **7.6 × 10⁻¹⁹** | 1.1 × 10⁻¹⁹ |
| across the full `(a, b)` grid | **8.7 × 10⁻¹⁹** | **0.195** | — |

`mean` and `rms` are *exactly* scale-equivariant and break under a location shift.
`mean+restore` hit rate falls to 0.580 at `a=1, b=200`, and its nmse rises from
0.00067 to 0.195 — a 291× degradation from an offset alone.

This is a limitation of the frozen configuration, found by this probe and reported
rather than fixed (rules 9, 12). It does not invalidate the M5 results: M5 demand
is non-negative and count-like, so a large negative location shift is not a regime
that occurs there, which is plausibly why `mean` was selected on validation in the
first place. But it does scope the claim.

## What this does and does not establish

**Establishes.** Scale restoration is not a no-op relabelling of normalisation. The
two stages are separately necessary: normalisation buys invariant *retrieval* and
nothing else; restoration is what converts an invariantly-retrieved analogue into a
forecast in the query's coordinates. Under an imposed affine transform the
composition is exact to floating-point precision.

**Does not establish.** Anything about end-to-end forecasting accuracy on real
data. The recorded negatives stand unchanged: ScaleRAG meets 0/3 pre-registered
criteria on the consumed M5 test split, loses WRMSSE to LightGBM and seasonal-naive,
and is −0.85% MSE against frozen Chronos-Bolt on ETTm2. A mechanism can be exactly
correct and still fail to help a strong backbone — which is the separate open
question this probe deliberately does not touch.

**Also does not establish** that `znorm` should replace `mean` in the frozen config.
That would be a method change selected after seeing results, and the M5 test split
is consumed. The finding is recorded as a scoping limitation, not acted on.

## Reproduce

```bash
uv run python scripts/affine_probe_run.py --seeds 5 --n-queries 200
```

Deterministic for a fixed `--base-seed` (default 20260726); a unit test asserts
run-to-run identity. Runtime is a few seconds; no GPU, no network, no dataset.
