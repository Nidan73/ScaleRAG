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

---

## ADDENDUM — 2026-08-28 (selection step executed; premises revised)

**Appended, not edited.** Nothing above this line has been altered. The
confirmation run has not begun.

**1. Premise 2 is retracted.** It cites the C3 claim that on ETTm2 "the analogues
are 2.3x better than the backbone once optimally rescaled". That comparison
applied an oracle correction to the retrieval branch and none to the backbone.
Under the *same* correction the backbone's shape floor is 0.0566 (test) / 0.0469
(val) against retrieval's 0.0647 / 0.0542 — the analogues are ~13% **worse**.
See `docs/retrieval-forecasting-gap.md`.

**2. Premise 1 is confirmed and is the operative one.** The affine probe's
location-term result — `mean` restoration carries no μ, so a pure offset degrades
it 291x — is what the selection step validated.

**3. A third premise, stronger than either, was added by the code audit.** On
ETTm2 the deployed Eq. (4) divisor is the *signed* window mean, negative for
65.4% of validation queries, so 26.05% of retrieved pairs are restored through a
negative ratio and emerge sign-inverted. See `docs/code-versus-text-audit.md`
finding B. This is a correctness defect in the deployed operator, but the
selection step showed it is **not** what costs the accuracy (see 4).

**4. Selection step: EXECUTED on ETTm2 validation, 2026-08-28.**
`docs/ettm2-scale-operator-selection.md`,
`reports/ettm2-znorm/`. Fused at the frozen k=20, w=0.25, against a
0.10602 backbone: Eq. (4) `mean` −4.548% (CI [−5.033, −4.077]), `rms` −5.361%,
Eq. (3) `znorm` **+1.522%** (CI [+1.253, +1.781]). Eq. (3) over Eq. (4) is
+5.806% fused and +30.967% on the retrieval branch, all intervals excluding zero.
**ETTm2 validation selects `znorm` decisively.**

`rms` removes the sign flip but has no location term and is the *worst* of the
three, so the binding constraint is the **missing μ**, not the sign pathology.

**5. M5 validation selects the opposite, and this was already on record.**
`docs/ablation-report.md`: `mean` 0.7425 RMSSE against `znorm` 0.8719 — Eq. (4)
wins by 17% on the sparse non-negative panel. The selection is therefore **not
global**. The operative axis is the panel's sign structure: divide by a signed
mean only where the data is non-negative.

**6. The confirmation plan is now in question, and this is a decision for the
user, not a mechanical step.** The one clean split this document names is
**Favorita test, origin 972**. Favorita is a retail count panel like M5 —
non-negative and sparse — so both theory and the M5 result predict Eq. (4) wins
there and Eq. (3) loses. Running the confirmation on the single panel where the
hypothesis is predicted to fail would confirm nothing.

Hypotheses H1–H3 above were written as though `znorm` were universally better.
They are **not** revised here, because revising hypotheses after seeing
selection evidence is precisely what a pre-registration exists to prevent. Either
a **new** pre-registration is written for the sign-structure-conditional
hypothesis — in which case Favorita test becomes its predicted-negative arm,
which is a legitimate and informative use of it — or the ETTm2 finding stays
**validation-only and unconfirmed**, and is reported that way.

**Status unchanged: the confirmation run has NOT been executed.**
