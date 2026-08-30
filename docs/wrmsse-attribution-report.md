# Why frozen Chronos-2 scores WRMSSE 1.76 while winning MASE and WAPE (Phase 14, Unit 1)

**Question.** The Phase-10 held-out run recorded frozen Chronos-2 at WRMSSE 1.9395
on M5 test against Seasonal-Naive 0.8697 and LightGBM 0.8663, while the same
forecasts were best on MASE (0.8932) and WAPE (0.6653). An external referee judged
this "almost certainly a bug" and warned that if the WRMSSE implementation is
faulty, pre-registered criterion 2 was never meaningfully tested — so Table VIII's
"0 of 3" would become "0 of 2 plus one untested".

**Answer.** The implementation is sound and the WRMSSE criterion was
meaningfully tested. **0 of 3 stands as written.**

> **Numbering note added 2026-08-30 (item 20).** "Criterion 2" above follows the
> numbering Table VIII used *before* it was corrected. Against the registration
> of record (`docs/scalerag-ts-method.md`, commit `8ad961a`) WRMSSE belongs to
> **criterion 1** — "≥3% RMSSE *or WRMSSE* over the strongest matched baseline"
> — and the registered criterion 2 is "≥5% over target-only Chronos-2 on **both**
> M5 and Favorita". Nothing in this report's finding changes; only the label
> does. The number is caused by a systematic **−29.0%
aggregate under-forecast**: Chronos-2's errors are shared across series rather than
idiosyncratic, so they do not cancel under hierarchical aggregation. Removing that
single scalar takes WRMSSE from **1.7568 to 0.9524** — one number accounts for
**45.8%** of the deficit. Known-future calendar covariates do **not** fix it.

Run: `scripts/wrmsse_attribution_run.py`, M5 validation origin, full 30,490-series
panel (a subset would break the hierarchy at the store and department levels). The
consumed test split was not touched. Data:
`reports/wrmsse-attribution/m5-val-wrmsse-attribution-n30490-seed42.json`.

## 1. The harness reproduces the recorded run exactly

| method | recorded val | this run | |
|---|---|---|---|
| `seasonal_naive` | 0.922781 | 0.922781 | **exact** |
| `chronos2_target` | 1.756787 | 1.756786 | **exact** |
| `recent_mean` | 1.101410 | 1.097484 | differs by 0.004 |

The two that matter reproduce to six decimals, so this diagnostic is measuring the
same thing the held-out run measured. `recent_mean` differs slightly because this
script uses a 28-day mean where the held-out harness used a different lookback; it
is context here, not a subject.

## 2. Two hypotheses were already dead before the run

- **Median point forecast.** `src/graphroute_ts/tsfm/chronos2.py:9` uses the
  pipeline's predictive **mean**, deliberately and with a comment saying so. The
  referee's leading hypothesis, and the one most of the M5 literature would
  suggest, does not apply.
- **Metric misconstruction.** Dollar weights come from the last 28 **training**
  days (`scripts/scalerag_test_final.py:153`), and aggregated-level scale
  denominators are computed on the aggregated series rather than summed from
  bottom-level ones (`src/graphroute_ts/hierarchy.py`). Both are correct, and both
  are shared by every method, so neither can produce a method-specific anomaly.

## 3. The per-level profile is inverted

WRMSSE by hierarchy level, M5 validation:

| level | seasonal-naive | recent-mean | **chronos2** | chronos2 + covariates |
|---|---|---|---|---|
| L1_total | 0.6677 | 1.2001 | **2.1590** | 2.0878 |
| L2_state | 0.7245 | 1.1513 | 1.9892 | 1.9235 |
| L3_store | 0.8252 | 1.1188 | 1.9101 | 1.8471 |
| L4_cat | 0.7543 | 1.1938 | 2.2241 | 2.1646 |
| L5_dept | 0.9738 | 1.2304 | 2.2671 | 2.2070 |
| L6_state_cat | 0.8347 | 1.1457 | 2.0126 | 1.9578 |
| L7_state_dept | 0.9844 | 1.1690 | 2.0157 | 1.9620 |
| L8_store_cat | 0.9205 | 1.1111 | 1.8345 | 1.7845 |
| L9_store_dept | 1.0290 | 1.1131 | 1.7711 | 1.7253 |
| L10_item | 1.1143 | 0.9549 | 1.0847 | 1.0756 |
| L11_item_state | 1.1192 | 0.9106 | 0.9458 | 0.9424 |
| L12_item_store | 1.1259 | 0.8709 | **0.8675** | 0.8684 |

Chronos-2 runs from **2.16 at the top of the hierarchy to 0.87 at the bottom** —
best where aggregation is deepest is exactly backwards. The healthy shape is
LightGBM's, recorded separately at 0.509 at L1 rising to 0.860 at L12: errors
cancel when summed, so aggregated levels score *better*. Seasonal-naive shows the
same healthy direction here (0.668 → 1.126).

At the bottom level Chronos-2 is the **best** method in the table (0.8675). Its
per-series forecasting is not the problem.

## 4. The cause: a shared −29% level bias

| method | relative signed bias | WRMSSE | after oracle level correction | share explained |
|---|---|---|---|---|
| seasonal_naive | −2.9% | 0.9228 | 0.9231 | −0.0% |
| recent_mean | −0.6% | 1.0975 | 1.0974 | 0.0% |
| **chronos2_target** | **−29.0%** | 1.7568 | **0.9524** | **45.8%** |
| chronos2_covariates | −27.9% | 1.7122 | 0.9621 | 43.8% |

Rescaling the whole forecast by the single ratio that removes the aggregate bias is
an **oracle** — it reads the realised total — and is reported as a diagnostic only.
Adopting it would be selection on an evaluation split (rules 9, 12). Its purpose is
to bound how much of the deficit is one number, and the answer is nearly half.

That the correction barely moves seasonal-naive or recent-mean, whose biases are
already near zero, is the control: the diagnostic is not simply flattering whatever
it touches.

**The error is shared, not idiosyncratic.** Ratio of absolute error at L1 to
absolute error at L12 — near 1 means every series errs in the same direction and
aggregation preserves it; near 0 means errors cancel:

| method | L1 / L12 absolute error |
|---|---|
| seasonal_naive | 0.086 |
| recent_mean | 0.188 |
| **chronos2_target** | **0.432** |
| chronos2_covariates | 0.415 |

Chronos-2's errors are five times more correlated across series than
seasonal-naive's. That is the mechanism: a per-series bias of a few units is
invisible against per-series noise and dominates once 30,490 series are summed.

## 5. Calendar covariates do not fix it

The pre-registered hypothesis was that the shared error is calendar-driven —
Chronos-2 runs target-only, so it cannot see SNAP days, events or day-of-week, and
every store misses the same event on the same day. Supplying those as known-future
covariates (`snap`, `wday`, `event_type_1`, through `forecast_with_covariates`,
guarded by `assert_no_future_covariates`) moves the bias from −29.0% to −27.9% and
WRMSSE from 1.7568 to 1.7122. That is **2.5%**, and per-series RMSSE and WAPE get
marginally *worse* (0.7765 → 0.7774, 0.6712 → 0.6720).

So the shared error is real but it is **not calendar structure**. It is a level
calibration failure: the frozen model systematically under-predicts the magnitude
of intermittent retail demand, and giving it the calendar does not tell it the
level.

## 6. Correction to the pre-registration

The rule fixed before the run had two branches: L1 ≫ L12 with the covariate arm
reducing it ⇒ correlated calendar error; levels roughly uniform ⇒ dollar-weight
concentration. **Reality gave a third outcome.** L1 ≫ L12 holds decisively, but the
covariate arm reduces it by only 2.5%, so neither branch describes what happened.

The rule is reported as written and the outcome as observed. Neither branch is
retrofitted, and no configuration was selected from any result here: the frozen
backbone is unchanged.

## 7. What this does not license

- **It does not rescore criterion 2.** The M5 test split stays consumed.
  `scripts/scalerag_test_final.py` stored no per-series predictions, so any test
  metric would require re-running inference on a closed split (rules 2, 9, 12).
  This explains a number on validation; it does not change a verdict on test.
- **It does not adopt the level correction.** The 0.9524 figure reads the realised
  total. It is a bound on how much of the deficit is one scalar, not a method.
- **It does not make Chronos-2 competitive.** Even fully debiased at 0.9524, it
  remains far behind LightGBM's recorded 0.7106 on the same split. The level bias
  is the largest single component of the gap, not the whole of it.
- **It does not license adding covariates to the frozen configuration.** They help
  WRMSSE by 2.5% and hurt RMSSE and WAPE. Changing the frozen backbone on the
  strength of that would be exactly the test-driven tuning the rules forbid.

## 8. Consequences for the manuscript

1. **Answer referee item 5 directly: it is not a bug.** State the per-level
   inversion, the −29% shared bias, and the 45.8% share. Table VIII's "0 of 3"
   stands; the "0 of 2 plus one untested" scenario does not arise.
2. **Report the mechanism as a finding about frozen TSFMs.** A zero-shot foundation
   model that is best-in-table at the bottom level (0.8675) and worst by a wide
   margin at the top is a clean statement about correlated error under hierarchical
   aggregation, and it is measured, not asserted.
3. **Note the covariate negative.** Known-future calendar covariates are the obvious
   fix and they do not work. That is worth a sentence, and it forecloses the first
   question a reviewer will ask.
4. **Cite the M5 organisers on metric divergence.** Makridakis, Spiliotis and
   Assimakopoulos, "The M5 Accuracy competition: Results, findings and conclusions",
   Section 5.1 and Figure 3: the winning method ranked 13th on the per-series MCB
   test and the runner-up 47th, because the top five optimised WRMSSE rather than
   per-series accuracy. Per-series and hierarchical rankings diverge in M5 by
   design; this result is an instance of that, not an anomaly peculiar to us.
