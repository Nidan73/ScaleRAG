# Sparse-end scale identifiability (Phase 14, Unit 2)

**Question.** The regime study reports retrieval utility as an inverted U in series
sparsity, with the win rate falling from a peak of 0.794 to 0.472 in the sparsest
bin. An external referee argued that this sparse-end collapse may be manufactured
by the scale operator rather than measured from retrieval: on a context window
that is entirely zero the deployed rule cannot identify a scale, so the forecast
is produced by a fallback regardless of what was retrieved.

**Answer.** The collapse is real. Removing the degenerate contexts raises the
sparsest-bin win rate from 0.472 to 0.520, which is outside the deployed
estimate's interval and so is a genuine artifact — but it accounts for roughly a
sixth of the fall from the peak. Under a symmetric oracle rescaling the retrieval
branch loses decisively in that bin (0.213), so the analogues there are genuinely
inferior, not merely mis-scaled. **C1 survives, narrowed.**

A second finding, from running that symmetric contrast across the whole profile
rather than only the sparsest bin (section 4b): the inverted U keeps its shape and
its peak location under symmetric rescaling, but **37% of its amplitude is the
backbone mis-levelling rather than analogue quality**. The mid-band shape advantage
is real and statistically significant, and it is about four points, not
twenty-nine.

Run: `scripts/scale_identifiability_run.py --subset 1000 --origins 50`, M5
validation, 50 non-overlapping origins from d_513 to d_1885, 1,000-series subset
drawn by `graphroute_ts.eval.select_subset` with seed 42. The consumed test split
was not touched. Data: `reports/scale-identifiability/m5-val-scale-identifiability-v3-s1000-seed42-lb365-50origins.json`.

## 1. What the deployed operator actually does

`retrieval_faiss._fit_params` for `scale="mean"` computes

```python
m = w.mean(1)
return np.stack([np.zeros(len(w)), np.where(m == 0, 1.0, m)], axis=1)
```

It substitutes a **scale of exactly 1.0** when the window mean is zero. There is
no `1e-8` guard anywhere in the implementation, though the manuscript declares one
(`new___main.tex:932`, and Eq. 4 at line 569). The restored continuation on such a
window is therefore the candidate rescaled to **unit mean**, not zero as both the
manuscript's equation and the referee's reading imply. The manuscript documents an
operator that did not run; correcting it is a separate defect, recorded in the
Phase-14 design spec.

The deployed context length is **L=56** (`scripts/scalerag_eval.py:38`), which the
manuscript never states numerically. The referee assumed 28.

## 2. The inverted U is uncontaminated except in its final bin

Pooled over 50 origins, deployed operator:

| zero-fraction bin | n/origin | degenerate | win rate (all) | win rate (identifiable) | mean ΔRMSSE (identifiable) |
|---|---|---|---|---|---|
| [0.0, 0.1] | 56 | 0.000 | 0.220 | 0.220 | −0.1049 |
| [0.1, 0.2] | 55 | 0.000 | 0.341 | 0.341 | −0.0322 |
| [0.2, 0.3] | 52 | 0.000 | 0.406 | 0.406 | −0.0191 |
| [0.3, 0.4] | 70 | 0.000 | 0.514 | 0.514 | −0.0014 |
| [0.4, 0.5] | 67 | 0.000 | 0.636 | 0.636 | +0.0302 |
| [0.5, 0.6] | 89 | 0.000 | 0.764 | 0.764 | +0.0577 |
| **[0.6, 0.7]** | 106 | 0.000 | **0.794** | **0.794** | **+0.0583** |
| [0.7, 0.8] | 98 | 0.000 | 0.770 | 0.770 | +0.0408 |
| [0.8, 0.9] | 126 | 0.000 | 0.700 | 0.700 | +0.0112 |
| [0.9, 1.0] | 151 | **0.426** | 0.472 | **0.520** | −0.0630 |

Degeneracy is confined to the last bin: every bin below 0.9 contains exactly
**zero** unidentifiable contexts. The rise from 0.220 to the 0.794 peak — the
entire ascending arm and the peak itself — is therefore unaffected by the
referee's concern.

The effect-size column answers a separate objection (win rate discards magnitude).
It traces the same inverted U and is negative at **both** ends, so the two arms
are not an artifact of thresholding at a coin flip.

## 3. Degeneracy explains about a sixth of the sparse-end fall

| quantity | value |
|---|---|
| deployed, all contexts | 0.4715 ± 0.0138 |
| deployed, identifiable only | 0.5199 ± 0.0171 |
| peak, bin [0.6, 0.7] | 0.794 |

Pre-registered rule: the identifiable-only rate moves outside the deployed
estimate's ±1.96 SEM band [0.4577, 0.4853], so the **first branch fires** — part of
the sparse arm is operator-driven. The magnitude is +0.048 against a peak-to-sparse
fall of 0.322, so degeneracy accounts for roughly **15%** of the collapse. The
remaining 0.27 is not explained by scale identifiability.

## 4. The symmetric oracle: the analogues are genuinely inferior there

Under the least-squares scale-only minimiser — the best any rule of the deployed
form could do — the retrieval branch wins **0.9186 ± 0.0096** of sparsest-bin
windows against the raw backbone. That figure is **not reportable**: it compares an
oracle-rescaled branch against an un-rescaled one, which is precisely the asymmetry
the referee raised as item 4 and which reversed a headline when corrected on ETTm2.

Giving the backbone the identical rescaling:

| contrast | win rate |
|---|---|
| oracle retrieval vs **raw** backbone | 0.9186 ± 0.0096 |
| oracle retrieval vs **oracle** backbone | **0.2133 ± 0.0123** |

The backbone gains more from optimal rescaling than the retrieval branch does. In
the sparsest regime the backbone has the better shape and the worse level; once
both levels are corrected, its shape advantage dominates. The deployed near-tie at
0.472 is two errors roughly offsetting, not two comparable forecasts.

This is the decisive result. The sparse-end collapse is not a scale-identifiability
artifact: the analogues are genuinely worse there, which is the mechanism the paper
already claims.

## 4b. The symmetric oracle across the whole profile — how much of the inverted U is shape?

Section 4 answers the sparsest bin. The same contrast run on **every** bin answers
a larger question a reviewer will ask before that one: does retrieval's mid-band
advantage survive when the backbone is given the same rescaling, or is it only that
the backbone mis-levels on intermittent series?

| bin | n/origin | deployed | symmetric oracle | ±95% | shift |
|---|---|---|---|---|---|
| [0.0, 0.1] | 56 | 0.220 | 0.180 | 0.0151 | −0.040 |
| [0.1, 0.2] | 55 | 0.341 | 0.300 | 0.0215 | −0.041 |
| [0.2, 0.3] | 52 | 0.406 | 0.330 | 0.0180 | −0.076 |
| [0.3, 0.4] | 70 | 0.514 | 0.387 | 0.0176 | −0.128 |
| [0.4, 0.5] | 67 | 0.636 | 0.429 | 0.0172 | −0.207 |
| [0.5, 0.6] | 89 | 0.764 | 0.495 | 0.0154 | −0.269 |
| **[0.6, 0.7]** | 106 | **0.794** | **0.542** | 0.0144 | −0.252 |
| [0.7, 0.8] | 98 | 0.770 | 0.520 | 0.0151 | −0.250 |
| [0.8, 0.9] | 126 | 0.700 | 0.435 | 0.0119 | −0.265 |
| [0.9, 1.0] | 151 | 0.472 | 0.213 | 0.0123 | −0.258 |

Three readings, in order of importance.

**The inverted U survives.** Under symmetric rescaling the profile still rises from
0.180, peaks at 0.542 in the same bin [0.6, 0.7], and falls to 0.213. The
non-monotonicity is a property of analogue *shape*, not an artifact of the
backbone's level errors. C1's shape claim is safe.

**Its amplitude is roughly two-thirds genuine.** Peak minus dense end is 0.574 as
deployed and 0.362 symmetric, so **63%** of the profile's amplitude is shape and
**37%** is the backbone mis-levelling on intermittent series. The paper must say
this: a material part of the measured utility is retrieval acting as a level patch
for a mis-calibrated backbone, not as pattern memory.

**The mid-band advantage is real but small.** At the peak the symmetric rate is
0.542 with a 95% interval of [0.528, 0.556], which excludes a coin flip. So the
analogues genuinely do carry better shape than the frozen backbone in the
intermittent mid-band — by about four points, not the twenty-nine the deployed
number suggests.

Note this differs by panel, and coherently so. On dense continuous ETTm2 the
backbone's oracle shape floor is *better* than the retrieval branch's (0.0566
against 0.0647). On intermittent M5 mid-band it is slightly worse. That is the same
regime story the paper already tells, now measured on shape alone with the level
advantage removed from both sides.

## 5. No alternative operator rescues it

Scored on one common population (the sparsest bin where every past-data operator
resolves), so that an operator cannot look better by silently dropping the series
it cannot handle:

| operator | win rate | attrition |
|---|---|---|
| `window_mean` (deployed) | 0.4714 ± 0.0141 | 0.000 |
| `long_lookback` (365 d) | 0.4247 ± 0.0165 | 0.000 |
| `hierarchy_prior` (store×dept) | 0.2423 ± 0.0161 | 0.000 |
| `oracle_future_mean` | 0.4921 ± 0.0187 | 0.423 |
| `oracle_least_squares` | 0.9186 ± 0.0096 | 0.000 |

Both alternatives are **worse** than the deployed rule, and the hierarchy prior is
far worse. Seeding a scale from a pooled statistic is the treatment RAID uses at
L=0 (Appendix D.3: *"Chronos uses mean-scaling that divides by zero on empty or
zero context"*). It does not transfer to this setting.

`oracle_future_mean` carries 42.3% attrition because it is undefined when the
realised future is all zero. It is reported for completeness and should not be read
as a bound: moment matching is not the squared-error minimiser, so it bounds
nothing. Only `oracle_least_squares` is a bound.

## 6. What happens on the degenerate contexts

Mean over origins, degenerate cohort only (~65 series per origin):

| | value |
|---|---|
| backbone forecast | 0.0117 |
| retrieval forecast | 0.0849 |
| realised future | 0.1742 |
| retrieval win rate | 0.4037 |

Both branches under-predict by a wide margin. The backbone collapses toward zero
because Chronos-2's `InstanceNorm` guards a zero scale with `eps=1e-5` and its
inverse multiplies by that epsilon (`chronos_bolt.py:95`). The retrieval branch
restores to unit scale and lands closer, but still wins only 40% of the time.
Neither is usable on these series; they are a measurement hazard, not a regime.

## 7. Measurement caveats

- **36.8%** of the raw sparsest bin cannot be scored at all: RMSSE is undefined
  when the in-sample naive denominator is zero, which it is for an all-zero
  training history. Those series never entered any win rate.
- Consequently the never-launched series — items with no sale anywhere before the
  origin, 36.7% of the raw sparsest bin, rising to 85.8% at the earliest origin —
  have a **0.000** share of the *scored* bin. Their presence in the panel is not a
  contaminant of these numbers. All shares in this report are over the scored
  population.
- The band is unchanged by the stratification: deployed [0.327, 0.949],
  identifiable-only [0.327, 0.947]. The upper edge moves by 0.002.

## 8. Correction to the pre-registration

The pre-registered rule's second clause read: *if the win rate stays inside the
band even under `oracle_future`, extreme sparsity genuinely provides no anchor.*
That clause was **invalid as written**. `oracle_future` set the scale to the
realised future's mean, which is moment matching, not the squared-error minimiser,
so it bounded nothing — `error_decomposition.py`'s own docstring says as much. It
was replaced during execution by `oracle_least_squares`, the true minimiser, and
the symmetric contrast in section 4 is the corrected test.

The first clause was applied as written and is reported unchanged in section 3.
No criterion was relaxed, and no configuration was selected from any result here:
the deployed operator is unchanged.

## 9. Consequences for the manuscript

1. **C1 stands, with the sparse arm qualified.** The inverted U is real and its
   ascending arm is uncontaminated. State that 42.6% of the sparsest scored bin has
   an unidentifiable scale, report the identifiable-only rate of 0.520 beside the
   pooled 0.472, and note that degeneracy accounts for ~15% of the fall.
2. **Correct the equation.** Eq. 4 and the guard-ε table entry describe an operator
   that did not run. The implementation substitutes 1.0 at a zero mean.
3. **State L=56.** The manuscript never gives it numerically.
4. **Report the fused profile.** The deployment recommendation concerns the fused
   system, whose win rates are uniformly higher and same-shaped: 0.381 against the
   branch's 0.220 on the densest bin, 0.780 against 0.636 mid-band, 0.546 against
   0.472 on the sparsest.
5. **Apply item 4 to ourselves everywhere.** An oracle-rescaled branch compared
   against a raw backbone reads as 0.92 and is 0.21 when made symmetric. Every
   oracle contrast in the paper needs the same treatment.
6. **Report the symmetric profile beside the deployed one** (section 4b), and state
   plainly that 37% of the inverted U's amplitude is the backbone's level error.
   A reviewer who computes this and finds it unreported will conclude the paper
   benchmarked against an uncalibrated backbone. Reporting it costs a table and
   converts the strongest available attack into a contribution: to our knowledge no
   prior work reports a symmetric oracle contrast between a frozen TSFM and
   retrieved analogues (checked against the 101-source corpus; RAFT, RAID, TS-RAG,
   kNN-MTS and the RevIN/normalisation analyses all lack it).

## References

- RAID, Appendix D.3, "Mean-seed input convention at L = 0" — precedent for seeding
  a scale from a pooled statistic, and the source of the Chronos mean-scaling
  characterisation. Note it describes v1-era mean-scaling, not Chronos-2.
- arXiv:2603.11869, Appendix B — constant windows have zero variance and cause
  exploding normalised values; the authors remove such windows entirely, an option
  unavailable here because on M5 they are the regime of interest.
