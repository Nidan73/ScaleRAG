# Referee item 6(b): does the scale rule manufacture the sparse-end collapse?

**Date:** 2026-08-29. **Split:** M5 **validation** only, 50 non-overlapping
origins (`d_513`–`d_1885`), 1,000 series, frozen `L=56`, `H=28`, `k=20`,
category filter. **Script:** `scripts/m5_sparse_scale_rule_run.py`.
**Artifacts:** `reports/sparse-scale-rule/`.

## The question

The referee's item 6 named a competing explanation for contribution C1 and asked
for three diagnostics. Unit 2 answered (a) and (c). This is (b), verbatim:

> *"Re-run the sparsest two bins with the RMS divisor and with the full rule (3),
> on validation origins only. You already have both implemented."*

Their reasoning: Eq. (4) divides by the window mean, which on a near-empty
context is zero or a high-variance estimate of a small quantity, so *"the
restoration operator degenerates on near-zero contexts, independently of whether
the retrieval was any good."* If so, the sparse arm of the inverted U is an
artifact of the operator, not a property of retrieval.

Unit 2 could not answer this. It held the retrieved set fixed and swapped the
restoration scale; `rms` and `znorm` change `_transform`, hence the index, hence
which neighbours return. This re-runs retrieval in full under each rule.

**The stakes rose after the ETTm2 work.** Eq. (3) beats Eq. (4) there by 5.8%
fused, and the audit found the deployed divisor degenerates exactly where the
sparse bin lives. A plausible unified story was available: the operator, not the
regime, explains both arms of the U, and C1 becomes a normalisation result.

## Answer: no. C1 survives, and Eq. (4) is vindicated on M5

Win rate per fixed zero-fraction bin, mean ± SEM over the 50 origins, all three
rules scored on **one common population** (finite RMSSE under every rule,
never-launched series excluded):

| zero-fraction bin | n | degenerate | Eq. (4) `mean` | `rms` | Eq. (3) `znorm` |
|---|---:|---:|---:|---:|---:|
| [0.0, 0.1] | 2,780 | 0.000 | 0.220 ± 0.009 | 0.206 ± 0.008 | **0.247 ± 0.010** |
| [0.1, 0.2] | 2,770 | 0.000 | **0.338** | 0.299 | 0.328 |
| [0.2, 0.3] | 2,587 | 0.000 | **0.404** | 0.363 | 0.378 |
| [0.3, 0.4] | 3,522 | 0.000 | **0.515** | 0.442 | 0.439 |
| [0.4, 0.5] | 3,337 | 0.000 | **0.636** | 0.565 | 0.535 |
| [0.5, 0.6] | 4,464 | 0.000 | **0.766** | 0.668 | 0.615 |
| [0.6, 0.7] | 5,314 | 0.000 | **0.794 ± 0.006** | 0.684 ± 0.008 | 0.615 ± 0.007 |
| [0.7, 0.8] | 4,887 | 0.000 | **0.771** | 0.633 | 0.568 |
| [0.8, 0.9] | 6,296 | 0.000 | **0.700 ± 0.005** | 0.513 ± 0.006 | 0.469 ± 0.005 |
| [0.9, 1.0] | 7,568 | 0.427 | **0.472 ± 0.007** | 0.387 ± 0.007 | 0.385 ± 0.007 |

**The sparsest bin, which is the referee's question.** The deployed Eq. (4)
scores 0.4715 ± 0.0070, a 95% band of [0.4577, 0.4853]. Eq. (3) scores
**0.3852 ± 0.0073** — far outside that band, and *below* it. The paired
per-origin difference is **−0.0863 ± 0.0042**, and Eq. (3) is worse at
**50 of 50 origins**. `rms` behaves the same (0.3874 ± 0.0067).

So the rule that cannot degenerate — Eq. (3) carries a location term and divides
by a non-negative RMS deviation — performs *worse* at the sparse end, not
better. The collapse is not manufactured by the operator.

**The inverted U is not an artifact of the scale rule either.** It appears under
all three: 0.220 → 0.794 → 0.472 under Eq. (4), 0.206 → 0.684 → 0.387 under
`rms`, 0.247 → 0.615 → 0.385 under Eq. (3). Same shape, same peak region.

**Eq. (4) has the highest win rate in 9 of the 10 bins.** The exception is the
densest bin, where Eq. (3) leads on win rate (0.247 vs 0.220) but is worse on
effect size (mean ΔRMSSE −0.1832 vs −0.1048) and both are losses to the
backbone, so no rule wins there.

The degenerate share reproduces the recorded value exactly: **0.427** in the
sparsest bin, 0.000 everywhere else.

## What this settles

1. **Referee item 6 is answered in the paper's favour.** Their own words: *"If
   the inverted U survives, C1 is much stronger than it currently is."* It
   survives a direct test with the two operators they named, at every origin.
2. **A hypothesis of ours was falsified, which is why it was worth running.**
   After the ETTm2 result we expected Eq. (3) might rescue the sparse end and
   dissolve C1 into a normalisation story. It does not. That story was written
   down as a possibility before the run and is now closed.
3. **The sign-structure account is confirmed at bin granularity.** Eq. (4) is
   the right rule on non-negative sparse counts — in *every* bin, including the
   sparsest — while Eq. (3) is the right rule on the zero-centred ETTm2 panel.
   The operative axis is the data's sign structure, not sparsity, and it is now
   measured at two granularities on M5 and one on ETTm2.
4. **The frozen configuration is unchanged.** Eq. (4) at k=20 stays deployed.
   Nothing was selected from this run; it happens to agree with what was already
   frozen, which is the only reason no rule-9 question arises.

## Also delivered, incidentally

The run records mean and median ΔRMSSE **and** the fused win rate per bin, which
is what referee item 13 asks for (*"win rate ignores effect size"*, and *"either
report the fused win rate in Fig. 8 as well, or restate the recommendation"*).
Both are in `reports/sparse-scale-rule/`.

## Reproduce

```fish
python scripts/m5_sparse_scale_rule_run.py --subset 1000 --origins 50 --resume
```

~15 minutes on the reference machine. Validation origins only; `--split test`
does not exist, and the harness refuses any origin whose evaluation window
reaches the consumed test split.
