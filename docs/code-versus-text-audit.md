# Code-versus-text audit of the extended manuscript (Phase 14)

**Date:** 2026-08-28. **Branch:** `phase14-unit2`. **Manuscript:**
`Journal paper/new___main.tex` (gitignored; tracked record in
`docs/manuscript-revision-log.md`).

## Why this pass exists

The Phase-13 audit checked ~145 manuscript numbers against the recorded JSON
artifacts. It verified the paper against the *artifacts* and never the artifacts
against the *code*. Every one of the five defects found earlier in Phase 14 —
the phantom guard, the gate being a classifier, the unstated `L=56`, the
`topk_exact` tie-break bug, the Favorita sign — fell through that gap, and every
one was found incidentally while chasing something else. This pass looks
deliberately, claim by claim, at the Methodology and Experimental Design
sections against the source that produced the numbers.

**Scope:** the Proposed Methodology and Experimental Design sections, Table II,
Table IV and the cost accounting. **Method:** each implementation claim traced to the
function that ran, with the discrepancy quantified on real data where the
magnitude was not obvious from reading.

**Headline:** no result changes. Finding B is a new scientific finding that
*explains* an existing negative rather than overturning it. The rest are
description and accounting defects. Under rule 9 none of the operators below
were altered to improve a score — the negative-mean behaviour in B is reported,
not fixed.

---

## A. Two different scale operators ran; the paper describes a third

**Claim.** Eq. (4) defines the deployed scale-only rule as

    s(x) = x̄  if x̄ > 0 ;   1  if x̄ = 0

and Table II gives one "Zero-mean rule" cell spanning both columns: *"scale set
to 1 when x̄ = 0"*.

**Code.** Two different operators ran.

- **M5** uses `retrieval_faiss._fit_params` (`src/graphroute_ts/retrieval_faiss.py:32-34`):
  `np.where(m == 0, 1.0, m)`. The substitution fires only at an *exact* zero and
  negatives pass straight through.
- **ETTm2** uses `NativeScaleRetriever`, which carries an undocumented threshold
  `scale_eps = 1e-2` (`src/graphroute_ts/scalerag_native.py:199`) and two
  branches the paper never mentions:
  - a query with `|x̄| < 1e-2` is **excluded from retrieval entirely** and
    receives a constant context-mean forecast
    (`scalerag_native.py:270-273`);
  - a candidate with `|c̄| < 1e-2` keeps its **unrestored** continuation inside
    the pooled mean (`scalerag_native.py:290-292`).

Neither branch is the Eq. (4) substitution. The ETTm2 path never sets a scale to 1.

**Magnitude** (`reports/phase11a/scalerag_native_ettm2_{val,test}.json`,
`invalid_scale_diag.mean`):

| split | queries | fallback queries | share | skipped candidate restorations |
|---|---|---|---|---|
| val  | 561,393 | 1,132 | 1.41% of windows | 378 |
| test | 561,393 |   931 | 0.17% of windows |  28 |

**Consequence.** Table II's zero-mean cell is wrong for the dense column, and
Eq. (4)'s "we state the rule as it ran" is true of M5 only. The mixed
restored/raw pooling also breaks the Eq. (8) identity (see C).

---

## B. On ETTm2 the deployed divisor is negative for two-thirds of queries

This is the substantive finding of the pass.

**Claim.** Eq. (4) defines `s(x)` for `x̄ > 0` and `x̄ = 0`. The case `x̄ < 0`
is not defined anywhere in the paper. The text hedges only that on "a
train-normalised, near zero-centred panel such as ETTm2 it is neither" cheap nor
natural, and Propositions 1 and 2 are both stated for `a > 0`.

**Code.** `_fit_params` passes a negative mean through unchanged, and
`_scale_invalid` gates on `|scale| < eps`, so a mean of −0.5 is treated as
perfectly valid. Restoration is then
`(cont − c_loc)/c_scale · q_scale` with `c_loc = q_loc = 0`, i.e. the analogue is
multiplied by the ratio `q̄/c̄`.

**Measured on the ETTm2 validation windows** (all 80,199 × 7 = 561,393):

- **65.4%** of query contexts have `x̄ < 0`;
- **64.5%** are negative *and* above the validity threshold, so they are used;
- exactly **0** windows have `x̄` exactly zero, so Eq. (4)'s stated guard never
  fires on this panel at all.

Per channel, share of query contexts with a negative mean: LUFL 97.9%,
HUFL 87.6%, HULL 85.9%, MUFL 82.7%, MULL 56.7%, LULL 37.3%, OT 9.9%.

**What it does to retrieval.** For a 400-query-per-channel sample of the
validation split at the deployed `k=20`, the share of retrieved
(query, candidate) pairs whose scales carry **opposite signs**:

| channel | mismatched pairs | queries with ≥1 mismatch |
|---|---|---|
| HUFL | 27.8% | 52.8% |
| HULL | 11.2% | 31.6% |
| MUFL |  4.2% | 10.2% |
| MULL | 27.9% | 48.2% |
| LUFL | **78.4%** | 85.0% |
| LULL | 24.0% | 33.4% |
| OT   | 10.7% | 14.8% |
| **pooled** | **26.4%** | — |

So on a quarter of all retrieved pairs the restoration multiplies the analogue
by a **negative** ratio and emits a sign-inverted continuation; on LUFL it does
so for four pairs in five. Separately, because the shape code is `x/x̄`, two
windows of identical shape whose means straddle zero map to opposite points, so
the L2 objective of Eq. (6) is not a shape metric on this panel.

**Status.** Reported, not fixed. This is a strong candidate mechanistic
explanation for the ETTm2 restored branch losing to its backbone (−0.714%) and
for the 85.1% unrecovered-magnitude residual, and it strengthens rather than
weakens the pre-registered `znorm` follow-up in
`docs/znorm-preregistration.md`. It must not be repaired and rescored, which
would be selection on a consumed split (rules 2, 9, 12).

### B-addendum: what the literature says (corpus check, 2026-08-28)

Queried against the 101-source NotebookLM corpus, with the ScaleRAG documents
excluded as circular. Two questions: is the pathology known, and does anyone
else divide by a signed quantity?

**The pathology is NOT IN SOURCES.** No paper in the corpus discusses dividing a
window by a signed statistic, sign inversion of the normalised representation,
exploding values at a near-zero denominator, or the resulting distance ceasing
to behave as a shape distance.

**But the reason it is unreported is that nobody else makes the choice.** Every
comparator either divides by a non-negative magnitude or does not divide at all:

| method | subtracts a location term | divides by |
|---|---|---|
| RAFT (arXiv:2505.04163, §3.2, Eq. 1–2) | yes — the last observed value `x_L` | nothing; Pearson correlation on the offset-subtracted patch |
| RAID | yes — the mean | the standard deviation (non-negative) |
| TS-RAG (§3.2) | delegated to the pretrained encoder | nothing; Euclidean distance in embedding space |
| kNN-MTS (arXiv:2505.11625, §III-B) | no | nothing; squared L2 on raw/channel-standardised series |
| TimeRAG (arXiv:2412.16643, §II-B, Eq. 4) | no | nothing; DTW on raw subsequences |
| RevIN (arXiv:2603.11869, App. B) | yes — the mean | `σ`, and *"in practice σ is replaced by σ + ε"* |

The corpus also confirms ETT is z-score normalised upstream, hence zero-centred
and sign-straddling, and that the field's standard guard is `σ + ε` plus outright
**removal** of degenerate windows — arXiv:2603.11869 App. B removes zero-variance
windows because they *"cause exploding normalized values"*.

**Framing consequence.** Do not present this as a general hazard of
retrieval-augmented forecasting. ScaleRAG is the only method in the comparison
set that divides by a signed quantity, so this is a diagnostic about *our*
operator on a panel where the choice was inappropriate — it explains our own
ETTm2 negative rather than establishing something about the field. The
defensible general contribution is narrower and still real: the field converged
on non-negative divisors without stating why, and this is the first quantified
measurement of what the alternative costs (26.4% of retrieved pairs inverted).

**Consequence for `docs/znorm-preregistration.md`.** Its premise 2 — the C3 claim
that the analogues are "2.3x better than the backbone once optimally rescaled" —
is **retracted**; under a symmetric correction they are ~13% worse. Premise 1
(the affine probe's 291x degradation under an offset) stands. Finding B is a
stronger third premise than either, because it is measured on the deployed panel
rather than a synthetic grid. The hypothesis is better motivated than when it was
written, for a different reason, and the pre-registration should carry a dated
addendum saying so rather than be rewritten.

---

## C. Undocumented non-negativity clip on M5

**Claim.** Eq. (7) is `y_restored = y*·σ_q + μ_q`; Eq. (8) states that because
restoration is affine, "averaging before or after restoration is identical".
Proposition 2 claims exact equivariance under `T(x) = ax + b`, `a > 0`.

**Code.** Both M5 paths clip the restored continuation at zero:
`scripts/scalerag_eval.py:79` and `src/graphroute_ts/retrieval_gpu.py:168`
(`np.clip(..., 0.0, None)`).

**Consequence.** Clipping is not affine, so for the deployed M5 pipeline the
Eq. (8) identity does not hold and Proposition 2's equivariance fails whenever
`b ≠ 0` drives a restored value negative. The clip is a defensible domain
constraint on a non-negative count panel — the objection is that it is
undeclared, and that the affine probe validates the *unclipped*
operator while the results section reports the clipped one.

Two further undeclared fallbacks sit in the same function: an empty candidate
set yields a constant context-mean forecast with the sentinel gate features
`nn_dist = 1e6`, `disagreement = 0.0` (`scalerag_eval.py:69-73`), and that
sentinel is fed to the gate as if it were a distance.

---

## D. Three of the six gate features are misdescribed

Eq. (11) against `scripts/scalerag_test_final.py:55-67` and
`scripts/scalerag_eval.py:82-83`:

| symbol | paper | code | verdict |
|---|---|---|---|
| `ζ` | `L⁻¹ Σ 1[x_t = 0]` | `(ctx == 0).mean(1)` | **matches** |
| `log v` | `v = Σ_t x_t` | `np.log(mean + 1.0)` | **mismatch** — window *mean*, offset by 1, not the sum |
| `s_σ` | "scale spread of the **retrieved neighbours'** context statistics" | `std / (mean + 1e-6)` on the **query's own** context | **mismatch** — different quantity entirely |
| `δ` | `(kH)⁻¹ Σ_j ‖y_j − y_restored‖²`, i.e. mean over the horizon of the **variance** | `conts.std(0).mean()`, mean over the horizon of the **standard deviation** | **mismatch** — differs by a square root inside the average |
| `d_1` | `min ‖q* − c*‖₂` | squared L2 (`topk_exact` and `IndexFlatL2` both return squared) | **mismatch, immaterial** — monotone, so tree splits are unchanged |
| `s_B` | "width of its central predictive interval" | `q90 − q10`, averaged over the horizon | **imprecise** — an 80% interval; state the level |

The `s_σ` error matters beyond bookkeeping: as written, the paper implies four
of six diagnostics summarise the retrieval; in the code only **two** (`d_1`,
`δ`) involve the retrieved set at all. That is also consistent with the
Phase-12 collinearity finding that the gate has ~1–2 effective dimensions.

---

## E. ETTm2 never used FAISS

**Claim.** `sec:search`: *"ETTm2 uses a flat FAISS `IndexFlatL2`"*, and Table II's
dense column gives Index = `FAISS IndexFlatL2`.

**Code.** The ETTm2 driver `scripts/scalerag_native_ettm2.py` builds a
`NativeScaleRetriever`, whose search is `topk_exact` — a NumPy BLAS GEMM with a
lexsort tie-break (`scalerag_native.py:67-104`). `faiss` is never imported on
this path. The module `retrieval_faiss` *is* imported, but only for the scale
primitives `_fit_params` / `_transform`, which is what its own docstring says.
FAISS appears in `scripts/scalable_retrieval_eval.py`, an M5-side script, and
in `ScaleAwareIndex`, which the ETTm2 path does not construct.

Both searches are exact, so no number changes; the claim is simply false as
stated and a referee can check it from the released code.

---

## F. The three gate seeds are one model

**Claim.** Table II: *"Gate parameters — 9,000 leaf values (3 seeds)"*, and the
study reports averaging over gate seeds 42/43/44.

**Code.** `GATE_SPEC = {n_estimators: 200, num_leaves: 15, learning_rate: 0.05,
min_child_samples: 50}` (`scripts/scalerag_test_final.py:47`) leaves LightGBM's
`subsample` and `colsample_bytree` at their 1.0 defaults, so nothing in fitting
consumes the RNG.

**Verified directly.** Fitting the spec at seeds 42, 43 and 44 on identical data
gives predictions differing by `max |Δp| = 0.0`, and the serialised boosters
differ **only** in the recorded `seed` / `bagging_seed` /
`feature_fraction_seed` / `extra_seed` metadata lines — the trees are identical.

**Consequence.** The distinct parameter count is **3,000**, not 9,000; Table II
triple-counts one model. Phase 12 already established that the seed average
carries no dispersion and must not be cited as robustness
(`docs/gate-transfer-report.md`); the inflated parameter count is new.

---

## G. The manuscript contradicts itself on the degenerate share

Line 592 (`sec:sicn`): *"On the M5 validation panel this affects 26.4% of the sparsest
zero-fraction bin"*. Line 1253 (`sec:regimes`): *"42.6% of the scored series there have
x̄ = 0"*. Same quantity, two values, ~660 lines apart.

The recorded artifact supports the second:
`reports/scale-identifiability/m5-val-scale-identifiability-v3-s1000-seed42-lb365-50origins.json`
gives `0.426` for the `[0.9, 1.0]` bin, and
`docs/scale-identifiability-report.md:229` instructs stating 42.6%. No `0.264`
degenerate share appears in any artifact. **26.4% is unsupported** and looks like
a transposition.

---

## H. Items 19, 24 and 9 were measured but never written into the manuscript

`docs/research-rules.md` and the design spec both list items 19, 24 and 27 among "12 of 27
closed", and `docs/manuscript-revision-log.md`'s "Not yet done" omits 19, 24
and 9. The manuscript contains none of their results.

- **Item 19.** `sec:cost`, Table IV and the `fig:pareto` caption still read
  0.607 ms CPU per window, 1,096.2 MB, and *"on ETTm2 we are Pareto-dominated"*;
  the Weaknesses subsection repeats it (line 1739). The measured replacements sit unused in
  `reports/ettm2-cost/ettm2-val-cost.json`: **0.073 ms/query on GPU against
  0.486 ms CPU (6.6×)**, below TS-RAG's 0.450 ms adapter, with candidate
  selection verified identical; and the searched index is float32 at
  **69.6 MB**, not the float64 139.2 MB the storage figure was derived from.
- **Item 24.** No occurrence of the k sweep (M5 k=50 at 0.7335 against k=20 at
  0.7425, turning back up at k=100).
- **Item 9.** No occurrence of the gate's out-of-origin AUC 0.763, accuracy
  0.711 against a 0.616 majority baseline, or Brier 0.190.

**The closed count should read 9 of 27, with 3 measured but unwritten.**

---

## Verified correct — do not re-check these

- **Horizon guard.** `WindowDatabase.legal_mask` uses strict `<`
  (`retrieval.py:109-111`), the pool is built at `t_r + H <= train_end`
  (`retrieval.py:87`), and `scalerag_eval.py:57` passes `o + 1` as the origin, so
  `t_r + H < o + 1 ⟺ t_r + H <= o`. Eq. (5) matches the code; no leak. The paper's
  "enforced at pool construction rather than at query time" understates it —
  both are enforced, and `assert_retrieval_horizon` re-checks after selection.
- **Restoration algebra.** Eq. (7)/(8) match `restore_continuation`
  (`retrieval_faiss.py:46-52`) and its vectorised twin.
- **Backbone parameters.** `reports/phase11a/compute_backbone.json` gives
  205,292,928 → Table II/IV's 205.3M ✓, and 0 trainable ✓; TS-RAG ARM
  4,775,425 → 4.78M ✓.
- **Gate spec.** 200 trees × 15 leaves ✓; 2 gate-training origins
  (`scalerag_test_final.py:145-148`) × 30,490 series = 60,980 examples ✓.
- **Bootstrap.** `n_boot = 2000` ✓ (`scalerag.py:70`). Note the docstring says the
  resample is **over series**, while Table II says "window level"; on M5 with one
  window per series per origin these coincide, so this is wording, not a defect.

---

## Recommended order of repair

1. **B** — it is new science and changes what the ETTm2 section *means*, not just
   how it reads. Write it as a mechanism finding, not an erratum.
2. **A, C, E, F, G** — corrections to Eq. (4), Table II, `sec:search`, `sec:restore` and the two
   conflicting degenerate shares. All are text edits against evidence in hand.
3. **D** — restate Eq. (11) as the six features are actually computed, and say
   plainly that only two of them see the retrieved set.
4. **H** — write items 19, 24 and 9 in, and correct the closed count.
