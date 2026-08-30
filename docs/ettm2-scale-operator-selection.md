# ETTm2 scale-operator selection: Eq. (3) against Eq. (4)

**Date:** 2026-08-28. **Split:** ETTm2 **validation** only. **Script:**
`scripts/ettm2_znorm_selection_run.py`. **Artifacts:**
`reports/ettm2-znorm/ettm2-znorm-selection-val.json`, `reports/ettm2-znorm/ci.json`.
**Pre-registration:** `docs/znorm-preregistration.md` names this comparison as its
selection step.

This closes a gap the manuscript already flags: it proves both propositions about
Eq. (3), draws it in two figures and calls it the canonical form of Stage 1,
while stating that Eq. (3) *"was never run on that panel"*. No new operator was
introduced — `_fit_params` has implemented `znorm` since Phase 5.

## Result

Frozen Chronos-Bolt backbone: **0.10602** MSE. 80,199 windows, 7 channels,
k=20, w=0.25, paired bootstrap over windows, 2,000 resamples.

| scale rule | restored retrieval MSE | fused w=0.25 | vs backbone | 95% CI | sign-flipped pairs | fallback queries |
|---|---:|---:|---:|---|---:|---:|
| Eq. (4) `mean` — **deployed** | 0.23309 | 0.11084 | **−4.548%** | [−5.033, −4.077] | **26.05%** | 1,132 |
| `rms` (divisor only) | 0.27149 | 0.11171 | −5.361% | [−5.918, −4.807] | 0.00% | 0 |
| Eq. (3) `znorm` — **canonical** | **0.16091** | **0.10441** | **+1.522%** | [+1.253, +1.781] | 0.00% | 0 |

Eq. (3) over Eq. (4): **+5.806%** fused (CI [+5.364, +6.231]) and **+30.967%** on
the restored retrieval branch alone (CI [+29.640, +32.183]). Every interval
excludes zero.

On validation, Eq. (3) turns a **4.5% loss** to the frozen backbone into a
**1.5% win**.

## The mechanism is not the one the audit predicted

Audit finding B established that Eq. (4)'s divisor is the *signed* window mean,
negative for 65.4% of ETTm2 queries, so 26.05% of retrieved pairs are restored
through a negative ratio and emerge sign-inverted. That is confirmed exactly:
26.05% under `mean`, **0.00%** under both `rms` and `znorm`.

**But removing the sign pathology in isolation makes things worse.** `rms` has a
non-negative divisor and zero sign-flips, and it is the *worst* of the three
(−5.361% against `mean`'s −4.548%). The two rules that lack a location term both
lose to the backbone; the one that has it wins.

So the binding constraint is the **missing location term μ**, not the sign flip.
The sign flip is real, measurable and worth reporting — it is a defect in the
deployed operator — but it is not what costs the accuracy. Finding B should be
written as a *correctness* defect, not as the explanation of the deficit.

Three prior results now agree on the same cause:

- the affine probe: `mean` restoration carries no location term, so a pure offset
  degrades it 291x (`docs/affine-probe-report.md`);
- the error decomposition: **85.1%** of the restored branch's error is unrecovered
  *magnitude* (`docs/retrieval-forecasting-gap.md`);
- this run: adding μ recovers 31% of the retrieval branch's error.

Unrecovered magnitude *is* the missing location term. The three measurements were
independent and converge.

## The M5 contrast — the selection is regime-dependent, on a new axis

The same comparison on M5 validation was already recorded
(`docs/ablation-report.md`, `reports/scalerag-matrix-m5-1000.json`), and it runs
the **opposite** way:

| panel | data | Eq. (4) `mean` | Eq. (3) `znorm` | winner |
|---|---|---:|---:|---|
| M5 val (RMSSE) | non-negative counts, sparse | **0.7425** | 0.8719 | Eq. (4), by 17% |
| ETTm2 val (fused MSE) | z-normalised, zero-centred | 0.11084 | **0.10441** | Eq. (3) |

*(The M5 ablation row is labelled `no_seasonal(znorm)`, which is a misleading
name — the code is `("znorm", True, "cat_id")` against `("mean", True, "cat_id")`,
so only the scale varies and the comparison is clean. The label should be fixed
in the ablation report.)*

Both directions are what theory predicts. On a non-negative count panel the mean
is a stable positive quantity and never straddles zero, and subtracting a location
term from sparse counts manufactures negative "demand" that no candidate can
match. On a z-normalised panel the mean is near zero, signed, and carries no
information, while the level *is* the thing that must be restored.

**This reframes the regime story.** The manuscript attributes the ETTm2 loss to
dense continuous channels — a claim about *sparsity*. The evidence says the
operative axis is the panel's **sign structure**: divide by a signed mean only
when the data is non-negative. That is a sharper, more defensible and more
testable claim than the sparsity one, and it is consistent with the corpus check
in `docs/code-versus-text-audit.md`, where every comparator method (RAFT, RAID,
TS-RAG, kNN-MTS, TimeRAG, RevIN) divides by a non-negative magnitude or does not
divide at all.

## What this does NOT change

- **Phase-11A condition 3 still fails and the verdict is frozen.** It was decided
  on ETTm2 **test** under Eq. (4). ETTm2 test was consumed by that gate. The
  +1.522% is a **validation** number and can never be confirmed on ETTm2 —
  rescoring test after selecting a rule on validation is exactly what rules 2, 9
  and 12 forbid. The runner refuses `--split test` unconditionally.
- **No headline is rescued.** This is a diagnostic that explains a negative
  result, not a result that reverses it. It must be reported as such.
- **One panel, one backbone, one weight, one k.** The comparison is at the frozen
  grid point (k=20, w=0.25); k was deliberately not extended here so that referee
  item 24's question stays separate.

## Consequence for the pre-registration

`docs/znorm-preregistration.md` needs a dated addendum, not a rewrite (its own
terms permit editing before the confirmation run begins, but editing a
pre-registration is the wrong instinct).

1. **Premise 2 is retracted.** It cites C3 — "the analogues are 2.3x better than
   the backbone once optimally rescaled" — which reversed under symmetric
   correction; they are ~13% worse.
2. **Premise 1 is confirmed and is the operative one.** The affine probe's
   location-term finding is what this run validates.
3. **The confirmation plan has a problem.** The single clean split it names is
   **Favorita test, origin 972**. Favorita is a retail *count* panel like M5 —
   non-negative, sparse. Theory and the M5 result both predict Eq. (4) wins there
   and Eq. (3) loses. Confirming the hypothesis on the one panel where it is
   predicted to fail would test nothing. This needs a decision before any
   confirmation run: either reframe the hypothesis as sign-structure-conditional
   and confirm *that* (which Favorita can test — as the predicted-negative arm),
   or accept that the ETTm2 finding stays validation-only and unconfirmed.

## Reproduce

```fish
python scripts/ettm2_znorm_selection_run.py --resume
```

~2.5 minutes, CPU. The run aborts unless it reproduces the recorded Phase-11A
validation anchors for `mean`/k=20 to 1e-9.

---

## Corpus check and a rival explanation, eliminated (2026-08-28)

Queried against the 101-source corpus with the ScaleRAG documents excluded.

**Novelty of the three claims.** All three come back NOT IN SOURCES:

1. **No source ablates instance normalisation into its location and scale
   components separately.** The closest, arXiv:2603.11869 (Table 1), ablates
   RevIN's *learnable affine* parameters α, β and its backpropagation variants,
   but treats `(x − μ)/σ` as indivisible. The μ-versus-σ separation is ours.
2. **No source claims the level shift dominates the variance shift.** RevIN and
   related work name mean and variance as joint symptoms of distribution shift
   without ranking them.
3. **No source reports that the same method needs a different normalisation rule
   on a sparse non-negative panel than on a z-normalised one.** The
   regime-dependent selection is unclaimed.

**One independent corroboration, from a completely different method.** The
Chronos sparse-autoencoder dissection (arXiv:2603.10071, §4) reports that causal
feature importance in Chronos-T5 on ETT data is *"dominated by level shifts
(1,024 features) and noise (413)"*, concluding that *"detecting abrupt
distributional changes rather than periodic patterns is central"*. That is
mechanistic-interpretability evidence, on the same backbone family and dataset
family, pointing at the same thing this run measures behaviourally. Worth citing
as convergent, not as support for priority.

**Vocabulary worth adopting.** arXiv:2603.11869 App. B taxonomises normalisation
into Min-Max `(x−m)/(M−m)`, **Relative `x/μ`**, and Standardization `(x−μ)/σ`.
Our Eq. (4) is "Relative" and our Eq. (3) is "Standardization" in that
vocabulary; using it lets a reader place both rules immediately. The taxonomy
exists; what does not exist is any rule for choosing between them.

**Support for audit finding A.** RAID (App. D.3) already observes that *"Chronos
uses mean-scaling that divides by zero on empty or zero context"* — the same
degenerate-denominator hazard, noted in passing by someone else.

### The "match the backbone" hypothesis is FALSIFIED

NotebookLM answered that Chronos-2 applies *"per-series mean absolute scaling"*,
citing a third-party guide and RAID's remark. **That is wrong**, and it matters,
because it would have produced a tidy but false unifying story (M5's backbone
mean-scales so Eq. (4) matches it; ETTm2's z-normalises so Eq. (3) matches it).
Checked in the installed library instead:

| variant | internal normalisation | source |
|---|---|---|
| Chronos v1 | mean **absolute** scaling, `Σ|x|·mask / Σmask`, no location term | `chronos/chronos.py:177` |
| Chronos-Bolt | `loc = nanmean(x)`, `scale = sqrt(nanmean((x−loc)²))`, `(x−loc)/scale`, `eps=1e-5` substituted at exact zero | `chronos/chronos_bolt.py:110-117` |
| Chronos-2 | **the same class** — `from chronos.chronos_bolt import InstanceNorm` | `chronos/chronos2/model.py:26,254` |

Only Chronos **v1** mean-scales; the guide and RAID were both describing v1's
tokenizer. **Chronos-Bolt and Chronos-2 both apply exactly Eq. (3).**

So on *both* panels the retrieval branch (Eq. (4)) was normalised differently
from the frozen backbone it was fused with (Eq. (3)) — and yet Eq. (4) wins on
M5 and loses on ETTm2. **Backbone-matching cannot be the rule.** That eliminates
the most obvious rival explanation a referee would raise ("you simply mismatched
the backbone's own normalisation"), and it does so with measurements already in
hand. The data-property explanation survives; this one does not.

*(Incidental: Chronos's own guard is `where(scale == 0, eps, scale)` — the same
substitute-at-exact-zero pattern as `_fit_params`, not the additive `σ + ε` the
manuscript declares. Audit finding A's ε defect mirrors what the backbone does.)*
