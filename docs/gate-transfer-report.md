# Does the Fusion Gate Transfer Between Datasets?

**Why this matters.** TS-RAG trains its Adaptive Retrieval Mixer on a multi-domain
corpus and applies it zero-shot to unseen benchmarks, so cross-dataset transfer is
**table stakes** in this literature, not a differentiator. ScaleRAG's gate had only
ever been fitted and evaluated on one dataset at a time, leaving the cell empty.
This fills it: train the gate on M5, apply it unchanged to Favorita, and the reverse.

**Method.** Four arms per evaluation dataset, sharing one backbone and one retrieval
run so the arms differ only in the fusion rule:

| Arm | Fusion weight |
|---|---|
| `chronos_only` | 0 (frozen backbone alone) |
| `fixed_0.5` | 0.5, untrained — the floor any gate must clear |
| `gate_in_domain` | learned on this dataset's own historical origins |
| `gate_transfer` | learned on the *other* dataset, applied unchanged |

Gates are fitted on historical origins only (rule 5) and evaluated on the validation
window. The consumed M5 test split is never read, and the script refuses to run if
the eval origin reaches it (rule 2). Nothing is selected from the outcome (rules
9, 12). 1,000 series each, RMSSE, paired bootstrap. 75 s.

- Code: `scripts/gate_transfer_run.py` (reuses frozen retrieval and gate features
  from `scripts/scalerag_eval.py`)
- Results: `reports/gate-transfer/gate-transfer.json`

## Results

| Arm | M5 (gate from Favorita) | Favorita (gate from M5) |
|---|---|---|
| `chronos_only` | 0.75396 | **0.61689** |
| `retrieval_only` | 0.74246 | 0.67055 |
| `fixed_0.5` | 0.72516 | 0.62342 |
| `gate_in_domain` | **0.71731** | **0.61162** |
| `gate_transfer` | 0.71827 | 0.61939 |

| Contrast | M5 | Favorita |
|---|---|---|
| transfer vs in-domain | −0.13% **CI [−0.39, +0.12] — includes 0** | −1.27% **CI [−2.01, −0.61] — excludes 0** |
| transfer vs `fixed_0.5` | +0.95% CI [+0.62, +1.31] | +0.65% CI [+0.24, +1.04] |

## The gate transfers, asymmetrically

**Favorita → M5 is free.** A gate that never saw M5 performs indistinguishably from
one fitted on it (−0.13%, CI spans zero). **M5 → Favorita costs 1.27%** and the CI
excludes zero. Both directions comfortably beat untrained fusion, so a transferred
gate is always better than no gate.

The asymmetry has a plausible reading. M5 is the regime where retrieval genuinely
pays, so an M5-trained gate learns to trust retrieval; carried to Favorita — where
retrieval is a *liability* — that learned optimism costs something. The reverse
direction is safe because a Favorita-trained gate is already sceptical, and
scepticism is merely suboptimal on M5 rather than harmful.

**On Favorita, retrieval is a net liability and only the gate rescues it.**
`retrieval_only` (0.67055) is much worse than the backbone alone (0.61689), and
`fixed_0.5` (0.62342) is *also* worse than the backbone. Only the learned gate gets
below it (0.61162). This sharpens the regime story: on dense data the gate's job is
not to blend two useful signals but to suppress a harmful one.

This is consistent with the finding in `docs/regime-threshold-report.md` that the six
gate features are near-collinear along a single demand-level axis. A gate that is
effectively a one-dimensional monotone rule on demand level is exactly the kind of
thing that ports across datasets — which is what the near-free Favorita → M5
direction shows.

## Incidental finding: the "3 gate seeds" carry no dispersion

Seed standard deviation is **exactly 0.0** in every arm. The frozen gate is
`LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
min_child_samples=50)` with `subsample=1.0` and `colsample_bytree=1.0` — no bagging,
no feature subsampling — so it is fully deterministic and `random_state` is inert.
Verified directly on synthetic data: seeds 42, 43 and 44 give bit-identical
predictions.

Phase 9 and Phase 10 report results "averaged over 3 gate seeds". That averaging is
real but vacuous: it is an average of three identical numbers and provides **no
evidence of robustness to gate initialisation**. It does not invalidate any recorded
result — the paired-bootstrap CIs over series are the actual uncertainty estimate and
are unaffected — but the seed-averaging should not be cited as dispersion. Genuine
gate-level dispersion would need bagging enabled, or resampling of the gate's
training origins.

## Limitations

Single validation origin per dataset, 1,000 series each, one direction of
metadata mapping (Favorita `family_id` → `cat_id`, `class_id` → `dept_id`, following
`scripts/scalerag_favorita.py`). The two panels also differ in horizon coverage and
sparsity, so "transfer" here confounds domain shift with regime shift; separating
them would need a third dataset in the same regime as one of these.

## Reproduce

```bash
uv run python scripts/gate_transfer_run.py --subset 1000 --favorita-subset 1000
```
