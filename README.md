# ScaleRAG

**Scale-Aware Retrieval Augmentation for Time-Series Foundation Models.**

A frozen time-series foundation model is augmented with a retrieval branch that
adds **zero trainable parameters**. Nearest neighbours are matched in a
normalized shape space, their continuations are restored to the query's own
scale in closed form, and the result is blended with the backbone's forecast.

This repository is the code and the full result set behind a study that asks a
narrower question than most retrieval papers: **when does retrieval actually
help a frozen backbone, and when does it not?** The answer is reported here in
full, including the parts that did not work.

> **Headline, stated honestly.** On the held-out M5 panel ScaleRAG beats its
> frozen Chronos-2 backbone by **5.49% RMSSE**, exceeds a tuned LightGBM by only
> **0.69%**, **loses** the competition's own WRMSSE metric, and meets **none of
> three pre-registered success criteria**. On dense ETTm2 it is **0.85% worse**
> than the backbone it augments. The mechanism works — retrieval quality
> improves by 85.4% — but that gain does not become a forecasting gain outside a
> narrow regime. Locating why is the contribution.

---

## Contents

- [The problem: catastrophic scale mismatch](#the-problem-catastrophic-scale-mismatch)
- [Method](#method)
- [Install](#install)
- [Quickstart — runs offline, no dataset needed](#quickstart--runs-offline-no-dataset-needed)
- [Results](#results)
- [Diagnostics](#diagnostics)
- [Cost](#cost)
- [Reproducing every number](#reproducing-every-number)
- [Repository layout](#repository-layout)
- [Protocol](#protocol)
- [Manuscript](#manuscript)
- [Citation](#citation)

---

## The problem: catastrophic scale mismatch

A foundation model sees only the context window handed to it at inference time.
Anything older — last quarter's promotion, last year's seasonal shape — is
unreachable. Retrieval is the natural remedy, and nearest-neighbour retrieval on
raw values needs no training at all. It also fails, for one identifiable reason.

Real series differ in magnitude by orders of magnitude, so a Euclidean index over
raw windows ranks candidates **by scale before it ranks them by shape**. Worse,
even after the index is repaired by normalizing the context, the retrieved
continuation is still expressed at the magnitude of the series it came from.

<p align="center">
  <img src="figures/scale-mismatch.png" width="62%" alt="Scale mismatch on a real ETTm2 query">
</p>

Panel (a) is a real ETTm2 query and its top-1 Euclidean neighbour. The candidate
tracks the query's future shape almost exactly — and sits at **0.54×** the
query's RMS, so using it as a forecast injects a systematic level error. Panel
(b) is the same continuation after restoration.

The repair is arithmetic the pipeline has already done. With the query's context
statistics `(μ_q, σ_q)`, mapping a normalized continuation back through
`y·σ_q + μ_q` places it on the query's scale exactly — two scalars, no
parameters.

**We claim no novelty for the restoration itself.** Removing and restoring a
level is standard preprocessing, RAFT already restores a location offset, and
RAID normalizes a retrieved trajectory by its own moments before mapping it onto
a target scale. What is unclaimed elsewhere is the composition, and what this
work adds is the measurement of *where it suffices*. See
[docs/literature-verification.md](docs/literature-verification.md) for the
prior-art checks behind that statement.

---

## Method

Three deterministic stages sit between a query and a frozen backbone, followed by
one fusion step. **Stages 1–3 contribute no trainable parameters.** The backbone
is never updated.

<p align="center">
  <img src="figures/pipeline.png" width="100%" alt="ScaleRAG pipeline">
</p>

```
(x*, μ_q, σ_q) = N(x)                          # Stage 1  normalize
J              = R_k(x*; P*),  |J| = k         # Stage 2  exact k-NN
y_restored     = S({y*_j}, μ_q, σ_q)           # Stage 3  restore
ŷ_TSFM         = B(x)                          # frozen backbone
ŷ              = (1−w)·ŷ_TSFM + w·y_restored   # Stage 4  fuse
```

Every symbol is observable at the forecast origin. Nothing on the right-hand
side depends on the realised future.

### Stage 1 — Scale-Invariant Context Normalization

<p align="center">
  <img src="figures/stage1-normalization.png" width="100%" alt="Stage 1: normalization">
</p>

The raw window is split into a **shape code**, which is all the index is allowed
to see, and a **two-scalar scale tag**, which the index never sees and Stage 3
re-applies. Applying the identical closed form to the query and to every pooled
candidate is what turns the retrieval metric into a shape metric.

The full rule subtracts the mean and divides by RMS deviation:

```
x* = (x − μ) / (σ_RMS + ε)          ε = 1e-8
```

This is exactly invariant to any positive affine map `x ↦ ax + b`. **Both
deployed configurations in this study instead use the cheaper scale-only
variant** `x* = x / (x̄ + ε)`, which is invariant to `x ↦ ax` but **not** to a
location shift. On M5 that was chosen against measured alternatives (the full
rule scored 0.8719 RMSSE, an RMS-only divisor 0.7951, against 0.7425). Carrying
it onto near zero-centred ETTm2 was a mistake, and
[Diagnostics](#diagnostics) quantifies what it cost.

### Stage 2 — Exact shape-space retrieval

```
J = argmin Σ ‖q* − c*_i‖²        subject to    t_i + H < τ
```

**The search is exact everywhere.** ETTm2 uses FAISS `IndexFlatL2`; the
30,490-series M5 panel uses a batched GPU implementation of the same exact
search, verified **bit-for-bit identical** to the CPU retriever (maximum absolute
difference `0.0`) before any held-out run — see
[`scripts/verify_gpu_retrieval.py`](scripts/verify_gpu_retrieval.py). No
approximate index such as HNSW is used anywhere, so no reported difference is
attributable to index quantisation.

The constraint `t_i + H < τ` is the **horizon guard**, enforced at pool
construction rather than at query time. A retrieved continuation cannot overlap
the window it predicts. `tests/leakage/` fails if it is violated.

### Stage 3 — Closed-form scale restoration

<p align="center">
  <img src="figures/stage3-restoration.png" width="100%" alt="Stage 3: restoration and the invariance probe">
</p>

```
y_restored = ȳ* · σ_q + μ_q          # 2 FLOPs per horizon step, 0 parameters
```

Because the operator is affine, averaging the `k` neighbours before or after
restoration is identical — which is why the neighbour budget and the restoration
rule can be ablated independently. A neural adapter reaches the same alignment by
learning a projection from a pretraining corpus; here it costs two scalars.

### Stage 4 — Diagnostic-gated fusion

<p align="center">
  <img src="figures/stage4-gated-fusion.png" width="100%" alt="Stage 4: gated fusion">
</p>

On dense data `w` is a constant chosen on validation (`w = 0.25`) and the whole
pipeline stays parameter-free. On sparse intermittent panels the value of
retrieval varies sharply across series, so `w = g(φ)` is predicted per series and
per origin by a gradient-boosted regressor over six origin-time diagnostics:
nearest-neighbour distance, retrieval disagreement, intermittency, log-volume,
backbone predictive spread, and neighbour scale spread.

The gate is fitted by least squares onto the per-origin oracle weight, which has
a closed-form solution, so fitting needs no inner optimisation loop. **The gate
is the only trained component anywhere in the system**, it is trained on
historical origins only, and it is counted as trainable rather than described
away.

---

## Install

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/). A CUDA GPU is
optional — everything except the backbone runs on CPU.

```bash
git clone https://github.com/Nidan73/ScaleRAG.git
cd ScaleRAG
uv sync --extra ml --extra retrieval --extra tsfm --extra viz
```

`torch` resolves from the CUDA 13.0 index pinned in `pyproject.toml`, matching
the RTX 5070 Ti (Blackwell, sm_120) this study ran on. Change the index in
`[[tool.uv.index]]` for a different accelerator, or drop `--extra ml` for a
CPU-only install.

---

## Quickstart — runs offline, no dataset needed

M5, Favorita and ETTm2 are all externally licensed and **none of them is
redistributed here**. The pipeline ships a deterministic synthetic fixture so the
whole path is runnable without downloading anything:

```bash
# 1) build + ingest the offline synthetic M5 fixture (idempotent)
uv run python scripts/make_synthetic.py --days 1941 \
      --raw data/raw_synth --processed data/processed

# 2) verify chronological-split integrity before anything else runs
uv run python scripts/leakage_audit.py --spec configs/split_check_val.json

# 3) classical baselines — declare first, then run
uv run python scripts/baseline_run.py --config configs/baseline_seasonal_naive.yaml --dry-run
uv run python scripts/baseline_run.py --config configs/baseline_seasonal_naive.yaml --confirm

# 4) the fast gate: format + lint + types + unit + leakage
make check
```

Real data uses the identical code path — point a config's `processed_dir` at the
ingested panel. Download links for all three datasets are in
[docs/method.md](docs/method.md).

```python
from scalerag.retrieval import WindowDatabase
from scalerag.native import NativeScaleRetriever, fixed_fusion
```

---

## Results

### Ablation — restoration is what makes retrieval usable

<p align="center">
  <img src="figures/ablation-restoration.png" width="58%" alt="Restoration ablation">
</p>

Removing restoration from an otherwise identical retriever destroys it. Retriever,
index, normalization and neighbour budget are unchanged across each row, so the
difference is attributable to the restoration step alone.

| Dataset | Metric ↓ | Raw | Restored | Improvement |
|---|---|---:|---:|---|
| M5 (1k, validation) | RMSSE | 2.7884 | **0.7425** | 73.4% (3.8×) |
| ETTm2 (test, n=80,199) | MSE | 2.9989 | **0.4376** | 85.4% &nbsp;`CI [85.2, 85.6]` |

All **7 of 7** ETTm2 channels improve, so no single series carries the aggregate.

### Held-out M5 — the honest table

`d_1914`–`d_1941`, all 30,490 series, **single locked run**. The split was frozen
before any result was seen and opened exactly once.

| Method | RMSSE ↓ | WRMSSE ↓ | MASE ↓ | WAPE ↓ | Cov.80 |
|---|---:|---:|---:|---:|---:|
| Seasonal-Naive | 0.9972 | 0.8697 | 1.211 | 0.862 | 0.458 |
| Recent-mean | 0.7669 | 1.0876 | 1.074 | 0.745 | n/a |
| LightGBM | 0.7665 | **0.8663** | 1.077 | 0.740 | n/a |
| Chronos-2 (frozen) | 0.8054 | 1.9395 | **0.893** | **0.665** | **0.786** |
| **ScaleRAG (gated)** | **0.7612** | 1.2231 | 1.020 | 0.698 | 0.698 |

<p align="center">
  <img src="figures/m5-metric-reordering.png" width="100%" alt="The ranking reorders with the metric">
</p>

**ScaleRAG leads on RMSSE and loses every other metric in the table.** That
reordering, not a leaderboard position, is the empirical result:

- vs. frozen Chronos-2: **+5.49% RMSSE**, 95% CI `[5.40, 5.59]` — real, excludes zero.
- vs. tuned LightGBM: **+0.69%**, CI `[0.57, 0.82]` — significant at 30,490 series, and operationally negligible.
- On **WRMSSE**, the dollar-weighted metric that defines M5, even Seasonal-Naive (0.8697) beats us (1.2231).
- The backbone is better on MASE, WAPE, MAE (0.960 vs 1.008), pinball (0.289 vs 0.298), and calibration — it attains 0.786 coverage at a nominal 80% where the gated fusion under-covers at 0.698.

#### Where the gain comes from

Every row below is from that same locked run.

| Configuration | RMSSE ↓ | Δ vs previous | Δ vs backbone |
|---|---:|---:|---:|
| Seasonal-Naive (floor) | 0.9972 | — | −23.8% |
| + frozen backbone only | 0.8054 | −0.1918 | — |
| + stages 1–3, no blend | 0.7795 | −0.0259 | +3.22% |
| + stage 4, fixed `w = 0.5` | 0.7692 | −0.0103 | +4.49% |
| + stage 4 gate (full ScaleRAG) | **0.7612** | −0.0080 | **+5.49%** |
| *reference:* tuned LightGBM | 0.7665 | — | +4.83% |

Roughly **four fifths of the retrieval gain is already available from the
parameter-free stages under a constant weight**; the learned gate is worth the
remaining ~1.0 percentage point. That is a weaker case for the learned component
than a comparison against the backbone alone would suggest, and it is reported
here rather than left out.

### Pre-registered criteria — 0 of 3 met

Fixed before the split was opened, reported unchanged.

| # | Criterion | Observed | Met |
|---|---|---|:--:|
| 1 | RMSSE improvement over tuned LightGBM ≥ 3% | 0.69% `CI [0.57, 0.82]` | ❌ |
| 2 | Competitive on the official WRMSSE metric | 1.2231 vs 0.8663 | ❌ |
| 3 | No regression on absolute and probabilistic accuracy | worse on MASE, WAPE, MAE, pinball, coverage | ❌ |

We report this instead of re-tuning. The split was consumed exactly once by
design, and any post-hoc adjustment would invalidate it.

### ETTm2 — the dense regime, where the method loses

n = 80,199 windows. ΔMSE is relative to the frozen target-only backbone.

| Method | MSE ↓ | MAE ↓ | ΔMSE | Trainable |
|---|---:|---:|---:|---:|
| Chronos-Bolt (frozen, target-only) | 0.1486 | 0.2235 | — | 0 |
| TS-RAG (official ARM) | **0.1465** | **0.2230** | +1.42% | 4.78M |
| ScaleRAG (restored, fixed fusion) | 0.1498 | 0.2407 | **−0.85%** | **0** |

The degradation is small but statistically significant, CI `[0.28, 1.39]`. The
official TS-RAG adapter beats our fusion by a further **2.30%** MSE, at the cost
of 4.78M trainable parameters and a pretraining corpus. Our TS-RAG reproduction
matches its published numbers to within **0.10%**, which is the condition under
which we treat it as a fair external comparison.

On Favorita, the third panel, the gain over the backbone is **+0.83%**.

---

## Diagnostics

Three analyses explain the gap between an 85.4% retrieval gain and a 0.85%
end-to-end loss. This is the part of the study that generalises.

### 1. Retrieval utility is not monotone in sparsity

<p align="center">
  <img src="figures/regime-band.png" width="58%" alt="Inverted-U regime profile">
</p>

Read at dataset level the relationship looks monotone — a 0.85% loss on
continuous ETTm2, 0.83% on Favorita, 0.13% on the dense M5 slice *(not
significant, CI contains zero)*, 5.08% on the full M5 validation panel, 5.63% on
its most intermittent slice.

**That reading is wrong at the sparse end.** Measured per series over ~1,000
series at each of **50 non-overlapping forecast origins**, the win rate forms an
inverted U:

| Context zero fraction | Win rate (mean ± SEM over 50 origins) |
|---|---:|
| 0.0–0.1 (densest) | 0.220 ± 0.009 |
| 0.6–0.7 (peak) | **0.794 ± 0.006** |
| 0.9–1.0 (sparsest) | 0.472 ± 0.007 |

The peak clears both extremes with **non-overlapping 95% intervals**, so this is
not a sampling artefact. Refitting an isotonic crossing at each origin separately
places the useful band at a zero fraction of **[0.33 ± 0.05, 0.95 ± 0.02]** —
the lower edge found at all 50 origins, the upper at 48.

Retrieval fails at both ends for different reasons: a dense series leaves the
backbone little to gain from an analogue, while a near-empty one offers too
little structure for any analogue to match. **Dataset-level averages and a linear
coefficient cannot express this shape**, which is the likeliest reason it has not
been reported before.

> **For a practitioner this band is the operative result, not the method.** If a
> panel sits outside a context zero fraction of roughly [0.33, 0.95], non-neural
> retrieval augmentation of a frozen backbone is unlikely to repay its index
> cost — and that can be decided from one statistic per series, before any model
> is built.

### 2. The residual error is magnitude, not shape

<p align="center">
  <img src="figures/error-decomposition.png" width="100%" alt="Oracle affine decomposition">
</p>

Fitting the per-window least-squares affine correction to the realised future
gives a **shape floor** — a lower bound, so whatever survives it is the analogue
genuinely having the wrong shape.

- Shape floor **0.0647** against a backbone at **0.1486**: optimally rescaled, the retrieved continuations would be **2.3× more accurate than the model they augment**.
- Restoration removes **2.56** of the **2.93** MSE of scale error, but the remaining **0.373** is 5.8× the shape floor and 2.5× the backbone's entire error.
- The split: **14.8% shape, 85.2% unrecovered magnitude.**

A practitioner reading only an aggregate MSE would conclude retrieval does not
work on ETTm2. Reading the decomposition, the correct conclusion is that
**retrieval works well and the two-scalar correction is the component that does
not**. Those two diagnoses imply opposite next experiments.

### 3. Invariance and equivariance, measured rather than asserted

<p align="center">
  <img src="figures/stage3-restoration.png" width="100%" alt="Invariance probe">
</p>

Normalization is routinely *described* as making retrieval scale-invariant. Under
a controlled affine transform `x' = ax + b` on synthetic queries with known
ground-truth matches:

| Retrieval space | Top-1 hit rate |
|---|---:|
| Normalized (full rule) | **1.000** |
| Raw | 0.176 |
| *chance* | *0.083* |

The invariance is not approximate and not statistical — it is an arithmetic
identity confirmed numerically, and restoration is **exactly equivariant**
(error constant to 8.7e-19 across the grid).

**The probe also bounds the claim, which is why it is diagnostic rather than
promotional.** The scale-only variant that both deployed configurations actually
use passes the multiplicative test and **fails the additive one** — a 291×
degradation in reconstruction error from an offset alone. That is precisely the
property whose absence costs 85.2% on near-zero-centred ETTm2. The probe
*predicts* the ETTm2 failure rather than rationalising it after the fact.

The implied repair — a location-aware scale rule — is
[**pre-registered and deliberately not executed**](docs/znorm-preregistration.md).
Choosing a scale rule after inspecting this attribution would be tuning on
evaluation data.

---

## Cost

Measured on one machine, batch 256, warm-up excluded.

| Component | Trainable | Latency / window | Storage |
|---|---:|---:|---:|
| Chronos-Bolt (frozen) | 0 (205.3M) | 0.431 ms | — |
| TS-RAG ARM (official) | 4.78M | 0.450 ms | 1,528.6 MB |
| ScaleRAG retrieval | **0** | 0.607 ms (CPU) | 1,096.2 MB |
| ScaleRAG total | **0** | 1.038 ms | 1,096.2 MB |

<p align="center">
  <img src="figures/pareto-ettm2.png" width="58%" alt="Accuracy against cost on ETTm2">
</p>

**On ETTm2 we are Pareto-dominated** — slower than the official neural adapter
*and* less accurate. The exact search that removes confounds from every ablation
is also what sets this cost floor. The configuration is attractive where a neural
adapter cannot be trained at all, not where it can.

---

## Reproducing every number

Each script regenerates a specific artifact. All write machine-readable JSON.

| Script | Reproduces |
|---|---|
| `scripts/make_synthetic.py` | offline M5 fixture (no download) |
| `scripts/leakage_audit.py` | chronological-split integrity report |
| `scripts/baseline_run.py` | Seasonal-Naive / LightGBM baselines |
| `scripts/retrieval_eval.py` | Phase-5 retrieval design sweep |
| `scripts/scalerag_matrix.py` | full M5 ablation matrix → `docs/ablation-report.md` |
| `scripts/scalerag_eval.py` | M5 gated-fusion evaluation |
| `scripts/scalerag_favorita.py` | Favorita cross-dataset check |
| `scripts/scalerag_test_final.py` | the locked held-out M5 run *(refuses while the consumption lock exists)* |
| `scripts/regime_threshold_run.py` | the 50-origin inverted U → `docs/regime-threshold-report.md` |
| `scripts/affine_probe_run.py` | invariance/equivariance probe → `docs/affine-probe-report.md` |
| `scripts/error_decomposition_run.py` | shape/magnitude split → `docs/retrieval-forecasting-gap.md` |
| `scripts/gate_transfer_run.py` | cross-dataset gate transfer → `docs/gate-transfer-report.md` |
| `scripts/scalerag_native_ettm2.py` | ETTm2 adapter → `docs/scalerag-native-dev-report.md` |
| `scripts/verify_gpu_retrieval.py` | GPU ≡ CPU retrieval, bit-for-bit |
| `scripts/make_decomposition_figure.py` | `figures/error-decomposition.png`, straight from the run record |
| `scripts/make_phase11a_figures.py` | the Phase-11A figure set |

**On the figures.** Every *number* in every figure above is regenerated by the
scripts in this table, which write the JSON in `docs/`. The figure *files* are
a mix:

- `figures/error-decomposition.png` is fully reproducible here — run
  `uv run python scripts/make_decomposition_figure.py`. It reads
  `docs/error-decomposition-ettm2-test.json`, hardcodes nothing, and refuses to
  render if the waterfall does not close to within `1e-6`.
- The four schematics (`pipeline`, `stage1-normalization`, `stage3-restoration`,
  `stage4-gated-fusion`, `gap-map`) are drawio diagrams whose sources are not in
  this repository.
- The remaining plotted figures were rendered by a manuscript-side script.
  `scripts/make_phase11a_figures.py` produces an earlier, overlapping set
  (`fig1_motivation`, `fig2_ablation`, `fig3_qualitative`, `fig4_regimes`,
  `fig5_pareto`, `fig6_sensitivity`) rather than these exact files.

Treat the rest of `figures/` as published artifacts whose underlying values are
reproducible, not as build outputs of this tree.

**Curated results** live in `docs/` and are tracked. `reports/` is regenerable
and gitignored. The tables above are machine-readable in
[`docs/scalerag-heldout-test-tables.json`](docs/scalerag-heldout-test-tables.json)
and [`docs/scalerag-native-dev-results.json`](docs/scalerag-native-dev-results.json).

---

## Repository layout

```
src/scalerag/
  retrieval.py  retrieval_faiss.py  retrieval_gpu.py   exact k-NN, three backends
  retrieval_forecast.py                                continuations → forecast
  fusion.py                                            convex blend + paired bootstrap
  native.py                                            dense-panel adapter (ETTm2)
  tsfm/chronos2.py                                     frozen backbone wrapper
  affine_probe.py  error_decomposition.py  regime.py   the three diagnostics
  leakage.py  splits.py                                horizon guard, chronological splits
  metrics.py  hierarchy.py                             RMSSE / WRMSSE / MASE / WAPE / pinball
  features.py  baselines/                              LightGBM + Seasonal-Naive
  data/                                                M5 schema, ingest, synthetic fixture
  config.py  reproducibility.py  eval.py  cli.py       run records, seeding, config schema

tests/
  unit/        11 files    component correctness
  leakage/      7 files    temporal-integrity guards — first-class, not an afterthought
  integration/  1 file     end-to-end baseline evaluation

scripts/   23 entry points, one per reproducible artifact
docs/      curated reports + machine-readable result tables
figures/   every figure in this README
paper/     the manuscript — main.tex, references.bib, figures/
configs/   YAML run definitions, schema-validated with extra="forbid"
```

Every splitting, scaling, retrieval and feature component has a leakage test that
**fails when a temporal violation is deliberately introduced** — asserting the
violation is caught, not merely that normal input passes.

```bash
make check     # format + lint + types + unit + leakage
make test      # everything
```

---

## Protocol

The full set is in [docs/research-rules.md](docs/research-rules.md). The ones
that shaped the results above:

- **Chronological splits only.** Never random, never shuffled.
- **Horizon guard** `candidate_end + H < target_forecast_origin`, enforced at pool construction.
- **Everything fitted is fitted on history only** — scalers, indices, utility labels, the gate.
- **No single-run headline numbers.** Paired-bootstrap CIs over series, 2,000 resamples.
- **The M5 test split was opened exactly once** and is not re-evaluated. The harness refuses while the consumption lock exists.
- **Pre-registered criteria are not edited after the fact**, including when none is met.
- **Negative results are preserved.** The relation-aware graph-routing hypothesis this project started from was falsified across two datasets with non-learned routers, learned routers, controls and confidence intervals. It is reported as a negative rather than quietly dropped.

Known limitations are catalogued in
[docs/threats-to-validity.md](docs/threats-to-validity.md) — including that the
regime profile covers 50 origins on one panel, that the held-out comparison
resamples at a single evaluation origin, and that the scale rule carried onto
ETTm2 was the wrong one.

---

## Manuscript

The full write-up lives in [`paper/`](paper/) — `main.tex`, `references.bib` and
the 11 figures it references, self-contained and buildable with:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

See [`paper/README.md`](paper/README.md) for the build verification, the package
requirements, and a per-figure table of what is and is not reproducible from
this repository.

## Citation

The manuscript is under peer review, so author and venue details are withheld
until notification.

```bibtex
@article{scalerag2026,
  title  = {ScaleRAG: When Does Retrieval Help a Time Series Foundation Model?
            Scale Mismatch and the Limits of Non-Neural Retrieval Augmentation},
  year   = {2026},
  note   = {Under review. Code: https://github.com/Nidan73/ScaleRAG}
}
```

## License

[MIT](LICENSE).

Datasets are **not** redistributed here and remain under their own terms: M5 and
Favorita via their Kaggle competitions, ETTm2 via the ETDataset repository.
Backbone checkpoints are the public Chronos releases from Amazon Science and are
referenced, never vendored.
