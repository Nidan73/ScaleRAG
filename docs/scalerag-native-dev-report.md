# ScaleRAG-TS Native-Protocol Development Report (Phase 11A · Parts 3–5)

**Date:** 2026-07-24 · **Dataset:** ETTm2 (development only) · **Context 512 / Horizon 64**
**Backbone:** frozen `amazon/chronos-bolt-base` · **Metric space:** train-only-`StandardScaler` normalised

> **Scope guard.** Everything here is on **ETTm2 only** (val = selection, test = mechanism
> checks). The four Phase-11B datasets (ETTh1, ETTm1, Weather, Electricity) were **not opened**.

## 0. Headline verdict

ScaleRAG's **scale-restoration mechanism is strongly validated** (restored retrieval beats
non-restored retrieval by **+85.4%** MSE, CI excludes zero, 7/7 series), **but the retrieval
signal is far weaker than the frozen Chronos-Bolt backbone on ETTm2**, so the restored fixed
fusion **cannot beat target-only Chronos-Bolt** (aggregate **−0.85%** MSE, CI [−1.39, −0.28],
significantly *worse*) and **loses to official TS-RAG** (**−2.30%** MSE). **Decision-gate
condition 3 fails → do NOT proceed to Phase 11B.** This mirrors the frozen M5-vs-Favorita
finding: the mechanism helps in sparse/scale-heterogeneous regimes, not in dense continuous
channels like ETT.

| # | Decision-gate condition | Result |
|---|---|---|
| 1 | Official TS-RAG reproduced within justified tolerance | **MET** (≤0.10%, Part 2) |
| 2 | Restored retrieval **significantly** beats non-restored on ETTm2 | **MET** (+85.4%, CI [+85.2, +85.6]) |
| 3 | Restored fixed fusion **improves** target-only Chronos-Bolt on MSE | **FAILED** (−0.85%, CI [−1.39, −0.28]) |
| 4 | Passes all leakage + numerical-equivalence tests | **MET** (13/13; FAISS≡NumPy) |
| 5 | Single config frozen before opening final datasets | **MET** (frozen on val) |

**Gate requires all five ⇒ NOT PASSED (4/5).** Phase 11B remains closed.

---

## 1. Part 3 — Native-protocol adapter (`src/graphroute_ts/scalerag_native.py`)

The adapter reuses ScaleRAG's **frozen scale math** — it imports `_fit_params`, `_transform`,
and the restoration formula from `retrieval_faiss` (no new neural retriever). Per variable it
builds an exact scale-aware k-NN over a **strictly train-only** candidate pool.

- **Restoration** (candidate→query): `restored = (cont − c_loc)/c_scale · q_scale + q_loc`,
  vectorised over the `(queries, k, H)` batch; bit-equivalent to `restore_continuation`.
- **Retrieval:** deterministic exact top-k L2 in transformed space (single BLAS GEMM,
  ascending-index tie-break). A FAISS `IndexFlatL2` path is verified **≡ the NumPy reference**
  (`test_faiss_equivalence_numpy_reference`, faiss 1.14.3, agreement > 0.999 with the only
  disagreements being exact-distance ties).
- **Leakage posture (rule 3 / rule 5):** candidate continuations satisfy `t_r + H ≤ 34560`
  (train end); each query keeps only candidates legal for the earliest origin in its batch
  (`legal_mask(min origin)`), raising `LeakageViolation` if none remain. Five leakage tests
  assert future-touching candidates are *excluded* (not merely that clean input passes).
- **Invalid-scale accounting:** near-zero scale denominators (common for `mean` scaling on
  zero-centred normalised data) are counted and served a constant context-mean fallback;
  candidate restorations with invalid candidate scale are counted and kept raw.

**Fairness caveat (important).** TS-RAG's KB (`ETTm2_minute_512.pkl`) is built from the
**full** ETTm2 series; the ScaleRAG adapter is deliberately restricted to the **train-only**
subset of the identical (context-512, continuation-64) pairs. ScaleRAG is therefore held to a
*stricter* leakage standard than TS-RAG. This can only disadvantage ScaleRAG and is reported
as such — never used to inflate it.

## 2. Part 4 — Frozen configuration (selected on the ETTm2 **validation** split)

Pre-registered grid: scale {mean, rms} × k {5, 10, 20} × weight {0.25, 0.50, 0.75}.
**Rule:** lowest val MSE among configs whose val MAE is within 2% of the grid-best val MAE.

**Frozen config** → `docs/scalerag-native-frozen-config.json`:

| field | value |
|---|---|
| method | `scalerag_restored_fixed_fusion` |
| scale strategy | **mean** |
| top-k | **20** |
| fusion weight | **0.25** |
| val MSE / MAE | 0.11102 / 0.22319 |
| MAE regression vs grid-best | **0.0%** (it *is* the grid-best MAE config) |
| learned gate | none |

Every fusion config on val had **higher** MSE than target-only Chronos (0.10602); the
selection minimises the damage rather than finding a winner — an early warning that
condition 3 would fail.

## 3. Part 5 — Mechanism checks (ETTm2 **test** split, 80 199 windows)

Per-window MSE loss, paired bootstrap (2 000 resamples, window-level; see §5 caveat). Frozen
config mean/k20/w0.25.

| Comparison | baseline MSE | method MSE | rel. MSE Δ | 95% CI | excl 0 | per-series |
|---|---|---|---|---|---|---|
| **A** restored vs raw retrieval | 2.99892 | 0.43763 | **+85.41%** | [+85.18, +85.64] | yes | **7/7** win |
| **B** fusion vs restored-only | 0.43763 | 0.14982 | +65.77% | [+65.32, +66.20] | yes | — |
| **C** fusion vs target-only Chronos | 0.14856 | 0.14982 | **−0.85%** | [−1.39, −0.28] | yes | 5/7 win |
| **D** fusion vs official TS-RAG | 0.14646 | 0.14982 | **−2.30%** | [−2.86, −1.70] | yes | 2/7 win |

MAE tells the same story: fusion MAE 0.24985 vs target 0.22354 vs TS-RAG 0.22302 (fusion is
worse on MAE too — the weight-0.25 blend still drags the sharp backbone toward the noisier
retrieval mean).

**Honest aggregation note.** Fusion beats the backbone on **5 of 7** channels *per series*,
yet **loses in the pooled mean** (−0.85%): the single high-variance channel **MULL** (test
MSE 0.366 → 0.392) dominates the element-weighted average. Reporting only the per-series
win-rate would be misleading; the pooled result is the honest headline.

**Restoration is the real finding.** A (restored vs raw) is enormous and unanimous (7/7):
scale restoration is doing exactly what it should. The mechanism is sound; the *regime* (dense
continuous ETT channels, where a frozen TSFM is already excellent) is unfavourable to
retrieval augmentation.

### Invalid-scale diagnostics (test)

| scale | fallback (invalid query) | invalid candidate restorations |
|---|---|---|
| mean | 931 / 80 199 (1.2%) | 28 |
| rms | 0 | 0 |

`mean` scaling is fragile on zero-centred normalised data (931 near-zero-mean contexts fell
back); `rms` is robust. `mean` was nonetheless selected because it marginally won val MSE.

## 4. Part 5 — Measured compute / storage / parameters (same machine, warm-up excluded)

RTX 5070 Ti (sm_120), batch 256. Sources: `reports/phase11a/compute_backbone.json`,
`reports/phase11a/compute_scalerag.json`, and the capture/reproduction runtimes.

| Component | Trainable params | Latency / window | Peak VRAM | Index build | DB storage | Peak RAM |
|---|---|---|---|---|---|---|
| Chronos-Bolt backbone (frozen) | **0** (205.3 M frozen) | 0.431 ms (GPU) | 1 562 MB | — | — | — |
| TS-RAG ARM (official) | **4.78 M** (16 tensors) | ~0.45 ms (+4.3% wall) | ~1.6 GB | — | 1 529 MB (`.pkl` KB) | — |
| ScaleRAG retrieval (frozen) | **0** (non-neural) | 0.607 ms (CPU, exact k-NN) | 0 (CPU) | 1.38 s (7 var) | 1 096 MB (float64) | 1 245 MB |

Full-test wall clock: backbone 34.2 s; TS-RAG 35.5 s; ScaleRAG ≈ 34.2 s (backbone) + ~49 s
(retrieval) ≈ **85 s**. ScaleRAG is ~2.4× slower and needs a ~1.1 GB in-RAM index — **for a
worse result**. The efficiency trade-off is also unfavourable.

## 5. Threats to validity

1. **Stricter leakage for ScaleRAG than TS-RAG** (train-only pool vs full-series KB, §1). Any
   remaining ScaleRAG deficit is a *lower bound* on its disadvantage under this constraint.
2. **Bootstrap unit = window** (n = 80 199), matching TS-RAG's `boot_res`. Stride-1 windows
   are autocorrelated, so the CIs are approximate (too narrow); a block bootstrap would widen
   them. The point estimates and signs are unaffected, and condition 3 fails on the point
   estimate regardless.
3. **`mean`-scale fallbacks** (1.2% of test queries) slightly favour the backbone within the
   fusion; `rms` (0 fallbacks) is close behind on val and also fails condition 3.
4. **Single dataset (ETTm2).** ETTm2 alone cannot establish generality — but it is the
   pre-registered *development* dataset and its negative gate result blocks the (multi-dataset)
   Phase-11B test by design.

## 6. Verdict

The scale-restoration mechanism transfers to the TS-RAG regime **as a retrieval-quality
improvement** (A: +85%), but **not as an end-to-end forecasting improvement** over a frozen
Chronos-Bolt on ETTm2 (C: −0.85%, D vs TS-RAG: −2.30%). **Decision gate NOT passed (4/5;
condition 3 fails).** Phase 11B is **not** initiated and the four test datasets remain
untouched, per the pre-registration and the project's non-negotiable rules (9, 12).

**Machine-readable artifacts:** `docs/scalerag-native-dev-results.json`,
`docs/scalerag-native-frozen-config.json`,
`reports/phase11a/scalerag_native_ettm2_{val,test}.json`,
`reports/phase11a/scalerag_native_ettm2_{val,test}_preds.npz`,
`reports/phase11a/chronos_target_ettm2_{val,test}.{json,npz}`,
`reports/phase11a/compute_{backbone,scalerag}.json`.
