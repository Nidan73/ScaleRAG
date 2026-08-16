# TS-RAG Official Reproduction on ETTm2 (Phase 11A · Part 2)

**Date:** 2026-07-14 · **Dataset:** ETTm2 (development only) · **Context 512 / Horizon 64**
**Verdict:** Official TS-RAG and its Chronos-Bolt backbone **reproduced to within ≈0.1%**
of the published numbers. **Decision-gate condition 1 (reproduction within a justified
tolerance): MET.**

> Scope guard: ETTm2 only. ETTh1/ETTm1/Weather/Electricity (Phase-11B final) not touched.

---

## 1. Isolated environment (unchanged main env)

| | |
|---|---|
| venv | `external/ts-rag/.venv-tsrag` (uv, **Python 3.9.25**) — separate from the project `.venv` |
| torch | **2.7.1+cu128** (cu12.8) — required to run the RTX 5070 Ti (sm_120); autogluon's default cu126 torch cannot (`no kernel image`) |
| key deps | `autogluon.timeseries==1.3.0`, `chronos-forecasting==1.5.1`, `transformers==4.49.0`, `gluonts==0.16.3`, `numpy==1.25.0`, `faiss-cpu==1.7.4`, `scikit-learn==1.6.1`, `statsmodels==0.14.6`, `safetensors==0.7.0` |
| full lock | `external/ts-rag/requirements.lock.txt` (167 packages, exact pins) |
| GPU | NVIDIA RTX 5070 Ti, sm_120, CUDA 12.8; Chronos-Bolt peak VRAM < 2 GB |

**Documented, results-neutral deviations from `requirements.txt`:**
1. `faiss_gpu==1.7.2` → `faiss-cpu==1.7.4`. `faiss_gpu==1.7.2` has no sm_120 build; with the
   **cached official retrieval CSV**, `do_retrieve` is skipped and FAISS is only *imported*
   (never executed), so this cannot change any reproduction number. `faiss-cpu>=1.8` breaks
   on numpy 1.25 (`numpy._core`), hence the 1.7.4 pin.
2. `torch` upgraded from autogluon's pinned 2.6/cu126 to **2.7.1+cu128** for sm_120 kernels.
   The full autogluon/chronos/TS-RAG import chain loads unchanged under 2.7.1; the backbone
   is frozen (§2), so this only affects arithmetic precision/latency, not the method.
3. Installed `autogluon.timeseries==1.3.0` (which pulls `autogluon.common/core==1.3.0`, the
   exact submodules `models/utils.py` imports) rather than the full `autogluon==1.3.0`
   metapackage — identical imported code paths.

## 2. Provenance & the "frozen backbone" verification

All assets downloaded from official HF repos over HTTPS with **sha256 cross-checked against
each repo's LFS oid** (`external/ts-rag/ettm2_asset_manifest.tsv`):

| Asset | Source | sha256 (prefix) | Bytes |
|---|---|---|---|
| `best.pth` (TS-RAG trained) | `nkh/TS-RAG-ChronosBolt/pytorch_model.bin` | `765e858d…` | 840,404,992 |
| base `model.safetensors` | `autogluon/chronos-bolt-base` | `31f87548…` | 821,203,576 |
| `ETTm2_minute_512.pkl` (KB) | `nkh/TS-RAG-Data` | `9f86bcb3…` | 1,528,600,844 |
| `ETTm2_retrieve_…csv` (indices) | `nkh/TS-RAG-Data` | `7058930a…` | 317,999,133 |
| `ETTm2.csv` (raw) | ETDataset | `db973ca2…` | 9,677,236 |

**Untrusted-pickle handling:** the 1.53 GB `.pkl` was (a) opcode-scanned with `pickletools`
(memo-aware) — the only import targets are `numpy.{ndarray,dtype}` and
`numpy.core.multiarray._reconstruct` (no `os`/`subprocess`/`eval`); (b) loaded via a
**restricted unpickler** (`find_class` numpy-allowlist) inside the isolated env. Its
`raw_data` for all 7 variables is **bit-exact equal** to `ETTm2.csv`, and its embeddings are
`(69169, 768) = (len−512+1, 768)` — confirming the official KB pairs are exactly the ETTm2
context/continuation windows.

**Frozen-backbone check (resolves the audit's open item).** best.pth (287 tensors) vs base
`model.safetensors` (269 tensors): the **269 shared backbone tensors are bit-identical**
(max abs diff `0.000e+00`); the 18 extra tensors are exactly the ARM
(`encode_mlp`/`mha`/`ffn`/`gate_layer`). **⇒ TS-RAG trains only the ARM on a genuinely frozen
Chronos-Bolt-base**, and the correct **target-only baseline is vanilla
`autogluon/chronos-bolt-base`** — no Google-Drive `autogluon_model.pth` is needed. At ARM load
`load_state_dict` reported `missing=0 unexpected=0` (best.pth fully populates the model).

## 3. Reproduction results (ETTm2 test, 80,199 windows, normalised space)

Metrics = mean over all (window × 64-horizon × channel) elements, exactly TS-RAG's
`utils.metrics.metric` on the train-only-`StandardScaler`-normalised series (their code, splits,
and metric reused unmodified).

| Method | Paper MSE | **Repro MSE** | Δ abs | Δ rel | Paper MAE | **Repro MAE** | Δ abs | Δ rel |
|---|---|---|---|---|---|---|---|---|
| Chronos-Bolt (target-only) | 0.1487 | **0.14856** | −0.00014 | **−0.09%** | 0.2236 | **0.22354** | −0.00006 | −0.03% |
| TS-RAG (moe, top_k=10) | 0.1466 | **0.14646** | −0.00014 | **−0.10%** | 0.2231 | **0.22302** | −0.00008 | −0.04% |

**TS-RAG improvement over its backbone (the quantity of interest) reproduces essentially
exactly:** MSE −1.41% (paper −1.41%), MAE −0.23% (paper −0.22%).

**Independent cross-check** — TS-RAG's **unmodified `zeroshot.py`** (retrieve path, config-only
ETTm2 wrapper) printed `mse:0.1465, mae:0.2230` over `(1, 80199, 64)`, matching the capture
runner (0.14646 / 0.22302). The runner only substitutes the cached official retrieval for the
FAISS `do_retrieve` step and additionally saves per-window arrays; it is otherwise their code.

Runtime: 34–36 s per method (full test set) on the RTX 5070 Ti at batch 256.

Artifacts: `reports/phase11a/repro_{chronos_bolt_target,tsrag_official}_ettm2.{json,npz}`
(the `.npz` hold per-window preds/trues/per-window MSE for Part-5 paired-bootstrap CIs).

## 4. Justified tolerance & verdict

Because the reproduction uses the **same frozen checkpoint, the same official retrieval
assets, and TS-RAG's own splits/metric/model code**, the only expected differences are
floating-point/device/precision noise. A pre-declared tolerance of **≤1% relative on MSE and
MAE** (and reproducing the sign and ~magnitude of the TS-RAG−backbone delta) is generous; the
observed deviations are **≤0.10%** on every metric, and the delta matches to two decimals, with
the unmodified script agreeing to 4 decimals.

**Decision-gate condition 1: MET.** No blocker. Proceeding to Part 3 (ScaleRAG native-protocol
adapter) is warranted — the remaining decision-gate conditions (2–5) are evaluated in Parts 3–5
and remain open.
