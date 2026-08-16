# ScaleRAG-TS — Method

**ScaleRAG-TS: Scale-Aware Retrieval Augmentation for Time-Series Foundation
Models.** A model-agnostic system that augments a **frozen** Chronos-2 backbone
with **scale-aware temporal retrieval** and **uncertainty-aware gated fusion**. No
graph/relational features (the typed-relation routing hypothesis was falsified
across M5 and Favorita in Phases 6–8). No retrieved future labels ever enter the
Chronos input.

## Components

### A. Scale-aware temporal retrieval (Phase 5, frozen)
Candidate windows are matched with **mean/RMS-normalised** context vectors (FAISS
`IndexFlatL2`, exact) and their continuations are **restored to the target's
scale** before use — the core fix that turns naive retrieval (which *hurt* in
Phase 4) into a competitive forecaster. Design levers evaluated: normalisation
(raw / z-norm / mean / RMS), scale restoration on/off, seasonal alignment, context
compatibility, intermittency-aware filtering, retrieval diversity, top-k, and
candidate reliability from historical forecast utility. Best frozen config:
`mean / L2 / category-filter / k=20` + restoration.

### B. Uncertainty-aware gated fusion (proposed, learned)
The target-only Chronos-2 predictive distribution and the retrieved-continuation
distribution are blended per series:

    forecast = (1 − α) · Chronos2(target) + α · Retrieval(target)

α is produced by a **small learned gate** trained on **historical origins only**.
Gate features (no relations): retrieval nearest-neighbour distance (retrieval
confidence), retrieved-continuation disagreement, intermittency, demand level,
Chronos predictive uncertainty (q90−q10 spread), context scale-spread. The gate
routes each series toward whichever source is historically more reliable for series
with its characteristics.

### C. Chronos-2 integration (two model-agnostic strategies)
1. **In-context grouping** — pass the target plus retrieved histories as grouped
   numerical series to Chronos-2 (uses its multivariate/grouping capability).
2. **Gated fusion** — combine the two distributions via the learned gate (B).

Both keep Chronos-2 frozen and inject **no** retrieved future labels.

### D. Parameter-efficient adaptation (deferred; evaluated last)
Only after frozen + gated-fusion systems are measured: a small **adapter** or
**LoRA** on a frozen Chronos-2, reported separately (adapter-only vs LoRA, with
trainable-parameter and memory accounting). LoRA is **never** used to conceal a
weak retrieval mechanism — retrieval-only and gated-fusion results stay separately
reported.

## Leakage rules (enforced)
For every retrieved candidate: `candidate_end + H < target_forecast_origin`.
All retriever fitting, utility labels, normalisation stats, and gate training use
**historical origins only**. Known-future covariates only when demonstrably
available at forecast time. The M5 test split `d_1914–d_1941` is never used during
development.

## Statistical protocol
Paired **bootstrap** 95% CIs over series (per-series relative-improvement
distributions); ≥3 rolling origins; ≥3 seeds for trainable components (the gate);
predefined sparse / intermittent / low-volume / reduced-history slices.
Significance is **not** claimed from seed variance alone.

## Reproductions vs. inspired reimplementations
- **Exact/frozen (ours):** recent-mean, Seasonal Naive, LightGBM, target-only
  Chronos-2, Chronos-2 + known-future covariates, Phase 5 scale-aware retrieval.
- **Inspired reimplementations** (clearly labelled, not exact reproductions):
  *RAFT-style* retrieved-continuation forecasting and *TS-RAG-style* retrieval with
  a simple reproduced fusion. These follow the published ideas, not released code,
  and are compared only under our identical splits/metrics/leakage protocol.

## Pre-registered success criteria (see validation report)
The method must meet ≥1 of: (1) ≥3% relative WRMSSE/RMSSE improvement over the
strongest matched baseline with a 95% CI excluding zero; (2) ≥5% over target-only
Chronos-2 on **both** M5 and Favorita; (3) ≥7% on predefined sparse/intermittent/
low-volume/reduced-history slices without materially degrading overall. If none
hold, no positive forecasting improvement is claimed and the paper is framed as a
controlled retrieval + relational-metadata study.

## Status
Increment 1 (this commit): scale-aware retrieval + **learned gated fusion** vs
target-only Chronos-2 / recent-mean / raw retrieval / Phase 5 retrieval on M5
1,000, with paired-bootstrap CIs. Remaining (progressive): in-context grouping,
RAFT/TS-RAG baselines, full ablation matrix, Favorita + 5k/full scales, adapter/
LoRA (D), protocol-comparability and compute tables.
