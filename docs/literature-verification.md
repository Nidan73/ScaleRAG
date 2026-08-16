# Literature Verification of Published Claims

Claims that appear in `paper/main.tex` and in the unpublished manuscript were checked
against a 98-source NotebookLM corpus of TSFM and retrieval-augmented forecasting
papers. The ScaleRAG project's own documents were **excluded from every query**, so
none of these verdicts is self-referential; all queries returned zero self-citations.

Checked 2026-07-26. Corpus: 98 sources (81 web, 13 PDF, 4 markdown).

## Verdicts

| # | Claim | Verdict | Grounding |
|---|---|---|---|
| 1 | RAID normalizes a retrieved trajectory by its own mean and standard deviation before mapping the aggregate onto a target scale | **SUPPORTED** | RAID ×9 |
| 2 | RAID reports results on M5 and Favorita | **SUPPORTED** | RAID |
| 3 | RAID targets true cold start; target statistics are regressed by a trained network from text metadata | **SUPPORTED** | RAID |
| 4 | RAID retrieves over frozen multilingual text embeddings by cosine similarity, not numeric windows | **SUPPORTED** | RAID |
| 5 | TS-RAG trains its ARM on a multi-domain corpus and applies it zero-shot without task-specific fine-tuning | **SUPPORTED**, and empirically demonstrated (a "Cross-Domain" ablation row) | TS-RAG ×3 |
| 6 | TS-RAG correlates properties at the dataset level, not per series | **SUPPORTED** — Table 13 is a Pearson coefficient over **4 points** (ETTh1/h2/m1/m2) | TS-RAG |
| 7 | No paper here reports the accuracy of the retrieved continuation separately from the fused forecast | **SUPPORTED** (negative, see caveat) | none possible |
| 8 | Only RAID evaluates on intermittent retail demand among retrieval papers | **CONTRADICTED** | GNBAN ×3 |

## What changed as a result

**Claim 8 was wrong and was never published.** GNBAN also evaluates on M5 and
Favorita and speaks of a "retrieved context vector". The manuscripts say only that
RAID *also* reports on M5 and Favorita, which is accurate and claims no
exclusivity. Any future wording must not claim RAID is the sole retrieval paper on
intermittent retail data.

**Claim 6 sharpened the regime result.** TS-RAG's correlation table rests on four
dataset-level averages, so it structurally cannot show a non-monotone relationship.
That is now stated in `docs/regime-threshold-report.md` as the reason the per-series
inverted U is visible here and not there.

**Claim 5 removed a hedge.** An earlier draft described TS-RAG's cross-dataset
mixer transfer as "plausible from its design, thin as evidence". It is in fact
demonstrated with a dedicated ablation row, so `docs/gate-transfer-report.md` is
correct to treat cross-dataset transfer as table stakes rather than a
differentiator.

## Second sweep, after expanding the corpus (2026-07-27)

The first sweep's negative rested on a corpus with known holes, so sixteen sources
were added: RAFT, ReTime, TimeRAG, RATD, kNN-MTS, Spectral RAG, Retrieval Mechanisms
vs Long-Context, Global Temporal Retrieval, two normalization papers, k-NN and analog
forecasting, and three intermittent-demand papers.

**A methodological trap, recorded because it nearly produced a false negative a
second time.** arXiv `/abs/` pages ingest as **abstract and metadata only**: RAFT's
landing page gave 6,760 characters against 108,339 for a `/pdf/` source. A sweep run
against those returned "not covered" for every question, which reads as absence of
prior art but is absence of *text*. The papers were re-added as `/pdf/` URLs (35k to
88k characters each) before any conclusion was drawn. Broad queries over a 130-source
notebook are also unreliable, since retrieval surfaces only a subset; the findings
below come from queries **scoped to specific source ids**.

### What the expanded corpus found

Two papers materially narrow the novelty claim.

**RAFT** (arXiv 2505.04163) retrieves over **raw numeric windows** by Pearson
correlation, chosen "to exclude the effects of scale variations", and already
performs the location half of our operation. It treats the final value of each patch
as an offset and subtracts it, `x̂ = {xt − xL}`, from the query and from every
retrieved patch, then restores the query's offset to produce the forecast,
`y = {ŷt + xL}`. It divides out **no** magnitude statistic, its backbone is a shallow
MLP trained from scratch rather than a frozen foundation model, and it evaluates only
on dense data (ETT, Electricity, Exchange, Illness, Solar, Traffic, Weather).

**kNN-MTS** (arXiv 2505.11625) augments a **frozen** pretrained forecaster with a
retrieval branch that "does not introduce any trainable parameters", fusing through
`Ŷ_final = (1−λ)Ŷ + λ Σ w_j Y_j`, the same convex blend we use. It retrieves over
learned embeddings and uses the retrieved futures **unmodified**, with no rescaling.

**ReTime** (arXiv 2209.13525) is not prior art here: its retrieval is relational
(random walk with restart), it normalizes only globally at dataset level, it never
rescales a retrieved segment, and it trains end to end.

### Consequence for the novelty claim

| Ingredient | Prior art |
|---|---|
| Scale-insensitive retrieval over raw numeric windows | RAFT (Pearson), TimeRAG (DTW) |
| Remove level, restore the query's level | **RAFT** |
| Remove mean and variance of a retrieved trajectory, restore both | **RAID** (statistics predicted, not measured) |
| Frozen backbone, zero-parameter retrieval, convex fusion | **kNN-MTS** |
| All of the above together | none found |

Every ingredient is published. The composition is not. The manuscripts now say so
explicitly and claim the composition and its empirical characterization rather than
the operation. Citing only RAID, as an earlier draft did, would have read as
selective citation to any reviewer who knows RAFT.

## First sweep, and why it was weak evidence

A systematic sweep asked for **any** source beyond TS-RAG, TimeRAF, SARAF, SERAF,
TRACE and RAID performing normalize-then-restore on a retrieved segment, phrased to
match a dozen synonyms (denormalization, shape-scale decomposition, scale transfer,
analog rescaling, and so on). It returned none, and reported that no source combines
a frozen backbone with a non-neural numeric retriever and a closed-form rescaling of
the retrieved future.

**Treat that as suggestive, not settled.** Three reasons:

1. **The answer carried zero citations.** A negative finding has nothing to cite, so
   it cannot be audited the way the supported claims above can.
2. **The corpus has known holes.** RAFT, ReTime, TimeRAG and RATD are named inside
   other papers' related-work sections but their own papers are **not** sources
   here. Any of them could contain the mechanism. Analog forecasting, nearest
   neighbour forecasting, and the intermittent-demand literature (Croston, SBA, TSB)
   are absent entirely, and the RevIN paper is cited by others rather than present.
3. **Absence from 98 sources is not absence from the field.**

Those gaps were closed on 2026-07-27, see the second sweep above. Still absent in
usable form: the RevIN paper (the OpenReview add returned a browser-verification
page), and the analog, k-NN and intermittent-demand papers, whose full texts are
loaded but which no scoped query has yet interrogated. The claim as currently stated is "no prior art in
the surveyed corpus", and it should not be strengthened beyond that without them.

## Method note

Queries used `notebooklm ask --json` so citations could be mapped back to source
titles. An earlier round, before the ScaleRAG documents were excluded, produced a
novelty verdict in which 7 of 22 citations were the project's own audit report; that
result was discarded as circular. Every verdict above comes from a query that
excluded those documents and returned zero self-citations.
