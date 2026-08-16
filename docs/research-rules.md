# Research rules

The evaluation protocol this codebase is held to. These were fixed before the
study ran, not written up afterwards. Several modules cite them by number in
their docstrings, and `tests/leakage/` exists to make rules 1, 3, 4 and 5
mechanically enforceable rather than aspirational.

## Splitting and leakage

1. **Chronological splits only.** Train, validation and test are separated by
   time boundaries. A random or shuffled split over a temporal panel leaks the
   future into the retrieval pool and is never used.

2. **Hidden evaluation labels are never read.** For M5 that means `d_1942+`
   is off-limits entirely. The `d_1914`–`d_1941` test window was frozen, opened
   exactly once, and is not re-evaluated; `scripts/scalerag_test_final.py`
   refuses to run while the consumption lock exists.

3. **Retrieval horizon guard.** Every retrieved candidate satisfies
   `candidate_end + H < target_forecast_origin`. The guard is enforced at pool
   construction, not at query time, so a retrieved continuation cannot overlap
   the window it is used to predict.

4. **Known-future covariates are distinguished from unavailable information.**
   Calendar features and prices announced ahead of time are admissible at the
   forecast origin; realised future targets never are.

5. **Everything fitted is fitted on history only.** Scalers, retrieval indices,
   utility labels and the fusion gate all see training or historical origins
   exclusively.

## Reporting

6. **No single-run headline numbers.** Results carry seeds and dispersion, or a
   paired-bootstrap confidence interval over series.

7. **A failed model is reported, not swapped.** If a component does not work,
   that is the result.

8. **No significance claim without a named test or interval.** "Better" on its
   own is not a finding.

9. **Evaluation code is never altered to improve a score.** Changing evaluation
   logic requires a reason recorded in the experiment log, and it is never
   "the number was too low".

10. **Every experiment records** seeds, library versions, config, commit,
    runtime and hardware. See [reproducibility-policy.md](reproducibility-policy.md).

11. **Negative results are preserved.** The relation-aware graph-routing
    hypothesis that this project started from was falsified across two datasets
    with non-learned routers, learned routers, controls and confidence
    intervals. It is reported as a negative rather than quietly dropped, and
    parameter-efficient adaptation is never used to rescue a weak retrieval
    result.

12. **Pre-registered criteria are not edited after the fact.** The three success
    criteria in [method.md](method.md) were fixed before the held-out split was
    opened and are reported unchanged, including the fact that none was met.

## Data handling

- Flow is one-directional: `data/raw` → `data/interim` → `data/processed`.
  Raw is read-only; processed is regenerated, never edited in place.
- Datasets, model weights and credentials are never committed. Dataset access
  uses environment-provided tokens that are never echoed or logged.
- Timestamp ordering and frequency are asserted per series; row and null counts
  are recorded at each stage.
