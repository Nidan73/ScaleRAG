"""Classical forecasting baselines for GraphRoute-TS (Phase 2, task 7).

Only Seasonal Naive and LightGBM. No TSFM/retrieval/graph/LoRA here — those are
out of scope until the baseline pipeline is verified (CLAUDE.md rule 11).
"""

from __future__ import annotations

BASELINES = ("seasonal_naive", "lightgbm")
