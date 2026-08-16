"""Simple, model-agnostic forecasting from retrieved context (Phase 4, tasks 8-9).

- ``knn_forecast``: point = mean of the top-k retrieved continuations; quantiles =
  empirical quantiles across those continuations (enables pinball loss).
- ``late_fusion``: blend two forecasts (e.g. target-only Chronos-2 + retrieved).

This is intentionally NOT TS-RAG's ARM — no learned fusion, no attention, no
graph. Just a k-NN over leakage-safe windows and a convex blend.
"""

from __future__ import annotations

import numpy as np

from scalerag.retrieval import BaseRetriever, WindowDatabase


def knn_forecast(
    db: WindowDatabase,
    retriever: BaseRetriever,
    query_context: np.ndarray,
    origin: int,
    horizon: int,
    k: int,
    series_idx: int,
    quantile_levels: list[float],
    fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast = aggregate of top-k retrieved continuations.

    Returns (point [H], quantiles [H, Q]). On empty/insufficient retrieval,
    falls back to ``fallback`` (default: constant = mean of the query context) —
    covers the empty-history case (task 12).
    """
    cands = retriever.retrieve(db, query_context, origin, k, query_series_idx=series_idx)
    if not cands:
        base = fallback if fallback is not None else np.full(horizon, float(np.mean(query_context)))
        base = np.clip(base, 0.0, None)
        quants = np.repeat(base[:, None], len(quantile_levels), axis=1)
        return base, quants
    conts = np.stack([c.continuation for c in cands])  # (k, H)
    point = np.clip(conts.mean(axis=0), 0.0, None)  # (H,)
    quants = np.clip(np.quantile(conts, quantile_levels, axis=0).T, 0.0, None)  # (H, Q)
    return point, quants


def late_fusion(
    point_a: np.ndarray,
    quants_a: np.ndarray,
    point_b: np.ndarray,
    quants_b: np.ndarray,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convex blend: ``alpha * A + (1 - alpha) * B`` for point and quantiles."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    point = alpha * point_a + (1 - alpha) * point_b
    quants = alpha * quants_a + (1 - alpha) * quants_b
    return np.clip(point, 0.0, None), np.clip(quants, 0.0, None)
