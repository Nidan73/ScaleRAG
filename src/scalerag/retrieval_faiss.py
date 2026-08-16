"""Scalable, scale-aware temporal retrieval with FAISS-CPU (Phase 5).

Builds on the leakage-safe ``WindowDatabase`` (Phase 4) and adds:
- **scale strategies** (raw / z-norm / mean / RMS) applied to context windows
  before indexing, with **scale restoration** of retrieved continuations back to
  the query's scale — the fix for Phase 4's cross-series scale mismatch;
- **FAISS-CPU** exact search (`IndexFlatL2` / `IndexFlatIP` for cosine) — identical
  top-k to brute force, but fast at scale;
- **metadata filters** (same store / category / department, seasonal position).

The retrieval invariant ``t_r + H < origin`` (rule 3) is asserted on every result.
No graphs, no ARM.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from scalerag import leakage
from scalerag.retrieval import WindowDatabase

SCALE_STRATEGIES = ("raw", "znorm", "mean", "rms")
METRICS = ("l2", "cosine")


def _fit_params(w: np.ndarray, strategy: str) -> np.ndarray:
    """Per-row scale parameters. Shape (N, 2): (loc, scale)."""
    if strategy == "raw":
        return np.stack([np.zeros(len(w)), np.ones(len(w))], axis=1)
    if strategy == "znorm":
        return np.stack([w.mean(1), np.where(w.std(1) == 0, 1.0, w.std(1))], axis=1)
    if strategy == "mean":
        m = w.mean(1)
        return np.stack([np.zeros(len(w)), np.where(m == 0, 1.0, m)], axis=1)
    if strategy == "rms":
        r = np.sqrt((w**2).mean(1))
        return np.stack([np.zeros(len(w)), np.where(r == 0, 1.0, r)], axis=1)
    raise ValueError(f"unknown scale strategy {strategy!r}")


def _transform(w: np.ndarray, params: np.ndarray) -> np.ndarray:
    loc = params[:, 0:1]
    scale = params[:, 1:2]
    return (w - loc) / scale


def restore_continuation(
    cont: np.ndarray, cand_params: np.ndarray, query_params: np.ndarray
) -> np.ndarray:
    """Map a candidate continuation from its own scale to the query's scale."""
    c_loc, c_scale = cand_params
    q_loc, q_scale = query_params
    return (cont - c_loc) / c_scale * q_scale + q_loc


class ScaleAwareIndex:
    """FAISS index over scale-transformed candidate contexts, with metadata."""

    def __init__(
        self,
        db: WindowDatabase,
        entities: pl.DataFrame,
        scale: str = "znorm",
        metric: str = "l2",
        season: int = 7,
    ) -> None:
        import faiss

        if scale not in SCALE_STRATEGIES:
            raise ValueError(f"scale must be one of {SCALE_STRATEGIES}")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        self.db = db
        self.scale = scale
        self.metric = metric
        self.season = season

        self.params = _fit_params(db.contexts, scale)
        vecs = _transform(db.contexts, self.params).astype(np.float32)
        if metric == "cosine":
            faiss.normalize_L2(vecs)
        self._vecs = vecs
        self.index = (
            faiss.IndexFlatIP(vecs.shape[1])
            if metric == "cosine"
            else faiss.IndexFlatL2(vecs.shape[1])
        )
        self.index.add(vecs)

        # per-candidate metadata codes (for filtering)
        def codes(col: str) -> np.ndarray:
            vals = entities[col].to_numpy()[db.series_idx]
            _, inv = np.unique(vals, return_inverse=True)
            return inv.astype(np.int64)

        self.meta = {c: codes(c) for c in ("store_id", "cat_id", "dept_id")}
        self.entities = entities
        self.fallback_count = 0  # times retrieval was empty -> constant fallback
        # candidates are grouped contiguously by series (WindowDatabase build order),
        # enabling O(1) per-series candidate ranges for graph-restricted retrieval.
        n_ser = entities.height
        self._ser_start = np.searchsorted(db.series_idx, np.arange(n_ser), side="left")
        self._ser_end = np.searchsorted(db.series_idx, np.arange(n_ser), side="right")

    def reset_stats(self) -> None:
        self.fallback_count = 0

    def candidates_for_series(self, series_ids: np.ndarray) -> np.ndarray:
        """Candidate ids belonging to the given series (graph-restricted pool)."""
        parts = [
            np.arange(self._ser_start[s], self._ser_end[s])
            for s in series_ids
            if self._ser_end[s] > self._ser_start[s]
        ]
        return np.concatenate(parts) if parts else np.empty(0, dtype=int)

    def _query_vec(self, query_window: np.ndarray):
        import faiss

        qp = _fit_params(query_window[None], self.scale)[0]
        qv = _transform(query_window[None], qp[None]).astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(qv)
        return qv, qp

    def _allowed(self, origin: int, query_series_idx: int | None, meta_filter: str | None):
        allowed = self.db.legal_mask(origin)  # t_r + H < origin (rule 3)
        if meta_filter in ("store_id", "cat_id", "dept_id"):
            if query_series_idx is None:
                raise ValueError(f"{meta_filter} filter needs query_series_idx")
            # match candidates whose meta code equals the query's
            qc = self.meta[meta_filter][self.db.series_idx == query_series_idx]
            target = qc[0] if qc.size else -1
            allowed = allowed & (self.meta[meta_filter] == target)
        elif meta_filter == "seasonal":
            allowed = allowed & ((self.db.t_r % self.season) == ((origin - 1) % self.season))
        elif meta_filter is not None:
            raise ValueError(f"unknown meta_filter {meta_filter!r}")
        return allowed

    def search(
        self,
        query_window: np.ndarray,
        origin: int,
        k: int,
        query_series_idx: int | None = None,
        meta_filter: str | None = None,
        allowed_series: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (candidate_ids, distances, query_params). Enforces rule 3.

        ``allowed_series`` restricts the candidate pool to those series (graph-
        guided retrieval, Phase 6) — takes precedence over ``meta_filter``.
        """
        qv, qp = self._query_vec(query_window)
        if allowed_series is not None:
            cand = self.candidates_for_series(allowed_series)
            legal = self.db.legal_mask(origin)
            allowed_ids = cand[legal[cand]] if cand.size else cand
        else:
            allowed = self._allowed(origin, query_series_idx, meta_filter)
            allowed_ids = np.flatnonzero(allowed)
        if allowed_ids.size == 0:
            return np.empty(0, int), np.empty(0), qp

        if meta_filter is None and allowed_series is None:
            # global FAISS search; over-fetch then filter to legal ids.
            over = min(len(self.db), max(k * 4, k + 64))
            dist, idx = self.index.search(qv, over)
            allowed_set = set(allowed_ids.tolist())
            keep = [(d, i) for d, i in zip(dist[0], idx[0], strict=True) if i in allowed_set]
            keep = keep[:k]
            ids = np.array([i for _, i in keep], dtype=int)
            dists = np.array([d for d, _ in keep])
        else:
            # restricted pool is small -> exact search on the subset vectors.
            sub = self._vecs[allowed_ids]
            # cosine: higher inner product = closer -> negate for ascending order
            d = -(sub @ qv[0]) if self.metric == "cosine" else ((sub - qv[0]) ** 2).sum(1)
            order = np.lexsort((allowed_ids, d))[:k]
            ids = allowed_ids[order]
            dists = d[order]

        if ids.size:
            leakage.assert_retrieval_horizon(int(self.db.t_r[ids].max()), self.db.horizon, origin)
        return ids, dists, qp

    def forecast(
        self,
        query_window: np.ndarray,
        origin: int,
        horizon: int,
        k: int,
        quantile_levels: list[float],
        query_series_idx: int | None = None,
        meta_filter: str | None = None,
        restore_scale: bool = True,
        allowed_series: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """k-NN forecast with optional scale restoration of continuations."""
        ids, _d, qp = self.search(
            query_window, origin, k, query_series_idx, meta_filter, allowed_series
        )
        if ids.size == 0:
            self.fallback_count += 1
            base = np.clip(np.full(horizon, float(query_window.mean())), 0.0, None)
            return base, np.repeat(base[:, None], len(quantile_levels), axis=1)
        conts = self.db.continuations[ids]
        if restore_scale and self.scale != "raw":
            conts = np.stack(
                [restore_continuation(conts[j], self.params[ids[j]], qp) for j in range(len(ids))]
            )
        point = np.clip(conts.mean(0), 0.0, None)
        quants = np.clip(np.quantile(conts, quantile_levels, axis=0).T, 0.0, None)
        return point, quants
