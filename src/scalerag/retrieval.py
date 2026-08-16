"""Leakage-safe temporal retrieval (Phase 4, tasks 6-9).

A historical-window database of (context, continuation) candidates built from
**training data only**, plus three simple, model-agnostic retrievers (random,
seasonal, Euclidean). Every retrieval enforces the horizon guard (rule 3):

    t_r + H < target_forecast_origin

for a candidate whose context ends at day ``t_r`` and whose continuation covers
``t_r+1 .. t_r+H``. This module has no dependency on any model (task 9) and does
not implement ARM/graph fusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scalerag import leakage


@dataclass(frozen=True)
class RetrievedContext:
    """One retrieved candidate. ``continuation`` is the leakage-safe numerical
    context (the H values that followed the candidate's context in training)."""

    series_idx: int
    t_r: int
    context: np.ndarray
    continuation: np.ndarray
    distance: float


def _znorm(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / np.where(sd == 0, 1.0, sd)


class WindowDatabase:
    """Candidate windows extracted from a training panel.

    Built from ``series[n_series, T]`` reading only days ``<= train_end``. A
    candidate has context day-range ``[t_r-L+1, t_r]`` and continuation
    ``[t_r+1, t_r+H]``; only candidates whose continuation lies entirely within
    training (``t_r + H <= train_end``) are stored (candidate-index fitting on
    training data only).
    """

    def __init__(
        self,
        contexts: np.ndarray,
        continuations: np.ndarray,
        t_r: np.ndarray,
        series_idx: np.ndarray,
        context_length: int,
        horizon: int,
        train_end: int,
    ) -> None:
        self.contexts = contexts
        self.continuations = continuations
        self.t_r = t_r
        self.series_idx = series_idx
        self.context_length = context_length
        self.horizon = horizon
        self.train_end = train_end

    def __len__(self) -> int:
        return int(self.t_r.shape[0])

    @classmethod
    def from_training(
        cls,
        series: np.ndarray,
        train_end: int,
        context_length: int,
        horizon: int,
        stride: int = 1,
    ) -> WindowDatabase:
        n, t = series.shape
        L, H = context_length, horizon  # noqa: N806 — L,H are standard TS notation
        if train_end > t:
            raise ValueError(f"train_end {train_end} exceeds series length {t}.")
        # 1-based day t_r: context [t_r-L+1, t_r] (arr [t_r-L, t_r)),
        # continuation [t_r+1, t_r+H] (arr [t_r, t_r+H)). Legal: t_r+H <= train_end, t_r >= L.
        cand_trs = list(range(L, train_end - H + 1, stride))
        if not cand_trs:
            empty = np.empty((0, L))
            return cls(empty, np.empty((0, H)), np.empty(0, int), np.empty(0, int), L, H, train_end)
        ctxs, conts, trs, sidx = [], [], [], []
        for s in range(n):
            row = series[s]
            for t_r in cand_trs:
                ctxs.append(row[t_r - L : t_r])
                conts.append(row[t_r : t_r + H])
                trs.append(t_r)
                sidx.append(s)
        return cls(
            np.asarray(ctxs, dtype=np.float64),
            np.asarray(conts, dtype=np.float64),
            np.asarray(trs, dtype=np.int64),
            np.asarray(sidx, dtype=np.int64),
            L,
            H,
            train_end,
        )

    def legal_mask(self, origin: int) -> np.ndarray:
        """Candidates satisfying ``t_r + H < origin`` (rule 3)."""
        return (self.t_r + self.horizon) < origin

    def assert_all_legal(self, indices: np.ndarray, origin: int) -> None:
        """Raise LeakageViolation if any selected candidate breaks the guard."""
        for i in np.atleast_1d(indices):
            leakage.assert_retrieval_horizon(int(self.t_r[i]), self.horizon, origin)

    def _make(self, i: int, distance: float) -> RetrievedContext:
        return RetrievedContext(
            series_idx=int(self.series_idx[i]),
            t_r=int(self.t_r[i]),
            context=self.contexts[i],
            continuation=self.continuations[i],
            distance=float(distance),
        )


class BaseRetriever:
    """Retrievers return up to ``k`` leakage-safe candidates for a query."""

    def retrieve(
        self,
        db: WindowDatabase,
        query_context: np.ndarray,
        origin: int,
        k: int,
        query_series_idx: int | None = None,
    ) -> list[RetrievedContext]:
        raise NotImplementedError


class RandomRetriever(BaseRetriever):
    """Uniform random legal candidates. Deterministic per (seed, origin, series)."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def retrieve(self, db, query_context, origin, k, query_series_idx=None):
        idx = np.flatnonzero(db.legal_mask(origin))
        if idx.size == 0:
            return []
        rng = np.random.default_rng([self.seed, origin, query_series_idx or 0])
        chosen = rng.choice(idx, size=min(k, idx.size), replace=False)
        db.assert_all_legal(chosen, origin)
        return [db._make(int(i), float("nan")) for i in sorted(chosen)]


class SeasonalRetriever(BaseRetriever):
    """Same-series candidates at seasonal offsets before the origin (heuristic).

    Picks the most-recent legal candidate for the query series, then steps back
    by ``season`` days, giving the "same phase, weeks/years ago" contexts.
    """

    def __init__(self, season: int = 7) -> None:
        self.season = season

    def retrieve(self, db, query_context, origin, k, query_series_idx=None):
        legal = db.legal_mask(origin)
        if query_series_idx is not None:
            legal = legal & (db.series_idx == query_series_idx)
        idx = np.flatnonzero(legal)
        if idx.size == 0:
            return []
        trs = db.t_r[idx]
        chosen: list[int] = []
        target = int(trs.max())  # most-recent legal t_r
        while len(chosen) < k:
            j = idx[np.argmin(np.abs(trs - target))]
            if j not in chosen:
                chosen.append(int(j))
            target -= self.season
            if target < int(trs.min()):
                break
        db.assert_all_legal(np.array(chosen), origin)
        return [db._make(i, float("nan")) for i in chosen]


class EuclideanRetriever(BaseRetriever):
    """Top-k by Euclidean distance on z-normalised context windows.

    Deterministic: ties broken by candidate index (stable ``lexsort``).
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def retrieve(self, db, query_context, origin, k, query_series_idx=None):
        idx = np.flatnonzero(db.legal_mask(origin))
        if idx.size == 0:
            return []
        q = _znorm(query_context) if self.normalize else query_context
        cand = db.contexts[idx]
        cand = _znorm(cand) if self.normalize else cand
        dist = np.linalg.norm(cand - q, axis=1)
        order = np.lexsort((idx, dist))[:k]  # distance asc, tie-break by index
        sel = idx[order]
        db.assert_all_legal(sel, origin)
        return [db._make(int(i), float(d)) for i, d in zip(sel, dist[order], strict=True)]


RETRIEVERS = {
    "random": RandomRetriever,
    "seasonal": SeasonalRetriever,
    "euclidean": EuclideanRetriever,
}
