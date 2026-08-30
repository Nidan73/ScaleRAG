"""ScaleRAG native-protocol adapter (Phase 11A).

Adapts the *existing* ScaleRAG scale-aware retrieval + continuation restoration to
the official TS-RAG benchmark regime (context length 512, horizon 64), operating in
the **train-only-normalised** ETT space so that MSE/MAE are directly comparable to a
frozen Chronos-Bolt backbone and the official TS-RAG ARM.

Design constraints (Phase 11A spec):

* The retriever is **non-neural** and reuses ScaleRAG's frozen mathematical
  definition — it imports the exact scale primitives (``_fit_params`` /
  ``_transform`` / :func:`restore_continuation`) from :mod:`retrieval_faiss`. No new
  neural retriever is invented.
* Query and candidate contexts are compared with the scale-aware transform; retrieved
  continuations are restored from candidate scale to query scale.
* The candidate pool is **strictly train-only** (``t_r + H <= train_end``) so no
  target validation/test future can enter retrieval (rule 5 / rule 3).
* Deterministic **exact** k-NN (BLAS distance, index-tie-break) is the reference;
  a FAISS path can be verified bit-for-bit against it (:func:`topk_exact`).
* Invalid-scale cases (near-zero scale denominators — common for ``mean`` scaling on
  zero-centred normalised data) and empty-pool fallbacks are **counted and reported**.

This module has no import-time I/O and does not depend on torch or any TSFM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scalerag import leakage
from scalerag.retrieval import WindowDatabase

# Preserve ScaleRAG's exact scale math (do not re-derive — import the frozen defs).
# The restoration used below is the vectorised form of retrieval_faiss.restore_continuation
# ((cont - c_loc)/c_scale * q_scale + q_loc) — same math, applied over the (m, k, H) batch.
from scalerag.retrieval_faiss import (
    SCALE_STRATEGIES,
    _fit_params,
    _transform,
)

__all__ = [
    "SCALE_STRATEGIES",
    "NativeRetrievalOutput",
    "NativeScaleRetriever",
    "TopKResult",
    "fixed_fusion",
    "topk_exact",
    "topk_exact_torch",
]


def _scale_invalid(params: np.ndarray, scale: str, eps: float) -> np.ndarray:
    """Boolean mask of rows whose scale denominator is unusable (|scale| < eps).

    ``raw`` never restores/normalises, so it is never invalid. For ``mean`` the scale
    is the (possibly near-zero or negative) window mean; for ``rms``/``znorm`` it is a
    non-negative magnitude/std. In all non-raw cases the restoration/transform divides
    by ``params[:, 1]``, so |params[:, 1]| < eps is the danger condition.
    """
    if scale == "raw":
        return np.zeros(params.shape[0], dtype=bool)
    return np.abs(params[:, 1]) < eps


def topk_exact(
    query_vecs: np.ndarray, cand_vecs: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic exact top-k L2 over transformed vectors (BLAS, index tie-break).

    Returns ``(ids, dists)`` of shape ``(n_query, k)``. Distances use the
    ``||q||^2 - 2 q·c + ||c||^2`` identity (a single GEMM); ties are broken by ascending
    candidate index via a stable lexsort, matching ``retrieval_faiss``.
    """
    q = np.ascontiguousarray(query_vecs, dtype=np.float64)
    c = np.ascontiguousarray(cand_vecs, dtype=np.float64)
    n_cand = c.shape[0]
    kk = min(k, n_cand)
    c_sq = np.einsum("ij,ij->i", c, c)  # (n_cand,)
    ids = np.empty((q.shape[0], kk), dtype=np.int64)
    dists = np.empty((q.shape[0], kk), dtype=np.float64)
    # process queries in chunks to bound the (chunk x n_cand) distance matrix in RAM
    chunk = max(1, int(4_000_000 // max(n_cand, 1)))
    idx_all = np.arange(n_cand)
    for start in range(0, q.shape[0], chunk):
        qb = q[start : start + chunk]
        q_sq = np.einsum("ij,ij->i", qb, qb)[:, None]
        d = q_sq - 2.0 * (qb @ c.T) + c_sq[None, :]  # (b, n_cand), may be slightly <0
        np.maximum(d, 0.0, out=d)
        part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        for r in range(qb.shape[0]):
            cols = part[r]
            # argpartition picks an arbitrary kk among equal distances, so a tie group
            # straddling the k boundary would be resolved by partition order rather
            # than by candidate index -- which is what this function promises. Widen
            # the slice to every candidate at or inside the boundary distance, then
            # let the lexsort break the tie by index over the whole group.
            thresh = d[r, cols].max()
            wide = np.flatnonzero(d[r] <= thresh)
            order = np.lexsort((idx_all[wide], d[r, wide]))[:kk]  # dist asc, index asc
            sel = wide[order]
            ids[start + r] = sel
            dists[start + r] = d[r, sel]
    return ids, dists


def topk_exact_torch(
    query_vecs: np.ndarray, cand_vecs: np.ndarray, k: int, device: str = "cuda"
) -> tuple[np.ndarray, np.ndarray]:
    """GPU-resident equivalent of :func:`topk_exact`, for same-device cost comparison.

    The ETTm2 index is CPU-resident, so a latency comparison against a GPU adapter
    is not like-for-like. This is the identical computation on the accelerator: the
    same ``||q||^2 - 2 q.c + ||c||^2`` identity, the same ascending-distance order,
    and the same tie-break by ascending candidate index. Float64 throughout, so the
    result is bit-comparable with the CPU path rather than merely close.

    Arithmetic acceleration only. It selects the same candidates, so no result
    computed with it differs from one computed with :func:`topk_exact`.
    """
    import torch

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    q = torch.as_tensor(np.ascontiguousarray(query_vecs, dtype=np.float64), device=dev)
    c = torch.as_tensor(np.ascontiguousarray(cand_vecs, dtype=np.float64), device=dev)
    n_cand = c.shape[0]
    kk = min(k, n_cand)
    c_sq = (c * c).sum(1)

    ids = torch.empty((q.shape[0], kk), dtype=torch.int64, device=dev)
    dists = torch.empty((q.shape[0], kk), dtype=torch.float64, device=dev)
    chunk = max(1, int(4_000_000 // max(n_cand, 1)))
    for start in range(0, q.shape[0], chunk):
        qb = q[start : start + chunk]
        d = (qb * qb).sum(1, keepdim=True) - 2.0 * (qb @ c.T) + c_sq.unsqueeze(0)
        d.clamp_(min=0.0)
        # torch.topk on -d is not order-stable across ties, so sort the full row:
        # ascending distance, and ties already ascend by index because sort is stable.
        # A stable full sort resolves ties by ascending index globally, matching
        # topk_exact's widened selection rather than argpartition's arbitrary one.
        order = torch.argsort(d, dim=1, stable=True)
        sel = order[:, :kk]
        ids[start : start + qb.shape[0]] = sel
        dists[start : start + qb.shape[0]] = torch.gather(d, 1, sel)
    return ids.cpu().numpy(), dists.cpu().numpy()


@dataclass
class NativeRetrievalOutput:
    """Per-query retrieval forecasts (normalised space) plus diagnostic counters."""

    point: np.ndarray  # (n_query, H)
    nn_dist: np.ndarray  # (n_query,) nearest-neighbour distance (inf where fallback)
    disagreement: np.ndarray  # (n_query,) mean over-horizon std across the k continuations
    fallback_count: int  # queries served by the constant (context-mean) fallback
    invalid_query_scale: int  # queries whose own scale was invalid -> fallback
    invalid_cand_restore: int  # (query, cand) restorations skipped (invalid cand scale)
    n_query: int


@dataclass
class TopKResult:
    """Precomputed exact top-k search shared across k-slices and raw/restored variants."""

    ids: np.ndarray  # (n_valid, k) candidate ids for valid-scale queries, distance-sorted
    dists: np.ndarray  # (n_valid, k) squared-L2 distances
    valid_idx: np.ndarray  # (n_valid,) indices into the query batch (scale-valid queries)
    qparams: np.ndarray  # (n, 2) per-query (loc, scale)
    q_invalid: np.ndarray  # (n,) bool — queries whose own scale was invalid


class NativeScaleRetriever:
    """Per-variable scale-aware exact k-NN over train-only (context, continuation) pairs.

    Parameters
    ----------
    series:
        1-D normalised series for a single variable (full length; only the train
        region ``[:train_end]`` is used to build candidates).
    train_end:
        Exclusive index of the end of the training region. Candidates satisfy
        ``t_r + horizon <= train_end`` (strictly train-only continuations).
    scale:
        One of :data:`SCALE_STRATEGIES` (``mean``/``rms`` are the pre-registered grid).
    scale_eps:
        Threshold below which a scale denominator is treated as invalid.
    """

    def __init__(
        self,
        series: np.ndarray,
        train_end: int,
        scale: str,
        context_length: int = 512,
        horizon: int = 64,
        stride: int = 1,
        scale_eps: float = 1e-2,
    ) -> None:
        if scale not in SCALE_STRATEGIES:
            raise ValueError(f"scale must be one of {SCALE_STRATEGIES}, got {scale!r}")
        if series.ndim != 1:
            raise ValueError(f"series must be 1-D (one variable), got shape {series.shape}")
        self.scale = scale
        self.context_length = context_length
        self.horizon = horizon
        self.scale_eps = scale_eps
        self.db = WindowDatabase.from_training(
            series[None, :], train_end, context_length, horizon, stride=stride
        )
        if len(self.db) == 0:
            raise ValueError("empty candidate pool — check train_end/context/horizon")
        self.params = _fit_params(self.db.contexts, scale)  # (N, 2): loc, scale
        self.vecs = _transform(self.db.contexts, self.params).astype(np.float64)  # (N, L)
        self.cand_invalid = _scale_invalid(self.params, scale, scale_eps)  # (N,)

    def retrieve(self, queries: np.ndarray, origins: np.ndarray, k: int) -> TopKResult:
        """Exact top-k retrieval for each query (rule-3 guarded). Distances are computed
        once here; :meth:`forecast_from_topk` derives forecasts for any ``k' <= k`` and any
        ``restore`` flag from this single result (so a whole grid reuses one search)."""
        n = queries.shape[0]
        if origins.shape[0] != n:
            raise ValueError("queries and origins length mismatch")
        qparams = _fit_params(queries, self.scale)  # (n, 2)
        q_invalid = _scale_invalid(qparams, self.scale, self.scale_eps)
        qvecs = _transform(queries, qparams)  # (n, L)
        valid_idx = np.flatnonzero(~q_invalid)
        # Restrict to candidates rule-3-legal for the EARLIEST origin in the batch
        # (t_r + H < min(origins)); such candidates are legal for every later origin too.
        # This filters the boundary candidate at the val split (origin == train_end) instead
        # of raising, mirroring retrieval_faiss.ScaleAwareIndex.
        legal_ids = np.flatnonzero(self.db.legal_mask(int(origins.min())))
        if legal_ids.size == 0:
            raise leakage.LeakageViolation(
                f"no rule-3-legal candidates for origin {int(origins.min())} "
                f"(train_end pool max t_r+H = {int(self.db.t_r.max()) + self.horizon})"
            )
        kk = min(k, legal_ids.size)
        if valid_idx.size:
            loc_ids, dists = topk_exact(qvecs[valid_idx], self.vecs[legal_ids], kk)
            ids = legal_ids[loc_ids]  # map local (filtered) ids back to global candidate ids
            # invariant check: every returned candidate satisfies rule 3 for every query origin
            leakage.assert_retrieval_horizon(
                int(self.db.t_r[ids].max()), self.horizon, int(origins.min())
            )
        else:
            ids = np.empty((0, kk), dtype=np.int64)
            dists = np.empty((0, kk), dtype=np.float64)
        return TopKResult(
            ids=ids, dists=dists, valid_idx=valid_idx, qparams=qparams, q_invalid=q_invalid
        )

    def forecast_from_topk(
        self, topk: TopKResult, queries: np.ndarray, restore: bool, k: int | None = None
    ) -> NativeRetrievalOutput:
        """Build the mean-of-continuations forecast from a precomputed :class:`TopKResult`.

        ``k`` slices the top candidates (``<=`` the k used in :meth:`retrieve`); invalid-scale
        queries get the constant context-mean fallback; restoration is skipped (kept raw and
        counted) for candidates whose own scale is invalid.
        """
        n = queries.shape[0]
        H = self.horizon  # noqa: N806 — L,H are standard TS notation
        point = np.empty((n, H), dtype=np.float64)
        nn_dist = np.full(n, np.inf, dtype=np.float64)
        disagreement = np.zeros(n, dtype=np.float64)
        invalid_restore = 0

        # invalid-scale queries -> constant context-mean fallback (counted)
        inv_q = np.flatnonzero(topk.q_invalid)
        fallback = int(inv_q.size)
        invalid_q = int(inv_q.size)
        if inv_q.size:
            point[inv_q] = queries[inv_q].mean(axis=1, keepdims=True)

        valid = topk.valid_idx
        if valid.size:
            kk = topk.ids.shape[1] if k is None else min(k, topk.ids.shape[1])
            ids = topk.ids[:, :kk]  # (m, kk) global candidate ids, distance-sorted
            conts = self.db.continuations[ids]  # (m, kk, H) candidate-scale
            if restore and self.scale != "raw":
                cparams = self.params[ids]  # (m, kk, 2)
                c_loc = cparams[:, :, 0:1]
                c_scale = cparams[:, :, 1:2]
                qp = topk.qparams[valid]  # (m, 2)
                q_loc = qp[:, None, 0:1]
                q_scale = qp[:, None, 1:2]
                restored = (conts - c_loc) / c_scale * q_scale + q_loc  # (m, kk, H)
                # candidates with invalid own scale keep their raw continuation (counted)
                cand_bad = self.cand_invalid[ids]  # (m, kk)
                invalid_restore = int(cand_bad.sum())
                conts = np.where(cand_bad[:, :, None], conts, restored)
            point[valid] = conts.mean(axis=1)
            nn_dist[valid] = topk.dists[:, :kk].min(axis=1)
            disagreement[valid] = conts.std(axis=1).mean(axis=1)

        return NativeRetrievalOutput(
            point=point,
            nn_dist=nn_dist,
            disagreement=disagreement,
            fallback_count=fallback,
            invalid_query_scale=invalid_q,
            invalid_cand_restore=invalid_restore,
            n_query=n,
        )

    def forecast_batch(
        self, queries: np.ndarray, origins: np.ndarray, k: int, restore: bool
    ) -> NativeRetrievalOutput:
        """Convenience: :meth:`retrieve` then :meth:`forecast_from_topk` in one call."""
        topk = self.retrieve(queries, origins, k)
        return self.forecast_from_topk(topk, queries, restore, k=k)


def fixed_fusion(
    chronos_point: np.ndarray, retrieval_point: np.ndarray, weight: float
) -> np.ndarray:
    """Fixed convex blend ``(1 - weight) * chronos + weight * retrieval`` (no learned gate).

    Both forecasts are in the same normalised space; no non-negativity clip (ETT values
    are real-valued, unlike M5 counts).
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    return (1.0 - weight) * chronos_point + weight * retrieval_point
