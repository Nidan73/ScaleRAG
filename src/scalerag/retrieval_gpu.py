"""GPU batched **exact** scale-aware temporal retrieval (Phase 10, scale-up only).

This is a faster arithmetic path for the *same* frozen retriever used in Phases 5
and 9 (``retrieval_faiss.ScaleAwareIndex`` with ``scale="mean"``, ``metric="l2"``,
``meta_filter="cat_id"``). It exists purely to make the **full 30,490-series M5
panel** tractable for the single locked held-out test: the frozen numpy path is
O(n_query x n_candidate) with a per-query copy and takes ~5 h per origin at full
panel, which is not feasible.

**No method change.** The nearest neighbours, scale restoration, and forecasts are
identical to the frozen path (verified bit-for-bit on the 1,000-series validation
subset by ``scripts/verify_gpu_retrieval.py`` before any test run). The GPU does
only a coarse over-fetch by float32 L2 distance; the final top-k and the frozen
tie-break (ascending global candidate id) are resolved on CPU in float32, exactly
as ``retrieval_faiss`` does. Continuation restoration is float64, as frozen.

The leakage invariant ``candidate_end + H < origin`` (rule 3) is enforced by the
window construction (``t_r + H <= train_end < origin``) and asserted here.
"""

# ruff: noqa: N806  # L,H (time-series) and C,Q,Cn (linear-algebra) are standard notation
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scalerag import leakage


@dataclass(frozen=True)
class GpuRetrievalResult:
    """Per-series retrieval forecast at one origin (shapes keyed to n queries)."""

    point: np.ndarray  # (n, H)
    quants: np.ndarray  # (n, H, Q)
    nn_dist: np.ndarray  # (n,) min transformed-space L2 distance
    disagreement: np.ndarray  # (n,) mean over horizon of std across restored conts


def build_windows(
    sales: np.ndarray, train_end: int, context_length: int, horizon: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised, order-identical replacement for ``WindowDatabase.from_training``.

    Returns ``(contexts[Nc, L] f64, continuations[Nc, H] f64, series_idx[Nc] i64,
    t_r[Nc] i64)`` in series-major, ascending-``t_r`` order — the exact build order
    of ``WindowDatabase`` (so a later lexsort tie-break on global id matches).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n, t = sales.shape
    L, H = context_length, horizon
    if train_end > t:
        raise ValueError(f"train_end {train_end} exceeds series length {t}.")
    # t_r (1-based end of context): context = row[t_r-L:t_r], continuation = row[t_r:t_r+H].
    # Legal build: L <= t_r <= train_end - H  (=> t_r + H <= train_end).
    cand_trs = np.arange(L, train_end - H + 1, stride, dtype=np.int64)
    if cand_trs.size == 0:
        return (
            np.empty((0, L), np.float64),
            np.empty((0, H), np.float64),
            np.empty(0, np.int64),
            np.empty(0, np.int64),
        )
    starts = cand_trs - L  # window start index into the row
    # all consecutive (L+H)-length windows over the training region
    train = sales[:, :train_end]
    win = sliding_window_view(train, L + H, axis=1)  # (n, train_end-(L+H)+1, L+H)
    sel = win[:, starts, :]  # (n, W, L+H)
    contexts = sel[:, :, :L].reshape(-1, L).astype(np.float64)
    continuations = sel[:, :, L:].reshape(-1, H).astype(np.float64)
    w = starts.size
    series_idx = np.repeat(np.arange(n, dtype=np.int64), w)
    t_r = np.tile(cand_trs, n)
    return (
        np.ascontiguousarray(contexts),
        np.ascontiguousarray(continuations),
        series_idx,
        t_r,
    )


def _mean_scale(x: np.ndarray) -> np.ndarray:
    """Per-row mean with 0 -> 1 (matches ``retrieval_faiss._fit_params('mean')``)."""
    m = x.mean(1)
    return np.where(m == 0.0, 1.0, m)


def retrieve_scaleaware_gpu(
    sales: np.ndarray,
    cat_codes: np.ndarray,
    train_end: int,
    origin: int,
    queries: np.ndarray,
    query_cat: np.ndarray,
    quantile_levels: list[float],
    *,
    context_length: int = 56,
    horizon: int = 28,
    stride: int = 7,
    k: int = 20,
    device: str = "cuda",
) -> GpuRetrievalResult:
    """Exact k-NN scale-aware retrieval forecast, GPU-accelerated coarse ranking.

    Equivalent to ``retrieval_faiss.ScaleAwareIndex(scale='mean', metric='l2')``
    searched with ``meta_filter='cat_id'`` at ``origin`` over a training window
    ``train_end``. ``queries`` is ``(n, L)`` and ``query_cat`` is ``(n,)``.
    """
    import torch

    L, H = context_length, horizon
    ctx, cont, series_idx, t_r = build_windows(sales, train_end, L, H, stride)
    if ctx.shape[0] == 0:
        raise ValueError("no candidate windows built; check train_end/context length")
    # frozen leakage guard: every built candidate must satisfy t_r + H < origin
    leakage.assert_retrieval_horizon(int(t_r.max()), H, origin)

    cand_scale = _mean_scale(ctx)  # (Nc,) f64
    cand_cat = cat_codes[series_idx]  # (Nc,)
    vecs = (ctx / cand_scale[:, None]).astype(np.float32)  # transformed space (frozen)
    del ctx

    n = queries.shape[0]
    q_scale = _mean_scale(queries)  # (n,) f64
    qv = (queries / q_scale[:, None]).astype(np.float32)  # (n, L)

    over = max(k * 4, k + 64)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    point = np.zeros((n, H), np.float64)
    quants = np.zeros((n, H, len(quantile_levels)), np.float64)
    nn_dist = np.full(n, 1e6, np.float64)
    disagreement = np.zeros(n, np.float64)

    uniq_cats = np.unique(cand_cat)
    for c in uniq_cats:
        q_in_cat = np.flatnonzero(query_cat == c)
        if q_in_cat.size == 0:
            continue
        cand_ids = np.flatnonzero(cand_cat == c)  # global ids, ascending
        if cand_ids.size == 0:
            continue
        C = torch.from_numpy(vecs[cand_ids]).to(dev)  # (Mc, L)
        Cn = (C * C).sum(1)  # (Mc,)
        take = min(over, cand_ids.size)
        # query blocks to bound the (Qb x Mc) distance matrix
        qblk = max(1, min(q_in_cat.size, 4_000_000 // max(1, cand_ids.size)))
        for start in range(0, q_in_cat.size, qblk):
            qsl = q_in_cat[start : start + qblk]
            Q = torch.from_numpy(qv[qsl]).to(dev)  # (b, L)
            # ||q-c||^2 = ||q||^2 + ||c||^2 - 2 q.c  (coarse GPU ranking, float32)
            d = (Q * Q).sum(1)[:, None] + Cn[None, :] - 2.0 * (Q @ C.T)
            _, top = torch.topk(d, take, dim=1, largest=False)  # (b, take)
            top_np = top.cpu().numpy()
            del Q, d, top
            for row, qi in enumerate(qsl):
                shortlist = cand_ids[top_np[row]]  # global ids
                # frozen finalize: exact float32 L2 on the shortlist, tie-break by id asc
                sub = vecs[shortlist]
                dd = ((sub - qv[qi]) ** 2).sum(1)
                order = np.lexsort((shortlist, dd))[:k]
                ids = shortlist[order]
                dists = dd[order]
                conts = cont[ids]
                # scale restoration (mean: loc=0) in float64, as frozen
                restored = np.clip(conts / cand_scale[ids, None] * q_scale[qi], 0.0, None)
                point[qi] = restored.mean(0)
                quants[qi] = np.quantile(restored, quantile_levels, axis=0).T
                nn_dist[qi] = float(dists.min())
                disagreement[qi] = float(restored.std(0).mean())
        del C, Cn
    return GpuRetrievalResult(point, quants, nn_dist, disagreement)
