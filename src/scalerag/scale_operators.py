"""Alternative scale operators for the sparse-end identifiability diagnostic.

The deployed retrieval restores a retrieved continuation to the query's scale
using the query window's own mean. On a window that is entirely zero that mean is
zero, and ``retrieval_faiss._fit_params`` substitutes a scale of exactly ``1.0``
(not the ``1e-8`` the manuscript declares). The restored continuation is then the
candidate rescaled to unit mean, on a series that has sold nothing.

At the M5 validation origin with L=56 this affects 995 series, every one of them
inside the sparsest zero-fraction bin. This module supplies three alternative
scale estimators so the sparse-end result can be attributed to the retrieval or
to the operator, plus an oracle upper bound.

Every operator returns ``(n, 2)`` params of ``(loc, scale)`` compatible with
``retrieval_faiss.restore_continuation``. Where a scale cannot be estimated the
operator returns ``NaN`` rather than substituting a constant: a window the
diagnostic cannot resolve must be counted, not hidden.

``oracle_future`` uses the realised future and is therefore not a forecaster. It
is a diagnostic upper bound only and refuses to run unless explicitly allowed
(rule 12).
"""

from __future__ import annotations

import numpy as np

from scalerag.retrieval_faiss import _fit_params

__all__ = [
    "OPERATORS",
    "degenerate_mask",
    "hierarchy_prior",
    "long_lookback",
    "oracle_future_mean",
    "oracle_least_squares",
    "restore_batch",
    "window_mean",
]

OPERATORS: dict[str, str] = {
    "window_mean": "deployed: mean of the L-day retrieval context, 1.0 substituted at zero",
    "long_lookback": "mean over a longer history window than the retrieval context",
    "hierarchy_prior": "mean over the series' hierarchy group, pooling across members",
    "oracle_future_mean": (
        "moment matching on the realised future's mean; oracle, and NOT a bound"
    ),
    "oracle_least_squares": (
        "scale minimising squared error against the realised future; a true bound, never deployable"
    ),
}


def _as2d(name: str, arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_series, n_steps), got shape {out.shape}")
    return out


def _params(scale: np.ndarray) -> np.ndarray:
    """Pack a scale vector into (loc, scale) params with a zero location.

    Every operator here is scale-only, matching the deployed rule: there is no
    location term. Non-positive scales are unusable and become NaN.
    """
    s = np.asarray(scale, dtype=np.float64)
    s = np.where(s > 0.0, s, np.nan)
    return np.stack([np.zeros_like(s), s], axis=1)


def degenerate_mask(contexts: np.ndarray) -> np.ndarray:
    """True where the retrieval context has a zero mean, so the scale is unidentifiable."""
    return _as2d("contexts", contexts).mean(axis=1) == 0.0


def window_mean(contexts: np.ndarray) -> np.ndarray:
    """The deployed operator, delegated so it cannot drift from the frozen path."""
    return _fit_params(_as2d("contexts", contexts), "mean")


def long_lookback(history: np.ndarray, origin: int, lookback: int = 365) -> np.ndarray:
    """Scale from a longer history window than the retrieval context uses.

    A series can be empty over 56 days and active over 365. Uses only days before
    ``origin``, so it introduces no leakage.
    """
    h = _as2d("history", history)
    if origin > h.shape[1]:
        raise ValueError(f"origin {origin} exceeds history length {h.shape[1]}")
    start = max(0, origin - lookback)
    if start >= origin:
        raise ValueError(f"empty lookback window for origin {origin}, lookback {lookback}")
    return _params(h[:, start:origin].mean(axis=1))


def hierarchy_prior(
    history: np.ndarray, origin: int, group_ids: np.ndarray, lookback: int = 56
) -> np.ndarray:
    """Scale from the mean of the series' hierarchy group over the same window.

    A dead series inherits the level of its group, which is the treatment RAID
    uses at ``L = 0`` (Appendix D.3). Uses only days before ``origin``.
    """
    h = _as2d("history", history)
    g = np.asarray(group_ids)
    if g.shape[0] != h.shape[0]:
        raise ValueError(f"{g.shape[0]} group ids for {h.shape[0]} series")
    if origin > h.shape[1]:
        raise ValueError(f"origin {origin} exceeds history length {h.shape[1]}")
    start = max(0, origin - lookback)
    if start >= origin:
        raise ValueError(f"empty lookback window for origin {origin}, lookback {lookback}")

    window = h[:, start:origin]
    per_series_sum = window.sum(axis=1)
    per_series_n = float(window.shape[1])

    uniq, inv = np.unique(g, return_inverse=True)
    group_sum = np.zeros(uniq.size, dtype=np.float64)
    group_n = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(group_sum, inv, per_series_sum)
    np.add.at(group_n, inv, per_series_n)
    group_mean = np.divide(group_sum, group_n, out=np.full(uniq.size, np.nan), where=group_n > 0)
    return _params(group_mean[inv])


def oracle_future_mean(future: np.ndarray, *, allow_oracle: bool = False) -> np.ndarray:
    """Scale matching the realised future's mean. An oracle, but NOT a bound.

    Moment matching is not the squared-error minimiser, so this cannot bound the
    error of any scale rule. It answers a different question: what would a rule
    that got the level exactly right have achieved. Use
    :func:`oracle_least_squares` for the bound.

    Refuses to run unless ``allow_oracle=True`` (rule 12).
    """
    if not allow_oracle:
        raise ValueError(
            "oracle_future_mean uses the realised future and is a diagnostic only; "
            "pass allow_oracle=True to acknowledge that it is not a forecaster"
        )
    return _params(_as2d("future", future).mean(axis=1))


def oracle_least_squares(
    unit_prediction: np.ndarray, future: np.ndarray, *, allow_oracle: bool = False
) -> np.ndarray:
    """The scale-only least-squares minimiser: a genuine lower bound on error.

    For a unit-scale prediction ``u`` and realised future ``y``, the scale
    minimising ``||s*u - y||^2`` is ``<u, y> / <u, u>``. No scale rule of the
    deployed form can beat it for this retrieved set, so whatever error survives is
    the analogues having the wrong shape.

    A scale of exactly zero is legitimate — it is the best a scale-only rule can do
    when the future is all zero — and is preserved rather than mapped to NaN, so
    this operator does not condition on the outcome by dropping those series.

    An identically-zero prediction (``<u, u> = 0``) is also resolvable: it predicts
    zero under every scale, so the scale is arbitrary rather than missing, and 1.0
    is returned. Only a prediction that is not finite at all — no candidate was
    retrieved — is genuinely unresolvable and returns NaN. Dropping the zero-
    prediction case instead would silently remove the sparsest series, which is
    where this diagnostic is aimed.

    Refuses to run unless ``allow_oracle=True`` (rule 12).
    """
    if not allow_oracle:
        raise ValueError(
            "oracle_least_squares uses the realised future and is a diagnostic bound only; "
            "pass allow_oracle=True to acknowledge that it is not a forecaster"
        )
    u = _as2d("unit_prediction", unit_prediction)
    y = _as2d("future", future)
    if u.shape != y.shape:
        raise ValueError(f"unit_prediction {u.shape} and future {y.shape} must align")
    finite_row = np.isfinite(u).all(axis=1)
    denom = np.sum(u * u, axis=1, where=finite_row[:, None], initial=0.0)
    numer = np.sum(u * y, axis=1, where=finite_row[:, None], initial=0.0)
    scale = np.divide(numer, denom, out=np.full(denom.shape, np.nan), where=denom > 0)
    # A zero prediction is scale-invariant, so the scale is arbitrary, not missing.
    scale = np.where(finite_row & (denom == 0.0), 1.0, scale)
    scale = np.where(~finite_row, np.nan, scale)
    scale = np.where(np.isfinite(scale) & (scale < 0.0), 0.0, scale)
    return np.stack([np.zeros_like(scale), scale], axis=1)


def restore_batch(
    continuations: np.ndarray, cand_params: np.ndarray, query_params: np.ndarray
) -> np.ndarray:
    """Restore a stack of candidate continuations to one query's scale.

    Vectorised equivalent of calling ``retrieval_faiss.restore_continuation`` per
    candidate. A NaN query scale propagates, so an unresolvable window is visibly
    NaN downstream rather than silently defaulted.
    """
    c = _as2d("continuations", continuations)
    p = _as2d("cand_params", cand_params)
    if p.shape != (c.shape[0], 2):
        raise ValueError(f"cand_params shape {p.shape} does not match {c.shape[0]} continuations")
    q = np.asarray(query_params, dtype=np.float64)
    if q.shape != (2,):
        raise ValueError(f"query_params must be shape (2,), got {q.shape}")
    c_loc, c_scale = p[:, 0:1], p[:, 1:2]
    q_loc, q_scale = q[0], q[1]
    return (c - c_loc) / c_scale * q_scale + q_loc
