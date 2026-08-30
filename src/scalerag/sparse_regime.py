"""Stratified retrieval-utility profile over fixed intermittency bins.

The regime study reports a win rate per zero-fraction bin. In the sparsest bin
that number is a mixture: at the M5 validation origin 26.37% of those contexts
have a zero mean, so their scale is unidentifiable and the restored forecast is
produced by a fallback rather than by retrieval. A single pooled rate cannot
distinguish "retrieval does not help here" from "the operator decided this".

This module splits every bin into the degenerate and identifiable cohorts and
reports both, alongside the effect size the win rate discards: two bins at 0.52
can carry very different mean gains.

Bins are **fixed absolute intervals**, not quantile deciles, so a bin denotes the
same thing at every origin and rates can be pooled across origins. This is the
design the multi-origin band relies on; the paper must use fixed-bin language
throughout.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FIXED_BIN_EDGES", "bin_index", "stratified_bin_summary"]

FIXED_BIN_EDGES = np.round(np.arange(0.0, 1.01, 0.1), 2)


def bin_index(intermittency: np.ndarray) -> np.ndarray:
    """Index of the fixed bin each value falls in. The last bin is closed."""
    x = np.asarray(intermittency, dtype=np.float64)
    if x.size and (x.min() < 0.0 or x.max() > 1.0):
        raise ValueError(f"intermittency must lie within [0, 1], got [{x.min()}, {x.max()}]")
    n_bins = FIXED_BIN_EDGES.size - 1
    return np.clip(np.floor(x * n_bins).astype(np.int64), 0, n_bins - 1)


def _rate(values: np.ndarray) -> float:
    return float((values > 0.0).mean()) if values.size else float("nan")


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def _median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else float("nan")


def stratified_bin_summary(
    intermittency: np.ndarray,
    utility: np.ndarray,
    delta: np.ndarray,
    degenerate: np.ndarray,
    *,
    fused_utility: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Per fixed bin, the win rate and effect size split by cohort.

    ``utility`` is the retrieval branch's advantage over the backbone; ``delta``
    is the signed per-series metric difference behind it. ``degenerate`` marks the
    series whose scale was unidentifiable. ``fused_utility``, when given, is the
    same statistic for the shipped fused system, which is what the deployment
    recommendation is actually about.

    Empty bins are omitted; an empty cohort inside a present bin reports NaN
    rather than removing the bin, so a thin stratum stays visible.
    """
    x = np.asarray(intermittency, dtype=np.float64)
    u = np.asarray(utility, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    g = np.asarray(degenerate, dtype=bool)
    if not (x.shape == u.shape == d.shape == g.shape):
        raise ValueError(f"inputs must align, got {x.shape}, {u.shape}, {d.shape}, {g.shape}")
    f = None
    if fused_utility is not None:
        f = np.asarray(fused_utility, dtype=np.float64)
        if f.shape != x.shape:
            raise ValueError(f"fused_utility must align, got {f.shape} and {x.shape}")

    idx = bin_index(x)
    rows: list[dict[str, float]] = []
    for b in range(FIXED_BIN_EDGES.size - 1):
        m = idx == b
        if not m.any():
            continue
        deg, ident = m & g, m & ~g
        row: dict[str, float] = {
            "lo": float(FIXED_BIN_EDGES[b]),
            "hi": float(FIXED_BIN_EDGES[b + 1]),
            "n": int(m.sum()),
            "n_degenerate": int(deg.sum()),
            "n_identifiable": int(ident.sum()),
            "degenerate_fraction": float(g[m].mean()),
            "win_rate_all": _rate(u[m]),
            "win_rate_degenerate": _rate(u[deg]),
            "win_rate_identifiable": _rate(u[ident]),
            "mean_delta_all": _mean(d[m]),
            "median_delta_all": _median(d[m]),
            "mean_delta_identifiable": _mean(d[ident]),
            "median_delta_identifiable": _median(d[ident]),
        }
        if f is not None:
            row["fused_win_rate_all"] = _rate(f[m])
            row["fused_win_rate_identifiable"] = _rate(f[ident])
        rows.append(row)
    return rows
