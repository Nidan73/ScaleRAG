"""ScaleRAG-TS: scale-aware retrieval + uncertainty-aware gated fusion (Phase 9).

Model-agnostic fusion of a frozen target-only Chronos-2 forecast with a
scale-restored retrieved-continuation forecast. A small learned gate predicts,
per series, how much to trust retrieval vs Chronos-2, from uncertainty/reliability
features. No graph or relational features (rejected in Phases 6-8). No retrieved
future labels enter the Chronos input.

Gate is trained on **historical** origins only; the leakage invariant
``candidate_end + H < target_forecast_origin`` is enforced by the retriever.
"""

from __future__ import annotations

import numpy as np

GATE_FEATURES = [
    "retr_nn_dist",  # distance to nearest retrieved neighbour (retrieval confidence)
    "retr_disagreement",  # spread across retrieved continuations
    "intermittency",  # context zero-fraction
    "log_volume",  # demand level
    "chronos_uncertainty",  # Chronos q90-q10 spread
    "scale_spread",  # context coefficient of variation
]


def gate_feature_row(
    retr_nn_dist: float,
    retr_disagreement: float,
    intermittency: float,
    log_volume: float,
    chronos_uncertainty: float,
    scale_spread: float,
) -> list[float]:
    return [
        retr_nn_dist,
        retr_disagreement,
        intermittency,
        log_volume,
        chronos_uncertainty,
        scale_spread,
    ]


def fuse(
    chronos_point: np.ndarray,
    chronos_quants: np.ndarray,
    retr_point: np.ndarray,
    retr_quants: np.ndarray,
    alpha: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convex blend with retrieval weight ``alpha`` (scalar or per-series array).

    Point forecasts are (n, H); quantiles (n, H, Q). A per-series ``alpha`` (n,)
    must broadcast over the horizon (and quantile) axes — not the horizon itself.
    """
    a = np.asarray(alpha, dtype=float)
    if a.ndim == 1:  # per-series
        ap, aq = a[:, None], a[:, None, None]
    else:  # scalar
        ap = aq = a
    point = np.clip((1 - ap) * chronos_point + ap * retr_point, 0.0, None)
    quants = np.clip((1 - aq) * chronos_quants + aq * retr_quants, 0.0, None)
    return point, quants


def paired_bootstrap_rel_improvement(
    loss_baseline: np.ndarray,
    loss_method: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Relative improvement of ``method`` over ``baseline`` (per-series losses),
    with a percentile paired bootstrap 95% CI over series. Positive = better."""
    rng = np.random.default_rng(seed)
    b = np.asarray(loss_baseline, float)
    m = np.asarray(loss_method, float)
    ok = np.isfinite(b) & np.isfinite(m)
    b, m = b[ok], m[ok]
    n = len(b)
    base_mean = b.mean()
    point = (base_mean - m.mean()) / base_mean
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        bb, mm = b[idx].mean(), m[idx].mean()
        boots[i] = (bb - mm) / bb
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "rel_improvement": float(point),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n": int(n),
    }
