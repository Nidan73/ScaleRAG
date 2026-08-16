"""Forecast accuracy metrics (Phase 2, task 8).

Scale-dependent metrics (MASE, RMSSE) take the training in-sample actuals and
derive their denominator from the naive forecast **on training data only**
(docs/research-rules.md rule 5) — never from the evaluation horizon. The official-style M5
scale ignores the leading run of zeros before a series' first observed sale.

All functions operate on 1-D NumPy arrays for a single (possibly aggregated)
series. WRMSSE lives in ``scalerag.hierarchy`` since it needs the M5
hierarchy and dollar weights.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray | list[float]


def _asarray(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _asarray(y_true), _asarray(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Weighted absolute percentage error = sum|e| / sum|y|."""
    yt, yp = _asarray(y_true), _asarray(y_pred)
    denom = float(np.sum(np.abs(yt)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / denom)


def _trim_leading_zeros(insample: np.ndarray) -> np.ndarray:
    nz = np.flatnonzero(insample != 0)
    return insample[nz[0] :] if nz.size else insample


def naive_scale(
    insample: ArrayLike, seasonality: int = 1, *, squared: bool, trim_leading_zeros: bool = True
) -> float:
    """Mean (abs or squared) one-step/seasonal naive error on training data."""
    ins = _asarray(insample)
    if trim_leading_zeros:
        ins = _trim_leading_zeros(ins)
    if ins.size <= seasonality:
        return float("nan")
    diff = ins[seasonality:] - ins[:-seasonality]
    return float(np.mean(diff**2 if squared else np.abs(diff)))


def mase(y_true: ArrayLike, y_pred: ArrayLike, insample: ArrayLike, seasonality: int = 1) -> float:
    scale = naive_scale(insample, seasonality, squared=False)
    if not np.isfinite(scale) or scale == 0.0:
        return float("nan")
    return mae(y_true, y_pred) / scale


def rmsse(y_true: ArrayLike, y_pred: ArrayLike, insample: ArrayLike, seasonality: int = 1) -> float:
    scale = naive_scale(insample, seasonality, squared=True)
    if not np.isfinite(scale) or scale == 0.0:
        return float("nan")
    yt, yp = _asarray(y_true), _asarray(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2) / scale))


def pinball_loss(
    y_true: ArrayLike, quantile_preds: np.ndarray, quantile_levels: list[float]
) -> float:
    """Mean pinball (quantile) loss over horizon and quantile levels.

    ``y_true`` is (H,); ``quantile_preds`` is (H, Q) aligned with
    ``quantile_levels`` (length Q). Lower is better.
    """
    yt = _asarray(y_true)
    qp = np.asarray(quantile_preds, dtype=np.float64)
    if qp.shape != (yt.shape[0], len(quantile_levels)):
        raise ValueError(
            f"quantile_preds shape {qp.shape} != ({yt.shape[0]}, {len(quantile_levels)})"
        )
    losses = []
    for i, q in enumerate(quantile_levels):
        err = yt - qp[:, i]
        losses.append(np.mean(np.maximum(q * err, (q - 1) * err)))
    return float(np.mean(losses))


def summary(
    y_true: ArrayLike, y_pred: ArrayLike, insample: ArrayLike, seasonality: int = 1
) -> dict[str, float]:
    """Convenience bundle of the per-series metrics."""
    return {
        "mae": mae(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "mase": mase(y_true, y_pred, insample, seasonality),
        "rmsse": rmsse(y_true, y_pred, insample, seasonality),
    }
