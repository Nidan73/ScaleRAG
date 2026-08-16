"""Tests for pinball loss, k-NN forecast, and late fusion (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from scalerag import metrics
from scalerag.retrieval import EuclideanRetriever, WindowDatabase
from scalerag.retrieval_forecast import knn_forecast, late_fusion

QL = [0.1, 0.5, 0.9]


@pytest.mark.unit
def test_pinball_known_value() -> None:
    # y=2, preds [1,2,3] at q=[.1,.5,.9] -> (0.1 + 0 + 0.1)/3
    val = metrics.pinball_loss([2.0], np.array([[1.0, 2.0, 3.0]]), QL)
    assert val == pytest.approx((0.1 + 0.0 + 0.1) / 3)


@pytest.mark.unit
def test_pinball_zero_when_perfect() -> None:
    yt = np.array([3.0, 1.0])
    qp = np.array([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]])  # all quantiles == truth
    assert metrics.pinball_loss(yt, qp, QL) == pytest.approx(0.0)


@pytest.mark.unit
def test_pinball_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        metrics.pinball_loss([1.0, 2.0], np.array([[1.0, 2.0, 3.0]]), QL)


@pytest.mark.unit
def test_late_fusion_blend() -> None:
    a = np.array([2.0, 4.0])
    b = np.array([0.0, 0.0])
    qa = np.ones((2, 3))
    qb = np.zeros((2, 3))
    p, q = late_fusion(a, qa, b, qb, alpha=0.25)
    assert np.allclose(p, [0.5, 1.0])
    assert np.allclose(q, 0.25)


@pytest.mark.unit
def test_late_fusion_alpha_bounds() -> None:
    with pytest.raises(ValueError):
        late_fusion(np.zeros(2), np.zeros((2, 1)), np.zeros(2), np.zeros((2, 1)), alpha=1.5)


@pytest.mark.unit
def test_knn_forecast_averages_continuations() -> None:
    # series with clean weekly pattern → euclidean picks similar windows
    rng = np.random.default_rng(0)
    t = np.arange(300)
    series = (np.sin(2 * np.pi * t / 7)[None, :] + rng.normal(0, 0.1, (3, 300))).astype(float)
    db = WindowDatabase.from_training(series, train_end=200, context_length=14, horizon=7)
    q = series[0, 200 - 14 : 200]
    point, quants = knn_forecast(
        db, EuclideanRetriever(), q, origin=220, horizon=7, k=5, series_idx=0, quantile_levels=QL
    )
    assert point.shape == (7,)
    assert quants.shape == (7, 3)
    assert np.all(point >= 0)  # clipped


@pytest.mark.unit
def test_knn_forecast_empty_uses_fallback() -> None:
    empty = WindowDatabase.from_training(
        np.zeros((2, 30)), train_end=10, context_length=14, horizon=7
    )
    q = np.array([5.0] * 14)
    point, quants = knn_forecast(
        empty, EuclideanRetriever(), q, origin=5, horizon=7, k=5, series_idx=0, quantile_levels=QL
    )
    assert point.shape == (7,) and np.allclose(point, 5.0)  # fallback = mean(query)
    assert quants.shape == (7, 3)
