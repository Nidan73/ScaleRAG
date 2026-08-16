"""Metric formula + hierarchy/WRMSSE tests (Phase 2, task 8-9)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scalerag import hierarchy, metrics

Y = [3.0, 1.0, 4.0]
YH = [2.0, 1.0, 5.0]
INS = [1.0, 2.0, 3.0, 4.0]


@pytest.mark.unit
def test_mae() -> None:
    assert metrics.mae(Y, YH) == pytest.approx(2 / 3)


@pytest.mark.unit
def test_wape() -> None:
    assert metrics.wape(Y, YH) == pytest.approx(0.25)
    assert np.isnan(metrics.wape([0, 0], [1, 2]))  # zero denom → nan


@pytest.mark.unit
def test_mase() -> None:
    assert metrics.mase(Y, YH, INS, seasonality=1) == pytest.approx(2 / 3)


@pytest.mark.unit
def test_rmsse() -> None:
    assert metrics.rmsse(Y, YH, INS, seasonality=1) == pytest.approx(np.sqrt(2 / 3))


@pytest.mark.unit
def test_rmsse_trims_leading_zeros() -> None:
    # Leading zeros (pre-sale) must not deflate the scale.
    assert metrics.naive_scale([0, 0, 1, 2, 3], squared=True) == pytest.approx(1.0)


@pytest.mark.unit
def test_scale_zero_returns_nan() -> None:
    assert np.isnan(metrics.rmsse([1, 2], [1, 2], [5, 5, 5]))  # flat insample → scale 0


def _tiny_entities() -> pl.DataFrame:
    rows = []
    data = [
        ("i1", "CA", "CA_1"),
        ("i2", "CA", "CA_1"),
        ("i1", "TX", "TX_1"),
        ("i2", "TX", "TX_1"),
    ]
    for item, state, store in data:
        rows.append(
            {
                "id": f"{item}_{store}",
                "item_id": item,
                "dept_id": "D1",
                "cat_id": "C1",
                "store_id": store,
                "state_id": state,
            }
        )
    return pl.DataFrame(rows)


@pytest.mark.unit
def test_aggregate_rows_total_equals_column_sum() -> None:
    mat = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    gid = np.zeros(4, dtype=np.int64)  # everything into one group
    agg = hierarchy.aggregate_rows(mat, gid, 1)
    assert np.allclose(agg[0], mat.sum(axis=0))


@pytest.mark.unit
def test_wrmsse_perfect_forecast_is_zero() -> None:
    ent = _tiny_entities()
    rng = np.random.default_rng(0)
    train = rng.integers(1, 10, size=(4, 60)).astype(float)
    actual = rng.integers(1, 10, size=(4, 28)).astype(float)
    weights = hierarchy.dollar_weights(train[:, -28:], np.ones((4, 28)))
    score, per_level = hierarchy.wrmsse(ent, train, actual, actual.copy(), weights)
    assert score == pytest.approx(0.0)
    assert len(per_level) == 12


@pytest.mark.unit
def test_wrmsse_worse_forecast_scores_higher() -> None:
    ent = _tiny_entities()
    rng = np.random.default_rng(1)
    train = rng.integers(1, 10, size=(4, 60)).astype(float)
    actual = rng.integers(1, 10, size=(4, 28)).astype(float)
    weights = hierarchy.dollar_weights(train[:, -28:], np.ones((4, 28)))
    good = actual + rng.normal(0, 0.5, actual.shape)
    bad = actual + rng.normal(0, 5.0, actual.shape)
    s_good, _ = hierarchy.wrmsse(ent, train, actual, good, weights)
    s_bad, _ = hierarchy.wrmsse(ent, train, actual, bad, weights)
    assert s_bad > s_good >= 0
