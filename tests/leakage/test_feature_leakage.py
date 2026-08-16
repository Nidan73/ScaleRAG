"""Feature-leakage guard (Phase 2, task 9).

Corrupting the *horizon* target must not change any training or horizon feature,
nor any train-fitted statistic. If it did, the model could see the future.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scalerag.baselines import seasonal_naive
from scalerag.data.m5_ingest import ingest_m5
from scalerag.data.synthetic import generate_m5
from scalerag.eval import load_processed
from scalerag.features import FEATURE_COLS, build_features
from scalerag.splits import RollingSplit

N = 120
SPLIT = RollingSplit("test", train_end=92, horizon=28)  # horizon 93..120


@pytest.fixture
def processed(tmp_path):
    raw = generate_m5(tmp_path / "raw", n_days=N, seed=3)
    out = tmp_path / "processed"
    ingest_m5(raw.root, out, n_days=N)
    return load_processed(out)


@pytest.mark.leakage
def test_corrupting_horizon_target_does_not_change_features(processed) -> None:
    entities, dynamic = processed
    feats_clean, stats_clean = build_features(dynamic, entities, SPLIT)

    # Blow up every horizon-day sale; features must be identical.
    corrupted = dynamic.with_columns(
        pl.when(pl.col("day_idx") > SPLIT.train_end)
        .then(pl.lit(99999))
        .otherwise(pl.col("sales"))
        .alias("sales")
    )
    feats_corrupt, stats_corrupt = build_features(corrupted, entities, SPLIT)

    ca = feats_clean.sort(["id", "day_idx"])
    cb = feats_corrupt.sort(["id", "day_idx"])

    # Sales-derived features are the leakage signal: they MUST be bit-identical.
    # (A real leak would inject the corrupted 99999 values — a huge, obvious diff.)
    sales_derived = [
        "lag_28",
        "lag_35",
        "lag_42",
        "rmean_7_lag28",
        "rmean_28_lag28",
        "series_train_mean",
    ]
    assert ca.select(sales_derived).equals(cb.select(sales_derived)), (
        "horizon target leaked into sales-derived features"
    )

    # Sales-independent features (calendar, price) match within float tolerance.
    for c in [c for c in FEATURE_COLS if c not in sales_derived]:
        assert np.allclose(ca[c].to_numpy(), cb[c].to_numpy(), atol=1e-9, equal_nan=True), c

    # Train-fitted stats are unaffected by horizon values (tolerant of ULP noise).
    for col, frame_c, frame_k in [
        (
            "item_price_mean",
            stats_clean.item_price_mean.sort("item_id"),
            stats_corrupt.item_price_mean.sort("item_id"),
        ),
        (
            "series_train_mean",
            stats_clean.series_train_mean.sort("id"),
            stats_corrupt.series_train_mean.sort("id"),
        ),
    ]:
        assert np.allclose(frame_c[col].to_numpy(), frame_k[col].to_numpy(), atol=1e-9)


@pytest.mark.leakage
def test_seasonal_naive_ignores_horizon(processed) -> None:
    _entities, dynamic = processed
    clean = seasonal_naive.predict(dynamic, SPLIT).sort(["id", "day_idx"])
    corrupted = dynamic.with_columns(
        pl.when(pl.col("day_idx") > SPLIT.train_end)
        .then(pl.lit(99999))
        .otherwise(pl.col("sales"))
        .alias("sales")
    )
    after = seasonal_naive.predict(corrupted, SPLIT).sort(["id", "day_idx"])
    assert clean.equals(after)
