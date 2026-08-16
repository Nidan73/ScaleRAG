"""LightGBM baseline (Phase 2, task 7).

Trains one gradient-boosted regressor on the leakage-safe features
(``scalerag.features``) using training-day rows only, then predicts all 28
horizon days directly (lags >= horizon make this leakage-safe). Predictions are
clipped at 0 (sales are non-negative). Deterministic given ``seed``.
"""

from __future__ import annotations

import gc
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from scalerag.features import FEATURE_COLS, TARGET, horizon_matrix, train_matrix
from scalerag.splits import RollingSplit

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "objective": "tweedie",  # good for intermittent retail demand
    "tweedie_variance_power": 1.1,
    "verbose": -1,
}


def fit_predict(
    features: pl.DataFrame,
    split: RollingSplit,
    *,
    params: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[pl.DataFrame, lgb.LGBMRegressor]:
    """Return (predictions ``(id, day_idx, y_pred)``, fitted model)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    tr = train_matrix(features, split)
    hz = horizon_matrix(features, split)
    if tr.height == 0:
        raise ValueError(f"{split.name}: no training rows with defined lags.")

    # float32 matrices (half the memory of float64) — matters at full-M5 scale
    # (~56M training rows). Free the Polars training frame before fitting.
    x_tr = tr.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y_tr = tr.select(TARGET).to_numpy().ravel().astype(np.float32)
    del tr
    gc.collect()

    with warnings.catch_warnings():
        # numpy input has no feature names — cosmetic sklearn warning only.
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        model = lgb.LGBMRegressor(random_state=seed, **p)
        model.fit(x_tr, y_tr)

        ids_days = hz.select("id", "day_idx")
        x_hz = hz.select(FEATURE_COLS).to_numpy().astype(np.float32)
        y_hat = np.clip(model.predict(x_hz), a_min=0.0, a_max=None)
    preds = ids_days.with_columns(pl.Series("y_pred", y_hat))
    return preds, model
