"""Evaluation harness: run a baseline on a split and score it (Phase 2).

Loads processed Parquet, builds bottom-level actual/price matrices ordered
consistently with the entities, runs a baseline, then computes per-series metrics
and the official-style WRMSSE. Captures a reproducibility fingerprint for every
run (CLAUDE.md rule 10). No metric is tuned here and evaluation never touches
days beyond the split horizon.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from scalerag import hierarchy, metrics
from scalerag.baselines import lightgbm_model, seasonal_naive
from scalerag.features import build_features
from scalerag.reproducibility import RunContext, set_seed
from scalerag.splits import RollingSplit


def load_processed(processed_dir: str | Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    processed = Path(processed_dir)
    entities = pl.read_parquet(processed / "entities.parquet").sort("id")
    dynamic = pl.read_parquet(processed / "dynamic.parquet").sort(["id", "day_idx"])
    return entities, dynamic


def select_subset(
    entities: pl.DataFrame, dynamic: pl.DataFrame, n_series: int, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deterministically pick a small subset of series (declared smoke runs)."""
    ids = entities["id"].sort()
    if n_series >= ids.len():
        chosen = ids
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(ids.len(), size=n_series, replace=False))
        chosen = ids.gather(idx)
    ent = entities.filter(pl.col("id").is_in(chosen)).sort("id")
    dyn = dynamic.filter(pl.col("id").is_in(chosen)).sort(["id", "day_idx"])
    return ent, dyn


def _matrix(dynamic: pl.DataFrame, col: str, n_series: int, n_days: int) -> np.ndarray:
    if dynamic.height != n_series * n_days:
        raise ValueError(
            f"Expected a full {n_series}x{n_days} grid ({n_series * n_days} rows), "
            f"got {dynamic.height}. Ingestion should produce a dense panel."
        )
    return dynamic[col].to_numpy().astype(np.float64).reshape(n_series, n_days)


def _prediction_matrix(preds: pl.DataFrame, n_series: int, split: RollingSplit) -> np.ndarray:
    if preds.height != n_series * split.horizon:
        raise ValueError(
            f"Expected {n_series * split.horizon} prediction rows, got {preds.height}."
        )
    ordered = preds.sort(["id", "day_idx"])
    return ordered["y_pred"].to_numpy().astype(np.float64).reshape(n_series, split.horizon)


def evaluate(
    processed_dir: str | Path,
    split: RollingSplit,
    baseline: str,
    *,
    subset: int | None = None,
    seed: int = 42,
    params: dict[str, Any] | None = None,
) -> dict:
    """Run a baseline on one split and return a full metrics + repro report."""
    set_seed(seed)
    entities, dynamic = load_processed(processed_dir)
    if subset is not None:
        entities, dynamic = select_subset(entities, dynamic, subset, seed)

    n_series = entities.height
    n_days = int(dynamic["day_idx"].max())  # type: ignore[arg-type]
    sales = _matrix(dynamic, "sales", n_series, n_days)
    price = _matrix(dynamic, "sell_price", n_series, n_days)

    if baseline == "seasonal_naive":
        preds = seasonal_naive.predict(dynamic, split)
        model_info = {"kind": "seasonal_naive", "season": 7}
    elif baseline == "lightgbm":
        features, _stats = build_features(dynamic, entities, split)
        del dynamic  # 59M-row panel no longer needed; matrices already built
        gc.collect()
        preds, _model = lightgbm_model.fit_predict(features, split, params=params, seed=seed)
        del features
        gc.collect()
        model_info = {
            "kind": "lightgbm",
            "params": {**lightgbm_model.DEFAULT_PARAMS, **(params or {})},
        }
    else:
        raise ValueError(f"Unknown baseline {baseline!r}. Choices: seasonal_naive, lightgbm.")

    pred_mat = _prediction_matrix(preds, n_series, split)
    train_actuals = sales[:, : split.train_end]
    horizon_actuals = sales[:, split.h_start - 1 : split.h_end]
    last28_sales = sales[:, split.train_end - 28 : split.train_end]
    last28_price = price[:, split.train_end - 28 : split.train_end]
    weights = hierarchy.dollar_weights(last28_sales, last28_price)

    # per-series scale metrics (macro mean over series)
    rmsse_i = np.array(
        [metrics.rmsse(horizon_actuals[i], pred_mat[i], train_actuals[i]) for i in range(n_series)]
    )
    mase_i = np.array(
        [metrics.mase(horizon_actuals[i], pred_mat[i], train_actuals[i]) for i in range(n_series)]
    )
    wrmsse_score, per_level = hierarchy.wrmsse(
        entities, train_actuals, horizon_actuals, pred_mat, weights
    )

    report = {
        "baseline": baseline,
        "model": model_info,
        "split": split.as_dict(),
        "seed": seed,
        "n_series": n_series,
        "metrics": {
            "mae": metrics.mae(horizon_actuals.ravel(), pred_mat.ravel()),
            "wape": metrics.wape(horizon_actuals.ravel(), pred_mat.ravel()),
            "mase_mean": float(np.nanmean(mase_i)),
            "rmsse_mean": float(np.nanmean(rmsse_i)),
            "wrmsse": wrmsse_score,
        },
        "wrmsse_per_level": per_level,
        "run_context": RunContext().to_dict(),
    }
    return report
