"""Leakage-safe feature engineering for the LightGBM baseline (Phase 2, task 6).

Design rule: every sales-derived feature uses a lag of **at least the horizon**
(28). Because the forecast horizon is 28 days, a lag-28 feature for any horizon
day references only days ``<= train_end`` — so a single model predicts all 28
horizon days directly, with no recursion and no future leakage.

Fitted statistics (item price means, per-series mean demand) are computed on the
**training slice only** (``day_idx <= split.train_end``) and then applied to all
rows (docs/research-rules.md rule 5). Prices are treated as known-future covariates (M5 prices
are announced ahead), never the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from scalerag.splits import RollingSplit

# Numeric feature columns fed to the model (order is stable for reproducibility).
FEATURE_COLS: list[str] = [
    "lag_28",
    "lag_35",
    "lag_42",
    "rmean_7_lag28",
    "rmean_28_lag28",
    "wday",
    "month",
    "snap",
    "is_event",
    "sell_price",
    "price_norm",
    "series_train_mean",
]
TARGET = "sales"
# Only item_id is needed (id is already on the dynamic panel). Joining the full
# entity set would add 4 unused string columns x tens of millions of rows.
_ENTITY_JOIN_COLS = ("id", "item_id")


@dataclass
class FittedStats:
    """Train-only statistics, kept for provenance/repro."""

    train_end: int
    item_price_mean: pl.DataFrame  # item_id -> item_price_mean
    series_train_mean: pl.DataFrame  # id -> series_train_mean


def build_features(
    dynamic: pl.DataFrame, entities: pl.DataFrame, split: RollingSplit
) -> tuple[pl.DataFrame, FittedStats]:
    """Return (feature frame, fitted stats). The frame contains FEATURE_COLS,
    the target, ``id`` and ``day_idx`` for every row of ``dynamic``."""
    ent = entities.select(_ENTITY_JOIN_COLS)
    df = dynamic.join(ent, on="id", how="left").sort(["id", "day_idx"])

    # Causal sales lags (>= horizon) and rolling means anchored at lag-28.
    df = df.with_columns(
        pl.col("sales").shift(28).over("id").alias("lag_28"),
        pl.col("sales").shift(35).over("id").alias("lag_35"),
        pl.col("sales").shift(42).over("id").alias("lag_42"),
    ).with_columns(
        pl.col("lag_28").rolling_mean(7).over("id").alias("rmean_7_lag28"),
        pl.col("lag_28").rolling_mean(28).over("id").alias("rmean_28_lag28"),
        pl.col("event_name_1").is_not_null().cast(pl.Int8).alias("is_event"),
    )

    # --- Fit transforms on the TRAINING slice only, then apply to all rows. ---
    train = df.filter(pl.col("day_idx") <= split.train_end)
    item_price_mean = train.group_by("item_id").agg(
        pl.col("sell_price").mean().alias("item_price_mean")
    )
    series_train_mean = train.group_by("id").agg(pl.col("sales").mean().alias("series_train_mean"))

    df = (
        df.join(item_price_mean, on="item_id", how="left")
        .join(series_train_mean, on="id", how="left")
        # round to kill ULP-level nondeterminism from parallel float sums in the
        # train-only mean — keeps features bit-reproducible (rule 10).
        .with_columns(
            (pl.col("sell_price") / pl.col("item_price_mean")).round(6).alias("price_norm")
        )
    )

    keep = ["id", "day_idx", TARGET, *FEATURE_COLS]
    stats = FittedStats(
        train_end=split.train_end,
        item_price_mean=item_price_mean,
        series_train_mean=series_train_mean,
    )
    return df.select(keep), stats


def train_matrix(features: pl.DataFrame, split: RollingSplit) -> pl.DataFrame:
    """Rows usable for training: training days with a defined primary lag."""
    return features.filter((pl.col("day_idx") <= split.train_end) & pl.col("lag_28").is_not_null())


def horizon_matrix(features: pl.DataFrame, split: RollingSplit) -> pl.DataFrame:
    """Rows to forecast: the split's horizon days."""
    return features.filter(
        (pl.col("day_idx") >= split.h_start) & (pl.col("day_idx") <= split.h_end)
    )
