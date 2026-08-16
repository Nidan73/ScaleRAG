"""Seasonal Naive baseline (Phase 2, task 7).

Leakage-safe by construction: it repeats the **last observed season** of training
data across the whole horizon. For horizon day ``train_end + k`` (k = 1..H) the
prediction is ``y[train_end - season + ((k-1) mod season) + 1]`` — always a
training day, never a horizon day. Default season = 7 (weekly).
"""

from __future__ import annotations

import polars as pl

from scalerag.splits import RollingSplit


def predict(dynamic: pl.DataFrame, split: RollingSplit, season: int = 7) -> pl.DataFrame:
    """Return predictions ``(id, day_idx, y_pred)`` for the split horizon."""
    te = split.train_end
    last = (
        dynamic.filter((pl.col("day_idx") > te - season) & (pl.col("day_idx") <= te))
        .sort(["id", "day_idx"])
        .group_by("id", maintain_order=True)
        .agg(pl.col("sales").cast(pl.Float64).alias("season_vals"))
    )
    offsets = pl.DataFrame({"k": list(range(split.horizon))})
    preds = (
        last.join(offsets, how="cross")
        .with_columns(
            (te + 1 + pl.col("k")).alias("day_idx"),
            # cycle through the last `season` observed values
            pl.col("season_vals").list.get(pl.col("k") % season, null_on_oob=True).alias("y_pred"),
        )
        .with_columns(pl.col("y_pred").fill_null(0.0).clip(lower_bound=0.0))
        .select("id", "day_idx", "y_pred")
    )
    return preds
