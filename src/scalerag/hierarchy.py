"""M5 hierarchy aggregation and official-style WRMSSE (Phase 2, task 8).

The 12 M5 aggregation levels, from Total down to item x store. WRMSSE averages the
weighted RMSSE across all 12 levels; within each level, weights are proportional
to each (aggregated) series' cumulative dollar sales over the last 28 training
days, and normalised to sum to 1. Each level contributes equally (1/12).

All scale denominators come from **training** actuals only (rule 5).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from scalerag import metrics

# The 12 official M5 levels: (name, entity key columns). Total = single group.
M5_LEVELS: list[tuple[str, tuple[str, ...]]] = [
    ("L1_total", ("__total__",)),
    ("L2_state", ("state_id",)),
    ("L3_store", ("store_id",)),
    ("L4_cat", ("cat_id",)),
    ("L5_dept", ("dept_id",)),
    ("L6_state_cat", ("state_id", "cat_id")),
    ("L7_state_dept", ("state_id", "dept_id")),
    ("L8_store_cat", ("store_id", "cat_id")),
    ("L9_store_dept", ("store_id", "dept_id")),
    ("L10_item", ("item_id",)),
    ("L11_item_state", ("item_id", "state_id")),
    ("L12_item_store", ("item_id", "store_id")),
]


def level_group_ids(entities: pl.DataFrame, cols: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """Map each bottom series (row of ``entities``) to a group index for a level."""
    if cols == ("__total__",):
        return np.zeros(entities.height, dtype=np.int64), ["TOTAL"]
    parts = [entities[c].to_list() for c in cols]
    combined = np.array(["|".join(map(str, vals)) for vals in zip(*parts, strict=True)])
    uniq, inv = np.unique(combined, return_inverse=True)
    return inv.astype(np.int64), uniq.tolist()


def aggregate_rows(matrix: np.ndarray, gid: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum bottom-level rows into their group (hierarchy roll-up)."""
    out = np.zeros((n_groups, matrix.shape[1]), dtype=np.float64)
    np.add.at(out, gid, matrix)
    return out


def aggregate_weights(weights: np.ndarray, gid: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.zeros(n_groups, dtype=np.float64)
    np.add.at(out, gid, weights)
    return out


def dollar_weights(sales_last28: np.ndarray, price_last28: np.ndarray) -> np.ndarray:
    """Per bottom series: cumulative dollar sales over the last 28 training days.
    Missing prices are treated as 0 contribution."""
    price = np.nan_to_num(price_last28, nan=0.0)
    return np.sum(sales_last28 * price, axis=1)


def wrmsse(
    entities: pl.DataFrame,
    train_actuals: np.ndarray,
    horizon_actuals: np.ndarray,
    horizon_preds: np.ndarray,
    bottom_dollar_weights: np.ndarray,
    seasonality: int = 1,
) -> tuple[float, dict[str, float]]:
    """Official-style WRMSSE and the per-level breakdown.

    Matrices are ordered consistently with ``entities`` rows (bottom level):
    ``train_actuals`` [n_series, T_train], ``horizon_actuals`` / ``horizon_preds``
    [n_series, H]. ``bottom_dollar_weights`` [n_series].
    """
    level_scores: dict[str, float] = {}
    for name, cols in M5_LEVELS:
        gid, labels = level_group_ids(entities, cols)
        ng = len(labels)
        tr = aggregate_rows(train_actuals, gid, ng)
        ah = aggregate_rows(horizon_actuals, gid, ng)
        ph = aggregate_rows(horizon_preds, gid, ng)
        w = aggregate_weights(bottom_dollar_weights, gid, ng)

        r = np.array(
            [metrics.rmsse(ah[g], ph[g], tr[g], seasonality) for g in range(ng)],
            dtype=np.float64,
        )
        finite = np.isfinite(r)
        w_eff = np.where(finite, w, 0.0)
        total_w = w_eff.sum()
        level_scores[name] = (
            float(np.sum(w_eff[finite] * r[finite]) / total_w) if total_w > 0 else float("nan")
        )

    score = float(np.nanmean(list(level_scores.values())))
    return score, level_scores
