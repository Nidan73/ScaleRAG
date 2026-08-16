"""Deterministic synthetic M5-shaped data for offline tests and smoke runs.

Produces the three real M5 files (`calendar.csv`, `sell_prices.csv`,
`sales_train_evaluation.csv`) with a structurally valid — but small — hierarchy,
so the *identical* ingestion/split/eval code path runs without downloading the
real dataset. Seeded for reproducibility. This is fixture data only; it is never
a substitute for real M5 results.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from scalerag.data.m5_schema import M5Paths, sales_day_columns

M5_START = date(2011, 1, 29)  # real M5 day d_1

# Small but valid hierarchy: 2 states x 2 stores, 2 cats x 1 dept x 3 items.
_STATES = {"CA": ["CA_1", "CA_2"], "TX": ["TX_1", "TX_2"]}
_DEPTS = {"FOODS": "FOODS_1", "HOBBIES": "HOBBIES_1"}
_ITEMS_PER_DEPT = 3


def _series_frame() -> pl.DataFrame:
    """Build the stable (id, item, dept, cat, store, state) grid."""
    rows = []
    for state, stores in _STATES.items():
        for store in stores:
            for cat, dept in _DEPTS.items():
                for k in range(1, _ITEMS_PER_DEPT + 1):
                    item = f"{dept}_{k:03d}"
                    rows.append(
                        {
                            "id": f"{item}_{store}_evaluation",
                            "item_id": item,
                            "dept_id": dept,
                            "cat_id": cat,
                            "store_id": store,
                            "state_id": state,
                        }
                    )
    return pl.DataFrame(rows)


def _calendar(n_days: int) -> pl.DataFrame:
    idx = np.arange(1, n_days + 1)
    dates = [M5_START + timedelta(days=int(i - 1)) for i in idx]
    week_of = (idx - 1) // 7
    wm_yr_wk = 11101 + week_of  # synthetic, internally consistent week key
    wday = np.array([(d.isoweekday() % 7) + 1 for d in dates])  # 1..7
    dom = np.array([d.day for d in dates])
    snap = (dom <= 10).astype(np.int64)  # SNAP roughly first 10 days
    return pl.DataFrame(
        {
            "date": [d.isoformat() for d in dates],
            "wm_yr_wk": wm_yr_wk.astype(np.int64),
            "weekday": [d.strftime("%A") for d in dates],
            "wday": wday.astype(np.int64),
            "month": np.array([d.month for d in dates], dtype=np.int64),
            "year": np.array([d.year for d in dates], dtype=np.int64),
            "d": [f"d_{i}" for i in idx],
            "event_name_1": [
                "Christmas" if (d.month == 12 and d.day == 25) else None for d in dates
            ],
            "event_type_1": [
                "National" if (d.month == 12 and d.day == 25) else None for d in dates
            ],
            "event_name_2": [None] * n_days,
            "event_type_2": [None] * n_days,
            "snap_CA": snap,
            "snap_TX": snap,
            "snap_WI": snap,
        }
    )


def _prices(series: pl.DataFrame, calendar: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """One price per (store, item, wm_yr_wk); some early weeks intentionally
    missing so 'missing price' handling is exercised."""
    weeks = calendar["wm_yr_wk"].unique().sort().to_list()
    store_items = series.select("store_id", "item_id").unique()
    rows = []
    for store, item in store_items.iter_rows():
        base = float(rng.uniform(1.5, 9.5))
        # first ~3 weeks have no listed price (item not yet sold) — like real M5
        first_week = weeks[min(int(rng.integers(0, 4)), len(weeks) - 1)]
        for wk in weeks:
            if wk < first_week:
                continue
            price = round(base * float(rng.uniform(0.9, 1.1)), 2)
            rows.append({"store_id": store, "item_id": item, "wm_yr_wk": wk, "sell_price": price})
    return pl.DataFrame(rows)


def _sales_matrix(series: pl.DataFrame, calendar: pl.DataFrame, rng: np.random.Generator):
    n = series.height
    n_days = calendar.height
    wday = calendar["wday"].to_numpy()
    weekend = np.isin(wday, [1, 2]).astype(float)  # boost two "weekend" wdays
    base = rng.lognormal(mean=0.2, sigma=0.6, size=n)  # per-series demand level
    mat = np.zeros((n, n_days), dtype=np.int64)
    for s in range(n):
        lam = base[s] * (1.0 + 0.6 * weekend)
        draws = rng.poisson(lam)
        # intermittency: zero out ~30% of days
        draws = np.where(rng.random(n_days) < 0.3, 0, draws)
        mat[s] = draws
    return mat


def generate_m5(dest: str | Path, *, n_days: int = 1941, seed: int = 0) -> M5Paths:
    """Write synthetic calendar/prices/sales CSVs into ``dest``. Idempotent per
    (n_days, seed): overwrites deterministically. Returns their paths."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    series = _series_frame()
    calendar = _calendar(n_days)
    prices = _prices(series, calendar, rng)
    mat = _sales_matrix(series, calendar, rng)

    day_cols = sales_day_columns(n_days)
    sales = series.clone().with_columns([pl.Series(day_cols[i], mat[:, i]) for i in range(n_days)])

    paths = M5Paths.under(dest)
    calendar.write_csv(paths.calendar)
    prices.write_csv(paths.prices)
    sales.write_csv(paths.sales)
    return paths
