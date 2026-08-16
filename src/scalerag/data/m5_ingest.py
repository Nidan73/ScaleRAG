"""Idempotent M5 ingestion: CSV → Parquet (Phase 2, tasks 1 & 3).

Separates **stable entities** (one row per series: item/dept/cat/store/state)
from **dynamic features** (long form: sales, price, SNAP, calendar events per
series-day). Idempotent via a content fingerprint of the three source files —
re-running with unchanged inputs is a no-op unless ``force=True``.

Outputs under ``processed_dir``:
- ``entities.parquet``   — stable hierarchy, one row per ``id``
- ``dynamic.parquet``    — long form ``(id, d, day_idx, date, sales, sell_price,
                            snap, event_name_1, event_type_1, wday, month, year)``
- ``_ingest_meta.json``  — fingerprint + row counts for idempotency/repro
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from scalerag.data import m5_schema as sch


def _fingerprint(paths: sch.M5Paths) -> dict:
    fp = {}
    for p in (paths.calendar, paths.prices, paths.sales):
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        fp[p.name] = {"sha256": h.hexdigest(), "bytes": p.stat().st_size}
    return fp


def _snap_expr() -> pl.Expr:
    return (
        pl.when(pl.col("state_id") == "CA")
        .then(pl.col("snap_CA"))
        .when(pl.col("state_id") == "TX")
        .then(pl.col("snap_TX"))
        .otherwise(pl.col("snap_WI"))
        .cast(pl.Int8)
        .alias("snap")
    )


def ingest_m5(
    raw_dir: str | Path,
    processed_dir: str | Path,
    *,
    n_days: int = sch.N_DAYS_EVAL,
    force: bool = False,
) -> dict:
    """Validate and ingest M5 into Parquet. Returns a summary dict."""
    paths = sch.M5Paths.under(raw_dir)
    sch.validate_all(paths, n_days=n_days)

    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    entities_out = processed / "entities.parquet"
    dynamic_out = processed / "dynamic.parquet"
    meta_out = processed / "_ingest_meta.json"

    fingerprint = _fingerprint(paths)
    if not force and entities_out.exists() and dynamic_out.exists() and meta_out.exists():
        prev = json.loads(meta_out.read_text())
        if prev.get("fingerprint") == fingerprint and prev.get("n_days") == n_days:
            return {**prev, "skipped": True}

    calendar = pl.read_csv(paths.calendar).with_columns(
        pl.col("d").str.strip_prefix("d_").cast(pl.Int32).alias("day_idx")
    )
    prices = pl.read_csv(paths.prices)
    sales = pl.read_csv(paths.sales)

    entities = sales.select(sch.ENTITY_COLS).unique(subset=["id"]).sort("id")

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.unpivot(
        on=day_cols, index=list(sch.ENTITY_COLS), variable_name="d", value_name="sales"
    ).with_columns(
        pl.col("sales").cast(pl.Int32),
        pl.col("d").str.strip_prefix("d_").cast(pl.Int32).alias("day_idx"),
    )

    cal_cols = [
        "day_idx",
        "date",
        "wm_yr_wk",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    dynamic = (
        long.join(calendar.select(cal_cols), on="day_idx", how="left")
        .with_columns(_snap_expr())
        .join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        .select(
            "id",
            "d",
            "day_idx",
            "date",
            "wm_yr_wk",
            "sales",
            "sell_price",
            "snap",
            "event_name_1",
            "event_type_1",
            "wday",
            "month",
            "year",
        )
        .sort(["id", "day_idx"])
    )

    entities.write_parquet(entities_out)
    dynamic.write_parquet(dynamic_out)

    summary = {
        "n_days": n_days,
        "n_series": entities.height,
        "n_rows": dynamic.height,
        "missing_price_rows": int(dynamic.select(pl.col("sell_price").is_null().sum()).item()),
        "fingerprint": fingerprint,
        "entities_parquet": str(entities_out),
        "dynamic_parquet": str(dynamic_out),
        "skipped": False,
    }
    meta_out.write_text(json.dumps(summary, indent=2))
    return summary
