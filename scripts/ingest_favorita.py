#!/usr/bin/env python3
"""Streaming subset ingestion of Corporación Favorita (Phase 8).

Favorita's train.csv is ~125M rows / 5 GB. We never load it whole: Polars
streaming filters to a fixed recent window and a declared sample of item-store
series, then densifies to a leakage-safe panel. Favorita quirks handled: negative
unit_sales (returns) clipped to 0; missing (store,item,day) rows = 0 sales;
variable series start dates; onpromotion nulls -> False.

Outputs data/processed/favorita/{entities.parquet, dynamic.parquet}.

    uv run python scripts/ingest_favorita.py --series 5000 --days 1000 --seed 42
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "favorita"
OUT = REPO / "data" / "processed" / "favorita"
LAST_DATE = date(2017, 8, 15)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", type=int, default=5000)
    ap.add_argument("--days", type=int, default=1000)
    ap.add_argument("--min-present", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    start = LAST_DATE - timedelta(days=args.days - 1)
    start_s = start.isoformat()
    print(f"window {start_s}..{LAST_DATE} ({args.days} days); target {args.series} series")

    items = pl.read_csv(RAW / "items.csv")
    stores = pl.read_csv(RAW / "stores.csv")

    scan = pl.scan_csv(RAW / "train.csv").filter(pl.col("date") >= start_s)

    # pass 1: per-series presence + promo fraction (streaming)
    grp = (
        scan.group_by(["store_nbr", "item_nbr"])
        .agg(
            n=pl.len(),
            promo=(pl.col("onpromotion") == "True").fill_null(False).sum(),
        )
        .filter(pl.col("n") >= args.min_present)
        .collect(engine="streaming")
    )
    print(f"  qualifying series (>= {args.min_present} days present): {grp.height}")
    rng = np.random.default_rng(args.seed)
    take = min(args.series, grp.height)
    sel = grp.sort(["store_nbr", "item_nbr"])[rng.permutation(grp.height)[:take]].sort(
        ["store_nbr", "item_nbr"]
    )
    sel = sel.with_columns((pl.col("promo") / pl.col("n")).alias("promo_frac"))

    # pass 2: pull selected series' rows (streaming inner join)
    long = (
        scan.join(sel.lazy().select("store_nbr", "item_nbr"), on=["store_nbr", "item_nbr"])
        .select("store_nbr", "item_nbr", "date", "unit_sales")
        .collect(engine="streaming")
        .with_columns(pl.col("unit_sales").clip(lower_bound=0.0))  # returns -> 0
    )

    # dense day grid
    cal = pl.DataFrame({"date": pl.date_range(start, LAST_DATE, "1d", eager=True)}).with_columns(
        pl.col("date").dt.strftime("%Y-%m-%d")
    )
    cal = cal.with_columns((pl.int_range(1, cal.height + 1)).alias("day_idx"))
    long = long.join(cal, on="date", how="inner")

    ids = sel.with_columns(
        (pl.col("store_nbr").cast(str) + "_" + pl.col("item_nbr").cast(str)).alias("id")
    )
    grid = ids.select("store_nbr", "item_nbr", "id").join(cal.select("day_idx"), how="cross")
    dyn = (
        grid.join(
            long.select("store_nbr", "item_nbr", "day_idx", "unit_sales"),
            on=["store_nbr", "item_nbr", "day_idx"],
            how="left",
        )
        .with_columns(pl.col("unit_sales").fill_null(0.0).alias("sales"))
        .select("id", "day_idx", "sales")
        .sort(["id", "day_idx"])
    )

    entities = (
        ids.join(items, on="item_nbr", how="left")
        .join(stores, on="store_nbr", how="left")
        .select(
            "id",
            pl.col("item_nbr").cast(str).alias("item_id"),
            pl.col("family").alias("family_id"),
            pl.col("class").cast(str).alias("class_id"),
            pl.col("perishable").cast(str).alias("perishable_id"),
            pl.col("store_nbr").cast(str).alias("store_id"),
            pl.col("type").alias("type_id"),
            pl.col("cluster").cast(str).alias("cluster_id"),
            pl.col("city").alias("city_id"),
            pl.col("state").alias("state_id"),
            "promo_frac",
        )
        .sort("id")
    )

    entities.write_parquet(OUT / "entities.parquet")
    dyn.write_parquet(OUT / "dynamic.parquet")
    z = float((dyn["sales"] == 0).mean())
    print(f"  series={entities.height} days={args.days} rows={dyn.height} zero_frac={z:.3f}")
    print(
        f"  families={entities['family_id'].n_unique()} classes={entities['class_id'].n_unique()} "
        f"cities={entities['city_id'].n_unique()} store_types={entities['type_id'].n_unique()}"
    )
    print(f"wrote {OUT}/entities.parquet , {OUT}/dynamic.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
