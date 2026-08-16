#!/usr/bin/env python3
"""Generate a synthetic M5-shaped dataset and ingest it (Phase 2 dev bootstrap).

This produces offline fixture data so the *identical* ingestion → split → eval
path can run without the real M5 download. It is NOT a substitute for real M5
results. Example:

    uv run python scripts/make_synthetic.py \
        --days 1941 --raw data/raw_synth --processed data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scalerag.data.m5_ingest import ingest_m5
from scalerag.data.synthetic import generate_m5

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=1941, help="Number of days (M5 = 1941).")
    ap.add_argument("--raw", type=Path, default=REPO / "data" / "raw_synth")
    ap.add_argument("--processed", type=Path, default=REPO / "data" / "processed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Generating synthetic M5 ({args.days} days, seed={args.seed}) → {args.raw}")
    paths = generate_m5(args.raw, n_days=args.days, seed=args.seed)
    print(f"  wrote {paths.calendar.name}, {paths.prices.name}, {paths.sales.name}")

    print(f"Ingesting → {args.processed}")
    summary = ingest_m5(args.raw, args.processed, n_days=args.days, force=True)
    print(
        f"  entities={summary['n_series']} series | dynamic={summary['n_rows']} rows | "
        f"missing_price_rows={summary['missing_price_rows']}"
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
