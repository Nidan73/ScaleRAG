"""Expected M5 file layout and schema validation (Phase 2, task 2).

We validate the three public M5 competition files *before* any processing and
fail loudly on a missing file, a missing column, or an unexpected day range. We
use ``sales_train_evaluation.csv`` (days ``d_1``..``d_1941``) because Phase 2 is
scoped to the publicly available labels through ``d_1941``.

No hidden evaluation labels are ever read (CLAUDE.md research rule 2): everything
here lives at or before ``d_1941``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------
N_DAYS_EVAL = 1941  # last publicly labelled day in sales_train_evaluation.csv

CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"
SALES_FILE = "sales_train_evaluation.csv"
REQUIRED_FILES = (CALENDAR_FILE, PRICES_FILE, SALES_FILE)

# Stable entity columns (one row per series) — kept separate from dynamics.
ENTITY_COLS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")

CALENDAR_COLS = (
    "date",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "d",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
)
PRICE_COLS = ("store_id", "item_id", "wm_yr_wk", "sell_price")

_DAY_RE = re.compile(r"^d_(\d+)$")


class SchemaError(ValueError):
    """Raised when an input file is missing or violates the expected schema."""


def day_index(d: str) -> int:
    """``'d_1913'`` -> ``1913``. Raises SchemaError on a malformed label."""
    m = _DAY_RE.match(d)
    if not m:
        raise SchemaError(f"Malformed day label: {d!r} (expected 'd_<int>').")
    return int(m.group(1))


def day_label(i: int) -> str:
    """``1913`` -> ``'d_1913'``."""
    if i < 1:
        raise SchemaError(f"Day index must be >= 1, got {i}.")
    return f"d_{i}"


def sales_day_columns(n_days: int = N_DAYS_EVAL) -> list[str]:
    """The ordered ``d_1``..``d_n`` value columns of the sales file."""
    return [day_label(i) for i in range(1, n_days + 1)]


@dataclass(frozen=True)
class M5Paths:
    """Resolved paths to the three required raw files under a directory."""

    root: Path
    calendar: Path
    prices: Path
    sales: Path

    @classmethod
    def under(cls, raw_dir: str | Path) -> M5Paths:
        root = Path(raw_dir)
        return cls(
            root=root,
            calendar=root / CALENDAR_FILE,
            prices=root / PRICES_FILE,
            sales=root / SALES_FILE,
        )

    def missing(self) -> list[str]:
        return [p.name for p in (self.calendar, self.prices, self.sales) if not p.is_file()]


def _require_columns(df: pl.DataFrame, required, *, what: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{what}: missing required columns {missing}. Found {df.columns}.")


def validate_calendar(df: pl.DataFrame) -> None:
    _require_columns(df, CALENDAR_COLS, what=CALENDAR_FILE)
    if df.height == 0:
        raise SchemaError(f"{CALENDAR_FILE}: empty.")


def validate_prices(df: pl.DataFrame) -> None:
    _require_columns(df, PRICE_COLS, what=PRICES_FILE)
    if df.select(pl.col("sell_price").is_null().all()).item():
        raise SchemaError(f"{PRICES_FILE}: sell_price is entirely null.")


def validate_sales(df: pl.DataFrame, n_days: int = N_DAYS_EVAL) -> None:
    _require_columns(df, ENTITY_COLS, what=SALES_FILE)
    day_cols = [c for c in df.columns if _DAY_RE.match(c)]
    if not day_cols:
        raise SchemaError(f"{SALES_FILE}: no d_* day columns found.")
    max_day = max(day_index(c) for c in day_cols)
    if max_day != n_days:
        raise SchemaError(
            f"{SALES_FILE}: expected last day d_{n_days} (public labels), found d_{max_day}. "
            "Refusing to proceed — do not use hidden evaluation labels."
        )
    if df.height == 0:
        raise SchemaError(f"{SALES_FILE}: no rows.")


def validate_all(paths: M5Paths, n_days: int = N_DAYS_EVAL) -> None:
    """Validate presence + schema of all three files. Reads headers cheaply."""
    missing = paths.missing()
    if missing:
        raise SchemaError(
            f"Missing required M5 files in {paths.root}: {missing}. "
            f"Expected: {list(REQUIRED_FILES)}."
        )
    validate_calendar(pl.read_csv(paths.calendar, n_rows=5))
    validate_prices(pl.read_csv(paths.prices, n_rows=5))
    # Sales: read only the header + a couple rows to check columns/day range.
    validate_sales(pl.read_csv(paths.sales, n_rows=2), n_days=n_days)
