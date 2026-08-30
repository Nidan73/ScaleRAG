"""Chronological rolling forecast splits for M5 (Phase 2, tasks 4-5).

Time is the M5 day index (``d_1``..``d_1941``). Splits are **expanding-window**
and strictly chronological (CLAUDE.md rule 1): a split trains on days
``1..train_end`` and forecasts the next ``horizon`` days
``[train_end+1, train_end+horizon]``. Only publicly labelled days (``<= 1941``)
are ever used as forecast targets (rule 2).

Canonical layout (horizon = 28):
- ``test``   : train <= d_1913, forecast d_1914..d_1941
- ``val``    : train <= d_1885, forecast d_1886..d_1913
- ``val_m1`` : train <= d_1857, forecast d_1858..d_1885
- ``val_m2`` : train <= d_1829, forecast d_1830..d_1857
"""

from __future__ import annotations

from dataclasses import dataclass

from scalerag import leakage
from scalerag.data.m5_schema import N_DAYS_EVAL

HORIZON = 28


@dataclass(frozen=True)
class RollingSplit:
    """One expanding-window split, expressed in M5 day indices."""

    name: str
    train_end: int
    horizon: int = HORIZON

    @property
    def h_start(self) -> int:
        return self.train_end + 1

    @property
    def h_end(self) -> int:
        return self.train_end + self.horizon

    def is_train_day(self, day_idx: int) -> bool:
        return day_idx <= self.train_end

    def is_horizon_day(self, day_idx: int) -> bool:
        return self.h_start <= day_idx <= self.h_end

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "train_end": self.train_end,
            "horizon": self.horizon,
            "h_start": self.h_start,
            "h_end": self.h_end,
        }


def make_rolling_splits(
    last_labeled_day: int = N_DAYS_EVAL,
    horizon: int = HORIZON,
    n_earlier_val: int = 2,
) -> list[RollingSplit]:
    """Build the canonical splits, earliest → latest. Requires ≥2 earlier
    validation origins (task 5). Every split is validated for leakage."""
    if n_earlier_val < 2:
        raise ValueError("Phase 2 requires at least two earlier validation origins.")

    test = RollingSplit("test", last_labeled_day - horizon, horizon)
    val = RollingSplit("val", last_labeled_day - 2 * horizon, horizon)
    earlier = [
        RollingSplit(f"val_m{k}", val.train_end - k * horizon, horizon)
        for k in range(1, n_earlier_val + 1)
    ]
    splits = sorted([test, val, *earlier], key=lambda s: s.train_end)
    validate_splits(splits, last_labeled_day=last_labeled_day)
    return splits


def validate_splits(splits: list[RollingSplit], last_labeled_day: int = N_DAYS_EVAL) -> None:
    """Fail loudly on any leakage in the split definitions (rules 1 & 3)."""
    for s in splits:
        if s.train_end < 1:
            raise leakage.LeakageViolation(f"{s.name}: train_end {s.train_end} < 1.")
        # train window and horizon window must not overlap.
        leakage.assert_no_window_overlap([(1, s.train_end)], [(s.h_start, s.h_end)])
        # horizon must stay within publicly labelled days.
        if s.h_end > last_labeled_day:
            raise leakage.LeakageViolation(
                f"{s.name}: forecast day d_{s.h_end} exceeds public labels d_{last_labeled_day}."
            )
    # origins strictly increasing in time.
    leakage.assert_strictly_increasing([(s.name, s.train_end) for s in splits])


def split_by_name(splits: list[RollingSplit], name: str) -> RollingSplit:
    for s in splits:
        if s.name == name:
            return s
    raise KeyError(f"No split named {name!r}. Available: {[s.name for s in splits]}")
