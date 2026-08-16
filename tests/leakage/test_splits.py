"""Chronological rolling-split integrity tests (Phase 2, tasks 4-5, 9)."""

from __future__ import annotations

import pytest

from scalerag import leakage
from scalerag.splits import (
    RollingSplit,
    make_rolling_splits,
    split_by_name,
    validate_splits,
)


@pytest.mark.leakage
def test_canonical_boundaries() -> None:
    splits = make_rolling_splits()
    test = split_by_name(splits, "test")
    val = split_by_name(splits, "val")
    assert (test.h_start, test.h_end) == (1914, 1941)
    assert (val.h_start, val.h_end) == (1886, 1913)
    assert test.train_end == 1913
    assert val.train_end == 1885


@pytest.mark.leakage
def test_at_least_two_earlier_val_origins() -> None:
    splits = make_rolling_splits()
    earlier = [s for s in splits if s.name.startswith("val_m")]
    assert len(earlier) >= 2
    # earlier origins strictly precede the primary validation origin
    val_end = split_by_name(splits, "val").train_end
    assert all(s.train_end < val_end for s in earlier)


@pytest.mark.leakage
def test_train_and_horizon_never_overlap() -> None:
    for s in make_rolling_splits():
        assert not s.is_train_day(s.h_start)  # first horizon day is not a train day
        assert s.is_train_day(s.train_end)
        assert s.h_start == s.train_end + 1


@pytest.mark.leakage
def test_horizons_within_public_labels() -> None:
    for s in make_rolling_splits(last_labeled_day=1941):
        assert s.h_end <= 1941  # never uses days beyond public labels (rule 2)


@pytest.mark.leakage
def test_origins_are_strictly_increasing() -> None:
    splits = make_rolling_splits()
    ends = [s.train_end for s in splits]
    assert ends == sorted(ends)
    assert len(set(ends)) == len(ends)


@pytest.mark.leakage
def test_horizon_beyond_labels_is_rejected() -> None:
    bad = [RollingSplit("bad", train_end=1920, horizon=28)]  # 1920+28=1948 > 1941
    with pytest.raises(leakage.LeakageViolation):
        validate_splits(bad, last_labeled_day=1941)


@pytest.mark.leakage
def test_requires_two_earlier_origins() -> None:
    with pytest.raises(ValueError):
        make_rolling_splits(n_earlier_val=1)
