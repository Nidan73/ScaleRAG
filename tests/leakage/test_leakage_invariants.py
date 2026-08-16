"""Leakage / split-integrity invariant tests (docs/research-rules.md rules 1, 3, 4, 5).

Each invariant is tested both ways: a valid case passes silently, and a
deliberately invalid case raises LeakageViolation. Covers smoke-test check 11.
"""

from __future__ import annotations

import pytest

from scalerag.leakage import (
    LeakageViolation,
    assert_chronological_split,
    assert_no_future_covariates,
    assert_no_window_overlap,
    assert_retrieval_horizon,
    assert_target_not_in_features,
)


@pytest.mark.leakage
def test_chronological_split_valid() -> None:
    assert assert_chronological_split(train_end=100, val_end=110, test_start=120) is None


@pytest.mark.leakage
def test_chronological_split_out_of_order_raises() -> None:
    # val_end precedes train_end — an invalid temporal split.
    with pytest.raises(LeakageViolation):
        assert_chronological_split(train_end=100, val_end=90, test_start=120)


@pytest.mark.leakage
def test_chronological_split_iso_dates() -> None:
    with pytest.raises(LeakageViolation):
        assert_chronological_split("2016-04-24", "2016-03-27", "2016-05-22")


@pytest.mark.leakage
def test_retrieval_horizon_valid() -> None:
    assert assert_retrieval_horizon(t_r=50, horizon=28, target_forecast_origin=100) is None


@pytest.mark.leakage
def test_retrieval_horizon_violation_raises() -> None:
    # t_r + H = 95 + 28 = 123 >= origin(110): future-horizon leakage.
    with pytest.raises(LeakageViolation):
        assert_retrieval_horizon(t_r=95, horizon=28, target_forecast_origin=110)


@pytest.mark.leakage
def test_window_overlap_raises() -> None:
    with pytest.raises(LeakageViolation):
        assert_no_window_overlap(train_windows=[(0, 100)], eval_windows=[(95, 120)])


@pytest.mark.leakage
def test_window_no_overlap_ok() -> None:
    assert assert_no_window_overlap([(0, 100)], [(101, 120)]) is None


@pytest.mark.leakage
def test_target_in_features_raises() -> None:
    with pytest.raises(LeakageViolation):
        assert_target_not_in_features(["price", "sales"], target_column="sales")


@pytest.mark.leakage
def test_future_covariate_raises() -> None:
    with pytest.raises(LeakageViolation):
        assert_no_future_covariates(
            covariate_columns=["price", "unknown_demand"],
            known_future_columns=["price", "calendar_event"],
        )
