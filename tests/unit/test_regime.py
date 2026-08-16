"""Unit tests for retrieval-utility regime estimation.

The estimators must recover a planted boundary, and must refuse to invent one when
the data contain none — a fabricated threshold would be worse than no threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from scalerag.regime import (
    estimate_band,
    estimate_threshold,
    utility_correlates,
)

pytestmark = pytest.mark.unit

N = 4000


def _planted_step(cut: float, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Utility that flips sign at ``cut``, with noise on both sides."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=N)
    # 85% correct side / 15% wrong, so the win rate crosses 0.5 exactly at `cut`.
    correct = rng.random(N) < 0.85
    wins = np.where(x > cut, correct, ~correct)
    utility = np.where(wins, 1.0, -1.0)
    return utility, x


def test_threshold_recovers_a_planted_crossing() -> None:
    utility, x = _planted_step(0.4)
    res = estimate_threshold(utility, x, "x", n_boot=200, seed=3)
    assert res.crosses
    assert res.threshold == pytest.approx(0.4, abs=0.05)
    assert res.win_rate_below < 0.5 < res.win_rate_above


def test_bootstrap_interval_is_narrow_and_not_relied_on_for_coverage() -> None:
    """Pins the documented caveat: these intervals are indicative, not calibrated.

    The naive bootstrap is inconsistent for an isotonic crossing (cube-root
    asymptotics), so on a sharply planted boundary it produces an interval tight
    enough to miss the very value it was built around. The test asserts what the
    interval *is* — narrow and centred on the estimate — not that it covers, so that
    a future switch to subsampling shows up here as an intentional change.
    """
    utility, x = _planted_step(0.4)
    res = estimate_threshold(utility, x, "x", n_boot=200, seed=3)
    assert res.ci95 is not None
    lo, hi = res.ci95
    assert lo <= res.threshold <= hi
    assert hi - lo < 0.05, "interval unexpectedly wide; the caveat may no longer apply"


def test_threshold_refuses_when_retrieval_never_wins() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(size=N)
    utility = -np.abs(rng.normal(size=N))  # backbone always better
    res = estimate_threshold(utility, x, "x", n_boot=50)
    assert not res.crosses
    assert res.threshold is None
    assert "never reaches 0.5" in res.note


def test_threshold_refuses_when_retrieval_always_wins() -> None:
    """Winning from the smallest x means no interior boundary exists."""
    rng = np.random.default_rng(2)
    x = rng.uniform(size=N)
    utility = np.abs(rng.normal(size=N))
    res = estimate_threshold(utility, x, "x", n_boot=50)
    assert not res.crosses
    assert res.threshold is None


def test_band_recovers_both_edges_of_a_planted_interval() -> None:
    """The M5 shape: retrieval pays in a middle band and fails at both extremes."""
    rng = np.random.default_rng(11)
    x = rng.uniform(0.0, 1.0, size=N)
    inside = (x > 0.3) & (x < 0.8)
    correct = rng.random(N) < 0.85
    wins = np.where(inside, correct, ~correct)
    utility = np.where(wins, 1.0, -1.0)

    band = estimate_band(utility, x, "x", n_boot=200, seed=5)
    assert band.bounded_above
    assert band.lower == pytest.approx(0.3, abs=0.06)
    assert band.upper == pytest.approx(0.8, abs=0.06)
    assert band.win_rate_inside > 0.7 > band.win_rate_outside


def test_band_reports_no_upper_edge_when_the_trend_is_monotone() -> None:
    """A genuinely monotone relationship must not be given a spurious ceiling."""
    utility, x = _planted_step(0.35, seed=9)
    band = estimate_band(utility, x, "x", n_boot=200, seed=6)
    assert band.lower == pytest.approx(0.35, abs=0.06)
    assert not band.bounded_above
    assert band.upper is None


def test_correlates_recover_the_sign_of_each_relationship() -> None:
    rng = np.random.default_rng(4)
    n = 2000
    up = rng.uniform(size=n)
    down = rng.uniform(size=n)
    utility = up - down + rng.normal(scale=0.05, size=n)
    corr = utility_correlates(
        utility, np.stack([up, down], axis=1), ["rises_with_utility", "falls_with_utility"]
    )
    assert corr.rho[0] > 0.4
    assert corr.rho[1] < -0.4
    assert all(p < 1e-10 for p in corr.p_value)


def test_constant_diagnostic_reports_nan_rather_than_crashing() -> None:
    rng = np.random.default_rng(5)
    utility = rng.normal(size=100)
    feats = np.stack([np.ones(100), rng.normal(size=100)], axis=1)
    corr = utility_correlates(utility, feats, ["constant", "varying"])
    assert np.isnan(corr.rho[0]) and np.isnan(corr.p_value[0])
    assert not np.isnan(corr.rho[1])


def test_too_few_series_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 20 series"):
        estimate_threshold(np.ones(10), np.arange(10.0), "x")


def test_misaligned_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="must align"):
        estimate_threshold(np.ones(50), np.arange(40.0), "x")
    with pytest.raises(ValueError, match="names for"):
        utility_correlates(np.ones(30), np.ones((30, 2)), ["only_one"])
    with pytest.raises(ValueError, match="features must be 2-D"):
        utility_correlates(np.ones(30), np.ones(30), ["x"])
