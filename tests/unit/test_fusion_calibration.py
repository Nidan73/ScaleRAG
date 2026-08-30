"""Item 10's mixture inversion: the numerics have to be right before the finding is.

The measurement compares the deployed quantile-average against a genuine mixture
law, so a wrong mixture would fabricate the very artifact the run is trying to
attribute. These tests pin the inversion against cases with a known answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import fusion_calibration_run as FC  # noqa: E402,N812 -- matches the convention in scripts/

TAU = np.array(FC.QL)


def test_cdf_is_right_continuous_at_a_mass_point() -> None:
    """A series that is zero half the time has P(X <= 0) = 0.5, not 0.05.

    This is the bug the first smoke run hit: taking the lowest level of a tie
    block put the mixture's lower bound above zero on intermittent M5 series and
    reported a 60% below-interval miss rate that does not exist.
    """
    v = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]])  # q(0.05)..q(0.5) all zero
    f = FC._cdf_at(np.array([[0.0]]), v, TAU)
    assert f[0, 0] == pytest.approx(0.5)


def test_cdf_clamps_outside_the_reported_levels() -> None:
    v = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    f = FC._cdf_at(np.array([[-10.0, 0.0, 6.0, 99.0]]), v, TAU)
    assert f[0, 0] == pytest.approx(TAU[0])  # below every knot
    assert f[0, 1] == pytest.approx(TAU[0])  # at the lowest knot
    assert f[0, 2] == pytest.approx(TAU[-1])  # at the top knot
    assert f[0, 3] == pytest.approx(TAU[-1])  # above every knot


def test_cdf_interpolates_linearly_between_knots() -> None:
    v = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    f = FC._cdf_at(np.array([[1.5]]), v, TAU)  # midway between q(0.1) and q(0.25)
    assert f[0, 0] == pytest.approx(0.5 * (TAU[1] + TAU[2]))


def test_identical_branches_mix_to_themselves() -> None:
    """If both branches are the same law, every combination rule must return it.

    The mixture, the barycentre and the branch coincide exactly here, so any
    drift is pure numerical error in the inversion.
    """
    rng = np.random.default_rng(0)
    v = np.sort(rng.random((32, len(TAU))) * 10.0, axis=1)
    mix = FC._mixture_quantiles(v, v.copy(), 0.5, TAU)
    assert np.allclose(mix, v, atol=1e-9)


def test_mixture_is_wider_than_the_quantile_average_when_locations_differ() -> None:
    """Two shifted Gaussians: the textbook case the deployed operator misses.

    Vincentization preserves the component width when the scales match, while the
    mixture has to span both modes. If this ordering ever flips, the claim the
    run is built on is wrong.
    """
    from scipy.stats import norm

    z = norm.ppf(TAU)
    a = (0.0 + z)[None, :]
    b = (4.0 + z)[None, :]
    vincent = 0.5 * a + 0.5 * b
    mixture = FC._mixture_quantiles(a, b, 0.5, TAU)
    lo, hi = FC.QL.index(0.1), FC.QL.index(0.9)
    assert (mixture[0, hi] - mixture[0, lo]) > (vincent[0, hi] - vincent[0, lo])
    # and the barycentre is exactly as wide as one component, which is the point
    assert (vincent[0, hi] - vincent[0, lo]) == pytest.approx(a[0, hi] - a[0, lo])


def test_mixture_matches_a_brute_force_inversion() -> None:
    """Check the vectorised inversion against a dense scalar solve."""
    rng = np.random.default_rng(7)
    a = np.sort(rng.random((16, len(TAU))) * 5.0, axis=1)
    b = np.sort(rng.random((16, len(TAU))) * 5.0 + 2.0, axis=1)
    w = 0.3
    got = FC._mixture_quantiles(a, b, w, TAU)
    for i in range(a.shape[0]):
        # Start the dense grid at the lowest knot either branch reports. Further
        # left both CDFs are clamped flat at tau[0], so inf{x : G(x) >= tau[0]}
        # is -inf there and a dense grid would just return its own left edge.
        xs = np.linspace(min(a[i, 0], b[i, 0]), 10.0, 200_001)
        fa = FC._cdf_at(xs[None, :], a[i : i + 1], TAU)[0]
        fb = FC._cdf_at(xs[None, :], b[i : i + 1], TAU)[0]
        g = (1 - w) * fa + w * fb
        for j, t in enumerate(TAU):
            want = xs[int(np.argmax(g >= t - 1e-12))]  # inf{x : G(x) >= tau}
            assert got[i, j] == pytest.approx(want, abs=2e-4)


def test_mixture_quantiles_are_non_decreasing() -> None:
    rng = np.random.default_rng(3)
    a = np.sort(rng.random((64, len(TAU))) * 5.0, axis=1)
    b = np.sort(rng.random((64, len(TAU))) * 5.0, axis=1)
    mix = FC._mixture_quantiles(a, b, 0.5, TAU)
    assert np.all(np.diff(mix, axis=1) >= -1e-12)


def test_mixture_lower_bound_stays_at_zero_on_intermittent_counts() -> None:
    """The regression that motivated these tests, at the shape the run sees.

    Both branches put more than 10% of their mass at zero, so the mixture's 10%
    quantile is zero and a non-negative actual can never fall below it.
    """
    a = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]])
    b = np.array([[0.0, 0.0, 0.0, 0.5, 1.5, 4.0, 6.0]])
    mix = FC._mixture_quantiles(a, b, 0.5, TAU)
    assert mix[0, FC.QL.index(0.1)] == pytest.approx(0.0)
    assert mix[0, FC.QL.index(0.05)] == pytest.approx(0.0)
