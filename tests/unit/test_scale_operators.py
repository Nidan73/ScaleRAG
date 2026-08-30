from __future__ import annotations

import numpy as np
import pytest

from scalerag import scale_operators as so
from scalerag.retrieval_faiss import _fit_params


@pytest.mark.unit
def test_window_mean_is_identical_to_deployed_fit_params():
    """The deployed operator must be reproduced exactly, substitution included."""
    rng = np.random.default_rng(0)
    ctx = rng.poisson(0.5, size=(64, 56)).astype(np.float64)
    ctx[3] = 0.0  # force a degenerate row
    ctx[17] = 0.0
    expected = _fit_params(ctx, "mean")
    np.testing.assert_array_equal(so.window_mean(ctx), expected)


@pytest.mark.unit
def test_window_mean_substitutes_one_not_epsilon_on_zero_mean():
    """Guards the fact the manuscript gets wrong: the substitution is 1.0."""
    ctx = np.zeros((2, 56))
    params = so.window_mean(ctx)
    np.testing.assert_array_equal(params[:, 0], np.zeros(2))
    np.testing.assert_array_equal(params[:, 1], np.ones(2))


@pytest.mark.unit
def test_degenerate_mask_flags_exactly_the_zero_mean_rows():
    ctx = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 2.0]])
    np.testing.assert_array_equal(so.degenerate_mask(ctx), [True, False, False])


@pytest.mark.unit
def test_long_lookback_returns_nan_rather_than_substituting():
    """Fail loudly: an unresolvable scale is NaN, never a silent constant."""
    hist = np.zeros((2, 400))
    # index 100 is inside the lookback window [400-365, 400) = [35, 400)
    hist[0, 100] = 5.0  # series 0 has history, series 1 does not
    params = so.long_lookback(hist, origin=400, lookback=365)
    assert np.isfinite(params[0, 1]) and params[0, 1] > 0
    assert np.isnan(params[1, 1])


@pytest.mark.unit
def test_hierarchy_prior_pools_across_the_group():
    """A dead series inherits a usable scale from its group."""
    hist = np.zeros((3, 100))
    hist[0, -56:] = 2.0
    hist[1, -56:] = 4.0
    # series 2 is dead; groups: 0,1,2 all in group 0
    params = so.hierarchy_prior(hist, origin=100, group_ids=np.array([0, 0, 0]), lookback=56)
    assert params[2, 1] == pytest.approx(2.0)


@pytest.mark.unit
def test_hierarchy_prior_nan_when_whole_group_is_dead():
    hist = np.zeros((2, 100))
    params = so.hierarchy_prior(hist, origin=100, group_ids=np.array([0, 0]), lookback=56)
    assert np.isnan(params[:, 1]).all()


@pytest.mark.unit
def test_oracles_are_refused_unless_explicitly_allowed():
    """Rule 12: neither oracle may be reachable by accident."""
    fut = np.ones((2, 28))
    with pytest.raises(ValueError, match="oracle"):
        so.oracle_future_mean(fut)
    with pytest.raises(ValueError, match="oracle"):
        so.oracle_least_squares(np.ones((2, 28)), fut)
    assert so.oracle_future_mean(fut, allow_oracle=True)[0, 1] == pytest.approx(1.0)


@pytest.mark.unit
def test_least_squares_oracle_is_the_squared_error_minimiser():
    """It must beat any other scale on its own objective, which moment matching does not."""
    rng = np.random.default_rng(11)
    u = rng.random((40, 28)) + 0.1
    y = 3.0 * u + rng.normal(0, 0.5, size=u.shape)
    s_ls = so.oracle_least_squares(u, y, allow_oracle=True)[:, 1]
    s_mm = so.oracle_future_mean(y, allow_oracle=True)[:, 1] / u.mean(axis=1)

    err_ls = ((s_ls[:, None] * u - y) ** 2).sum(axis=1)
    err_mm = ((s_mm[:, None] * u - y) ** 2).sum(axis=1)
    assert (err_ls <= err_mm + 1e-9).all(), "least squares must never lose on squared error"
    assert (err_ls < err_mm - 1e-9).any(), "the two oracles must actually differ"


@pytest.mark.unit
def test_least_squares_oracle_keeps_a_zero_scale_instead_of_dropping_the_series():
    """An all-zero future has a legitimate optimum of 0; NaN there would condition on the outcome."""
    u = np.ones((1, 28))
    y = np.zeros((1, 28))
    params = so.oracle_least_squares(u, y, allow_oracle=True)
    assert params[0, 1] == pytest.approx(0.0)
    assert np.isfinite(params[0, 1])


@pytest.mark.unit
def test_restore_batch_matches_the_scalar_reference():
    from scalerag.retrieval_faiss import restore_continuation

    rng = np.random.default_rng(1)
    conts = rng.random((5, 28))
    cand = np.stack([np.zeros(5), np.array([1.0, 2.0, 0.5, 4.0, 0.25])], axis=1)
    query = np.array([0.0, 3.0])
    got = so.restore_batch(conts, cand, query)
    want = np.stack([restore_continuation(conts[j], cand[j], query) for j in range(5)])
    np.testing.assert_allclose(got, want)


@pytest.mark.unit
def test_restore_batch_propagates_a_nan_scale():
    conts = np.ones((3, 28))
    cand = np.stack([np.zeros(3), np.ones(3)], axis=1)
    got = so.restore_batch(conts, cand, np.array([0.0, np.nan]))
    assert np.isnan(got).all()


@pytest.mark.unit
def test_least_squares_oracle_resolves_an_identically_zero_prediction():
    """A zero prediction is scale-invariant, so the scale is arbitrary, not missing.

    NaN here would silently drop the sparsest series -- exactly the population the
    diagnostic exists to measure.
    """
    params = so.oracle_least_squares(np.zeros((1, 28)), np.ones((1, 28)), allow_oracle=True)
    assert np.isfinite(params[0, 1])


@pytest.mark.unit
def test_least_squares_oracle_is_nan_only_when_no_prediction_exists():
    params = so.oracle_least_squares(np.full((1, 28), np.nan), np.ones((1, 28)), allow_oracle=True)
    assert np.isnan(params[0, 1])
