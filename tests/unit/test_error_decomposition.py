"""Unit tests for the retrieval-error attribution.

The load-bearing property is that the oracle affine correction really is a lower
bound: the whole "how much of this error is shape, how much is magnitude" split is
meaningless if it is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from scalerag.error_decomposition import (
    decompose_errors,
    optimal_fusion_weight,
    oracle_rescale,
    paired_bootstrap_mean_diff,
)

pytestmark = pytest.mark.unit


def _rng() -> np.random.Generator:
    return np.random.default_rng(20260726)


def test_oracle_rescale_is_the_minimiser_over_affine_maps() -> None:
    """No (alpha, beta) may beat the fitted correction on any window."""
    rng = _rng()
    truth = rng.normal(size=(64, 24))
    pred = 3.0 * truth + 7.0 + rng.normal(scale=0.5, size=truth.shape)
    fitted = oracle_rescale(pred, truth)
    best = np.mean((fitted - truth) ** 2, axis=1)
    for alpha in (-1.0, 0.0, 0.31, 1.0, 2.5):
        for beta in (-5.0, 0.0, 2.0):
            competitor = np.mean((alpha * pred + beta - truth) ** 2, axis=1)
            assert np.all(best <= competitor + 1e-9), f"beaten by alpha={alpha}, beta={beta}"


def test_oracle_rescale_beats_moment_matching() -> None:
    """Guards the reason least squares replaced moment matching here."""
    rng = _rng()
    truth = rng.normal(size=(128, 32))
    pred = rng.normal(size=truth.shape)  # uncorrelated: moment matching is actively harmful
    fitted_mse = np.mean((oracle_rescale(pred, truth) - truth) ** 2)
    p_mu, p_sd = pred.mean(1, keepdims=True), pred.std(1, keepdims=True)
    t_mu, t_sd = truth.mean(1, keepdims=True), truth.std(1, keepdims=True)
    moment_matched = (pred - p_mu) / p_sd * t_sd + t_mu
    assert fitted_mse < np.mean((moment_matched - truth) ** 2)


def test_oracle_rescale_recovers_an_exact_affine_image() -> None:
    rng = _rng()
    truth = rng.normal(size=(32, 16))
    pred = -2.5 * truth + 11.0
    assert np.allclose(oracle_rescale(pred, truth), truth, atol=1e-9)


def test_oracle_rescale_handles_a_constant_prediction() -> None:
    truth = _rng().normal(size=(8, 12))
    const = np.full_like(truth, 4.2)
    out = oracle_rescale(const, truth)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, truth.mean(axis=1, keepdims=True))


def test_shape_floor_never_exceeds_the_restored_error() -> None:
    """The attribution splits a positive quantity; a negative part would be nonsense."""
    rng = _rng()
    truth = rng.normal(size=(256, 32))
    restored = 1.4 * truth + 0.3 + rng.normal(scale=0.8, size=truth.shape)
    dec = decompose_errors(
        truth, truth + rng.normal(size=truth.shape), 9.0 * restored, restored, 0.25
    )
    assert dec.e_shape <= dec.e_res
    assert dec.scale_error_remaining >= 0.0
    assert 0.0 <= dec.shape_fraction_of_restored <= 1.0


def test_decomposition_components_sum_to_the_raw_error() -> None:
    rng = _rng()
    truth = rng.normal(size=(100, 20))
    restored = truth + rng.normal(scale=0.4, size=truth.shape)
    dec = decompose_errors(
        truth, truth + rng.normal(size=truth.shape), 5.0 * restored, restored, 0.3
    )
    total = dec.e_shape + dec.scale_error_remaining + dec.scale_error_removed
    assert total == pytest.approx(dec.e_raw, rel=1e-12)


def test_optimal_fusion_weight_recovers_a_planted_optimum() -> None:
    """With orthogonal residuals the optimum is analytic: w* = s_c/(s_c + s_r)."""
    rng = _rng()
    truth = rng.normal(size=(4000, 8))
    e_c = rng.normal(scale=1.0, size=truth.shape)
    e_r = rng.normal(scale=2.0, size=truth.shape)
    w = optimal_fusion_weight(truth + e_c, truth + e_r, truth)
    expected = 1.0 / (1.0 + 4.0)  # var_c / (var_c + var_r) with var_c=1, var_r=4
    assert w == pytest.approx(expected, abs=0.02)


def test_optimal_weight_is_near_zero_when_retrieval_is_useless() -> None:
    rng = _rng()
    truth = rng.normal(size=(2000, 8))
    backbone = truth + rng.normal(scale=0.05, size=truth.shape)
    retrieval = rng.normal(scale=5.0, size=truth.shape)
    assert optimal_fusion_weight(backbone, retrieval, truth) < 0.02


def test_fused_matches_the_declared_weight() -> None:
    rng = _rng()
    truth = rng.normal(size=(50, 10))
    backbone = truth + rng.normal(scale=0.2, size=truth.shape)
    restored = truth + rng.normal(scale=0.9, size=truth.shape)
    dec = decompose_errors(truth, backbone, 3.0 * restored, restored, 0.0)
    assert dec.e_fused == pytest.approx(dec.e_backbone, rel=1e-12)
    dec_one = decompose_errors(truth, backbone, 3.0 * restored, restored, 1.0)
    assert dec_one.e_fused == pytest.approx(dec_one.e_res, rel=1e-12)


def test_paired_bootstrap_interval_brackets_a_known_shift() -> None:
    rng = _rng()
    a = rng.normal(loc=1.5, scale=1.0, size=3000)
    b = a - 0.5
    delta, lo, hi = paired_bootstrap_mean_diff(a, b, n_boot=500, seed=1)
    assert delta == pytest.approx(0.5, abs=1e-9)
    assert lo <= 0.5 <= hi
    assert lo > 0.0, "a constant paired shift must give a CI excluding zero"


@pytest.mark.parametrize("weight", [-0.1, 1.5])
def test_invalid_fusion_weight_is_rejected(weight: float) -> None:
    arr = np.ones((4, 3))
    with pytest.raises(ValueError, match="fusion_weight"):
        decompose_errors(arr, arr, arr, arr, weight)


def test_shape_mismatch_between_arrays_is_rejected() -> None:
    truth = np.ones((4, 3))
    with pytest.raises(ValueError, match="does not match truth shape"):
        decompose_errors(truth, np.ones((4, 5)), truth, truth, 0.25)


def test_identical_branches_make_the_weight_undefined() -> None:
    arr = np.ones((4, 3))
    with pytest.raises(ValueError, match="undefined"):
        optimal_fusion_weight(arr, arr, np.zeros((4, 3)))
