from __future__ import annotations

import numpy as np
import pytest

from scalerag import sparse_regime as sr


@pytest.mark.unit
def test_bin_index_uses_fixed_edges_and_closes_the_last_bin():
    x = np.array([0.0, 0.05, 0.35, 0.95, 1.0])
    np.testing.assert_array_equal(sr.bin_index(x), [0, 0, 3, 9, 9])


@pytest.mark.unit
def test_bin_index_rejects_values_outside_zero_one():
    with pytest.raises(ValueError, match="within"):
        sr.bin_index(np.array([-0.01]))
    with pytest.raises(ValueError, match="within"):
        sr.bin_index(np.array([1.01]))


@pytest.mark.unit
def test_summary_splits_a_bin_into_cohorts_and_reports_both():
    inter = np.array([0.95, 0.95, 0.95, 0.95])
    util = np.array([1.0, 1.0, -1.0, -1.0])
    delta = np.array([0.5, 0.5, -0.5, -0.5])
    degen = np.array([True, True, False, False])

    rows = sr.stratified_bin_summary(inter, util, delta, degen)
    row = next(r for r in rows if r["lo"] == pytest.approx(0.9))

    assert row["n"] == 4
    assert row["win_rate_all"] == pytest.approx(0.5)
    assert row["degenerate_fraction"] == pytest.approx(0.5)
    assert row["win_rate_degenerate"] == pytest.approx(1.0)
    assert row["win_rate_identifiable"] == pytest.approx(0.0)
    assert row["mean_delta_identifiable"] == pytest.approx(-0.5)
    assert row["median_delta_identifiable"] == pytest.approx(-0.5)


@pytest.mark.unit
def test_summary_reports_nan_for_an_absent_cohort_rather_than_dropping_the_bin():
    inter = np.array([0.05, 0.05])
    util = np.array([1.0, -1.0])
    delta = np.array([0.1, -0.1])
    degen = np.array([False, False])

    row = next(r for r in sr.stratified_bin_summary(inter, util, delta, degen) if r["lo"] == 0.0)
    assert row["degenerate_fraction"] == pytest.approx(0.0)
    assert np.isnan(row["win_rate_degenerate"])
    assert row["win_rate_identifiable"] == pytest.approx(0.5)


@pytest.mark.unit
def test_summary_carries_the_fused_win_rate_when_supplied():
    """Referee item 13: the recommendation is about the fused system."""
    inter = np.array([0.45, 0.45])
    util = np.array([1.0, -1.0])
    delta = np.array([0.2, -0.2])
    degen = np.array([False, False])
    fused = np.array([1.0, 1.0])

    row = next(
        r
        for r in sr.stratified_bin_summary(inter, util, delta, degen, fused_utility=fused)
        if r["lo"] == pytest.approx(0.4)
    )
    assert row["win_rate_all"] == pytest.approx(0.5)
    assert row["fused_win_rate_all"] == pytest.approx(1.0)


@pytest.mark.unit
def test_summary_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="align"):
        sr.stratified_bin_summary(
            np.array([0.1, 0.2]), np.array([1.0]), np.array([1.0]), np.array([True])
        )


@pytest.mark.unit
def test_summary_omits_empty_bins():
    inter = np.array([0.95, 0.95])
    rows = sr.stratified_bin_summary(
        inter, np.array([1.0, -1.0]), np.array([0.1, -0.1]), np.array([False, False])
    )
    assert [r["lo"] for r in rows] == [pytest.approx(0.9)]
