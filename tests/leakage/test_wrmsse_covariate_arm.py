from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import wrmsse_attribution_run as WA  # noqa: E402,N812 -- matches the convention in scripts/
from scalerag.leakage import LeakageViolation  # noqa: E402

H = 28


def _panel(n: int = 5, days: int = 200):
    rng = np.random.default_rng(0)
    sales = rng.poisson(2.0, size=(n, days)).astype(np.float64)
    covs = {
        "snap": rng.integers(0, 2, size=(n, days)).astype(np.float64),
        "wday": np.tile(np.arange(days) % 7, (n, 1)).astype(np.float64),
    }
    return sales, covs


@pytest.mark.leakage
def test_past_covariates_match_the_target_length_exactly():
    """chronos2/preprocess.py raises otherwise; catching it here is cheaper."""
    sales, covs = _panel()
    o = 150
    inputs = WA.build_covariate_inputs(sales, covs, o, H)
    for item in inputs:
        assert len(item["target"]) == o
        for v in item["past_covariates"].values():
            assert len(v) == o
        for v in item["future_covariates"].values():
            assert len(v) == H


@pytest.mark.leakage
def test_target_never_includes_a_value_at_or_after_the_origin():
    """The forecast target must stop strictly before the origin."""
    sales, covs = _panel()
    o = 150
    sentinel = 999999.0
    poisoned = sales.copy()
    poisoned[:, o:] = sentinel
    clean = WA.build_covariate_inputs(sales, covs, o, H)
    dirty = WA.build_covariate_inputs(poisoned, covs, o, H)
    for a, b in zip(clean, dirty, strict=True):
        np.testing.assert_array_equal(a["target"], b["target"])
        for k in a["past_covariates"]:
            np.testing.assert_array_equal(a["past_covariates"][k], b["past_covariates"][k])


@pytest.mark.leakage
def test_a_future_only_covariate_is_rejected():
    """Deliberately violate rule 4 and require it to raise, not pass quietly."""
    sales, covs = _panel()
    with pytest.raises(LeakageViolation):
        WA.build_covariate_inputs(sales, covs, 150, H, future_only_names=["promo_next_week"])


@pytest.mark.leakage
def test_horizon_reaching_past_the_panel_is_refused():
    sales, covs = _panel(days=200)
    with pytest.raises(ValueError, match="beyond"):
        WA.build_covariate_inputs(sales, covs, 190, H)


@pytest.mark.leakage
def test_future_covariate_window_is_the_horizon_immediately_after_the_origin():
    sales, covs = _panel()
    o = 150
    inputs = WA.build_covariate_inputs(sales, covs, o, H)
    np.testing.assert_array_equal(
        inputs[0]["future_covariates"]["wday"], covs["wday"][0, o : o + H]
    )


@pytest.mark.leakage
def test_string_covariate_stays_categorical_and_nulls_become_a_sentinel():
    """M5 event types have no order, so ordinal encoding would assert one that
    does not exist. Chronos-2 takes categorical covariates as string arrays."""
    import polars as pl

    s = pl.Series("event_type_1", ["Sporting", None, "Religious", None, None, "National"])
    arr = WA.reshape_covariate(s, 2, 3)
    assert arr.shape == (2, 3)
    assert arr.dtype.kind in ("U", "S")
    assert "none" in arr.tolist()[0] or "none" in arr.tolist()[1]
    assert "Sporting" in arr.tolist()[0]


@pytest.mark.leakage
def test_numeric_covariate_is_float_with_a_null_sentinel():
    import polars as pl

    s = pl.Series("snap", [0, 1, None, 1, 0, 1], dtype=pl.Float64)
    arr = WA.reshape_covariate(s, 2, 3)
    assert arr.dtype == np.float64
    assert (arr == -1.0).any()


@pytest.mark.leakage
def test_categorical_covariate_survives_the_input_builder():
    """A string covariate must slice through build_covariate_inputs unchanged."""
    sales, covs = _panel(n=2, days=100)
    covs["event_type_1"] = np.array([["none"] * 100, ["Sporting"] * 100])
    inputs = WA.build_covariate_inputs(sales, covs, 60, H)
    assert inputs[1]["past_covariates"]["event_type_1"][0] == "Sporting"
    assert len(inputs[1]["future_covariates"]["event_type_1"]) == H
