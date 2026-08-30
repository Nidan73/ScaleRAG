from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import scale_identifiability_run as SI  # noqa: E402,N812 -- matches the convention in scripts/


def _record(origin: int, n: int = 60, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    zf = np.clip(rng.random(n), 0.0, 1.0)
    # degenerate contexts live only in the sparsest bin, but not all of it -- both
    # cohorts must be populated or the contrast has nothing to contrast
    degen = (zf > 0.9) & (rng.random(n) < 0.6)
    rec = {
        "origin": np.array([origin]),
        "zero_fraction": zf,
        "degenerate": degen,
        "context_mean": np.where(degen, 0.0, rng.random(n) + 0.1),
        "rmsse_backbone": rng.random(n) + 0.5,
        "rmsse_fused": rng.random(n) + 0.5,
        "backbone_pt_mean": rng.random(n) * 0.01,
        "realised_future_mean": rng.random(n),
        "never_launched": degen & (rng.random(n) < 0.5),
        "history_mean_365": np.where(degen, 0.0, rng.random(n)),
        "rmsse_backbone_oracle_ls": rng.random(n) + 0.3,
    }
    for op in SI.OPERATOR_ORDER:
        rec[f"rmsse_{op}"] = rng.random(n) + 0.5
        rec[f"pt_mean_{op}"] = rng.random(n)
        rec[f"unresolved_{op}"] = np.zeros(n, dtype=bool)
    return rec


@pytest.mark.unit
def test_summarise_emits_every_operator_and_is_json_serialisable():
    import json

    out = SI.summarise([_record(1000, seed=1), _record(972, seed=2)])
    assert set(out["scale_operator_profiles"]) == set(SI.OPERATOR_ORDER)
    for op in SI.OPERATOR_ORDER:
        assert len(out["scale_operator_profiles"][op]["per_origin_bins"]) == 2
    json.dumps(out)  # raises if any numpy scalar leaked through


@pytest.mark.unit
def test_summarise_reports_the_fused_rate_beside_the_branch_rate():
    out = SI.summarise([_record(1000, seed=3)])
    row = out["fused_bin_profile"][0][0]
    assert "win_rate_all" in row and "fused_win_rate_all" in row


@pytest.mark.unit
def test_backbone_degeneracy_arm_is_recorded_when_degenerate_series_exist():
    out = SI.summarise([_record(1000, seed=4)])
    assert out["backbone_degeneracy"], "expected degenerate series in the synthetic record"
    entry = out["backbone_degeneracy"][0]
    for key in (
        "backbone_mean_forecast",
        "retrieval_mean_forecast",
        "realised_future_mean",
        "win_rate_degenerate",
        "share_of_sparsest_bin_wins_from_degenerate",
    ):
        assert key in entry


@pytest.mark.unit
def test_identifiable_band_excludes_the_degenerate_cohort():
    """The two bands must be computed on different populations."""
    recs = [_record(1000, seed=5), _record(972, seed=6)]
    assert SI._bands(recs, identifiable_only=True) != SI._bands(recs, identifiable_only=False)


@pytest.mark.unit
def test_group_codes_pools_store_and_dept():
    entities = pl.DataFrame(
        {
            "store_id": ["a", "a", "b", "b"],
            "dept_id": ["x", "y", "x", "x"],
        }
    )
    codes = SI.group_codes(entities)
    assert codes[2] == codes[3]  # same store and dept
    assert codes[0] != codes[1]  # same store, different dept


@pytest.mark.unit
def test_oracle_operator_is_reachable_only_through_the_explicit_path():
    """Rule 12: query_params_for must not smuggle the oracle in unacknowledged."""
    sales = np.ones((4, 200))
    groups = np.zeros(4, dtype=np.int64)
    unit = np.ones((4, 28))
    for op in ("oracle_future_mean", "oracle_least_squares"):
        params = SI.query_params_for(op, sales, 100, groups, 365, unit)
        assert np.isfinite(params[:, 1]).all()
    with pytest.raises(ValueError, match="unknown operator"):
        SI.query_params_for("something_else", sales, 100, groups, 365, unit)


@pytest.mark.unit
def test_safe_nanmean_handles_an_all_nan_slice_without_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning becomes a failure
        assert np.isnan(SI._safe_nanmean(np.array([np.nan, np.nan])))
        assert SI._safe_nanmean(np.array([1.0, np.nan, 3.0])) == pytest.approx(2.0)
        assert np.isnan(SI._safe_nanmean(np.array([])))


@pytest.mark.unit
def test_contrasts_score_every_operator_on_one_population():
    """The whole point: an operator must not look good by dropping hard series."""
    recs = [_record(1000, seed=7), _record(972, seed=8)]
    # make one oracle unresolvable on half the series
    for rec in recs:
        rec["rmsse_oracle_future_mean"][::2] = np.nan

    c = SI.sparse_bin_contrasts(recs)
    assert set(c["operators_on_common_population"]) == set(SI.OPERATOR_ORDER)
    assert c["operators_on_common_population"]["oracle_future_mean"]["residual_attrition"] > 0
    assert c["operators_on_common_population"]["window_mean"]["residual_attrition"] == 0.0
    for key in ("deployed_all_contexts", "deployed_identifiable_only", "deployed_launched_only"):
        assert 0.0 <= c[key]["mean"] <= 1.0


@pytest.mark.unit
def test_contrasts_report_the_never_launched_share_separately():
    recs = [_record(1000, seed=9), _record(972, seed=10)]
    c = SI.sparse_bin_contrasts(recs)
    assert 0.0 <= c["never_launched_fraction_of_raw_bin"]["mean"] <= 1.0
    assert (
        c["never_launched_fraction_of_raw_bin"]["mean"]
        <= c["degenerate_fraction_of_scored_bin"]["mean"]
    ), "never-launched series are a subset of the degenerate ones"


@pytest.mark.unit
def test_symmetric_oracle_contrast_is_reported():
    """Referee item 4 applied to ourselves: both branches must get the same rescaling."""
    c = SI.sparse_bin_contrasts([_record(1000, seed=21), _record(972, seed=22)])
    sym = c["oracle_ls_retrieval_vs_oracle_ls_backbone"]
    assert 0.0 <= sym["mean"] <= 1.0
    assert sym["n_origins"] == 2


@pytest.mark.unit
def test_shares_are_reported_over_the_scored_population():
    """A series RMSSE cannot score never entered the win rate; counting it overstates."""
    recs = [_record(1000, seed=23)]
    recs[0]["rmsse_window_mean"][recs[0]["never_launched"]] = np.nan
    c = SI.sparse_bin_contrasts(recs)
    assert c["never_launched_fraction_of_scored_bin"]["mean"] == pytest.approx(0.0)
    assert c["never_launched_fraction_of_raw_bin"]["mean"] > 0.0
    assert c["fraction_of_bin_unscorable"]["mean"] > 0.0


@pytest.mark.unit
def test_symmetric_oracle_by_bin_covers_every_populated_bin():
    """The symmetric contrast must be reported across the whole profile, not just
    the sparsest bin -- otherwise the mid-band peak goes unchecked."""
    rows = SI.symmetric_oracle_by_bin([_record(1000, seed=31), _record(972, seed=32)])
    assert len(rows) >= 8
    for r in rows:
        assert 0.0 <= r["symmetric_oracle_win_rate"] <= 1.0
        assert r["shift"] == pytest.approx(r["symmetric_oracle_win_rate"] - r["deployed_win_rate"])
