"""Guard the id-ordered reshape used to build dataset panels.

`scripts/gate_transfer_run.py` turns a long (id, day_idx, sales) frame into a dense
(n_series, n_days) matrix by sorting and reshaping. If the sort and the reshape ever
disagree, every row silently belongs to the wrong series and every downstream number
is wrong without anything raising. This pins the contract.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.unit


def _long_frame(n_series: int, n_days: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": np.repeat(np.arange(n_series), n_days),
            "day_idx": np.tile(np.arange(1, n_days + 1), n_series),
            # Encode (series, day) in the value so misalignment is detectable.
            "sales": np.repeat(np.arange(n_series) * 1000.0, n_days)
            + np.tile(np.arange(1, n_days + 1), n_series),
        }
    )


def test_sorted_reshape_puts_each_series_on_its_own_row() -> None:
    n_series, n_days = 7, 5
    dyn = _long_frame(n_series, n_days).sample(fraction=1.0, shuffle=True, seed=3)
    dyn = dyn.sort(["id", "day_idx"])
    sales = dyn["sales"].to_numpy().astype(np.float64).reshape(n_series, n_days)
    for s in range(n_series):
        assert np.allclose(sales[s], s * 1000.0 + np.arange(1, n_days + 1))


def test_reshape_without_sorting_is_detectably_wrong() -> None:
    """The test above must be capable of failing; shuffled input must not pass."""
    n_series, n_days = 7, 5
    dyn = _long_frame(n_series, n_days).sample(fraction=1.0, shuffle=True, seed=3)
    sales = dyn["sales"].to_numpy().astype(np.float64).reshape(n_series, n_days)
    ok = all(np.allclose(sales[s], s * 1000.0 + np.arange(1, n_days + 1)) for s in range(n_series))
    assert not ok, "unsorted reshape happened to align; the guard would be vacuous"


def test_entity_filter_and_dynamic_filter_select_the_same_series() -> None:
    """Subsetting must be applied to both frames from one id list, not independently."""
    n_series, n_days = 9, 4
    ents = pl.DataFrame({"id": np.arange(n_series), "cat_id": np.arange(n_series) % 3})
    dyn = _long_frame(n_series, n_days)
    keep = ents["id"].head(4)
    ents_k = ents.filter(pl.col("id").is_in(keep))
    dyn_k = dyn.filter(pl.col("id").is_in(keep)).sort(["id", "day_idx"])
    assert ents_k.height == 4
    assert dyn_k.height == 4 * n_days
    sales = dyn_k["sales"].to_numpy().astype(np.float64).reshape(ents_k.height, n_days)
    assert np.allclose(sales[:, 0], ents_k["id"].to_numpy() * 1000.0 + 1)
