from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import scalerag_eval as SE  # noqa: E402,N812 -- matches the convention in scripts/
from scalerag import scale_operators as so  # noqa: E402


def _panel():
    rng = np.random.default_rng(3)
    n, days = 8, 400
    sales = rng.poisson(1.5, size=(n, days)).astype(np.float64)
    entities = pl.DataFrame(
        {
            "id": [f"s{i}" for i in range(n)],
            "item_id": [f"i{i}" for i in range(n)],
            "dept_id": ["d0"] * n,
            "cat_id": ["c0"] * n,
            "store_id": ["st0"] * n,
            "state_id": ["CA"] * n,
        }
    )
    return sales, entities


@pytest.mark.unit
def test_default_return_is_unchanged():
    sales, entities = _panel()
    o = 300
    queries = [sales[i, o - SE.L : o] for i in range(sales.shape[0])]
    out = SE.retrieval_all(sales, entities, o, queries)
    assert len(out) == 4


@pytest.mark.unit
def test_artifacts_reproduce_the_point_forecast_exactly():
    """Re-restoring the returned artifacts must recover the frozen forecast bit-for-bit."""
    sales, entities = _panel()
    o = 300
    queries = [sales[i, o - SE.L : o] for i in range(sales.shape[0])]
    pt, _qt, _nnd, _dis, arts = SE.retrieval_all(sales, entities, o, queries, return_artifacts=True)
    db = SE.WindowDatabase.from_training(sales, o, SE.L, SE.H, stride=7)
    checked = 0
    for i, art in enumerate(arts):
        if art["ids"].size == 0:
            continue
        conts = db.continuations[art["ids"]]
        restored = so.restore_batch(conts, art["cand_params"], art["query_params"])
        np.testing.assert_allclose(np.clip(restored, 0.0, None).mean(0), pt[i], rtol=0, atol=0)
        checked += 1
    assert checked > 0, "no series had retrieved candidates; the test proved nothing"
