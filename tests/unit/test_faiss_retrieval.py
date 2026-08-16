"""Scale-aware FAISS retrieval tests (Phase 5, task 13)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scalerag.retrieval import WindowDatabase
from scalerag.retrieval_faiss import ScaleAwareIndex, restore_continuation

L, H, TRAIN_END = 14, 7, 150
ORIGIN = 200  # all training candidates legal


@pytest.fixture
def panel():
    rng = np.random.default_rng(1)
    t = np.arange(260)
    base = np.sin(2 * np.pi * t / 7)[None, :]
    # 4 series with different absolute scales (tests scale-awareness)
    scales = np.array([1.0, 10.0, 100.0, 5.0])[:, None]
    return (scales * (base + 2) + rng.normal(0, 0.2, (4, 260))).astype(np.float64)


@pytest.fixture
def entities():
    return pl.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "item_id": ["i1", "i2", "i1", "i2"],
            "dept_id": ["D1", "D1", "D2", "D2"],
            "cat_id": ["C1", "C1", "C2", "C2"],
            "store_id": ["S1", "S1", "S2", "S2"],
            "state_id": ["X", "X", "Y", "Y"],
        }
    )


@pytest.fixture
def db(panel):
    return WindowDatabase.from_training(panel, TRAIN_END, L, H)


# --- FAISS vs brute-force equivalence ---
@pytest.mark.unit
def test_faiss_matches_bruteforce_l2(db, entities, panel):
    idx = ScaleAwareIndex(db, entities, scale="raw", metric="l2")
    q = panel[0, TRAIN_END - L : TRAIN_END]
    ids, _d, _qp = idx.search(q, ORIGIN, k=5)

    legal = np.flatnonzero(db.legal_mask(ORIGIN))
    dd = ((db.contexts[legal] - q) ** 2).sum(1)
    bf = legal[np.lexsort((legal, dd))[:5]]
    assert list(ids) == list(bf)


@pytest.mark.unit
def test_faiss_deterministic(db, entities, panel):
    idx = ScaleAwareIndex(db, entities, scale="znorm", metric="l2")
    q = panel[1, TRAIN_END - L : TRAIN_END]
    a, _, _ = idx.search(q, ORIGIN, k=5, query_series_idx=1)
    b, _, _ = idx.search(q, ORIGIN, k=5, query_series_idx=1)
    assert list(a) == list(b)


# --- scale restoration ---
@pytest.mark.unit
def test_restore_continuation_maps_scale():
    cont = np.array([1.0, 2.0, 3.0])
    restored = restore_continuation(cont, cand_params=(0.0, 2.0), query_params=(0.0, 4.0))
    assert np.allclose(restored, [2.0, 4.0, 6.0])  # /2 * 4


@pytest.mark.unit
def test_scale_restoration_aligns_level(db, entities, panel):
    # query series 0 (scale ~1); without restoration, retrieved continuations from
    # high-scale series would inflate the forecast. With znorm + restoration the
    # forecast should sit near series 0's own level.
    idx = ScaleAwareIndex(db, entities, scale="znorm", metric="l2")
    q = panel[0, TRAIN_END - L : TRAIN_END]
    ql = [0.5]
    restored, _ = idx.forecast(
        q, ORIGIN, H, k=5, quantile_levels=ql, query_series_idx=0, restore_scale=True
    )
    raw, _ = idx.forecast(
        q, ORIGIN, H, k=5, quantile_levels=ql, query_series_idx=0, restore_scale=False
    )
    series0_level = panel[0, :TRAIN_END].mean()
    # restored forecast is much closer to series 0's level than the unrestored one
    assert abs(restored.mean() - series0_level) < abs(raw.mean() - series0_level)


# --- metadata-filtered candidate sets ---
@pytest.mark.unit
def test_same_store_filter_restricts_candidates(db, entities, panel):
    idx = ScaleAwareIndex(db, entities, scale="znorm", metric="l2")
    q = panel[0, TRAIN_END - L : TRAIN_END]  # series 0 -> store S1 (series 0,1)
    ids, _d, _qp = idx.search(q, ORIGIN, k=10, query_series_idx=0, meta_filter="store_id")
    stores = entities["store_id"].to_numpy()[db.series_idx[ids]]
    assert set(stores) == {"S1"}


@pytest.mark.unit
def test_seasonal_filter_matches_phase(db, entities, panel):
    idx = ScaleAwareIndex(db, entities, scale="znorm", metric="l2", season=7)
    q = panel[2, TRAIN_END - L : TRAIN_END]
    ids, _d, _qp = idx.search(q, ORIGIN, k=10, query_series_idx=2, meta_filter="seasonal")
    assert np.all((db.t_r[ids] % 7) == ((ORIGIN - 1) % 7))


@pytest.mark.unit
def test_cosine_metric_runs(db, entities, panel):
    idx = ScaleAwareIndex(db, entities, scale="znorm", metric="cosine")
    q = panel[3, TRAIN_END - L : TRAIN_END]
    ids, _d, _qp = idx.search(q, ORIGIN, k=5, query_series_idx=3)
    assert 0 < len(ids) <= 5
