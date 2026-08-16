"""Temporal-retrieval leakage & correctness tests (Phase 4, task 12)."""

from __future__ import annotations

import numpy as np
import pytest

from scalerag import leakage
from scalerag.retrieval import (
    EuclideanRetriever,
    RandomRetriever,
    SeasonalRetriever,
    WindowDatabase,
)

L, H = 14, 7
TRAIN_END = 150
ORIGIN = 180  # a val-like origin after train_end


@pytest.fixture
def panel():
    rng = np.random.default_rng(0)
    # 5 series x 200 days, weekly seasonality + noise
    t = np.arange(200)
    base = np.sin(2 * np.pi * t / 7)[None, :]
    return (base + rng.normal(0, 0.3, size=(5, 200))).astype(np.float64)


@pytest.fixture
def db(panel):
    return WindowDatabase.from_training(panel, TRAIN_END, L, H, stride=1)


# --- 1. illegal retrieval horizons -----------------------------------------
@pytest.mark.leakage
def test_all_db_candidates_are_training_only(db):
    # candidate-index fitting on training data only: every continuation ends <= train_end
    assert len(db) > 0
    assert np.all(db.t_r + db.horizon <= TRAIN_END)


@pytest.mark.leakage
def test_legal_mask_enforces_horizon_guard(db):
    origin = 160  # closer origin: some candidates become illegal
    mask = db.legal_mask(origin)
    assert np.all(db.t_r[mask] + db.horizon < origin)
    assert np.all(db.t_r[~mask] + db.horizon >= origin)


@pytest.mark.leakage
def test_assert_all_legal_raises_on_violation(db):
    # force an origin that makes the latest candidate illegal
    latest = int(db.t_r.max())
    bad_origin = latest + db.horizon  # t_r + H == origin -> NOT < origin
    idx = np.flatnonzero(db.t_r == latest)[:1]
    with pytest.raises(leakage.LeakageViolation):
        db.assert_all_legal(idx, bad_origin)


@pytest.mark.parametrize("retriever_cls", [RandomRetriever, SeasonalRetriever, EuclideanRetriever])
@pytest.mark.leakage
def test_retrievers_never_return_illegal(db, panel, retriever_cls):
    q = panel[0, ORIGIN - L : ORIGIN]
    got = retriever_cls().retrieve(db, q, ORIGIN, k=5, query_series_idx=0)
    for c in got:
        assert c.t_r + H < ORIGIN  # rule 3 holds for every returned candidate


# --- 2. duplicate / overlapping windows across splits ----------------------
@pytest.mark.leakage
def test_retrieved_windows_do_not_overlap_target_horizon(db, panel):
    q = panel[0, ORIGIN - L : ORIGIN]
    got = EuclideanRetriever().retrieve(db, q, ORIGIN, k=5, query_series_idx=0)
    target_horizon = [(ORIGIN + 1, ORIGIN + H)]
    for c in got:
        # candidate context AND continuation must be disjoint from target horizon
        leakage.assert_no_window_overlap([(c.t_r - L + 1, c.t_r)], target_horizon)
        leakage.assert_no_window_overlap([(c.t_r + 1, c.t_r + H)], target_horizon)


# --- 3. candidate-index fitting on training data only ----------------------
@pytest.mark.leakage
def test_db_ignores_post_train_data(panel):
    # Corrupt everything after train_end; the DB must be identical.
    clean = WindowDatabase.from_training(panel, TRAIN_END, L, H)
    corrupt = panel.copy()
    corrupt[:, TRAIN_END:] = 999999.0
    after = WindowDatabase.from_training(corrupt, TRAIN_END, L, H)
    assert np.array_equal(clean.contexts, after.contexts)
    assert np.array_equal(clean.continuations, after.continuations)


# --- 4. deterministic top-k ------------------------------------------------
@pytest.mark.leakage
def test_euclidean_topk_deterministic(db, panel):
    q = panel[2, ORIGIN - L : ORIGIN]
    a = EuclideanRetriever().retrieve(db, q, ORIGIN, k=5, query_series_idx=2)
    b = EuclideanRetriever().retrieve(db, q, ORIGIN, k=5, query_series_idx=2)
    assert [(c.series_idx, c.t_r) for c in a] == [(c.series_idx, c.t_r) for c in b]
    # distances are non-decreasing
    dists = [c.distance for c in a]
    assert dists == sorted(dists)


@pytest.mark.leakage
def test_random_deterministic_with_seed(db, panel):
    q = panel[1, ORIGIN - L : ORIGIN]
    a = RandomRetriever(seed=7).retrieve(db, q, ORIGIN, k=5, query_series_idx=1)
    b = RandomRetriever(seed=7).retrieve(db, q, ORIGIN, k=5, query_series_idx=1)
    assert [c.t_r for c in a] == [c.t_r for c in b]


# --- 5. empty or insufficient history --------------------------------------
@pytest.mark.leakage
def test_insufficient_history_yields_empty_db(panel):
    # train_end so small no candidate fits (need t_r >= L and t_r+H <= train_end)
    tiny = WindowDatabase.from_training(panel, train_end=L + H - 1, context_length=L, horizon=H)
    assert len(tiny) == 0


@pytest.mark.leakage
def test_no_legal_candidates_returns_empty(db, panel):
    q = panel[0, :L]
    # origin so early that no candidate satisfies t_r + H < origin
    got = EuclideanRetriever().retrieve(db, q, origin=L, k=5, query_series_idx=0)
    assert got == []


# --- 6. unavailable future covariates --------------------------------------
@pytest.mark.leakage
def test_unavailable_future_covariate_rejected():
    # Only genuinely known-future covariates may be used at forecast time (task 4).
    with pytest.raises(leakage.LeakageViolation):
        leakage.assert_no_future_covariates(
            covariate_columns=["sell_price", "future_demand"],
            known_future_columns=["sell_price", "snap"],
        )


@pytest.mark.leakage
def test_known_future_covariates_accepted():
    assert (
        leakage.assert_no_future_covariates(
            covariate_columns=["sell_price", "snap"],
            known_future_columns=["sell_price", "snap", "event"],
        )
        is None
    )
