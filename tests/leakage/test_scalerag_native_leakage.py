"""Leakage / split-integrity guards for the ScaleRAG native adapter (Phase 11A).

Each test asserts that a temporal violation is *caught* — illegal (future-touching)
candidates are excluded from every result, and an origin with no rule-3-legal candidate
raises — not merely that clean input passes (tests/ rules).
"""

from __future__ import annotations

import numpy as np
import pytest

from scalerag import leakage
from scalerag.native import NativeScaleRetriever
from scalerag.reproducibility import set_seed


def _retriever(seed: int, train_end: int = 3000, ctx: int = 128, hor: int = 32):
    series = np.random.default_rng(seed).standard_normal(4000)
    return NativeScaleRetriever(
        series, train_end=train_end, scale="rms", context_length=ctx, horizon=hor
    ), series


@pytest.mark.leakage
def test_candidate_pool_is_strictly_train_only() -> None:
    set_seed(0)
    r, _ = _retriever(0)
    # every candidate continuation ends at/before train_end (no val/test future in the pool)
    assert (r.db.t_r + r.horizon <= 3000).all()
    # and the whole pool is rule-3-legal for a test-region origin
    assert r.db.legal_mask(3000 + 1).all()


@pytest.mark.leakage
def test_returned_candidates_are_all_rule3_legal() -> None:
    # At a mid-range origin, higher-t_r candidates are illegal (t_r + H >= origin) and MUST
    # be excluded from the retrieved set — the guard filters them out.
    set_seed(1)
    r, series = _retriever(1)
    origin = 2000  # candidates with t_r + 32 >= 2000 are illegal
    ctx = series[origin - 128 : origin][None, :]
    tk = r.retrieve(ctx, np.array([origin]), k=20)
    returned = tk.ids.ravel()
    assert returned.size == 20
    assert (r.db.t_r[returned] + r.horizon < origin).all()  # no future leak in any result


@pytest.mark.leakage
def test_origin_with_no_legal_candidates_raises() -> None:
    # An origin so early that every candidate continuation reaches into/at it must fail loudly.
    set_seed(2)
    r, series = _retriever(2)
    early_origin = 160  # min candidate is t_r=128 -> t_r+H=160, not < 160 -> none legal
    ctx = series[early_origin - 128 : early_origin][None, :]
    with pytest.raises(leakage.LeakageViolation):
        r.retrieve(ctx, np.array([early_origin]), k=5)


@pytest.mark.leakage
def test_valid_test_origin_passes() -> None:
    set_seed(3)
    r, series = _retriever(3)
    origin = 3500  # well past all candidate continuations
    ctx = series[origin - 128 : origin][None, :]
    out = r.forecast_batch(ctx, np.array([origin]), k=5, restore=True)
    assert out.point.shape == (1, r.horizon)
    assert out.fallback_count == 0


@pytest.mark.leakage
def test_pool_ignores_all_future_days() -> None:
    # Monotone index series: any future leak into the pool would be numerically obvious.
    set_seed(4)
    series = np.arange(2000, dtype=float)
    train_end = 1500
    r = NativeScaleRetriever(
        series, train_end=train_end, scale="rms", context_length=64, horizon=16
    )
    assert (r.db.t_r + r.horizon <= train_end).all()
    assert r.db.continuations.max() < train_end  # values are indices; all strictly < train_end
