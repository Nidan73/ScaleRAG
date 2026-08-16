"""Unit tests for the ScaleRAG native-protocol adapter (Phase 11A)."""

from __future__ import annotations

import numpy as np
import pytest

from scalerag.native import (
    NativeScaleRetriever,
    fixed_fusion,
    topk_exact,
)
from scalerag.reproducibility import set_seed
from scalerag.retrieval_faiss import restore_continuation


@pytest.mark.unit
def test_topk_exact_matches_brute_force() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    q = rng.standard_normal((7, 32))
    c = rng.standard_normal((300, 32))
    ids, dists = topk_exact(q, c, 10)
    brute = ((c[None, :, :] - q[:, None, :]) ** 2).sum(-1)  # (7, 300)
    ref_ids = np.argsort(brute, axis=1, kind="stable")[:, :10]
    assert (ids == ref_ids).all()
    assert np.abs(dists - np.take_along_axis(brute, ids, 1)).max() < 1e-9


@pytest.mark.unit
def test_topk_exact_tie_break_by_index() -> None:
    # identical candidate vectors -> distances tie -> must return the lowest indices in order
    c = np.ones((20, 4))
    q = np.zeros((1, 4))
    ids, _ = topk_exact(q, c, 5)
    assert ids[0].tolist() == [0, 1, 2, 3, 4]


@pytest.mark.unit
def test_faiss_equivalence_numpy_reference() -> None:
    faiss = pytest.importorskip("faiss")
    set_seed(1)
    rng = np.random.default_rng(1)
    c = rng.standard_normal((500, 48)).astype(np.float32)
    q = rng.standard_normal((16, 48)).astype(np.float32)
    index = faiss.IndexFlatL2(48)
    index.add(c)
    _fd, fi = index.search(q, 10)
    ni, _nd = topk_exact(q.astype(np.float64), c.astype(np.float64), 10)
    # exact index agreement where there are no distance ties (random continuous data)
    assert (fi == ni).mean() > 0.999


@pytest.mark.unit
def test_fixed_fusion_endpoints_and_validation() -> None:
    a = np.full((3, 4), 2.0)
    b = np.full((3, 4), 6.0)
    assert np.allclose(fixed_fusion(a, b, 0.0), a)
    assert np.allclose(fixed_fusion(a, b, 1.0), b)
    assert np.allclose(fixed_fusion(a, b, 0.25), 0.75 * a + 0.25 * b)
    with pytest.raises(ValueError, match="weight"):
        fixed_fusion(a, b, 1.5)


@pytest.mark.unit
def test_restore_continuation_maps_to_query_scale() -> None:
    # rms scaling: a candidate continuation restored to a query 3x larger should scale ~3x
    cont = np.array([1.0, 2.0, 3.0, 4.0])
    cand_params = np.array([0.0, 1.0])  # loc 0, scale 1
    query_params = np.array([0.0, 3.0])  # loc 0, scale 3
    restored = restore_continuation(cont, cand_params, query_params)
    assert np.allclose(restored, cont * 3.0)
    # identity when scales match
    assert np.allclose(restore_continuation(cont, cand_params, cand_params), cont)


@pytest.mark.unit
def test_restored_differs_from_raw_and_shapes() -> None:
    set_seed(2)
    rng = np.random.default_rng(2)
    series = rng.standard_normal(3000)
    r = NativeScaleRetriever(series, train_end=2000, scale="rms", context_length=64, horizon=16)
    origins = np.array([2100, 2200, 2300])
    queries = np.stack([series[o - 64 : o] for o in origins])
    raw = r.forecast_batch(queries, origins, k=8, restore=False)
    res = r.forecast_batch(queries, origins, k=8, restore=True)
    assert raw.point.shape == (3, 16)
    assert not np.allclose(raw.point, res.point)  # restoration changes the forecast
    assert raw.fallback_count == 0 and res.invalid_query_scale == 0


@pytest.mark.unit
def test_invalid_scale_is_counted_and_falls_back() -> None:
    # 'mean' scaling with a near-zero-mean query context -> invalid scale -> constant fallback
    set_seed(3)
    rng = np.random.default_rng(3)
    series = rng.standard_normal(2000)
    r = NativeScaleRetriever(
        series, train_end=1500, scale="mean", context_length=64, horizon=16, scale_eps=1e-2
    )
    # near-zero (but nonzero) mean -> tiny 'mean' denominator, NOT caught by _fit_params'
    # exact-zero guard -> flagged invalid and served by the constant fallback.
    w = rng.standard_normal(64)
    near_zero_ctx = w - w.mean() + 0.003  # mean 0.003 < scale_eps 1e-2
    origins = np.array([1600])
    out = r.forecast_batch(near_zero_ctx[None, :], origins, k=8, restore=True)
    assert out.invalid_query_scale == 1
    assert out.fallback_count == 1
    assert np.allclose(out.point[0], near_zero_ctx.mean())  # constant fallback


@pytest.mark.unit
def test_unknown_scale_and_bad_series_raise() -> None:
    series = np.zeros(1000)
    with pytest.raises(ValueError, match="scale must be one of"):
        NativeScaleRetriever(series, 800, scale="bogus", context_length=32, horizon=8)
    with pytest.raises(ValueError, match="1-D"):
        NativeScaleRetriever(np.zeros((2, 100)), 80, scale="rms", context_length=16, horizon=4)
