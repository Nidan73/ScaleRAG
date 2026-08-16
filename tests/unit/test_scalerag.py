"""ScaleRAG fusion + paired-bootstrap tests (Phase 9)."""

from __future__ import annotations

import numpy as np
import pytest

from scalerag import fusion


@pytest.mark.unit
def test_fuse_scalar_alpha():
    c = np.array([[2.0, 4.0]])
    r = np.array([[0.0, 0.0]])
    cq = np.ones((1, 2, 3))
    rq = np.zeros((1, 2, 3))
    p, q = fusion.fuse(c, cq, r, rq, alpha=0.25)
    assert np.allclose(p, [[1.5, 3.0]])
    assert np.allclose(q, 0.75)


@pytest.mark.unit
def test_fuse_per_series_alpha():
    c = np.array([[4.0, 4.0], [4.0, 4.0]])
    r = np.array([[0.0, 0.0], [0.0, 0.0]])
    cq = np.ones((2, 2, 3))
    rq = np.zeros((2, 2, 3))
    p, _ = fusion.fuse(c, cq, r, rq, alpha=np.array([0.0, 1.0]))
    assert np.allclose(p[0], 4.0) and np.allclose(p[1], 0.0)


@pytest.mark.unit
def test_bootstrap_detects_real_improvement():
    rng = np.random.default_rng(0)
    base = rng.uniform(1.0, 2.0, 500)
    method = base * 0.9  # uniform 10% improvement
    out = fusion.paired_bootstrap_rel_improvement(base, method, n_boot=500)
    assert out["rel_improvement"] == pytest.approx(0.10, abs=0.02)
    assert out["excludes_zero"] and out["ci95_low"] > 0


@pytest.mark.unit
def test_bootstrap_no_improvement_includes_zero():
    rng = np.random.default_rng(1)
    base = rng.uniform(1.0, 2.0, 500)
    method = base + rng.normal(0, 0.3, 500)  # noisy, no real gain
    out = fusion.paired_bootstrap_rel_improvement(base, method, n_boot=500)
    assert not out["excludes_zero"] or abs(out["rel_improvement"]) < 0.05
