"""Leakage guards for the GPU exact retriever (Phase 10 scale-up path).

Mirrors the guarantees of ``retrieval_faiss``: built candidate windows never reach
into or past the training boundary, and the retriever refuses an origin that would
admit a leaking candidate (``t_r + H < origin`` must hold).
"""

from __future__ import annotations

import numpy as np
import pytest

from scalerag.leakage import LeakageViolation
from scalerag.retrieval_gpu import build_windows, retrieve_scaleaware_gpu

L, H, STRIDE = 56, 28, 7


@pytest.mark.leakage
def test_build_windows_never_leak_into_future() -> None:
    sales = np.random.default_rng(0).integers(0, 10, size=(5, 300)).astype(float)
    train_end = 200
    _ctx, _cont, sidx, t_r = build_windows(sales, train_end, L, H, STRIDE)
    assert t_r.size > 0
    # every candidate continuation ends within training: t_r + H <= train_end
    assert int(t_r.max()) + H <= train_end
    assert sidx.max() == 4 and sidx.min() == 0


@pytest.mark.leakage
def test_retrieve_raises_on_leaking_origin() -> None:
    """An origin at the last candidate's continuation-end violates t_r + H < origin."""
    sales = np.random.default_rng(1).integers(0, 10, size=(6, 300)).astype(float)
    cat = np.zeros(6, dtype=np.int64)
    train_end = 200
    _c, _k, _s, t_r = build_windows(sales, train_end, L, H, STRIDE)
    leaking_origin = int(t_r.max()) + H  # exactly the last candidate's end -> not < origin
    queries = np.stack([sales[i, train_end - L : train_end] for i in range(6)])
    # must raise BEFORE any GPU work (the guard runs right after window construction).
    with pytest.raises(LeakageViolation):
        retrieve_scaleaware_gpu(
            sales,
            cat,
            train_end,
            leaking_origin,
            queries,
            cat,
            [0.1, 0.5, 0.9],
            context_length=L,
            horizon=H,
            stride=STRIDE,
            k=5,
            device="cpu",
        )
