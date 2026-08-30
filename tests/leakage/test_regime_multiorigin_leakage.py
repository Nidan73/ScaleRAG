from __future__ import annotations

import numpy as np
import pytest

from scalerag.leakage import LeakageViolation
from scalerag.retrieval import WindowDatabase

L, H = 56, 28


def _panel(n: int = 6, days: int = 900) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.poisson(1.0, size=(n, days)).astype(np.float64)


@pytest.mark.leakage
def test_pool_built_at_earliest_origin_is_legal_at_every_origin():
    """A pool built once must be built for the EARLIEST origin to stay legal."""
    panel = _panel()
    origins = [500 - i * H for i in range(8)]
    earliest = min(origins)
    db = WindowDatabase.from_training(panel, earliest, L, H, stride=7)
    for o in origins:
        legal = db.legal_mask(o)
        assert legal.all(), f"pool built at {earliest} has illegal candidates at origin {o}"
        db.assert_all_legal(np.flatnonzero(legal), o)


@pytest.mark.leakage
def test_pool_built_at_latest_origin_is_illegal_at_earlier_origins():
    """The violation this guards against must actually be caught, not merely absent."""
    panel = _panel()
    origins = [500 - i * H for i in range(8)]
    latest = max(origins)
    db = WindowDatabase.from_training(panel, latest, L, H, stride=7)
    earliest = min(origins)
    assert not db.legal_mask(earliest).all(), "expected illegal candidates at the earliest origin"
    illegal = np.flatnonzero(~db.legal_mask(earliest))
    with pytest.raises(LeakageViolation):
        db.assert_all_legal(illegal, earliest)


@pytest.mark.leakage
def test_per_origin_pool_never_reaches_the_evaluation_window():
    """Rebuilding per origin, the frozen path, keeps t_r + H strictly below the origin."""
    panel = _panel()
    for o in [500 - i * H for i in range(8)]:
        db = WindowDatabase.from_training(panel, o, L, H, stride=7)
        legal = np.flatnonzero(db.legal_mask(o))
        assert legal.size > 0
        assert (db.t_r[legal] + H < o).all()
