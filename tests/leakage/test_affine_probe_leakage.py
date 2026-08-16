"""Leakage guards for the affine probe's candidate pool.

The probe builds its own retrieval pool, so it needs its own horizon-guard tests
(rule 3): every candidate must satisfy ``t_r + H < origin``. Each test here
asserts a deliberately introduced violation is *caught*, not merely that clean
input passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from scalerag import leakage
from scalerag.affine_probe import (
    AffineCondition,
    _build_database,
    make_motif_panel,
    run_condition,
)

pytestmark = pytest.mark.leakage

PERIOD = 12
CONTEXT = 24
HORIZON = 6
LENGTH = CONTEXT + HORIZON + PERIOD * 8


def _panel_and_db() -> tuple[object, object, int]:
    panel = make_motif_panel(4, LENGTH, PERIOD, 0.01, np.random.default_rng(3), rows_per_motif=3)
    origin = LENGTH - HORIZON
    db = _build_database(panel, LENGTH - HORIZON - 1, CONTEXT, HORIZON, stride=3)
    return panel, db, origin


def test_every_pool_candidate_closes_before_the_origin() -> None:
    _, db, origin = _panel_and_db()
    legal = db.t_r[db.legal_mask(origin)]  # type: ignore[attr-defined]
    assert legal.size > 0, "pool is empty; the guard test would be vacuous"
    assert np.all(legal + HORIZON < origin)


def test_guard_rejects_a_candidate_whose_continuation_reaches_the_origin() -> None:
    """A candidate ending exactly at the origin is a violation, not a boundary case."""
    _, _, origin = _panel_and_db()
    with pytest.raises(leakage.LeakageViolation):
        leakage.assert_retrieval_horizon(origin - HORIZON, HORIZON, origin)


def test_broken_legality_filter_is_caught_downstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defence in depth: if the mask were subverted, the assertion must still fire.

    ``run_condition`` filters by ``legal_mask`` and then re-asserts the guard on
    what it actually selected. The pool here deliberately extends past the origin,
    so forcing the mask to admit everything really does hand future-bearing
    candidates to the retriever; the second check has to catch them.
    """
    panel = make_motif_panel(4, LENGTH, PERIOD, 0.01, np.random.default_rng(3), rows_per_motif=3)
    # Pool spans the WHOLE series, so candidates exist whose continuation covers the origin.
    wide_db = _build_database(panel, LENGTH, CONTEXT, HORIZON, stride=3)
    origin = LENGTH // 2
    assert np.any(~wide_db.legal_mask(origin)), "pool must contain an illegal candidate"

    monkeypatch.setattr(
        type(wide_db), "legal_mask", lambda self, _origin: np.ones(len(self), dtype=bool)
    )
    rows = np.zeros(4, dtype=np.int64)
    t_r = np.full(4, origin, dtype=np.int64)
    with pytest.raises(leakage.LeakageViolation):
        run_condition(
            panel,
            wide_db,
            AffineCondition("znorm", True),
            a=2.0,
            b=1.0,
            origin=origin,
            k=3,
            query_rows=rows,
            query_t_r=t_r,
        )


def test_empty_legal_pool_raises_instead_of_silently_forecasting() -> None:
    """With no legal candidate the probe must fail loudly, never fall back quietly."""
    panel, db, _ = _panel_and_db()
    rows = np.zeros(4, dtype=np.int64)
    early_origin = CONTEXT  # nothing can have closed this early
    t_r = np.full(4, LENGTH - HORIZON, dtype=np.int64)
    with pytest.raises(leakage.LeakageViolation, match="no rule-3-legal candidates"):
        run_condition(
            panel,  # type: ignore[arg-type]
            db,  # type: ignore[arg-type]
            AffineCondition("znorm", True),
            a=1.0,
            b=0.0,
            origin=early_origin,
            k=3,
            query_rows=rows,
            query_t_r=t_r,
        )


def test_the_query_future_is_never_a_candidate_continuation() -> None:
    """The window the query is built from must not be retrievable as its own answer."""
    _, db, origin = _panel_and_db()
    legal_ids = np.flatnonzero(db.legal_mask(origin))  # type: ignore[attr-defined]
    # Continuations span [t_r, t_r + H); the query's own future spans [origin, origin + H).
    assert np.all(db.t_r[legal_ids] + HORIZON <= origin - 1)  # type: ignore[attr-defined]
