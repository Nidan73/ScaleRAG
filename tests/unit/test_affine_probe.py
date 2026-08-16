"""Unit tests for the controlled synthetic affine probe.

These assert the two properties the probe exists to measure — invariance of
retrieval and equivariance of reconstruction — as numerical predictions, not as
smoke checks. A regression in the scale primitives imported from
``retrieval_faiss`` should fail here.
"""

from __future__ import annotations

import numpy as np
import pytest

from scalerag.affine_probe import (
    AffineCondition,
    make_motif_panel,
    run_affine_grid,
)

pytestmark = pytest.mark.unit

# Small but non-degenerate: enough motifs that chance retrieval is ~1/8.
GRID_KW = {
    "n_motifs": 8,
    "rows_per_motif": 4,
    "period": 12,
    "context_length": 24,
    "horizon": 6,
    "n_queries": 96,
    "noise": 0.02,
    "k": 3,
    "seed": 20260726,
}
A_VALUES = [1.0, 10.0, 100.0]
B_VALUES = [0.0, 200.0]


def _cells(condition: AffineCondition, **overrides: object) -> dict[tuple[float, float], object]:
    kw = {**GRID_KW, **overrides}
    grid = run_affine_grid([condition], A_VALUES, B_VALUES, **kw)  # type: ignore[arg-type]
    return {(c.a, c.b): c for c in grid.cells}


def test_znorm_restore_is_exactly_affine_equivariant() -> None:
    """Query-scale restoration in z-space must be invariant to BOTH a and b.

    The scale-free error should be bit-identical across the whole (a, b) grid: the
    reconstruction maps the retrieved future into the query's coordinate system
    exactly, so the only residual is the observation noise, which the transform
    does not touch.
    """
    cells = _cells(AffineCondition("znorm", True))
    nmse = np.array([c.nmse for c in cells.values()])  # type: ignore[attr-defined]
    assert np.ptp(nmse) < 1e-12, f"nmse varied across the affine grid: {nmse}"
    # The residual is noise, not method error: well below predicting the mean (1.0).
    assert nmse.max() < 0.01


def test_znorm_retrieval_is_affine_invariant() -> None:
    """Normalised retrieval finds the correct motif regardless of the transform."""
    for cell in _cells(AffineCondition("znorm", False)).values():
        assert cell.hit_rate == 1.0, f"znorm lost the analogue at a={cell.a}, b={cell.b}"  # type: ignore[attr-defined]


def test_raw_retrieval_breaks_under_affine_transform() -> None:
    """Raw-space retrieval is the failure the mechanism exists to fix.

    It must be correct in the untransformed control and must degrade sharply once
    the query's magnitude or level is moved away from its own row's.
    """
    cells = _cells(AffineCondition("raw", False))
    assert cells[(1.0, 0.0)].hit_rate == 1.0, "raw should be exact in the control cell"  # type: ignore[attr-defined]
    transformed = [c.hit_rate for (a, b), c in cells.items() if (a, b) != (1.0, 0.0)]  # type: ignore[attr-defined]
    assert max(transformed) < 0.6, f"raw retrieval did not break: {transformed}"


def test_invariance_alone_does_not_fix_the_forecast() -> None:
    """Finding the right analogue is not enough — the future is in donor units.

    This is the cell that separates the two claims. Retrieval is perfect (hit rate
    1.0) yet the forecast is no better than predicting the mean, until restoration
    is applied.
    """
    without = _cells(AffineCondition("znorm", False))
    with_ = _cells(AffineCondition("znorm", True))
    for key in [(10.0, 0.0), (100.0, 200.0)]:
        assert without[key].hit_rate == 1.0  # type: ignore[attr-defined]
        assert without[key].nmse > 0.5, "un-restored error should approach the truth variance"  # type: ignore[attr-defined]
        assert with_[key].nmse < without[key].nmse / 100.0  # type: ignore[attr-defined]


def test_mean_scaling_is_scale_but_not_location_equivariant() -> None:
    """The frozen M5 configuration uses ``mean``, which carries no location term.

    It must therefore be exactly equivariant under a pure rescaling and must break
    under an added offset. Recording this keeps the paper's equivariance claim
    scoped to ``znorm`` rather than to the shipped configuration.
    """
    cells = _cells(AffineCondition("mean", True))
    at_b0 = [c.nmse for (a, b), c in cells.items() if b == 0.0]  # type: ignore[attr-defined]
    assert np.ptp(at_b0) < 1e-12, f"mean should be exactly scale-equivariant: {at_b0}"
    assert cells[(1.0, 200.0)].nmse > 100 * cells[(1.0, 0.0)].nmse, (  # type: ignore[attr-defined]
        "mean scaling must NOT survive a location shift"
    )


def test_panel_gives_each_motif_several_magnitudes() -> None:
    """Guards the design flaw that made an earlier version of this probe useless.

    If every row were unit-scaled, ``||c||`` would be constant across candidates,
    raw L2 would reduce to shape correlation, and raw retrieval would look
    accidentally scale-invariant.
    """
    panel = make_motif_panel(6, 240, 12, 0.01, np.random.default_rng(0), rows_per_motif=4)
    for motif in range(6):
        amps = panel.row_amplitude[panel.motif_of_row == motif]
        assert amps.size == 4
        assert amps.max() / amps.min() > 1.5, f"motif {motif} lacks magnitude spread: {amps}"
    scales = np.abs(panel.series).mean(axis=1)
    assert scales.max() / scales.min() > 10.0, "panel is not scale-heterogeneous"


def test_affine_condition_rejects_meaningless_and_unknown_settings() -> None:
    with pytest.raises(ValueError, match="meaningless"):
        AffineCondition("raw", True)
    with pytest.raises(ValueError, match="scale must be one of"):
        AffineCondition("minmax", False)


def test_run_affine_grid_rejects_zero_scaling() -> None:
    with pytest.raises(ValueError, match="a must be non-zero"):
        run_affine_grid([AffineCondition("znorm", True)], [0.0], [0.0], **GRID_KW)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_motifs": 1}, "at least 2 motifs"),
        ({"rows_per_motif": 0}, "rows_per_motif"),
        ({"period": 1}, "2 <= period <= length"),
        ({"length": 5}, "2 <= period <= length"),
        ({"noise": -0.1}, "noise must be non-negative"),
        ({"amp_log10_range": (2.0, 1.0)}, "amp_log10_range"),
        ({"offset_range": (5.0, 1.0)}, "offset_range"),
    ],
)
def test_make_motif_panel_fails_loudly(kwargs: dict[str, object], match: str) -> None:
    base: dict[str, object] = {"n_motifs": 4, "length": 120, "period": 12, "noise": 0.01}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        make_motif_panel(rng=np.random.default_rng(0), **base)  # type: ignore[arg-type]


def test_grid_is_deterministic_for_a_fixed_seed() -> None:
    first = run_affine_grid([AffineCondition("znorm", True)], [3.0], [7.0], **GRID_KW)  # type: ignore[arg-type]
    second = run_affine_grid([AffineCondition("znorm", True)], [3.0], [7.0], **GRID_KW)  # type: ignore[arg-type]
    assert first.to_dict() == second.to_dict()
