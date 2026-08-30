"""Temporal-leakage / split-integrity invariants.

These are the enforceable core of CLAUDE.md research rules 1, 3, 4, 5. Functions
either return ``None`` (invariant holds) or raise :class:`LeakageViolation`.
Time is expressed as ordered, comparable values — integer step indices are the
canonical form; ISO-date strings also compare correctly. This module is
dependency-light on purpose (no torch, no pandas) so leakage tests run fast.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise


class LeakageViolation(ValueError):  # noqa: N818 — "Violation" reads better than "Error" here
    """Raised when a temporal-leakage or split-integrity invariant is violated."""


def assert_strictly_increasing(labelled_boundaries: Sequence[tuple[str, object]]) -> None:
    """Require boundaries to be strictly increasing in time order.

    ``labelled_boundaries`` is an ordered sequence of ``(label, value)`` pairs.
    """
    for (a_label, a), (b_label, b) in pairwise(labelled_boundaries):
        if not a < b:  # type: ignore[operator]
            raise LeakageViolation(
                f"Chronological order violated: {a_label}={a!r} must be strictly "
                f"before {b_label}={b!r}."
            )


def assert_chronological_split(train_end: object, val_end: object, test_start: object) -> None:
    """Require ``train_end < val_end < test_start`` (rule 1)."""
    assert_strictly_increasing(
        [("train_end", train_end), ("val_end", val_end), ("test_start", test_start)]
    )


def assert_retrieval_horizon(t_r: int, horizon: int, target_forecast_origin: int) -> None:
    """Require ``t_r + H < target_forecast_origin`` (rule 3).

    ``t_r`` is the last step index of the retrieved context; ``horizon`` is H.
    A retrieved context whose forecast window could reach into or past the
    target's forecast origin leaks future information.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if not (t_r + horizon < target_forecast_origin):
        raise LeakageViolation(
            f"Illegal future-horizon retrieval: t_r({t_r}) + H({horizon}) = "
            f"{t_r + horizon} is not < target_forecast_origin({target_forecast_origin})."
        )


def assert_no_window_overlap(
    train_windows: Sequence[tuple[int, int]],
    eval_windows: Sequence[tuple[int, int]],
) -> None:
    """Require that no training window overlaps any evaluation window (rule 5).

    Windows are inclusive ``(start, end)`` step-index intervals.
    """
    for ts, te in train_windows:
        for es, ee in eval_windows:
            if ts <= ee and es <= te:
                raise LeakageViolation(
                    f"Duplicate/overlapping window across splits: train "
                    f"({ts},{te}) overlaps eval ({es},{ee})."
                )


def assert_target_not_in_features(feature_columns: Sequence[str], target_column: str) -> None:
    """Require the raw target not to appear among feature columns (rule 4/5)."""
    if target_column in feature_columns:
        raise LeakageViolation(f"Target column {target_column!r} leaked into feature columns.")


def assert_no_future_covariates(
    covariate_columns: Sequence[str], known_future_columns: Sequence[str]
) -> None:
    """Require every covariate used ahead of the origin to be known-future (rule 4)."""
    known = set(known_future_columns)
    unknown = [c for c in covariate_columns if c not in known]
    if unknown:
        raise LeakageViolation(
            f"Covariates used beyond the forecast origin are not known-future: {unknown}."
        )
