"""Controlled synthetic affine probe for the scale-restoration mechanism.

The ScaleRAG critique this module answers is that ``y_restored = y* * sigma_q + mu_q``
is "just denormalisation". Observational results on M5/ETTm2 cannot settle that,
because the affine relationship between a query and its analogue is never known
there. Here it is *imposed*: a query is built as an exact affine image
``x' = a*x + b`` of a known donor motif, so the true analogue and the true
continuation are both known in closed form, and each stage of the pipeline can be
credited or blamed separately.

The probe separates two properties that the observational studies conflate:

``invariance``
    Does the *retrieval* stage still find the correct analogue when the query is
    affinely transformed? Measured as top-1 hit rate against the known donor.

``equivariance``
    Does the *reconstruction* stage map the retrieved future back into the query's
    coordinate system? Measured as forecast error against the known continuation
    ``a*y + b``.

Crossing {raw, znorm} retrieval with {no restoration, query-scale restoration}
isolates which stage is responsible for which failure. Note that only ``znorm``
carries a location term, so only ``znorm`` can be equivariant under a location
shift ``b != 0``; ``mean`` and ``rms`` are scale-only. The frozen M5 configuration
uses ``mean``, so it is scale-equivariant but *not* affine-equivariant — a
distinction this probe measures rather than assumes.

The scale primitives are imported from :mod:`retrieval_faiss` so the probe tests
the frozen definitions rather than a re-derivation of them. No I/O at import.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scalerag import leakage
from scalerag.native import topk_exact
from scalerag.retrieval import WindowDatabase
from scalerag.retrieval_faiss import SCALE_STRATEGIES, _fit_params, _transform

__all__ = [
    "AffineCell",
    "AffineCondition",
    "MotifPanel",
    "ProbeGrid",
    "make_motif_panel",
    "run_affine_grid",
    "run_condition",
]


@dataclass(frozen=True)
class AffineCondition:
    """One arm of the 2x2: a retrieval space and a reconstruction rule."""

    scale: str
    restore: bool

    def __post_init__(self) -> None:
        if self.scale not in SCALE_STRATEGIES:
            raise ValueError(f"scale must be one of {SCALE_STRATEGIES}, got {self.scale!r}")
        if self.restore and self.scale == "raw":
            raise ValueError(
                "restore=True is meaningless for scale='raw': the raw transform is the "
                "identity, so restoration would be a no-op. Use restore=False."
            )

    @property
    def name(self) -> str:
        return f"{self.scale}{'+restore' if self.restore else ''}"


@dataclass(frozen=True)
class MotifPanel:
    """A donor panel whose every window has a known shape identity.

    ``series`` is ``(n_rows, length)`` and ``motif_of_row`` says which shape each
    row carries. Every shape appears on several rows at *different* magnitudes, so
    "same shape" and "same magnitude" are independent — which is what makes the
    top-1 hit rate a test of shape matching rather than of magnitude matching.
    """

    series: np.ndarray
    motif_of_row: np.ndarray
    row_amplitude: np.ndarray
    row_offset: np.ndarray
    period: int

    def __post_init__(self) -> None:
        if self.series.ndim != 2:
            raise ValueError(f"series must be 2-D (n_rows, length), got {self.series.shape}")
        for name, arr in (
            ("motif_of_row", self.motif_of_row),
            ("row_amplitude", self.row_amplitude),
            ("row_offset", self.row_offset),
        ):
            if arr.shape[0] != self.series.shape[0]:
                raise ValueError(f"{name} must have one entry per series row")


def make_motif_panel(
    n_motifs: int,
    length: int,
    period: int,
    noise: float,
    rng: np.random.Generator,
    rows_per_motif: int = 4,
    amp_log10_range: tuple[float, float] = (-1.0, 3.0),
    offset_range: tuple[float, float] = (-50.0, 500.0),
) -> MotifPanel:
    """Build ``n_motifs`` recurring shapes, each instantiated at several magnitudes.

    A shape is a random harmonic mixture over ``period`` steps, centred and
    unit-scaled, then tiled to ``length``. Each of its ``rows_per_motif`` rows is
    then given its own amplitude (log-uniform, spanning ``amp_log10_range`` orders
    of magnitude) and additive offset, mimicking the scale heterogeneity of a retail
    panel where a slow item and a fast item can share a weekly shape.

    That heterogeneity is the point. If every row were unit-scaled, ``||c||`` would
    be constant across candidates, the L2 ranking would collapse to shape
    correlation, and a raw-space retriever would be *accidentally* scale-invariant —
    hiding the very failure this probe is built to expose. With heterogeneous
    magnitudes, raw L2 is dominated by ``||q||^2 - 2 q.c + ||c||^2`` and prefers
    candidates whose magnitude resembles the query's, whatever their shape.

    Observation noise is scaled by each row's amplitude so the signal-to-noise ratio
    is constant across rows, keeping the difficulty of the task independent of scale.
    """
    if n_motifs < 2:
        raise ValueError(f"need at least 2 motifs to make retrieval non-trivial, got {n_motifs}")
    if rows_per_motif < 1:
        raise ValueError(f"rows_per_motif must be >= 1, got {rows_per_motif}")
    if period < 2 or length < period:
        raise ValueError(f"require 2 <= period <= length, got period={period}, length={length}")
    if noise < 0.0:
        raise ValueError(f"noise must be non-negative, got {noise}")
    if amp_log10_range[0] > amp_log10_range[1]:
        raise ValueError(f"amp_log10_range must be (lo, hi), got {amp_log10_range}")
    if offset_range[0] > offset_range[1]:
        raise ValueError(f"offset_range must be (lo, hi), got {offset_range}")

    phase = np.arange(period) / period * 2.0 * np.pi
    n_rows = n_motifs * rows_per_motif
    rows = np.empty((n_rows, length), dtype=np.float64)
    motif_of_row = np.empty(n_rows, dtype=np.int64)
    amplitude = np.empty(n_rows, dtype=np.float64)
    offset = np.empty(n_rows, dtype=np.float64)

    for m in range(n_motifs):
        shape = np.zeros(period, dtype=np.float64)
        for harmonic in (1, 2, 3):
            amp = rng.uniform(0.5, 1.5)
            off = rng.uniform(0.0, 2.0 * np.pi)
            shape += amp * np.sin(harmonic * phase + off)
        shape -= shape.mean()
        sd = shape.std()
        if sd < 1e-12:
            raise ValueError(f"degenerate motif {m}: zero standard deviation")
        shape /= sd
        tiled = np.tile(shape, int(np.ceil(length / period)))[:length]

        for j in range(rows_per_motif):
            r = m * rows_per_motif + j
            a_r = float(10.0 ** rng.uniform(*amp_log10_range))
            c_r = float(rng.uniform(*offset_range))
            rows[r] = a_r * tiled + c_r + rng.normal(0.0, noise * a_r, size=length)
            motif_of_row[r] = m
            amplitude[r] = a_r
            offset[r] = c_r

    return MotifPanel(
        series=rows,
        motif_of_row=motif_of_row,
        row_amplitude=amplitude,
        row_offset=offset,
        period=period,
    )


@dataclass
class AffineCell:
    """Outcome of one (condition, a, b) cell of the grid."""

    condition: str
    a: float
    b: float
    hit_rate: float
    mse: float
    nmse: float
    n_query: int
    scale: str = field(default="", repr=False)
    restore: bool = field(default=False, repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "condition": self.condition,
            "scale": self.scale,
            "restore": self.restore,
            "a": self.a,
            "b": self.b,
            "hit_rate": self.hit_rate,
            "mse": self.mse,
            "nmse": self.nmse,
            "n_query": self.n_query,
        }


@dataclass
class ProbeGrid:
    """All cells of a sweep, plus the configuration that produced them."""

    cells: list[AffineCell]
    context_length: int
    horizon: int
    k: int
    seed: int
    n_motifs: int
    noise: float

    def to_dict(self) -> dict[str, object]:
        return {
            "config": {
                "context_length": self.context_length,
                "horizon": self.horizon,
                "k": self.k,
                "seed": self.seed,
                "n_motifs": self.n_motifs,
                "noise": self.noise,
            },
            "cells": [c.to_dict() for c in self.cells],
        }


def _build_database(
    panel: MotifPanel, train_end: int, context_length: int, horizon: int, stride: int
) -> WindowDatabase:
    db = WindowDatabase.from_training(
        panel.series, train_end, context_length, horizon, stride=stride
    )
    if len(db) == 0:
        raise ValueError(
            f"empty candidate pool for train_end={train_end}, L={context_length}, H={horizon}"
        )
    return db


def run_condition(
    panel: MotifPanel,
    db: WindowDatabase,
    condition: AffineCondition,
    a: float,
    b: float,
    origin: int,
    k: int,
    query_rows: np.ndarray,
    query_t_r: np.ndarray,
) -> AffineCell:
    """Evaluate one condition at one affine transform ``(a, b)``.

    Each query is the affine image ``a*x + b`` of a donor window taken from the
    panel; the ground-truth continuation is the identically transformed future
    ``a*y + b``. Retrieval runs against the *untransformed* pool, so a retriever
    that is not scale-invariant sees the query drift away from its own analogue.

    Both metrics are reported in the query's coordinate system, which is what a
    forecaster is actually judged on.
    """
    if a == 0.0:
        raise ValueError("a must be non-zero: a=0 collapses the query to a constant")
    if query_rows.shape != query_t_r.shape:
        raise ValueError("query_rows and query_t_r must have the same shape")

    L, H = db.context_length, db.horizon  # noqa: N806 — L,H are standard TS notation
    contexts = np.stack(
        [panel.series[r, t - L : t] for r, t in zip(query_rows, query_t_r, strict=True)]
    )
    futures = np.stack(
        [panel.series[r, t : t + H] for r, t in zip(query_rows, query_t_r, strict=True)]
    )
    queries = a * contexts + b
    truth = a * futures + b

    # Rule 3: only candidates whose continuation closes before the origin.
    legal_ids = np.flatnonzero(db.legal_mask(origin))
    if legal_ids.size == 0:
        raise leakage.LeakageViolation(
            f"no rule-3-legal candidates for origin {origin} "
            f"(pool max t_r+H = {int(db.t_r.max()) + db.horizon})"
        )
    kk = min(k, legal_ids.size)

    cand_params = _fit_params(db.contexts[legal_ids], condition.scale)
    cand_vecs = _transform(db.contexts[legal_ids], cand_params)
    q_params = _fit_params(queries, condition.scale)
    q_vecs = _transform(queries, q_params)

    local_ids, _ = topk_exact(q_vecs, cand_vecs, kk)
    ids = legal_ids[local_ids]  # (n_query, kk) global candidate ids
    leakage.assert_retrieval_horizon(int(db.t_r[ids].max()), db.horizon, origin)

    # Invariance: did the nearest neighbour come from the motif we transformed?
    top1_motif = panel.motif_of_row[db.series_idx[ids[:, 0]]]
    hit_rate = float(np.mean(top1_motif == panel.motif_of_row[query_rows]))

    # Equivariance: reconstruct the retrieved futures into the query's coordinates.
    conts = db.continuations[ids]  # (n_query, kk, H) donor coordinates
    if condition.restore:
        cp = _fit_params(db.contexts, condition.scale)[ids]  # (n_query, kk, 2)
        c_loc, c_scale = cp[:, :, 0:1], cp[:, :, 1:2]
        q_loc = q_params[:, None, 0:1]
        q_scale = q_params[:, None, 1:2]
        conts = (conts - c_loc) / c_scale * q_scale + q_loc
    point = conts.mean(axis=1)  # (n_query, H)

    # Raw MSE is not comparable across ``a`` — it grows like a^2 for any method,
    # including a perfect one. Normalising by the variance of the truth removes that
    # trivial growth, so a flat nmse across the sweep is the signature of an
    # equivariant reconstruction rather than of an easy transform.
    mse = float(np.mean((point - truth) ** 2))
    truth_var = float(np.var(truth))
    if truth_var < 1e-30:
        raise ValueError("degenerate ground truth (zero variance): cannot form a scale-free error")

    return AffineCell(
        condition=condition.name,
        a=float(a),
        b=float(b),
        hit_rate=hit_rate,
        mse=mse,
        nmse=mse / truth_var,
        n_query=int(query_rows.shape[0]),
        scale=condition.scale,
        restore=condition.restore,
    )


def run_affine_grid(
    conditions: list[AffineCondition],
    a_values: list[float],
    b_values: list[float],
    *,
    n_motifs: int = 12,
    rows_per_motif: int = 4,
    period: int = 24,
    context_length: int = 48,
    horizon: int = 12,
    n_queries: int = 200,
    noise: float = 0.05,
    k: int = 5,
    stride: int = 4,
    seed: int = 42,
) -> ProbeGrid:
    """Sweep every ``(condition, a, b)`` cell against one shared donor panel.

    The panel, the query windows and the candidate pool are built once and reused
    across all cells, so differences between cells are attributable to the
    condition and the transform alone, not to resampling.

    The query's own row stays in the candidate pool, as it would in deployment.
    That makes ``a=1, b=0`` a control in which every condition should retrieve
    correctly, and it makes the failure at large ``a`` a genuine one: the transform
    pushes the query's magnitude away from its own row and towards rows carrying a
    different shape.
    """
    rng = np.random.default_rng(seed)
    length = context_length + horizon + period * 12
    panel = make_motif_panel(n_motifs, length, period, noise, rng, rows_per_motif=rows_per_motif)

    # Candidate pool closes strictly before the query origin (rule 3, rule 5).
    train_end = length - horizon - 1
    origin = length - horizon
    db = _build_database(panel, train_end, context_length, horizon, stride)

    # Queries sit at the held-out origin, so their own future is never a candidate.
    query_rows = rng.integers(0, panel.series.shape[0], size=n_queries)
    query_t_r = np.full(n_queries, origin, dtype=np.int64)

    cells = [
        run_condition(panel, db, cond, a, b, origin, k, query_rows, query_t_r)
        for cond in conditions
        for a in a_values
        for b in b_values
    ]
    return ProbeGrid(
        cells=cells,
        context_length=context_length,
        horizon=horizon,
        k=k,
        seed=seed,
        n_motifs=n_motifs,
        noise=noise,
    )
