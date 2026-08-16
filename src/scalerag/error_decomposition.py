"""Attribute retrieval-augmented forecast error to the stage that caused it.

Phase 11A left an unexplained result: restoration improves the *retrieved
continuation* by +85.4% MSE, and the *end-to-end* forecast still loses to the
frozen backbone by 0.85%. A single aggregate number cannot say why, because it
sums over four distinct failure modes. This module separates them.

For one window with truth ``y``, backbone forecast ``c``, raw retrieved mean
``r_raw`` and restored retrieved mean ``r_res``:

``E_raw``
    MSE of the un-restored retrieval — the retrieved future in the *donor's*
    coordinate system.
``E_shape``
    MSE of the restored retrieval after the **oracle** affine correction — the
    least-squares fit of ``alpha * r_res + beta`` to the realised future. Being the
    minimiser over all affine maps, it is a true lower bound: no scale-restoration
    rule can beat it for this retrieved set. Whatever remains is the analogue simply
    having the wrong shape.
``E_res``
    MSE of the restored retrieval as actually produced.
``E_bb`` / ``E_fused``
    Backbone alone, and the shipped convex blend.

which gives the additive attribution

    E_raw  =  E_shape  +  (E_res - E_shape)  +  (E_raw - E_res)
              ^^^^^^^     ^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^
              shape       scale error left      scale error the
              mismatch    by restoration        mechanism removed

The diagnostic question is whether ``E_res`` sits near ``E_shape`` (the scale
mechanism is saturated and the bottleneck is analogue quality) or well above it
(there is still scale error to recover).

:func:`optimal_fusion_weight` reports the blend weight that *would* have minimised
error. It is a diagnostic only. Selecting a weight from it would be tuning on an
evaluation split, which the project rules forbid; the frozen weight stays frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ErrorDecomposition",
    "decompose_errors",
    "optimal_fusion_weight",
    "oracle_rescale",
    "paired_bootstrap_mean_diff",
]


def _as2d(name: str, arr: np.ndarray, shape: tuple[int, int] | None) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_windows, horizon), got shape {out.shape}")
    if shape is not None and out.shape != shape:
        raise ValueError(f"{name} shape {out.shape} does not match truth shape {shape}")
    return out


def oracle_rescale(pred: np.ndarray, truth: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Apply the per-window affine correction that minimises MSE against the truth.

    This is an oracle: it fits ``alpha * pred + beta`` by least squares *using the
    realised future*, so it is not a forecaster. Because it is the minimiser over
    all affine maps, its error is a genuine lower bound — no scale-restoration rule,
    however clever, can do better for this retrieved set. Whatever error survives is
    the analogue having the wrong *shape*.

    Note that moment matching (forcing pred onto the truth's mean and standard
    deviation) is *not* this minimiser and gives no such bound, which is why the
    least-squares fit is used instead.

    Windows whose prediction is constant admit no slope; they collapse to the
    truth's mean, which is the best a location-only correction can do.
    """
    p = _as2d("pred", pred, None)
    t = _as2d("truth", truth, p.shape)
    p_mu = p.mean(axis=1, keepdims=True)
    t_mu = t.mean(axis=1, keepdims=True)
    pc = p - p_mu
    var = np.sum(pc * pc, axis=1, keepdims=True)
    degenerate = var < eps
    alpha = np.sum(pc * (t - t_mu), axis=1, keepdims=True) / np.where(degenerate, 1.0, var)
    return np.where(degenerate, t_mu, alpha * pc + t_mu)


def optimal_fusion_weight(backbone: np.ndarray, retrieval: np.ndarray, truth: np.ndarray) -> float:
    """Weight ``w`` minimising MSE of ``(1-w)*backbone + w*retrieval`` (unclipped).

    Closed form. Writing ``d_c = backbone - truth`` and ``d_r = retrieval - truth``,
    the blended residual is ``d_c + w (d_r - d_c)``, so the minimiser is

        w* = -<d_c, d_r - d_c> / ||d_r - d_c||^2

    A ``w*`` at or below zero means the retrieval branch carries no usable signal
    beyond the backbone on this data.
    """
    c = _as2d("backbone", backbone, None)
    t = _as2d("truth", truth, c.shape)
    r = _as2d("retrieval", retrieval, c.shape)
    d_c = c - t
    gap = r - c
    denom = float(np.sum(gap * gap))
    if denom < 1e-30:
        raise ValueError("backbone and retrieval are identical: fusion weight is undefined")
    return float(-np.sum(d_c * gap) / denom)


@dataclass
class ErrorDecomposition:
    """Per-stage MSE attribution over a set of forecast windows."""

    n_windows: int
    horizon: int
    fusion_weight: float
    e_raw: float
    e_res: float
    e_shape: float
    e_backbone: float
    e_fused: float
    optimal_weight: float
    per_window: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def scale_error_removed(self) -> float:
        """MSE the restoration mechanism eliminated (``E_raw - E_res``)."""
        return self.e_raw - self.e_res

    @property
    def scale_error_remaining(self) -> float:
        """MSE still attributable to imperfect magnitude (``E_res - E_shape``)."""
        return self.e_res - self.e_shape

    @property
    def shape_fraction_of_restored(self) -> float:
        """Share of the restored retrieval's error that is pure shape mismatch."""
        return self.e_shape / self.e_res if self.e_res > 0 else float("nan")

    @property
    def retrieval_backbone_ratio(self) -> float:
        """How many times worse the restored retrieval is than the backbone."""
        return self.e_res / self.e_backbone if self.e_backbone > 0 else float("nan")

    @property
    def fusion_penalty(self) -> float:
        """Fused error minus the better of the two branches. Positive means the
        blend is worse than simply picking the stronger branch."""
        return self.e_fused - min(self.e_backbone, self.e_res)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_windows": self.n_windows,
            "horizon": self.horizon,
            "fusion_weight": self.fusion_weight,
            "mse": {
                "raw_retrieval": self.e_raw,
                "restored_retrieval": self.e_res,
                "shape_floor_oracle_rescaled": self.e_shape,
                "backbone": self.e_backbone,
                "fused": self.e_fused,
            },
            "attribution": {
                "scale_error_removed_by_restoration": self.scale_error_removed,
                "scale_error_remaining": self.scale_error_remaining,
                "shape_fraction_of_restored_error": self.shape_fraction_of_restored,
                "restored_retrieval_vs_backbone_ratio": self.retrieval_backbone_ratio,
                "fusion_penalty_vs_best_branch": self.fusion_penalty,
            },
            "optimal_fusion_weight_diagnostic_only": self.optimal_weight,
        }


def decompose_errors(
    truth: np.ndarray,
    backbone: np.ndarray,
    retrieval_raw: np.ndarray,
    retrieval_restored: np.ndarray,
    fusion_weight: float,
) -> ErrorDecomposition:
    """Split forecast error across the retrieval, scale, shape and fusion stages."""
    if not 0.0 <= fusion_weight <= 1.0:
        raise ValueError(f"fusion_weight must be in [0, 1], got {fusion_weight}")
    t = _as2d("truth", truth, None)
    c = _as2d("backbone", backbone, t.shape)
    r_raw = _as2d("retrieval_raw", retrieval_raw, t.shape)
    r_res = _as2d("retrieval_restored", retrieval_restored, t.shape)

    fused = (1.0 - fusion_weight) * c + fusion_weight * r_res
    shape_only = oracle_rescale(r_res, t)

    def per_window(pred: np.ndarray) -> np.ndarray:
        return np.mean((pred - t) ** 2, axis=1)

    pw = {
        "raw_retrieval": per_window(r_raw),
        "restored_retrieval": per_window(r_res),
        "shape_floor": per_window(shape_only),
        "backbone": per_window(c),
        "fused": per_window(fused),
    }
    return ErrorDecomposition(
        n_windows=int(t.shape[0]),
        horizon=int(t.shape[1]),
        fusion_weight=float(fusion_weight),
        e_raw=float(pw["raw_retrieval"].mean()),
        e_res=float(pw["restored_retrieval"].mean()),
        e_shape=float(pw["shape_floor"].mean()),
        e_backbone=float(pw["backbone"].mean()),
        e_fused=float(pw["fused"].mean()),
        optimal_weight=optimal_fusion_weight(c, r_res, t),
        per_window=pw,
    )


def paired_bootstrap_mean_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    """Percentile CI for ``mean(a - b)`` with windows resampled as pairs.

    Pairing matters: both arms score the identical windows, so resampling them
    together removes between-window variance that would otherwise swamp the effect.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"paired inputs must align, got {x.shape} and {y.shape}")
    rng = np.random.default_rng(seed)
    diff = x - y
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot[i] = diff[rng.integers(0, diff.size, diff.size)].mean()
    return float(diff.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
