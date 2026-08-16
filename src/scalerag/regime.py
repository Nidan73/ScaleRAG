"""Where retrieval stops helping: utility correlates and a crossing threshold.

TS-RAG reports that retrieval effectiveness correlates with autocorrelation, noise
ratio, volatility and stationarity, but stops at correlations — it names no value
of any property at which retrieval ceases to pay. That leaves "retrieval sometimes
helps" as the state of the art. This module estimates the crossing point.

Retrieval utility for one series is

    U = Error(backbone) - Error(retrieval)

so ``U > 0`` means retrieval was the better of the two. Given a diagnostic ``x``,
the quantity of interest is the value ``x*`` at which the probability of ``U > 0``
crosses one half: below it the backbone should be trusted, above it retrieval
carries information the backbone lacks.

The crossing is estimated by isotonic regression rather than a logistic fit. The
relationship is expected to be monotone but there is no reason to expect it to be
logistic in shape, and a misspecified parametric curve would place the threshold
wherever its own tails demanded. Isotonic assumes only monotonicity; where the data
do not actually cross one half the estimator says so instead of extrapolating.

**On the reported intervals.** The accompanying ``ci95`` is a percentile bootstrap
over the crossing, and it should be read as a rough indication of sampling spread
rather than as a calibrated 95% interval. Isotonic regression has non-standard
(cube-root) asymptotics at a point, and the naive bootstrap is known to be
*inconsistent* for it, so these intervals are typically too narrow: on planted data
with a sharp boundary the interval can be tight enough to exclude the true value it
was built around. Treat the point estimate as the result and the interval as
indicative only; a calibrated interval would need subsampling or a smoothed
bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

__all__ = [
    "RegimeBand",
    "RegimeThreshold",
    "UtilityCorrelates",
    "estimate_band",
    "estimate_threshold",
    "utility_correlates",
]


@dataclass(frozen=True)
class UtilityCorrelates:
    """Rank correlation between retrieval utility and each diagnostic."""

    names: tuple[str, ...]
    rho: tuple[float, ...]
    p_value: tuple[float, ...]
    n: int

    def to_dict(self) -> dict[str, object]:
        return {
            "n_series": self.n,
            "spearman": [
                {"feature": f, "rho": r, "p_value": p}
                for f, r, p in zip(self.names, self.rho, self.p_value, strict=True)
            ],
        }


def utility_correlates(
    utility: np.ndarray, features: np.ndarray, names: list[str]
) -> UtilityCorrelates:
    """Spearman rho between utility and each column of ``features``."""
    u = np.asarray(utility, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be 2-D (n_series, n_features), got {x.shape}")
    if x.shape[0] != u.shape[0]:
        raise ValueError(f"utility has {u.shape[0]} rows, features {x.shape[0]}")
    if len(names) != x.shape[1]:
        raise ValueError(f"{len(names)} names for {x.shape[1]} feature columns")

    rhos, ps = [], []
    for j in range(x.shape[1]):
        col = x[:, j]
        if np.ptp(col) == 0.0:
            # A constant diagnostic has no rank information; report it rather than
            # emitting a silent NaN from the correlation routine.
            rhos.append(float("nan"))
            ps.append(float("nan"))
            continue
        res = spearmanr(col, u)
        rhos.append(float(res.statistic))
        ps.append(float(res.pvalue))
    return UtilityCorrelates(tuple(names), tuple(rhos), tuple(ps), int(u.shape[0]))


@dataclass(frozen=True)
class RegimeThreshold:
    """Estimated crossing of ``P(U > 0) = 0.5`` along one diagnostic."""

    feature: str
    threshold: float | None
    ci95: tuple[float, float] | None
    win_rate_below: float
    win_rate_above: float
    n_below: int
    n_above: int
    n: int
    crosses: bool
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "crosses_half": self.crosses,
            "threshold": self.threshold,
            "ci95": list(self.ci95) if self.ci95 else None,
            "win_rate_below": self.win_rate_below,
            "win_rate_above": self.win_rate_above,
            "n_below": self.n_below,
            "n_above": self.n_above,
            "n_series": self.n,
            "note": self.note,
        }


def _crossing(x_sorted: np.ndarray, fitted: np.ndarray) -> float | None:
    """First ``x`` at which the monotone fitted win-rate reaches one half."""
    above = np.flatnonzero(fitted >= 0.5)
    if above.size == 0 or above[0] == 0:
        # Never reaches half, or is already above it at the smallest x: in both
        # cases no interior crossing exists and interpolating would invent one.
        return None
    i = int(above[0])
    x0, x1 = x_sorted[i - 1], x_sorted[i]
    y0, y1 = fitted[i - 1], fitted[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))


def estimate_threshold(
    utility: np.ndarray,
    feature: np.ndarray,
    name: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> RegimeThreshold:
    """Estimate where retrieval starts winning more often than not, along ``feature``.

    Returns ``crosses=False`` with a null threshold when the fitted win rate never
    reaches one half over the observed range — the honest answer when retrieval does
    not become preferable anywhere in the data, rather than an extrapolated number.
    """
    u = np.asarray(utility, dtype=np.float64)
    x = np.asarray(feature, dtype=np.float64)
    if u.shape != x.shape:
        raise ValueError(f"utility and feature must align, got {u.shape} and {x.shape}")
    if u.size < 20:
        raise ValueError(f"need at least 20 series for a threshold, got {u.size}")

    wins = (u > 0.0).astype(np.float64)
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], wins[order]

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    fitted = iso.fit_transform(xs, ws)
    point = _crossing(xs, fitted)

    ci: tuple[float, float] | None = None
    if point is not None:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(n_boot):
            idx = rng.integers(0, x.size, x.size)
            bx, bw = x[idx], wins[idx]
            bo = np.argsort(bx, kind="stable")
            bfit = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
                bx[bo], bw[bo]
            )
            got = _crossing(bx[bo], bfit)
            if got is not None:
                draws.append(got)
        if len(draws) >= max(20, n_boot // 10):
            ci = (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))

    split = point if point is not None else float(np.median(x))
    below, above = x < split, x >= split
    return RegimeThreshold(
        feature=name,
        threshold=point,
        ci95=ci,
        win_rate_below=float(wins[below].mean()) if below.any() else float("nan"),
        win_rate_above=float(wins[above].mean()) if above.any() else float("nan"),
        n_below=int(below.sum()),
        n_above=int(above.sum()),
        n=int(u.size),
        crosses=point is not None,
        note=(
            ""
            if point is not None
            else "fitted win rate never reaches 0.5; split reported at the median instead"
        ),
    )


@dataclass(frozen=True)
class RegimeBand:
    """Interval of a diagnostic over which retrieval wins more often than not.

    A single increasing threshold presumes retrieval keeps getting more useful as
    the diagnostic grows. On M5 that is false: utility peaks at moderate
    intermittency and falls away again at the sparse extreme, where a series has too
    little signal for any analogue to match. Fitting only an increasing curve would
    report the lower edge and silently miss the upper one.
    """

    feature: str
    lower: float | None
    upper: float | None
    lower_ci95: tuple[float, float] | None
    upper_ci95: tuple[float, float] | None
    win_rate_inside: float
    win_rate_outside: float
    n_inside: int
    n_outside: int
    bounded_above: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "lower": self.lower,
            "upper": self.upper,
            "lower_ci95": list(self.lower_ci95) if self.lower_ci95 else None,
            "upper_ci95": list(self.upper_ci95) if self.upper_ci95 else None,
            "win_rate_inside": self.win_rate_inside,
            "win_rate_outside": self.win_rate_outside,
            "n_inside": self.n_inside,
            "n_outside": self.n_outside,
            "bounded_above": self.bounded_above,
        }


def estimate_band(
    utility: np.ndarray,
    feature: np.ndarray,
    name: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> RegimeBand:
    """Find both edges of the region where ``P(U > 0) > 0.5``.

    The lower edge comes from an increasing isotonic fit, the upper edge from the
    same fit run on the reflected axis. ``bounded_above`` is False when the win rate
    never falls back below one half, i.e. the relationship really is monotone and a
    single threshold would have sufficed.
    """
    x = np.asarray(feature, dtype=np.float64)
    low = estimate_threshold(utility, x, name, n_boot=n_boot, seed=seed)
    high = estimate_threshold(utility, -x, name, n_boot=n_boot, seed=seed + 1)

    lower = low.threshold
    # `crosses` is exactly `threshold is not None`, but test the field mypy can narrow.
    upper = -high.threshold if high.threshold is not None else None
    if lower is not None and upper is not None and upper <= lower:
        # The two fits disagree about there being an interior region at all; report
        # the lower edge alone rather than an empty or inverted band.
        upper = None

    inside = np.ones(x.shape, dtype=bool)
    if lower is not None:
        inside &= x >= lower
    if upper is not None:
        inside &= x <= upper
    wins = np.asarray(utility, dtype=np.float64) > 0.0
    return RegimeBand(
        feature=name,
        lower=lower,
        upper=upper,
        lower_ci95=low.ci95,
        upper_ci95=(-high.ci95[1], -high.ci95[0]) if (upper is not None and high.ci95) else None,
        win_rate_inside=float(wins[inside].mean()) if inside.any() else float("nan"),
        win_rate_outside=float(wins[~inside].mean()) if (~inside).any() else float("nan"),
        n_inside=int(inside.sum()),
        n_outside=int((~inside).sum()),
        bounded_above=upper is not None,
    )
