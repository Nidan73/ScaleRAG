#!/usr/bin/env python3
"""Referee item 10: what the fused predictive distribution actually is.

The referee's complaint is that the paper reports coverage and pinball from a
fused predictive distribution it never defines. Reading the code says the
definition is worse than missing -- it is two separate constructions, neither of
which is a predictive law in the usual sense:

  1. ``fusion.fuse`` blends **quantile by quantile**,
     ``q_fused(tau) = (1-w) q_c(tau) + w q_r(tau)``. That is Vincentization (the
     Wasserstein-2 barycentre of the two branches), not the mixture
     ``(1-w) F_c + w F_r``. Whenever the branches disagree in location the
     barycentre is strictly narrower than the mixture, so blending can lose
     coverage that neither input lacked.
  2. The retrieval branch's quantiles are ``np.quantile(conts, ...)`` over the
     k=20 restored neighbours. That is *retrieval disagreement* -- how much the
     analogues differ from each other -- not forecast uncertainty. With k=20 the
     0.05 and 0.95 levels sit essentially on the 2nd and 19th order statistics,
     so the outer interval cannot reach past the neighbour sample range.

Both predict undercoverage, and the recorded runs show it: on validation
Cov.80 is 0.7906 for the backbone and 0.7393 for retrieval, but only 0.7218
fused -- **below both inputs**. Test mirrors it (0.7864, 0.7280, 0.7147, and
0.6975 gated -- the 0.698 the paper reports).

This run separates the two causes on validation by holding the retrieved set and
the weight fixed and swapping only the construction:

  * ``fused_vincent``   -- deployed: quantile-average of backbone and empirical
                          retrieval quantiles.
  * ``fused_mixture``   -- same two branches, combined as a mixture CDF.
  * ``retrieval_gauss`` -- same neighbour cloud, quantiles from mean +/- z*sd
                          instead of ``np.quantile``, which can leave the sample
                          range.
  * ``fused_vincent_gauss`` -- the deployed operator over the parametric branch,
                          so operator and estimator are attributable separately.

It also decomposes every miss into *below the lower bound* and *above the upper
bound*. The motivating guess was that the loss would be one-sided, since a lower
bound sitting at zero cannot be violated by a non-negative actual. **It is not**
-- the bound is above zero often enough that the barycentre loses more coverage
below (0.1568 against the backbone's 0.1099) than above (0.1340 against 0.0980).

Reported, never selected from. The frozen configuration is unchanged whatever
comes back (rules 9, 12). **Validation origins only** -- the consumed M5 test
split is refused.

Usage (fish):
    uv run python scripts/fusion_calibration_run.py --subset 1000 --origins 20
    uv run python scripts/fusion_calibration_run.py --subset 1000 --origins 20 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import scalerag_eval as SE  # noqa: E402, N812  (sibling holds the frozen constants)
from scalerag import metrics  # noqa: E402
from scalerag.eval import load_processed, select_subset  # noqa: E402
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

OUT = REPO / "reports" / "fusion-calibration"
CKPT = OUT / "ckpt"

# The grid the deployed harness scores on (`scalerag_test_final.py`), not the
# nine-level grid `scalerag_eval` uses internally -- item 10 is about the numbers
# the paper prints, so the diagnostic has to sit on the same levels.
QL: list[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
LEVELS: dict[str, tuple[float, float]] = {"50": (0.25, 0.75), "80": (0.1, 0.9), "90": (0.05, 0.95)}
W = 0.5  # the fixed fusion weight; `fusion_fixed0.5` is the recorded comparator
CHUNK = 200_000  # rows per block in the mixture inversion, to bound peak memory


def _monotone(q: np.ndarray) -> np.ndarray:
    """Enforce a non-decreasing quantile function along the last axis.

    `np.quantile` is already sorted; a neural quantile head is not guaranteed to
    be. Crossing quantiles would make the CDF inversion below meaningless, so we
    repair them and report how often the repair bound.
    """
    return np.maximum.accumulate(q, axis=-1)


def _cdf_at(grid: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Piecewise-linear CDF implied by quantile values ``v`` at levels ``tau``.

    ``grid`` is (N, M), ``v`` is (N, Q). Outside ``[v_0, v_-1]`` the CDF is
    clamped to ``[tau_0, tau_-1]`` -- neither branch says anything about its own
    tails beyond the outermost level it reports, and inventing a tail there would
    be the opposite of what item 10 asks for.

    The count is ``v <= x``, not ``v < x``, which is the whole game on count
    data: a series that is zero half the time has ``q(0.05) = ... = q(0.5) = 0``,
    and ``P(X <= 0)`` is then 0.5, not 0.05. Taking the lowest level of a tie
    block instead of the highest puts the mixture's lower bound above zero and
    manufactures a coverage failure that is not there.
    """
    q = v.shape[1]
    cnt = (v[:, None, :] <= grid[:, :, None]).sum(axis=2)  # (N, M) in [0, Q]
    hi = np.clip(cnt, 1, q - 1)
    lo = hi - 1
    vlo = np.take_along_axis(v, lo, axis=1)
    vhi = np.take_along_axis(v, hi, axis=1)
    span = vhi - vlo
    t = np.where(span > 0, (grid - vlo) / np.where(span > 0, span, 1.0), 0.0)
    y = tau[lo] + np.clip(t, 0.0, 1.0) * (tau[hi] - tau[lo])
    y = np.where(cnt == 0, tau[0], y)  # below every knot
    y = np.where(cnt >= q, tau[-1], y)  # at or above the top knot
    return np.clip(y, tau[0], tau[-1])


def _mixture_quantiles(vc: np.ndarray, vr: np.ndarray, w: float, tau: np.ndarray) -> np.ndarray:
    """Quantiles of ``(1-w) F_c + w F_r`` -- the mixture, not the barycentre.

    Both branch CDFs are linear between the knots of either branch, so evaluating
    the mixture on the union of the two knot sets and inverting linearly there is
    exact, not an approximation.

    At the lowest reported level both branches are clamped, so the generalized
    inverse is degenerate and this returns the smaller of the two branches'
    lowest quantiles. That is the widest value the reported levels justify;
    extrapolating a tail below it would be inventing one.
    """
    out = np.empty_like(vc)
    for s in range(0, vc.shape[0], CHUNK):
        c, r = vc[s : s + CHUNK], vr[s : s + CHUNK]
        grid = np.sort(np.concatenate([c, r], axis=1), axis=1)  # (n, 2Q)
        g = (1.0 - w) * _cdf_at(grid, c, tau) + w * _cdf_at(grid, r, tau)
        # generalized inverse: leftmost x with G(x) >= level
        j = (g[:, :, None] < tau[None, None, :]).sum(axis=1)  # (n, Q) in [0, 2Q]
        hi = np.clip(j, 1, grid.shape[1] - 1)
        lo = hi - 1
        glo = np.take_along_axis(g, lo, axis=1)
        ghi = np.take_along_axis(g, hi, axis=1)
        xlo = np.take_along_axis(grid, lo, axis=1)
        xhi = np.take_along_axis(grid, hi, axis=1)
        span = ghi - glo
        t = np.where(span > 0, (tau[None, :] - glo) / np.where(span > 0, span, 1.0), 0.0)
        x = xlo + np.clip(t, 0.0, 1.0) * (xhi - xlo)
        out[s : s + CHUNK] = np.where(j[:, :] == 0, grid[:, :1], x)
    return _monotone(out)


def _gaussian_quantiles(mean: np.ndarray, sd: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """mean + z_tau * sd over the same neighbour cloud, floored at zero.

    The point of the contrast is that this estimator is *not* confined to the
    k=20 sample range, so it isolates how much of the narrowness is the empirical
    estimator rather than the magnitude of neighbour disagreement itself.
    """
    from scipy.stats import norm

    z = norm.ppf(tau)
    return np.clip(mean[:, :, None] + sd[:, :, None] * z[None, None, :], 0.0, None)


def run_origin(sales: np.ndarray, entities, oi: int, fc) -> dict[str, np.ndarray]:
    """One origin: both branches once, then every construction off the same inputs."""
    n = sales.shape[0]
    tau = np.asarray(QL, dtype=float)
    queries = [sales[i, oi - SE.L : oi] for i in range(n)]

    c_pt, c_q = fc.forecast([sales[i, :oi] for i in range(n)], SE.H, QL)
    c_pt = np.clip(c_pt, 0.0, None)
    c_q_raw = _monotone(np.asarray(c_q, dtype=float))  # as the harness scores it: NOT floored
    c_q = np.clip(c_q_raw, 0.0, None)  # common footing with everything downstream

    r_pt, r_q, _nnd, _dis, r_sd = SE.retrieval_all(
        sales, entities, oi, queries, scale="mean", quantile_levels=QL, return_spread=True
    )
    r_q = _monotone(np.asarray(r_q, dtype=float))
    g_q = _gaussian_quantiles(r_pt, r_sd, tau)

    flat = (n * SE.H, len(QL))
    mix = _mixture_quantiles(c_q.reshape(flat), r_q.reshape(flat), W, tau).reshape(c_q.shape)

    rec: dict[str, np.ndarray] = {
        "origin": np.array([oi]),
        "actual": sales[:, oi : oi + SE.H],
        "zero_fraction": (sales[:, oi - SE.L : oi] == 0).mean(axis=1),
        "never_launched": sales[:, :oi].sum(axis=1) == 0,
        # the deployed point forecast, so pinball and coverage refer to one method
        "point_fused": np.clip((1 - W) * c_pt + W * r_pt, 0.0, None),
        "point_retrieval": r_pt,
        "neighbour_sd": r_sd,
        "q_backbone_unclipped": c_q_raw,
        "q_backbone": c_q,
        "q_retrieval_emp": r_q,
        "q_retrieval_gauss": g_q,
        "q_fused_vincent": np.clip((1 - W) * c_q + W * r_q, 0.0, None),
        "q_fused_vincent_gauss": np.clip((1 - W) * c_q + W * g_q, 0.0, None),
        "q_fused_mixture": mix,
    }
    return rec


CONSTRUCTIONS = (
    "q_backbone_unclipped",
    "q_backbone",
    "q_retrieval_emp",
    "q_retrieval_gauss",
    "q_fused_vincent",
    "q_fused_vincent_gauss",
    "q_fused_mixture",
)


def score(rec: dict[str, np.ndarray], key: str, mask: np.ndarray) -> dict:
    """Coverage, width, one-sided miss rates and pinball for one construction."""
    q = rec[key][mask]
    y = rec["actual"][mask]
    out: dict = {}
    for lv, (qlo, qhi) in LEVELS.items():
        lo, hi = q[:, :, QL.index(qlo)], q[:, :, QL.index(qhi)]
        out[f"cov{lv}"] = float(np.mean((y >= lo) & (y <= hi)))
        out[f"width{lv}"] = float(np.mean(hi - lo))
        out[f"miss_below{lv}"] = float(np.mean(y < lo))
        out[f"miss_above{lv}"] = float(np.mean(y > hi))
    out["pinball"] = float(
        np.mean([metrics.pinball_loss(y[i], q[i], QL) for i in range(q.shape[0])])
    )
    return out


def per_series_cov80(rec: dict[str, np.ndarray], key: str, mask: np.ndarray) -> np.ndarray:
    q = rec[key][mask]
    y = rec["actual"][mask]
    lo, hi = q[:, :, QL.index(0.1)], q[:, :, QL.index(0.9)]
    return ((y >= lo) & (y <= hi)).mean(axis=1)


def paired_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of ``b - a`` with a percentile paired bootstrap over series."""
    rng = np.random.default_rng(seed)
    d = b - a
    boots = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "delta": float(d.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n": int(d.size),
    }


def pooled(records: list[dict[str, np.ndarray]]) -> dict:
    """Pool every origin into one population and score each construction on it."""
    keys = (
        "actual",
        "never_launched",
        "zero_fraction",
        "point_fused",
        "neighbour_sd",
        *CONSTRUCTIONS,
    )
    rec = {k: np.concatenate([r[k] for r in records], axis=0) for k in keys}
    mask = ~rec["never_launched"]

    table = {k: score(rec, k, mask) for k in CONSTRUCTIONS}

    # Is the neighbour spread even the right size? Compare it with the error the
    # retrieval point forecast actually makes on the same windows.
    pf = np.concatenate([r["point_retrieval"] for r in records], axis=0)[mask]
    err = np.abs(rec["actual"][mask] - pf)
    sd = rec["neighbour_sd"][mask]
    # Ratio and benchmark are reciprocals of one another, so reporting one against
    # the other reads as "slightly wide" when the truth is "a third too narrow".
    # Both are stated in the same direction, with the share spelled out.
    calib = float(np.sqrt(2.0 / np.pi))  # E|y - mu| = 0.798 sd for a Gaussian
    ratio = float(err.mean() / sd.mean()) if sd.mean() > 0 else float("nan")
    spread = {
        "mean_neighbour_sd": float(sd.mean()),
        "mean_abs_error_of_retrieval_point": float(err.mean()),
        "ratio_abs_error_over_sd": ratio,
        "calibrated_ratio_abs_error_over_sd": calib,
        "spread_as_share_of_calibrated": calib / ratio if ratio > 0 else float("nan"),
        "note": (
            "A calibrated spread satisfies E|y - mu| = sd * sqrt(2/pi) = 0.798 sd. "
            "An observed ratio ABOVE 0.798 means the neighbour cloud is NARROWER "
            "than the error it is meant to describe. "
            "`spread_as_share_of_calibrated` is how wide it is as a fraction of "
            "the spread that would be calibrated."
        ),
    }

    v = per_series_cov80(rec, "q_fused_vincent", mask)
    contrasts = {
        "mixture_minus_vincent_cov80": paired_ci(v, per_series_cov80(rec, "q_fused_mixture", mask)),
        "gauss_branch_minus_vincent_cov80": paired_ci(
            v, per_series_cov80(rec, "q_fused_vincent_gauss", mask)
        ),
        "backbone_minus_vincent_cov80": paired_ci(v, per_series_cov80(rec, "q_backbone", mask)),
    }
    return {
        "n_series_origin_pairs": int(mask.sum()),
        "table": table,
        "neighbour_spread_vs_error": spread,
        "contrasts_cov80": contrasts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=int, default=1000)
    ap.add_argument("--origins", type=int, default=20)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    started = time.time()
    set_seed(args.seed)

    from scalerag.tsfm.chronos2 import Chronos2Forecaster

    entities, dynamic = load_processed(REPO / "data" / "processed")
    entities, dynamic = select_subset(entities, dynamic, args.subset, args.seed)
    n = entities.height
    n_days = int(dynamic["day_idx"].max())
    sales = dynamic["sales"].to_numpy().astype(np.float64).reshape(n, n_days)

    splits = make_rolling_splits(last_labeled_day=n_days)
    o_eval = split_by_name(splits, "val").train_end
    test_start = split_by_name(splits, "test").train_end
    if o_eval >= test_start:
        raise SystemExit(f"refusing: eval origin {o_eval} reaches the consumed test split")
    origins = [o_eval - i * args.stride for i in range(args.origins)]
    if min(origins) - SE.L < 0:
        raise SystemExit(f"origin {min(origins)} has no room for a length-{SE.L} context")
    if max(o + SE.H for o in origins) > test_start:
        raise SystemExit("refusing: an evaluation window reaches the consumed test split")

    CKPT.mkdir(parents=True, exist_ok=True)
    tag = f"s{args.subset}-seed{args.seed}"
    fc = None
    records: list[dict[str, np.ndarray]] = []
    bar = tqdm(origins, desc="item 10: fused distribution", unit="origin", dynamic_ncols=True)
    for oi in bar:
        path = CKPT / f"{tag}-origin{oi}.npz"
        if args.resume and path.exists():
            with np.load(path) as z:
                records.append({k: z[k] for k in z.files})
            bar.set_postfix_str(f"origin {oi}: cached")
            continue
        if fc is None:  # lazily loaded so a fully cached resume needs no GPU
            fc = Chronos2Forecaster()
        rec = run_origin(sales, entities, oi, fc)
        np.savez_compressed(path, **rec)
        records.append(rec)
        m = ~rec["never_launched"]
        bar.set_postfix_str(
            f"origin {oi}: vincent cov80 {score(rec, 'q_fused_vincent', m)['cov80']:.3f}"
        )
    bar.close()

    res = pooled(records)
    payload = {
        "experiment": "m5-fusion-calibration",
        "referee_item": "10",
        "dataset": "M5",
        "split": "validation",
        "guard": "test split d_1914-d_1941 is consumed and untouched (rule 2)",
        "question": (
            "Is the reported Cov.80 a property of fusion, or an artifact of two "
            "construction choices -- quantile averaging instead of a mixture, and "
            "empirical quantiles of k=20 neighbours as the retrieval branch?"
        ),
        "constructions": {
            "q_backbone_unclipped": "Chronos-2 quantiles as the harness scores them (no zero floor)",
            "q_backbone": "Chronos-2 quantiles floored at zero, for a common footing",
            "q_retrieval_emp": "deployed: np.quantile over the k=20 restored neighbours",
            "q_retrieval_gauss": "same cloud, mean + z*sd -- not confined to the sample range",
            "q_fused_vincent": "deployed: (1-w) q_c + w q_r, the Wasserstein-2 barycentre",
            "q_fused_vincent_gauss": "deployed operator over the parametric retrieval branch",
            "q_fused_mixture": "quantiles of (1-w) F_c + w F_r, the mixture law",
        },
        "fusion_weight_w": W,
        "quantile_levels": QL,
        "interval_levels": {k: list(v) for k, v in LEVELS.items()},
        "n_origins": len(origins),
        "origins": origins,
        "stride": args.stride,
        "subset_draw": {
            "n_requested": args.subset,
            "n_selected": int(n),
            "seed": args.seed,
            "method": "scalerag.eval.select_subset",
        },
        "context_length_L": SE.L,
        "horizon_H": SE.H,
        "top_k": SE.K,
        "population": "never-launched series excluded; all constructions scored on it",
        "results": res,
        "runtime_sec": round(time.time() - started, 1),
        "timestamp": datetime.now(UTC).isoformat(),
        "run_context": RunContext().to_dict(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"m5-fusion-calibration-{tag}-{len(origins)}origins.json"
    dest.write_text(json.dumps(payload, indent=2))

    print(f"\nwrote {dest.relative_to(REPO)}   ({payload['runtime_sec']}s)\n")
    hdr = f"{'construction':24} {'cov50':>7} {'cov80':>7} {'cov90':>7} {'w80':>7} {'>hi80':>7} {'<lo80':>7} {'pinball':>8}"
    print(hdr)
    print("-" * len(hdr))
    for k in CONSTRUCTIONS:
        r = res["table"][k]
        print(
            f"{k.removeprefix('q_'):24} {r['cov50']:>7.4f} {r['cov80']:>7.4f} {r['cov90']:>7.4f} "
            f"{r['width80']:>7.4f} {r['miss_above80']:>7.4f} {r['miss_below80']:>7.4f} {r['pinball']:>8.4f}"
        )
    s = res["neighbour_spread_vs_error"]
    print(
        f"\nneighbour sd {s['mean_neighbour_sd']:.4f} vs mean |error| "
        f"{s['mean_abs_error_of_retrieval_point']:.4f}  ->  E|err|/sd "
        f"{s['ratio_abs_error_over_sd']:.4f} against a calibrated "
        f"{s['calibrated_ratio_abs_error_over_sd']:.4f}, i.e. the neighbour spread is "
        f"{100 * s['spread_as_share_of_calibrated']:.1f}% of a calibrated one"
    )
    for name, c in res["contrasts_cov80"].items():
        print(
            f"{name:38} {c['delta']:+.4f}  CI [{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]  "
            f"{'excludes 0' if c['excludes_zero'] else 'includes 0'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
