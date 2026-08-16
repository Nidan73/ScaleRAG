#!/usr/bin/env python3
"""Where does retrieval stop helping? A crossing threshold on M5 validation.

TS-RAG reports correlations between retrieval effectiveness and series properties
but names no value at which retrieval ceases to pay, leaving "retrieval sometimes
helps" as the state of the art. This estimates the crossing point.

Per series at the M5 **validation** origin (d_1886-1913) it computes retrieval
utility ``U = RMSSE(Chronos-2) - RMSSE(scale-restored retrieval)`` alongside the
six frozen gate diagnostics, then fits where ``P(U > 0)`` crosses one half.

Validation only. The test split d_1914-d_1941 is consumed and is never touched
(rule 2); nothing here selects or tunes a configuration (rules 9, 12) — the frozen
scale=mean / k=20 / cat_id retrieval is imported from `scripts/scalerag_eval.py`
rather than re-derived.

    uv run python scripts/regime_threshold_run.py --subset 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import scalerag_eval as SE  # noqa: E402,N812 — frozen M5 retrieval + gate-feature defs
from scalerag.eval import load_processed, select_subset  # noqa: E402
from scalerag.regime import (  # noqa: E402
    estimate_band,
    estimate_threshold,
    utility_correlates,
)
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

OUT = REPO / "reports" / "regime-threshold"
LOCK = REPO / "M5_TEST_CONSUMED.lock"
FEATURES = [
    "retr_nn_dist",
    "retr_disagreement",
    "intermittency",
    "log_volume",
    "chronos_uncertainty",
    "scale_spread",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument(
        "--origins",
        type=int,
        default=1,
        help="number of forecast origins, stepping back from the validation origin",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=28,
        help="days between origins; the default equals H so windows do not overlap",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    started = time.time()

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
        raise SystemExit(
            f"refusing to run: eval origin {o_eval} reaches the consumed test split "
            f"(starts {test_start}). See {LOCK.name}."
        )

    # Origins step back from validation so every evaluation window closes at or
    # before the consumed test boundary. Stride defaults to H, so the windows are
    # non-overlapping and each origin is an independent temporal sample.
    origins = [o_eval - i * args.stride for i in range(args.origins)]
    if min(origins) - SE.L < 0:
        raise SystemExit(f"origin {min(origins)} has no room for a length-{SE.L} context")
    if max(o + SE.H for o in origins) > test_start:
        raise SystemExit("refusing to run: an evaluation window reaches the consumed test split")

    fc = Chronos2Forecaster()
    per_origin: list[dict[str, np.ndarray]] = []
    for oi in origins:
        queries = [sales[i, oi - SE.L : oi] for i in range(n)]
        c_pt, c_q = fc.forecast([sales[i, :oi] for i in range(n)], SE.H, SE.QL)
        c_pt = np.clip(c_pt, 0.0, None)
        r_pt, _r_q, nnd, dis = SE.retrieval_all(sales, entities, oi, queries, scale="mean")
        rmsse_c = SE.rmsse_series(c_pt, sales, oi)
        rmsse_r = SE.rmsse_series(r_pt, sales, oi)
        finite = np.isfinite(rmsse_c) & np.isfinite(rmsse_r)
        per_origin.append(
            {
                "origin": oi,
                "utility": (rmsse_c - rmsse_r)[finite],
                "feats": SE.gate_features(sales, oi, nnd, dis, c_q)[finite],
            }
        )
        print(
            f"  origin {oi}: {int(finite.sum())} series, "
            f"win rate {float((per_origin[-1]['utility'] > 0).mean()):.3f}"
        )

    # The primary (single-origin) analysis stays anchored at the validation origin
    # so it remains comparable with the earlier run.
    utility = per_origin[0]["utility"]
    feats = per_origin[0]["feats"]

    corr = utility_correlates(utility, feats, FEATURES)
    thresholds = {
        name: estimate_threshold(
            utility, feats[:, j], name, n_boot=args.n_boot, seed=args.seed
        ).to_dict()
        for j, name in enumerate(FEATURES)
    }

    # The diagnostics are not independent, so a marginal correlation with utility can
    # be inherited from a neighbour. Record the inter-diagnostic ranks so any
    # single-feature reading can be checked for confounding rather than trusted.
    from scipy.stats import spearmanr as _spearmanr

    feat_corr = {
        FEATURES[i]: {
            FEATURES[j]: float(_spearmanr(feats[:, i], feats[:, j]).statistic)
            for j in range(len(FEATURES))
            if j != i
        }
        for i in range(len(FEATURES))
    }

    # A single increasing threshold assumes retrieval keeps getting more useful.
    # Fit both edges so a non-monotone relationship is visible rather than hidden.
    bands = {
        name: estimate_band(
            utility, feats[:, j], name, n_boot=args.n_boot, seed=args.seed
        ).to_dict()
        for j, name in enumerate(FEATURES)
    }

    # Decile profile along intermittency: the headline regime axis.
    inter = feats[:, FEATURES.index("intermittency")]
    edges = np.quantile(inter, np.linspace(0, 1, 11))
    deciles = []
    for lo, hi in pairwise(edges):
        m = (inter >= lo) & (inter <= hi if hi == edges[-1] else inter < hi)
        if not m.any():
            continue
        deciles.append(
            {
                "intermittency_lo": float(lo),
                "intermittency_hi": float(hi),
                "n": int(m.sum()),
                "win_rate": float((utility[m] > 0).mean()),
                "mean_utility": float(utility[m].mean()),
            }
        )

    # Fixed zero-fraction bins, identical across origins, so a bin means the same
    # thing everywhere and win rates can be averaged. Quantile deciles would move
    # with each origin's own distribution and could not be pooled.
    edges = np.round(np.arange(0.0, 1.01, 0.1), 2)
    inter_idx = FEATURES.index("intermittency")
    bins: list[dict[str, object]] = []
    for lo, hi in pairwise(edges):
        rates, utils, counts = [], [], []
        for rec in per_origin:
            f = rec["feats"][:, inter_idx]
            m = (f >= lo) & (f < hi if hi < 1.0 else f <= hi)
            # Keep thin strata rather than dropping them. Dropping would silently
            # remove exactly the extreme-sparsity bins the non-monotonicity claim
            # rests on; a noisy bin should instead show up as a wide interval.
            if m.sum() == 0:
                continue
            rates.append(float((rec["utility"][m] > 0).mean()))
            utils.append(float(rec["utility"][m].mean()))
            counts.append(int(m.sum()))
        if not rates:
            continue
        k = len(rates)
        bins.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n_origins": k,
                "mean_series_per_origin": float(np.mean(counts)),
                "min_series_at_any_origin": int(np.min(counts)),
                "win_rate_mean": float(np.mean(rates)),
                "win_rate_sem": float(np.std(rates, ddof=1) / np.sqrt(k))
                if k > 1
                else float("nan"),
                "mean_utility": float(np.mean(utils)),
                "mean_utility_sem": float(np.std(utils, ddof=1) / np.sqrt(k))
                if k > 1
                else float("nan"),
            }
        )

    # Band edges refitted per origin, so their spread is a temporal sampling spread.
    band_lo, band_hi = [], []
    for rec in per_origin:
        bd = estimate_band(rec["utility"], rec["feats"][:, inter_idx], "intermittency", n_boot=0)
        if bd.lower is not None:
            band_lo.append(bd.lower)
        if bd.upper is not None:
            band_hi.append(bd.upper)

    payload = {
        "experiment": "retrieval-utility-regime-threshold",
        "n_origins": len(origins),
        "origins": origins,
        "stride": args.stride,
        "fixed_bin_profile_across_origins": bins,
        "band_across_origins": {
            "lower_mean": float(np.mean(band_lo)) if band_lo else None,
            "lower_sd": float(np.std(band_lo, ddof=1)) if len(band_lo) > 1 else None,
            "upper_mean": float(np.mean(band_hi)) if band_hi else None,
            "upper_sd": float(np.std(band_hi, ddof=1)) if len(band_hi) > 1 else None,
            "n_origins_with_lower": len(band_lo),
            "n_origins_with_upper": len(band_hi),
        },
        "dataset": "M5",
        "split": "validation (d_1886-d_1913)",
        "guard": "test split d_1914-d_1941 is consumed and untouched (rule 2)",
        "note": "no configuration is selected or tuned; frozen retrieval imported from scalerag_eval",
        "n_series": int(utility.size),
        "subset": args.subset,
        "seed": args.seed,
        "eval_origin": int(o_eval),
        "frozen_retrieval": {
            "scale": "mean",
            "k": SE.K,
            "meta_filter": "cat_id",
            "L": SE.L,
            "H": SE.H,
        },
        "overall_win_rate": float((utility > 0).mean()),
        "mean_utility_rmsse": float(utility.mean()),
        "correlates": corr.to_dict(),
        "bands": bands,
        "diagnostic_intercorrelation_spearman": feat_corr,
        "thresholds": thresholds,
        "intermittency_deciles": deciles,
        "timestamp": datetime.now(UTC).isoformat(),
        "runtime_sec": round(time.time() - started, 2),
        "run_context": RunContext().to_dict(),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.subset}" + (f"-{len(origins)}origins" if len(origins) > 1 else "")
    path = OUT / f"m5-val-regime-threshold-{suffix}.json"
    path.write_text(json.dumps(payload, indent=2))

    print(
        f"\nM5 val, {utility.size} series. Retrieval beats Chronos-2 on "
        f"{100 * (utility > 0).mean():.1f}% of them (mean dU = {utility.mean():+.4f} RMSSE)\n"
    )
    print(f"{'diagnostic':22s} {'spearman':>9s} {'threshold':>12s} {'win<':>7s} {'win>':>7s}")
    for j, name in enumerate(FEATURES):
        t = thresholds[name]
        thr = f"{t['threshold']:.4f}" if t["crosses_half"] else "no crossing"
        print(
            f"{name:22s} {corr.rho[j]:+9.3f} {thr:>12s} "
            f"{t['win_rate_below']:7.2f} {t['win_rate_above']:7.2f}"
        )
    print(f"\n{'diagnostic':22s} {'band lower':>12s} {'band upper':>12s} {'in':>6s} {'out':>6s}")
    for name in FEATURES:
        b = bands[name]
        lo = f"{b['lower']:.4f}" if b["lower"] is not None else "-inf"
        hi = f"{b['upper']:.4f}" if b["bounded_above"] else "none"
        print(
            f"{name:22s} {lo:>12s} {hi:>12s} {b['win_rate_inside']:6.2f} {b['win_rate_outside']:6.2f}"
        )
    print(f"\n{'intermittency band':>24s}  {'n':>5s}  {'win rate':>9s}  {'mean dU':>9s}")
    for d in deciles:
        band = f"{d['intermittency_lo']:.2f}-{d['intermittency_hi']:.2f}"
        print(f"{band:>24s}  {d['n']:5d}  {d['win_rate']:9.2f}  {d['mean_utility']:+9.4f}")
    if len(origins) > 1:
        print(f"\n{'zero fraction':>16s} {'origins':>8s} {'win rate':>16s} {'mean dU':>16s}")
        for b in bins:
            print(
                f"{b['lo']:.1f}-{b['hi']:.1f}".rjust(16)
                + f" {b['n_origins']:8d}"
                + f" {b['win_rate_mean']:9.3f}+-{b['win_rate_sem']:<5.3f}"
                + f" {b['mean_utility']:+9.4f}+-{b['mean_utility_sem']:<5.4f}"
            )
        ba = payload["band_across_origins"]
        if ba["lower_mean"] is not None:
            print(
                f"\nband lower {ba['lower_mean']:.3f} +- {ba['lower_sd']:.3f} "
                f"({ba['n_origins_with_lower']}/{len(origins)} origins)"
            )
        if ba["upper_mean"] is not None:
            print(
                f"band upper {ba['upper_mean']:.3f} +- {ba['upper_sd']:.3f} "
                f"({ba['n_origins_with_upper']}/{len(origins)} origins)"
            )
    print(f"\nwrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
