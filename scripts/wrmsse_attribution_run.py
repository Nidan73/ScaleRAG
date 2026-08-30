#!/usr/bin/env python3
"""Why does a frozen Chronos-2 score WRMSSE 1.94 while winning MASE and WAPE?

The Phase-10 held-out run recorded frozen Chronos-2 at WRMSSE 1.9395 on M5 test
against Seasonal-Naive 0.8697 and LightGBM 0.8663, while the same forecasts were
best on MASE (0.8932) and WAPE (0.6653). The anomaly reproduces on validation
(1.7568 vs LightGBM 0.7106), so it can be diagnosed without reopening the consumed
test split.

Two hypotheses are already dead. The point forecast is **not** the median:
`tsfm/chronos2.py` uses the pipeline's predictive mean, deliberately, for exactly
this reason. And the metric is not misconstructed: dollar weights come from the
last 28 **training** days and aggregated-level scale denominators are computed on
the aggregated series, not summed from bottom-level ones.

The surviving hypothesis is **correlated error**. Chronos-2 runs target-only, so it
cannot see SNAP days, calendar events or day-of-week. Every store misses the same
event on the same day, and correlated error does not cancel under aggregation,
whereas idiosyncratic error does. LightGBM and Seasonal-Naive both capture that
common component. If this is right, Chronos-2's per-level WRMSSE should be
*inverted* -- worst at the top of the hierarchy, where the healthy shape is best at
the top -- and supplying known-future calendar covariates should reduce it.

Decision rule, fixed before the run:
  - L1/L2 >> L12 for Chronos-2, and the covariate arm reduces it -> correlated
    calendar error, reported as a property of target-only frozen TSFMs on
    hierarchical retail panels.
  - levels roughly uniform -> the deficit is dollar-weight concentration on
    high-volume series; report that instead.

Neither outcome rescues pre-registered criterion 2, which stays 0/3. The M5 test
split cannot be rescored: `scripts/scalerag_test_final.py` saved no per-series
predictions, so any test metric would require re-running inference on a consumed
split (rules 2, 9, 12). This explains a number; it does not change a verdict.

Usage (fish):
    uv run python scripts/wrmsse_attribution_run.py
    uv run python scripts/wrmsse_attribution_run.py --resume
    uv run python scripts/wrmsse_attribution_run.py --n-series 2000   # faster smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scalerag import hierarchy, metrics  # noqa: E402
from scalerag.eval import load_processed, select_subset  # noqa: E402
from scalerag.leakage import assert_no_future_covariates  # noqa: E402
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

OUT = REPO / "reports" / "wrmsse-attribution"
CKPT = OUT / "checkpoints"
LOCK = REPO / "M5_TEST_CONSUMED.lock"
H = 28
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Known-future at an M5 forecast origin: the calendar is published ahead, and SNAP
# days and event labels with it. Prices are excluded -- they are also announced
# ahead, but carry nulls for unlaunched items and the null handling would confound
# the very comparison this script makes.
CALENDAR_COVARIATES = ("snap", "wday", "event_type_1")


def reshape_covariate(series: pl.Series, n: int, n_days: int) -> np.ndarray:
    """Reshape one covariate column to (n_series, n_days), preserving its kind.

    Chronos-2 accepts categorical covariates as numpy string arrays and rejects
    them as torch tensors, so a string column stays a string array rather than
    being ordinal-encoded: M5 event types have no natural order, and encoding them
    as 0..k would assert one.

    Nulls are explicit sentinels, not dropped. ``event_type_1`` is null on the
    ~92% of days with no event, which is information, not missingness.
    """
    if series.dtype == pl.String:
        return series.fill_null("none").to_numpy().astype(str).reshape(n, n_days)
    arr = series.to_numpy().astype(np.float64)
    return np.nan_to_num(arr, nan=-1.0).reshape(n, n_days)


def build_covariate_inputs(
    sales: np.ndarray,
    covariates: dict[str, np.ndarray],
    origin: int,
    horizon: int,
    future_only_names: list[str] | None = None,
) -> list[dict[str, object]]:
    """Per-series Chronos-2 input dicts: target plus aligned calendar covariates.

    The target stops strictly before ``origin``. Past covariates must have exactly
    the target's length -- ``chronos2/preprocess.py`` raises otherwise, and catching
    it here is cheaper than at inference. Future covariates cover the horizon
    immediately after the origin and must be a subset of the past ones, which
    :func:`assert_no_future_covariates` enforces (rule 4).
    """
    n, n_days = sales.shape
    if origin + horizon > n_days:
        raise ValueError(
            f"horizon {horizon} from origin {origin} reaches beyond the panel ({n_days} days)"
        )
    names = list(covariates)
    # A name present only in the future is by definition not known-future data.
    assert_no_future_covariates(names + list(future_only_names or []), names)

    out: list[dict[str, object]] = []
    for i in range(n):
        out.append(
            {
                "target": sales[i, :origin],
                "past_covariates": {k: covariates[k][i, :origin] for k in names},
                "future_covariates": {
                    k: covariates[k][i, origin : origin + horizon] for k in names
                },
            }
        )
    return out


def per_level_bias(entities, actuals: np.ndarray, preds: np.ndarray) -> dict[str, dict[str, float]]:
    """Signed aggregate bias at each hierarchy level.

    Note that **signed** bias is level-invariant by construction: summing the errors
    over any grouping gives the same total, so it is reported once rather than per
    level. What varies is the *absolute* error, because idiosyncratic errors cancel
    when summed into a group and shared ones do not. A ratio of the L1 absolute
    error to the L12 absolute error near 1 therefore means the error is almost
    entirely shared; near 0 means it is almost entirely idiosyncratic.
    """
    out: dict[str, dict[str, float]] = {}
    for name, cols in hierarchy.M5_LEVELS:
        gid, labels = hierarchy.level_group_ids(entities, cols)
        ng = len(labels)
        a = hierarchy.aggregate_rows(actuals, gid, ng)
        p = hierarchy.aggregate_rows(preds, gid, ng)
        denom = np.abs(a).sum()
        out[name] = {
            "n_groups": int(ng),
            "signed_bias": float((p - a).sum()),
            "relative_signed_bias": float((p - a).sum() / denom) if denom > 0 else float("nan"),
            "mean_abs_relative_error": float(np.abs(p - a).sum() / denom)
            if denom > 0
            else float("nan"),
        }
    return out


def score_method(entities, sales, o, pred, weights) -> dict[str, object]:
    """WRMSSE with its per-level breakdown, error structure, and per-series means."""
    ha, tr = sales[:, o : o + H], sales[:, :o]
    score, per_level = hierarchy.wrmsse(entities, tr, ha, pred, weights)

    # How much of the WRMSSE deficit is a single scalar level error? Rescaling the
    # whole forecast by the ratio that removes the aggregate bias is an ORACLE (it
    # reads the realised total) and is reported as a diagnostic only: adopting it
    # would be selection on an evaluation split (rules 9, 12).
    total_pred, total_true = float(pred.sum()), float(ha.sum())
    bias_ratio = total_true / total_pred if total_pred > 0 else float("nan")
    debiased = pred * bias_ratio if np.isfinite(bias_ratio) else pred
    score_debiased, _lvl_db = hierarchy.wrmsse(entities, tr, ha, debiased, weights)
    rmsse = np.array([metrics.rmsse(ha[i], pred[i], tr[i]) for i in range(pred.shape[0])])
    mase = np.array([metrics.mase(ha[i], pred[i], tr[i]) for i in range(pred.shape[0])])
    return {
        "wrmsse": score,
        "wrmsse_per_level": per_level,
        "relative_signed_bias": float((total_pred - total_true) / abs(total_true)),
        "level_bias_correction_ratio_DIAGNOSTIC": bias_ratio,
        "wrmsse_after_oracle_level_correction_DIAGNOSTIC": score_debiased,
        "wrmsse_share_explained_by_level_bias": (
            float((score - score_debiased) / score) if score > 0 else float("nan")
        ),
        "per_level_error_structure": per_level_bias(entities, ha, pred),
        "rmsse_mean": float(np.nanmean(rmsse)),
        "mase_mean": float(np.nanmean(mase)),
        "wape": float(np.abs(ha - pred).sum() / np.abs(ha).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-series", type=int, default=0, help="0 = full panel (needed for a valid hierarchy)"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=4000, help="series per inference chunk")
    ap.add_argument("--resume", action="store_true", help="reuse cached per-method forecasts")
    args = ap.parse_args()

    set_seed(args.seed)
    started = time.time()

    from scalerag.tsfm.chronos2 import Chronos2Forecaster

    entities, dynamic = load_processed(REPO / "data" / "processed")
    if args.n_series:
        entities, dynamic = select_subset(entities, dynamic, args.n_series, args.seed)
    n = entities.height
    n_days = int(dynamic["day_idx"].max())

    def col(name: str) -> np.ndarray:
        return dynamic[name].to_numpy().reshape(n, n_days)

    sales = col("sales").astype(np.float64)
    price = col("sell_price").astype(np.float64)

    splits = make_rolling_splits(last_labeled_day=n_days)
    o = split_by_name(splits, "val").train_end
    test_start = split_by_name(splits, "test").train_end
    if o >= test_start or o + H > test_start:
        raise SystemExit(
            f"refusing to run: validation window from origin {o} reaches the consumed "
            f"test split (starts {test_start}). See {LOCK.name}."
        )

    weights = hierarchy.dollar_weights(sales[:, o - 28 : o], price[:, o - 28 : o])

    covariates = {name: reshape_covariate(dynamic[name], n, n_days) for name in CALENDAR_COVARIATES}

    CKPT.mkdir(parents=True, exist_ok=True)
    tag = f"n{n}-seed{args.seed}"

    def cached(name: str, fn) -> np.ndarray:
        path = CKPT / f"{tag}-{name}.npy"
        if args.resume and path.exists():
            return np.load(path)
        arr = fn()
        np.save(path, arr)
        return arr

    def chronos_forecast(with_covariates: bool) -> np.ndarray:
        fc = Chronos2Forecaster()
        preds = np.zeros((n, H))
        label = "chronos+cov" if with_covariates else "chronos"
        for lo in tqdm(range(0, n, args.chunk), desc=label, unit="chunk", dynamic_ncols=True):
            hi = min(lo + args.chunk, n)
            ctx = [sales[i, :o] for i in range(lo, hi)]
            if with_covariates:
                items = build_covariate_inputs(
                    sales[lo:hi], {k: v[lo:hi] for k, v in covariates.items()}, o, H
                )
                pt, _q = fc.forecast_with_covariates(
                    [it["target"] for it in items],
                    [it["past_covariates"] for it in items],
                    [it["future_covariates"] for it in items],
                    H,
                    QL,
                )
            else:
                pt, _q = fc.forecast(ctx, H, QL)
            preds[lo:hi] = np.clip(pt, 0.0, None)
        return preds

    methods: dict[str, np.ndarray] = {}
    methods["seasonal_naive"] = cached(
        "seasonal_naive", lambda: sales[:, o - 7 : o][:, np.arange(H) % 7]
    )
    methods["recent_mean"] = cached(
        "recent_mean", lambda: np.repeat(sales[:, o - 28 : o].mean(1)[:, None], H, axis=1)
    )
    methods["chronos2_target"] = cached("chronos2_target", lambda: chronos_forecast(False))
    methods["chronos2_covariates"] = cached("chronos2_cov", lambda: chronos_forecast(True))

    rows = {m: score_method(entities, sales, o, p, weights) for m, p in methods.items()}

    payload = {
        "experiment": "wrmsse-attribution",
        "dataset": "M5",
        "split": "validation",
        "eval_origin": int(o),
        "n_series": int(n),
        "guard": "test split is consumed and untouched (rule 2)",
        "note": (
            "Diagnostic only. No configuration is selected and no criterion is rescored. "
            "The M5 test split cannot be rescored: no per-series test predictions were "
            "stored, so any test metric would require re-running inference on a consumed "
            "split (rules 2, 9, 12)."
        ),
        "point_forecast": "Chronos-2 predictive MEAN (not the median); see tsfm/chronos2.py",
        "calendar_covariates": list(CALENDAR_COVARIATES),
        "dollar_weights": "last 28 training days before the origin",
        "methods": rows,
        "timestamp": datetime.now(UTC).isoformat(),
        "runtime_sec": round(time.time() - started, 2),
        "run_context": RunContext().to_dict(),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"m5-val-wrmsse-attribution-{tag}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path.relative_to(REPO)}\n")

    print(f"{'method':>22s} {'WRMSSE':>8s} {'RMSSE':>8s} {'MASE':>7s} {'WAPE':>7s}")
    for m, r in rows.items():
        print(
            f"{m:>22s} {r['wrmsse']:8.4f} {r['rmsse_mean']:8.4f} {r['mase_mean']:7.4f} {r['wape']:7.4f}"
        )

    print("\nper-level WRMSSE (L1 = Total, L12 = item x store)")
    names = [nm for nm, _ in hierarchy.M5_LEVELS]
    print(f"{'level':>16s} " + " ".join(f"{m[:11]:>11s}" for m in rows))
    for lv in names:
        print(f"{lv:>16s} " + " ".join(f"{rows[m]['wrmsse_per_level'][lv]:11.4f}" for m in rows))

    print("\naggregate level bias, and how much of WRMSSE it explains")
    print(f"{'method':>22s} {'rel bias':>10s} {'WRMSSE':>9s} {'debiased':>10s} {'share':>8s}")
    for m, r in rows.items():
        print(
            f"{m:>22s} {r['relative_signed_bias']:+10.4f} {r['wrmsse']:9.4f} "
            f"{r['wrmsse_after_oracle_level_correction_DIAGNOSTIC']:10.4f} "
            f"{r['wrmsse_share_explained_by_level_bias']:8.1%}"
        )
    print("  (debiased is an ORACLE diagnostic -- it reads the realised total. Not adopted.)")

    print("\nerror structure: L1 absolute error / L12 absolute error")
    print("  near 1 = error is shared across series; near 0 = idiosyncratic and cancels")
    for m, r in rows.items():
        st = r["per_level_error_structure"]
        ratio = (
            st["L1_total"]["mean_abs_relative_error"]
            / st["L12_item_store"]["mean_abs_relative_error"]
        )
        print(f"{m:>22s} {ratio:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
