#!/usr/bin/env python3
"""Sparse-end scale identifiability: is the collapse retrieval, or the operator?

The regime study reports a retrieval win rate per zero-fraction bin and finds an
inverted U. At the M5 validation origin, with the deployed context length L=56,
26.37% of the sparsest bin has a context window whose mean is exactly zero. There
``retrieval_faiss._fit_params`` substitutes a scale of 1.0 (not the 1e-8 the
manuscript declares), so the restored continuation is the candidate rescaled to
unit mean rather than anything retrieval chose. Every other bin is at 0.00%.

This script separates the two explanations. For each origin it retrieves **once**
at the frozen configuration, then re-restores the identical retrieved candidates
under four scale operators. Holding the retrieved set fixed means any difference
is attributable to restoration alone.

Decision rule, fixed before the run (see the design spec):
  - identifiable-only win rate in [0.9, 1.0] moves outside the +/-1.96 SEM band of
    the deployed estimate -> the sparse arm was operator-driven;
  - it stays inside even under oracle_future -> extreme sparsity genuinely offers
    no anchor for k-NN, and the original mechanism holds on a clean population.

No configuration is selected from this. ``oracle_future`` is a bound, not a
candidate (rule 12). Validation origins only; the consumed test split is refused.

Usage (fish):
    uv run python scripts/scale_identifiability_run.py --subset 1000 --origins 50
    uv run python scripts/scale_identifiability_run.py --subset 1000 --origins 50 --resume
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

import scalerag_eval as SE  # noqa: E402,N812 -- matches the convention in scripts/
from scalerag import scale_operators as so  # noqa: E402
from scalerag.eval import load_processed, select_subset  # noqa: E402
from scalerag.regime import estimate_band  # noqa: E402
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402
from scalerag.sparse_regime import (  # noqa: E402
    FIXED_BIN_EDGES,
    bin_index,
    stratified_bin_summary,
)
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

OUT = REPO / "reports" / "scale-identifiability"
CKPT = OUT / "checkpoints"
LOCK = REPO / "M5_TEST_CONSUMED.lock"
OPERATOR_ORDER = (
    "window_mean",
    "long_lookback",
    "hierarchy_prior",
    "oracle_future_mean",
    "oracle_least_squares",
)
CKPT_VERSION = "v3"  # bumped when the record schema changes, so --resume cannot load stale files


def group_codes(entities) -> np.ndarray:
    """store_id|dept_id as integer codes, for the hierarchy-prior operator."""
    pair = np.array(
        [
            f"{a}|{b}"
            for a, b in zip(
                entities["store_id"].to_list(), entities["dept_id"].to_list(), strict=True
            )
        ]
    )
    _uniq, inv = np.unique(pair, return_inverse=True)
    return inv.astype(np.int64)


def query_params_for(
    op: str,
    sales: np.ndarray,
    oi: int,
    group_ids: np.ndarray,
    lookback: int,
    unit_pred: np.ndarray,
) -> np.ndarray:
    """Per-series (loc, scale) under one operator at one origin."""
    if op == "window_mean":
        return so.window_mean(sales[:, oi - SE.L : oi])
    if op == "long_lookback":
        return so.long_lookback(sales, oi, lookback=lookback)
    if op == "hierarchy_prior":
        return so.hierarchy_prior(sales, oi, group_ids, lookback=SE.L)
    # Both oracles are diagnostics, explicitly acknowledged, never deployable.
    if op == "oracle_future_mean":
        return so.oracle_future_mean(sales[:, oi : oi + SE.H], allow_oracle=True)
    if op == "oracle_least_squares":
        return so.oracle_least_squares(unit_pred, sales[:, oi : oi + SE.H], allow_oracle=True)
    raise ValueError(f"unknown operator {op!r}")


def run_origin(
    sales: np.ndarray, entities, oi: int, group_ids: np.ndarray, lookback: int, fc
) -> dict[str, np.ndarray]:
    """One origin: retrieve once, re-restore under every operator, score each."""
    n = sales.shape[0]
    queries = [sales[i, oi - SE.L : oi] for i in range(n)]

    c_pt, _c_q = fc.forecast([sales[i, :oi] for i in range(n)], SE.H, SE.QL)[:2]
    c_pt = np.clip(c_pt, 0.0, None)
    r_pt, _r_q, _nnd, _dis, arts = SE.retrieval_all(
        sales, entities, oi, queries, scale="mean", return_artifacts=True
    )

    db = SE.WindowDatabase.from_training(sales, oi, SE.L, SE.H, stride=7)
    rmsse_bb = SE.rmsse_series(c_pt, sales, oi)
    fused_pt = 0.5 * c_pt + 0.5 * r_pt
    rmsse_fused = SE.rmsse_series(fused_pt, sales, oi)

    # Aggregate each retrieved set once at unit scale. Every operator here is
    # scale-only with a zero location, so its forecast is this multiplied by a
    # scalar -- clipping commutes with a positive scale, so this is exact, not an
    # approximation, and it replaces four aggregation passes with one.
    unit_pred = np.full((n, SE.H), np.nan)
    unit = np.array([0.0, 1.0])
    for i, art in enumerate(arts):
        if art["ids"].size == 0:
            continue
        u = so.restore_batch(db.continuations[art["ids"]], art["cand_params"], unit)
        unit_pred[i] = np.clip(u, 0.0, None).mean(0)

    ctx = sales[:, oi - SE.L : oi]
    out: dict[str, np.ndarray] = {
        "origin": np.array([oi]),
        "zero_fraction": (ctx == 0).mean(axis=1),
        "degenerate": so.degenerate_mask(ctx),
        # A series with no sale anywhere before the origin has not launched yet.
        # Its zeros are pre-launch padding, not intermittent demand, and pooling it
        # with dormant series conflates two different populations.
        "never_launched": sales[:, :oi].sum(axis=1) == 0,
        "context_mean": ctx.mean(axis=1),
        "history_mean_365": sales[:, max(0, oi - 365) : oi].mean(axis=1),
        "rmsse_backbone": rmsse_bb,
        "rmsse_fused": rmsse_fused,
        "backbone_pt_mean": c_pt.mean(axis=1),
        "realised_future_mean": sales[:, oi : oi + SE.H].mean(axis=1),
    }

    # Referee item 4, applied to ourselves: an oracle-rescaled retrieval compared
    # against a RAW backbone is not like-for-like. Give the backbone the identical
    # least-squares rescaling so the oracle contrast is symmetric. On ETTm2 that
    # correction reversed a headline; it must not be skipped here.
    bb_scale = so.oracle_least_squares(c_pt, sales[:, oi : oi + SE.H], allow_oracle=True)[:, 1]
    out["rmsse_backbone_oracle_ls"] = SE.rmsse_series(c_pt * bb_scale[:, None], sales, oi)

    for op in OPERATOR_ORDER:
        qp = query_params_for(op, sales, oi, group_ids, lookback, unit_pred)
        pred = unit_pred * qp[:, 1][:, None]
        out[f"rmsse_{op}"] = SE.rmsse_series(pred, sales, oi)
        valid = np.isfinite(pred).any(axis=1)
        pt_mean = np.full(n, np.nan)
        if valid.any():
            pt_mean[valid] = np.nanmean(pred[valid], axis=1)
        out[f"pt_mean_{op}"] = pt_mean
        out[f"unresolved_{op}"] = ~np.isfinite(qp[:, 1])

    # Self-check: the deployed operator must reproduce the frozen retrieval path.
    frozen = SE.rmsse_series(r_pt, sales, oi)
    both = np.isfinite(frozen) & np.isfinite(out["rmsse_window_mean"])
    if both.any() and not np.allclose(frozen[both], out["rmsse_window_mean"][both], atol=1e-12):
        raise RuntimeError(
            f"origin {oi}: re-restoration under window_mean does not reproduce the frozen "
            "retrieval forecast; the artifacts path has diverged and results are not comparable"
        )
    return out


def _safe_nanmean(values: np.ndarray) -> float:
    """Mean over the finite entries, or NaN if there are none. Never warns."""
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


PAST_DATA_OPS = ("window_mean", "long_lookback", "hierarchy_prior")


def _mean_ci(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "n_origins": 0}
    ci = 1.96 * float(x.std(ddof=1)) / np.sqrt(x.size) if x.size > 1 else float("nan")
    return {"mean": float(x.mean()), "ci95": ci, "n_origins": int(x.size)}


def sparse_bin_contrasts(records: list[dict[str, np.ndarray]]) -> dict[str, object]:
    """Sparsest-bin win rates on ONE population, so the operators are comparable.

    Scoring each operator only where it happens to resolve compares different
    populations: an operator that cannot resolve the hardest series looks better
    for having dropped them. Worse, an oracle keyed to the realised future drops
    exactly the all-zero-future series, which conditions on the outcome.

    So the population is fixed by the operators that use **past data only**, and
    the oracles are reported on that same population with their residual attrition
    stated rather than hidden.
    """
    sparsest = FIXED_BIN_EDGES.size - 2
    per_op: dict[str, list[float]] = {op: [] for op in OPERATOR_ORDER}
    attrition: dict[str, list[float]] = {op: [] for op in OPERATOR_ORDER}
    deployed_all, deployed_ident, deployed_launched = [], [], []
    deg_frac, never_frac, n_pop = [], [], []
    never_in_bin, dropped, sym = [], [], []

    for rec in records:
        bb = rec["rmsse_backbone"]
        in_bin = bin_index(rec["zero_fraction"]) == sparsest
        base = in_bin & np.isfinite(bb)

        dep = base & np.isfinite(rec["rmsse_window_mean"])
        util_dep = bb - rec["rmsse_window_mean"]
        deployed_all.append(float((util_dep[dep] > 0).mean()) if dep.any() else float("nan"))
        ident = dep & ~rec["degenerate"]
        deployed_ident.append(float((util_dep[ident] > 0).mean()) if ident.any() else float("nan"))
        launched = dep & ~rec["never_launched"]
        deployed_launched.append(
            float((util_dep[launched] > 0).mean()) if launched.any() else float("nan")
        )
        # Shares are reported over the SCORED population, not the raw bin. A series
        # with an all-zero training history has a zero naive-scale denominator, so
        # RMSSE is NaN and it never entered the win rate; counting it in the share
        # would overstate the contamination of a number it cannot affect.
        deg_frac.append(float(rec["degenerate"][dep].mean()) if dep.any() else float("nan"))
        never_frac.append(float(rec["never_launched"][dep].mean()) if dep.any() else float("nan"))
        never_in_bin.append(
            float(rec["never_launched"][in_bin].mean()) if in_bin.any() else float("nan")
        )
        dropped.append(
            float((in_bin & ~dep).mean() / max(1e-12, in_bin.mean()))
            if in_bin.any()
            else float("nan")
        )
        bb_or = rec.get("rmsse_backbone_oracle_ls")
        if bb_or is not None:
            m = dep & np.isfinite(bb_or) & np.isfinite(rec["rmsse_oracle_least_squares"])
            sym.append(
                float(((bb_or - rec["rmsse_oracle_least_squares"])[m] > 0).mean())
                if m.any()
                else float("nan")
            )

        common = base.copy()
        for op in PAST_DATA_OPS:
            common = common & np.isfinite(rec[f"rmsse_{op}"])
        n_pop.append(int(common.sum()))
        for op in OPERATOR_ORDER:
            r_op = rec[f"rmsse_{op}"]
            m = common & np.isfinite(r_op)
            per_op[op].append(float(((bb - r_op)[m] > 0).mean()) if m.any() else float("nan"))
            attrition[op].append(
                float((~np.isfinite(r_op[common])).mean()) if common.any() else float("nan")
            )

    return {
        "population": (
            "sparsest fixed bin [0.9, 1.0]; series where every past-data operator "
            "resolves, so all operators are scored on the same population"
        ),
        "mean_series_per_origin": float(np.mean(n_pop)),
        "degenerate_fraction_of_scored_bin": _mean_ci(deg_frac),
        "never_launched_fraction_of_scored_bin": _mean_ci(never_frac),
        "never_launched_fraction_of_raw_bin": _mean_ci(never_in_bin),
        "fraction_of_bin_unscorable": _mean_ci(dropped),
        "oracle_ls_retrieval_vs_oracle_ls_backbone": {
            **_mean_ci(sym),
            "note": (
                "symmetric: both branches given the same least-squares rescaling. The "
                "asymmetric figure, oracle retrieval against a raw backbone, is not "
                "like-for-like (referee item 4)."
            ),
        },
        "deployed_all_contexts": _mean_ci(deployed_all),
        "deployed_identifiable_only": _mean_ci(deployed_ident),
        "deployed_launched_only": _mean_ci(deployed_launched),
        "operators_on_common_population": {
            op: {**_mean_ci(per_op[op]), "residual_attrition": _mean_ci(attrition[op])["mean"]}
            for op in OPERATOR_ORDER
        },
    }


def symmetric_oracle_by_bin(records: list[dict[str, np.ndarray]]) -> list[dict[str, float]]:
    """Win rate per bin when BOTH branches receive the same least-squares rescaling.

    The deployed win rate confounds two advantages: the analogues may have the
    better shape, or the backbone may simply be mis-levelled. Rescaling only the
    retrieval branch measures the sum of both and flatters retrieval; rescaling
    neither is the deployed number. Rescaling both isolates shape, and the gap
    between the two profiles is how much of retrieval's apparent utility was the
    backbone's level error rather than the analogues' quality.
    """
    out: list[dict[str, float]] = []
    for b in range(FIXED_BIN_EDGES.size - 1):
        dep, sym, ns = [], [], []
        for rec in records:
            m = bin_index(rec["zero_fraction"]) == b
            d = m & np.isfinite(rec["rmsse_backbone"]) & np.isfinite(rec["rmsse_window_mean"])
            s_ = (
                m
                & np.isfinite(rec["rmsse_backbone_oracle_ls"])
                & np.isfinite(rec["rmsse_oracle_least_squares"])
            )
            if d.any():
                dep.append(
                    float(((rec["rmsse_backbone"] - rec["rmsse_window_mean"])[d] > 0).mean())
                )
            if s_.any():
                sym.append(
                    float(
                        (
                            (rec["rmsse_backbone_oracle_ls"] - rec["rmsse_oracle_least_squares"])[
                                s_
                            ]
                            > 0
                        ).mean()
                    )
                )
                ns.append(int(s_.sum()))
        if not sym:
            continue
        d_ci, s_ci = _mean_ci(dep), _mean_ci(sym)
        out.append(
            {
                "lo": float(FIXED_BIN_EDGES[b]),
                "hi": float(FIXED_BIN_EDGES[b + 1]),
                "mean_series_per_origin": float(np.mean(ns)),
                "deployed_win_rate": d_ci["mean"],
                "symmetric_oracle_win_rate": s_ci["mean"],
                "symmetric_oracle_ci95": s_ci["ci95"],
                "shift": s_ci["mean"] - d_ci["mean"],
            }
        )
    return out


def summarise(records: list[dict[str, np.ndarray]]) -> dict[str, object]:
    """Pool per-origin arrays into the reported profiles."""
    operator_profiles: dict[str, object] = {}
    for op in OPERATOR_ORDER:
        rows, unresolved = [], []
        for rec in records:
            r_op, r_bb = rec[f"rmsse_{op}"], rec["rmsse_backbone"]
            ok = np.isfinite(r_op) & np.isfinite(r_bb)
            util = (r_bb - r_op)[ok]
            rows.append(
                stratified_bin_summary(rec["zero_fraction"][ok], util, util, rec["degenerate"][ok])
            )
            unresolved.append(float(rec[f"unresolved_{op}"].mean()))
        operator_profiles[op] = {
            "description": so.OPERATORS[op],
            "unresolved_scale_fraction_mean": float(np.mean(unresolved)),
            "per_origin_bins": rows,
        }

    fused_profile, scale_tags, backbone_degen = [], [], []
    sparsest_bin = FIXED_BIN_EDGES.size - 2
    for rec in records:
        r_dep, r_bb, r_fu = rec["rmsse_window_mean"], rec["rmsse_backbone"], rec["rmsse_fused"]
        ok = np.isfinite(r_dep) & np.isfinite(r_bb) & np.isfinite(r_fu)
        fused_profile.append(
            stratified_bin_summary(
                rec["zero_fraction"][ok],
                (r_bb - r_dep)[ok],
                (r_bb - r_dep)[ok],
                rec["degenerate"][ok],
                fused_utility=(r_bb - r_fu)[ok],
            )
        )

        idx_b = bin_index(rec["zero_fraction"])
        mu = rec["context_mean"]
        for b in range(FIXED_BIN_EDGES.size - 1):
            m = idx_b == b
            if not m.any():
                continue
            scale_tags.append(
                {
                    "origin": int(rec["origin"][0]),
                    "lo": float(FIXED_BIN_EDGES[b]),
                    "hi": float(FIXED_BIN_EDGES[b + 1]),
                    "n": int(m.sum()),
                    "mean_x_p05": float(np.quantile(mu[m], 0.05)),
                    "mean_x_median": float(np.median(mu[m])),
                    "mean_x_p95": float(np.quantile(mu[m], 0.95)),
                    "frac_mean_x_below_0p1": float((mu[m] < 0.1).mean()),
                }
            )

        # Section 2.6: on a zero-mean context the backbone's instance norm falls
        # back to eps=1e-5 and collapses toward zero, while the deployed operator
        # restores to unit scale. Record both so the sparsest bin's wins can be
        # attributed rather than assumed.
        deg = rec["degenerate"] & ok
        sparse = (idx_b == sparsest_bin) & ok
        if deg.any():
            util_dep = r_bb - r_dep
            backbone_degen.append(
                {
                    "origin": int(rec["origin"][0]),
                    "n_degenerate": int(deg.sum()),
                    "n_in_sparsest_bin": int(sparse.sum()),
                    "backbone_mean_forecast": _safe_nanmean(rec["backbone_pt_mean"][deg]),
                    "retrieval_mean_forecast": _safe_nanmean(rec["pt_mean_window_mean"][deg]),
                    "realised_future_mean": _safe_nanmean(rec["realised_future_mean"][deg]),
                    "win_rate_degenerate": float((util_dep[deg] > 0).mean()),
                    "share_of_sparsest_bin_wins_from_degenerate": float(
                        (util_dep[deg] > 0).sum() / max(1, int((util_dep[sparse] > 0).sum()))
                    ),
                }
            )

    band_all, band_ident = (
        _bands(records, identifiable_only=False),
        _bands(records, identifiable_only=True),
    )
    return {
        "scale_operator_profiles": operator_profiles,
        "fused_bin_profile": fused_profile,
        "scale_tag_distribution": scale_tags,
        "backbone_degeneracy": backbone_degen,
        "band_deployed": band_all,
        "band_identifiable_only": band_ident,
        "sparse_bin_contrasts": sparse_bin_contrasts(records),
        "symmetric_oracle_by_bin": symmetric_oracle_by_bin(records),
    }


def _bands(records: list[dict[str, np.ndarray]], *, identifiable_only: bool) -> dict[str, object]:
    lo, hi = [], []
    for rec in records:
        r_dep, r_bb = rec["rmsse_window_mean"], rec["rmsse_backbone"]
        ok = np.isfinite(r_dep) & np.isfinite(r_bb)
        if identifiable_only:
            ok = ok & ~rec["degenerate"]
        if ok.sum() < 20:
            continue
        bd = estimate_band((r_bb - r_dep)[ok], rec["zero_fraction"][ok], "intermittency", n_boot=0)
        if bd.lower is not None:
            lo.append(bd.lower)
        if bd.upper is not None:
            hi.append(bd.upper)
    return {
        "lower_mean": float(np.mean(lo)) if lo else None,
        "lower_sd": float(np.std(lo, ddof=1)) if len(lo) > 1 else None,
        "upper_mean": float(np.mean(hi)) if hi else None,
        "upper_sd": float(np.std(hi, ddof=1)) if len(hi) > 1 else None,
        "n_origins_with_lower": len(lo),
        "n_origins_with_upper": len(hi),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--origins", type=int, default=50)
    ap.add_argument("--stride", type=int, default=28, help="days between origins; default is H")
    ap.add_argument("--lookback", type=int, default=365, help="window for long_lookback")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="reuse per-origin checkpoints already on disk instead of recomputing them",
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
    origins = [o_eval - i * args.stride for i in range(args.origins)]
    if min(origins) - SE.L < 0:
        raise SystemExit(f"origin {min(origins)} has no room for a length-{SE.L} context")
    if max(o + SE.H for o in origins) > test_start:
        raise SystemExit("refusing to run: an evaluation window reaches the consumed test split")

    CKPT.mkdir(parents=True, exist_ok=True)
    tag = f"{CKPT_VERSION}-s{args.subset}-seed{args.seed}-lb{args.lookback}"
    group_ids = group_codes(entities)

    fc = None
    records: list[dict[str, np.ndarray]] = []
    bar = tqdm(origins, desc="origins", unit="origin", dynamic_ncols=True)
    for oi in bar:
        path = CKPT / f"{tag}-origin{oi}.npz"
        if args.resume and path.exists():
            with np.load(path) as z:
                records.append({k: z[k] for k in z.files})
            bar.set_postfix_str(f"origin {oi}: cached")
            continue
        if fc is None:  # loaded lazily so a fully-cached resume needs no GPU
            fc = Chronos2Forecaster()
        rec = run_origin(sales, entities, oi, group_ids, args.lookback, fc)
        np.savez_compressed(path, **rec)
        records.append(rec)
        util = rec["rmsse_backbone"] - rec["rmsse_window_mean"]
        ok = np.isfinite(util)
        bar.set_postfix_str(f"origin {oi}: win {float((util[ok] > 0).mean()):.3f}")
    bar.close()

    payload = {
        "experiment": "sparse-end-scale-identifiability",
        "dataset": "M5",
        "split": "validation",
        "guard": "test split d_1914-d_1941 is consumed and untouched (rule 2)",
        "note": (
            "retrieval runs once per origin; the identical retrieved candidates are "
            "re-restored under each operator, so any difference is attributable to "
            "restoration. No configuration is selected (rules 9, 12)."
        ),
        "n_origins": len(origins),
        "origins": origins,
        "stride": args.stride,
        "subset_draw": {
            "n_requested": args.subset,
            "n_selected": int(n),
            "seed": args.seed,
            "method": "scalerag.eval.select_subset",
            "note": "recorded so the bin rates can be read as conditional on this draw",
        },
        "frozen_retrieval": {
            "scale": "mean",
            "k": SE.K,
            "meta_filter": "cat_id",
            "L": SE.L,
            "H": SE.H,
        },
        "context_length_L": SE.L,
        "horizon_H": SE.H,
        "top_k": SE.K,
        "deployed_zero_mean_substitution": 1.0,
        "long_lookback_days": args.lookback,
        "binning": "fixed absolute zero-fraction intervals; quantile deciles not used",
        **summarise(records),
        "timestamp": datetime.now(UTC).isoformat(),
        "runtime_sec": round(time.time() - started, 2),
        "run_context": RunContext().to_dict(),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"m5-val-scale-identifiability-{tag}-{len(origins)}origins.json"
    path.write_text(json.dumps(payload, indent=2))

    print(f"\nwrote {path.relative_to(REPO)}\n")
    c = payload["sparse_bin_contrasts"]
    print(
        f"sparsest fixed bin [0.9, 1.0], {len(origins)} origins, "
        f"mean n={c['mean_series_per_origin']:.0f} per origin"
    )
    print(f"  degenerate share (scored) : {c['degenerate_fraction_of_scored_bin']['mean']:.3f}")
    print(
        f"  never-launched (raw bin)  : {c['never_launched_fraction_of_raw_bin']['mean']:.3f}"
        f"  -> of scored: {c['never_launched_fraction_of_scored_bin']['mean']:.3f}"
    )
    print(f"  bin unscorable (NaN RMSSE): {c['fraction_of_bin_unscorable']['mean']:.3f}")
    for label, key in (
        ("deployed, all contexts", "deployed_all_contexts"),
        ("deployed, identifiable", "deployed_identifiable_only"),
        ("deployed, launched only", "deployed_launched_only"),
    ):
        r = payload["sparse_bin_contrasts"][key]
        print(f"  {label:24s}: {r['mean']:.4f} +/- {r['ci95']:.4f}")

    print(
        f"\n{'operator (common population)':>30s} {'win rate':>10s} {'+/-95%':>9s} {'attrition':>10s}"
    )
    for op in OPERATOR_ORDER:
        r = c["operators_on_common_population"][op]
        print(f"{op:>30s} {r['mean']:10.4f} {r['ci95']:9.4f} {r['residual_attrition']:10.3f}")

    print(f"\n{'bin':>10s} {'deployed':>10s} {'symmetric':>11s} {'+/-95%':>8s} {'shift':>8s}")
    for row in payload["symmetric_oracle_by_bin"]:
        print(
            f"[{row['lo']:.1f},{row['hi']:.1f}] {row['deployed_win_rate']:10.3f} "
            f"{row['symmetric_oracle_win_rate']:11.3f} {row['symmetric_oracle_ci95']:8.4f} "
            f"{row['shift']:+8.3f}"
        )

    sym_r = c["oracle_ls_retrieval_vs_oracle_ls_backbone"]
    print(
        f"\nSYMMETRIC oracle contrast (both branches rescaled): "
        f"{sym_r['mean']:.4f} +/- {sym_r['ci95']:.4f}"
    )

    bd, bi = payload["band_deployed"], payload["band_identifiable_only"]
    print(f"\nband, deployed         : [{bd['lower_mean']}, {bd['upper_mean']}]")
    print(f"band, identifiable only: [{bi['lower_mean']}, {bi['upper_mean']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
