#!/usr/bin/env python3
"""ScaleRAG-TS — single LOCKED held-out evaluation (Phase 10).

Runs the FROZEN ScaleRAG method (no tuning, no changes) on either the primary
validation origin (``--split val``, for the val<->test comparison, item 8) or the
reserved held-out test origin (``--split test`` = d_1914..d_1941), on the FULL M5
panel (30,490 series). Method, hyperparameters, gate features, gate seeds, gate
training origins, retrieval config, and QL are identical to the frozen Phase-9
matrix (``scripts/scalerag_matrix.py``); only the eval origin and the exact-but-
GPU-accelerated retriever (verified bit-identical by ``verify_gpu_retrieval.py``)
differ, so the full panel is tractable.

Test-consumption discipline (rules 2, 9, 12): ``--split test`` refuses to run if
``M5_TEST_CONSUMED.lock`` already exists, and writes that lock after a successful
run. This blocks any further test-driven tuning.

    uv run python scripts/scalerag_test_final.py --split val    # matched-pop reference
    uv run python scripts/scalerag_test_final.py --split test   # the one-shot (locks after)
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import gc
import json
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scalerag import fusion, hierarchy, metrics  # noqa: E402
from scalerag.eval import load_processed, select_subset  # noqa: E402
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402
from scalerag.retrieval_gpu import retrieve_scaleaware_gpu  # noqa: E402
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

QL = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
L, H, K, STRIDE = 56, 28, 20, 7
LEVELS = {"50": (0.25, 0.75), "80": (0.1, 0.9), "90": (0.05, 0.95)}
GATE_SPEC = {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 50}
TEST_LOCK = REPO / "M5_TEST_CONSUMED.lock"


def const_quants(pt: np.ndarray) -> np.ndarray:
    return np.repeat(pt[:, :, None], len(QL), axis=2)


def gate_features(sales, o, nnd, dis, chr_q):
    ctx = sales[:, o - L : o]
    mean, std = ctx.mean(1), ctx.std(1)
    unc = (chr_q[:, :, QL.index(0.9)] - chr_q[:, :, QL.index(0.1)]).mean(1)
    cols = {
        "nn_dist": nnd,
        "disagreement": dis,
        "intermittency": (ctx == 0).mean(1),
        "log_volume": np.log(mean + 1.0),
        "chronos_uncertainty": unc,
        "scale_spread": std / (mean + 1e-6),
    }
    return np.stack(list(cols.values()), axis=1), list(cols)


def per_series_rmsse(pred, sales, o):
    ha, tr = sales[:, o : o + H], sales[:, :o]
    return np.array([metrics.rmsse(ha[i], pred[i], tr[i]) for i in range(pred.shape[0])])


def scores(pred, quants, sales, o, entities, weights, mask=None, with_wrmsse=True):
    idx = np.arange(pred.shape[0]) if mask is None else np.flatnonzero(mask)
    ha, tr = sales[:, o : o + H], sales[:, :o]
    ri = per_series_rmsse(pred, sales, o)[idx]
    mase = np.array([metrics.mase(ha[i], pred[i], tr[i]) for i in idx])
    pin = np.array([metrics.pinball_loss(ha[i], quants[i], QL) for i in idx])
    out = {
        "rmsse": float(np.nanmean(ri)),
        "mase": float(np.nanmean(mase)),
        "wape": metrics.wape(ha[idx].ravel(), pred[idx].ravel()),
        "mae": metrics.mae(ha[idx].ravel(), pred[idx].ravel()),
        "pinball": float(np.nanmean(pin)),
    }
    for lv, (qlo, qhi) in LEVELS.items():
        lo, hi = quants[idx][:, :, QL.index(qlo)], quants[idx][:, :, QL.index(qhi)]
        out[f"cov{lv}"] = float(np.mean((ha[idx] >= lo) & (ha[idx] <= hi)))
        out[f"width{lv}"] = float(np.mean(hi - lo))
    if with_wrmsse and mask is None:
        score, _lvl = hierarchy.wrmsse(entities, tr, ha, pred, weights)
        out["wrmsse"] = score
    return out


def chronos_batched(fc, contexts, batch):
    pts, qs = [], []
    for s in range(0, len(contexts), batch):
        p, q = fc.forecast(contexts[s : s + batch], H, QL)
        pts.append(np.clip(p, 0.0, None))
        qs.append(q)
    return np.vstack(pts), np.vstack(qs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--subset", type=int, default=0, help="0 = full panel")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate-seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--chronos-batch", type=int, default=1000)
    args = ap.parse_args()

    if args.split == "test" and TEST_LOCK.exists():
        print(
            f"REFUSED: {TEST_LOCK.name} exists — the M5 test split is already consumed.\n"
            f"Further test-driven tuning is blocked (rules 2, 9, 12). "
            f"Delete the lock only with an explicit, logged authorization.",
            file=sys.stderr,
        )
        return 2

    set_seed(args.seed)
    import lightgbm as lgb

    from scalerag.baselines.lightgbm_model import fit_predict as lgb_fit_predict
    from scalerag.features import build_features
    from scalerag.tsfm.chronos2 import Chronos2Forecaster

    entities, dynamic = load_processed(REPO / "data" / "processed")
    if args.subset > 0:
        entities, dynamic = select_subset(entities, dynamic, args.subset, args.seed)
    n = entities.height
    n_days = int(dynamic["day_idx"].max())
    sales = dynamic["sales"].to_numpy().astype(np.float64).reshape(n, n_days)
    price = dynamic["sell_price"].to_numpy().astype(np.float64).reshape(n, n_days)
    cat_codes = np.unique(entities["cat_id"].to_numpy(), return_inverse=True)[1].astype(np.int64)

    splits = make_rolling_splits(last_labeled_day=n_days)
    ev = split_by_name(splits, args.split)
    o = ev.train_end
    # FROZEN gate-training origins (unchanged from the Phase-9 study): val_m2, val_m1.
    gate_origins = [
        split_by_name(splits, "val_m2").train_end,
        split_by_name(splits, "val_m1").train_end,
    ]
    if not all(go < o for go in gate_origins):
        raise ValueError(f"gate origins {gate_origins} must all precede eval origin {o}")

    # WRMSSE dollar weights from last-28 TRAIN days only (rule 5)
    weights = hierarchy.dollar_weights(sales[:, o - 28 : o], price[:, o - 28 : o])

    # LightGBM baseline FIRST — it holds the ~58M-row `dynamic` panel; compute it
    # and free `dynamic` before the window-heavy retrieval, so the two large host
    # allocations never overlap (15 GB machine).
    features, _stats = build_features(dynamic, entities, ev)
    del dynamic
    gc.collect()
    lgb_pred, _m = lgb_fit_predict(features, ev, seed=args.seed)
    lgb_mat = lgb_pred.sort(["id", "day_idx"])["y_pred"].to_numpy().reshape(n, H)
    del features
    gc.collect()

    fc = Chronos2Forecaster()

    def chronos_at(origin):
        contexts = [sales[i, :origin] for i in range(n)]
        return chronos_batched(fc, contexts, args.chronos_batch)

    # --- eval-origin forecasts ---
    t0 = time.perf_counter()
    c_pt, c_q = chronos_at(o)
    chr_ms = 1000 * (time.perf_counter() - t0)
    vram = fc.cuda_peak_gib()

    queries = np.stack([sales[i, o - L : o] for i in range(n)])
    t0 = time.perf_counter()
    rr = retrieve_scaleaware_gpu(sales, cat_codes, o, o + 1, queries, cat_codes, QL, k=K)
    retr_ms = 1000 * (time.perf_counter() - t0)
    r_pt, r_q, nnd, dis = rr.point, rr.quants, rr.nn_dist, rr.disagreement

    # --- gate: trained ONLY on historical origins (frozen procedure) ---
    Xg, yg, gate_names = [], [], None
    for go in gate_origins:
        cg, cgq = chronos_at(go)
        gq = np.stack([sales[i, go - L : go] for i in range(n)])
        rg = retrieve_scaleaware_gpu(sales, cat_codes, go, go + 1, gq, cat_codes, QL, k=K)
        act = sales[:, go : go + H]
        lc = np.sqrt(((act - cg) ** 2).mean(1))
        lr = np.sqrt(((act - rg.point) ** 2).mean(1))
        f, gate_names = gate_features(sales, go, rg.nn_dist, rg.disagreement, cgq)
        Xg.append(f)
        yg.append((lr < lc).astype(int))
    Xg, yg = np.vstack(Xg), np.concatenate(yg)
    Xe, _ = gate_features(sales, o, nnd, dis, c_q)

    def train_gate(seed):
        g = lgb.LGBMClassifier(**GATE_SPEC, verbose=-1, random_state=seed)
        g.fit(Xg, yg)
        return g

    gates = [train_gate(s) for s in args.gate_seeds]
    alphas = np.mean([g.predict_proba(Xe)[:, 1] for g in gates], 0)
    fused_pt, fused_q = fusion.fuse(c_pt, c_q, r_pt, r_q, alphas)
    fixed_pt, fixed_q = fusion.fuse(c_pt, c_q, r_pt, r_q, 0.5)

    # --- simple point baselines (lightgbm already computed above) ---
    rm_pt = np.clip(np.stack([np.full(H, queries[i].mean()) for i in range(n)]), 0, None)
    sn_pt = np.clip(sales[:, o - 7 : o][:, np.arange(H) % 7], 0, None)

    methods = {
        "recent_mean": (rm_pt, const_quants(rm_pt), "exact-baseline"),
        "seasonal_naive": (sn_pt, const_quants(sn_pt), "exact-baseline"),
        "lightgbm": (lgb_mat, const_quants(lgb_mat), "exact-baseline"),
        "chronos2_target": (c_pt, c_q, "exact-frozen-backbone"),
        "retrieval_scaleaware": (r_pt, r_q, "ours-component"),
        "fusion_fixed0.5": (fixed_pt, fixed_q, "ours-ablation"),
        "ScaleRAG_gated": (fused_pt, fused_q, "ours-proposed"),
    }

    rmsse_i = {m: per_series_rmsse(p, sales, o) for m, (p, _q, _t) in methods.items()}
    rows = {
        m: {**scores(p, q, sales, o, entities, weights), "kind": t}
        for m, (p, q, t) in methods.items()
    }
    strongest = min([m for m in methods if m != "ScaleRAG_gated"], key=lambda m: rows[m]["rmsse"])
    boot = {
        f"vs_{name}": fusion.paired_bootstrap_rel_improvement(
            rmsse_i[base], rmsse_i["ScaleRAG_gated"]
        )
        for name, base in [("chronos", "chronos2_target"), ("strongest", strongest)]
    }
    boot_per_method = {
        m: fusion.paired_bootstrap_rel_improvement(rmsse_i["chronos2_target"], rmsse_i[m])
        for m in methods
        if m != "chronos2_target"
    }

    # --- pre-registered slices ---
    tr_e = sales[:, :o]
    zf, vol, nz = (tr_e == 0).mean(1), tr_e.mean(1), (tr_e != 0).sum(1)
    slices = {
        "intermittent(z>0.8)": zf > 0.8,
        "low_volume(<med)": vol < np.median(vol),
        "reduced_history(<100nz)": nz < 100,
        "dense(z<0.3)": zf < 0.3,
    }
    slice_rows = {}
    for sn_, sm in slices.items():
        if sm.sum() < 20:
            continue
        si = np.flatnonzero(sm)
        slice_rows[sn_] = {
            "n": int(sm.sum()),
            "chronos_rmsse": float(np.nanmean(rmsse_i["chronos2_target"][si])),
            "scalerag_rmsse": float(np.nanmean(rmsse_i["ScaleRAG_gated"][si])),
            "strongest_rmsse": float(np.nanmean(rmsse_i[strongest][si])),
            "scalerag_scores": scores(
                fused_pt, fused_q, sales, o, entities, weights, mask=sm, with_wrmsse=False
            ),
            "vs_chronos": fusion.paired_bootstrap_rel_improvement(
                rmsse_i["chronos2_target"][si], rmsse_i["ScaleRAG_gated"][si]
            ),
            "vs_strongest": fusion.paired_bootstrap_rel_improvement(
                rmsse_i[strongest][si], rmsse_i["ScaleRAG_gated"][si]
            ),
        }

    gate_beh = {
        "alpha_mean": float(alphas.mean()),
        "alpha_std": float(alphas.std()),
        "alpha_pct_prefer_chronos(<0.5)": float((alphas < 0.5).mean()),
        "corr_alpha_chronos_unc": float(
            np.corrcoef(alphas, Xe[:, gate_names.index("chronos_uncertainty")])[0, 1]
        ),
        "corr_alpha_intermittency": float(
            np.corrcoef(alphas, Xe[:, gate_names.index("intermittency")])[0, 1]
        ),
    }

    # gate "trainable parameters": frozen backbone updates 0 params; the gate is a
    # GBDT ensemble (report its size, not a gradient-parameter count).
    gate_leaves = int(sum(g.booster_.num_trees() for g in gates)) * GATE_SPEC["num_leaves"]
    ram_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)

    report = {
        "phase": "10-locked-heldout-test",
        "split": args.split,
        "eval_window": f"d_{o + 1}-d_{o + H}",
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": "M5",
        "n_series": n,
        "full_panel": args.subset == 0,
        "seed": args.seed,
        "gate_seeds": args.gate_seeds,
        "eval_origin": o,
        "gate_origins": gate_origins,
        "frozen_config": "ScaleRAG_gated: mean/L2/cat-filter/k=20 + scale restoration + LGBM gate",
        "methods": rows,
        "strongest_baseline": strongest,
        "bootstrap_scalerag": boot,
        "bootstrap_per_method_vs_chronos": boot_per_method,
        "slices": slice_rows,
        "gate_behaviour": gate_beh,
        "profiling": {
            "chronos_inference_ms": round(chr_ms),
            "retrieval_ms": round(retr_ms),
            "gpu_vram_gib": round(vram, 3),
            "host_ram_peak_gib": round(ram_gib, 3),
            "backbone_params_updated": 0,
            "backbone_frozen": True,
            "gate_kind": "LightGBM GBDT",
            "gate_ensemble_leaf_values_approx": gate_leaves,
            "gate_seeds_averaged": len(args.gate_seeds),
        },
        "run_context": RunContext().to_dict(),
    }
    out = REPO / "reports" / f"scalerag-heldout-{args.split}-{n}.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    print(f"\n=== ScaleRAG-TS LOCKED {args.split.upper()} ({report['eval_window']}, n={n}) ===")
    print(f"{'method':22s} {'RMSSE':>7s} {'WRMSSE':>7s} {'MASE':>6s} {'WAPE':>6s} {'cov80':>6s}")
    for m in sorted(rows, key=lambda m: rows[m]["rmsse"]):
        r = rows[m]
        print(
            f"{m:22s} {r['rmsse']:7.4f} {r.get('wrmsse', float('nan')):7.4f} "
            f"{r['mase']:6.3f} {r['wape']:6.3f} {r['cov80']:6.3f}"
        )
    b = boot["vs_chronos"]
    bs = boot["vs_strongest"]
    print(
        f"\nScaleRAG vs chronos {b['rel_improvement']:+.2%} "
        f"[{b['ci95_low']:+.2%},{b['ci95_high']:+.2%}] | "
        f"vs strongest({strongest}) {bs['rel_improvement']:+.2%} "
        f"[{bs['ci95_low']:+.2%},{bs['ci95_high']:+.2%}]"
    )
    print(f"report -> {out.relative_to(REPO)}")

    if args.split == "test":
        TEST_LOCK.write_text(
            json.dumps(
                {
                    "consumed_at": datetime.now(UTC).isoformat(),
                    "commit": RunContext().to_dict().get("git_commit"),
                    "eval_window": report["eval_window"],
                    "report": out.name,
                    "note": "M5 test split d_1914-d_1941 consumed by the single locked "
                    "Phase-10 run. No further test-driven tuning (rules 2, 9, 12).",
                },
                indent=2,
            )
        )
        print(f"\nM5 TEST SPLIT CONSUMED — wrote {TEST_LOCK.name}. Further test runs blocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
