#!/usr/bin/env python3
"""M5 validation: k sweep beyond the deployed value, gate validation, and cost.

Three referee items that share one retrieval pass.

**Item 24 (k sweep).** Both configurations deploy k=20, which was the top of the
swept range, so "k=20 is good" could not be distinguished from "20 is where we
stopped looking". Retrieval here searches once at ``--kmax`` and evaluates every
smaller k by prefix: the restricted-pool path in ``retrieval_faiss`` sorts the
legal candidates exactly and slices ``[:k]``, so the top-20 is the first 20 of the
top-100 and the sweep costs one search rather than five.

**Item 9 (gate validation).** The gate is fitted on two historical origins and its
quality has never been reported. It is trained here on the same two frozen origins
and scored on a *third* origin used for neither training nor evaluation, giving an
out-of-origin AUC, accuracy and Brier score.

**Item 19 (M5 cost row).** Table X has no M5 row at all, despite M5 being the panel
where the method wins and where the index is larger. Index sizes are reported in
the dtype FAISS actually holds (float32), separately from the float64 candidate
matrix the process also retains.

Nothing here selects a configuration. The frozen k stays 20 whatever the sweep
shows; reporting where the deployed value sits in a wider range is the point
(rules 9, 12).

Usage (fish):
    uv run python scripts/m5_ksweep_gate_run.py --subset 1000 --resume
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
from scalerag.eval import load_processed, select_subset  # noqa: E402
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402
from scalerag.retrieval import WindowDatabase  # noqa: E402
from scalerag.retrieval_faiss import ScaleAwareIndex, restore_continuation  # noqa: E402
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

OUT = REPO / "reports" / "ksweep-gate"
CKPT = OUT / "checkpoints"
LOCK = REPO / "M5_TEST_CONSUMED.lock"
K_GRID = (5, 10, 20, 50, 100)


def retrieve_prefix_forecasts(
    sales: np.ndarray, entities, o: int, kmax: int, k_grid: tuple[int, ...]
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, dict[str, float]]:
    """Search once at ``kmax``; return a forecast per k, plus gate features and cost.

    Returns ``(forecasts_by_k, nn_dist, disagreement, cost)``.
    """
    n = sales.shape[0]
    t0 = time.time()
    db = WindowDatabase.from_training(sales, o, SE.L, SE.H, stride=7)
    idx = ScaleAwareIndex(db, entities, scale="mean", metric="l2")
    build_s = time.time() - t0

    preds = {k: np.zeros((n, SE.H)) for k in k_grid}
    nnd, dis = np.zeros(n), np.zeros(n)
    t1 = time.time()
    for i in range(n):
        q = sales[i, o - SE.L : o]
        ids, dists, qp = idx.search(q, o + 1, kmax, query_series_idx=i, meta_filter="cat_id")
        if ids.size == 0:
            base = float(q.mean())
            for k in k_grid:
                preds[k][i] = base
            nnd[i], dis[i] = 1e6, 0.0
            continue
        conts = np.stack(
            [restore_continuation(db.continuations[j], idx.params[j], qp) for j in ids]
        )
        conts = np.clip(conts, 0.0, None)
        for k in k_grid:
            preds[k][i] = conts[: min(k, len(ids))].mean(0)
        nnd[i] = float(dists.min())
        dis[i] = float(conts[: min(20, len(ids))].std(0).mean())
    search_s = time.time() - t1

    cost = {
        "index_build_seconds": build_s,
        "search_seconds": search_s,
        "search_latency_ms_per_query": 1000.0 * search_s / max(1, n),
        "candidate_pool_windows": len(db),
        # FAISS holds float32; the float64 candidate matrix is retained separately
        # by the WindowDatabase and is an implementation artifact, not index storage.
        "index_vectors_mb_float32": idx._vecs.nbytes / 1e6,
        "continuations_mb_float64": db.continuations.nbytes / 1e6,
        "contexts_retained_mb_float64": db.contexts.nbytes / 1e6,
    }
    return preds, nnd, dis, cost


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kmax", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    started = time.time()
    k_grid = tuple(k for k in K_GRID if k <= args.kmax)

    from scalerag.tsfm.chronos2 import Chronos2Forecaster

    entities, dynamic = load_processed(REPO / "data" / "processed")
    entities, dynamic = select_subset(entities, dynamic, args.subset, args.seed)
    n = entities.height
    n_days = int(dynamic["day_idx"].max())
    sales = dynamic["sales"].to_numpy().astype(np.float64).reshape(n, n_days)

    splits = make_rolling_splits(last_labeled_day=n_days)
    o_eval = split_by_name(splits, "val").train_end
    test_start = split_by_name(splits, "test").train_end
    if o_eval + SE.H > test_start:
        raise SystemExit(
            f"refusing to run: window reaches the consumed test split. See {LOCK.name}"
        )

    # Frozen gate origins, plus one origin used for neither training nor evaluation.
    gate_origins = [o_eval - 2 * SE.H, o_eval - SE.H]
    holdout_origin = o_eval - 3 * SE.H
    origins = [holdout_origin, *gate_origins, o_eval]

    CKPT.mkdir(parents=True, exist_ok=True)
    tag = f"s{args.subset}-seed{args.seed}-kmax{args.kmax}"
    fc = None
    per_origin: dict[int, dict] = {}

    for oi in tqdm(origins, desc="origins", unit="origin", dynamic_ncols=True):
        path = CKPT / f"{tag}-origin{oi}.npz"
        if args.resume and path.exists():
            with np.load(path) as z:
                per_origin[oi] = {k: z[k] for k in z.files}
            continue
        if fc is None:
            fc = Chronos2Forecaster()
        t0 = time.time()
        c_pt, c_q = fc.forecast([sales[i, :oi] for i in range(n)], SE.H, SE.QL)
        chronos_s = time.time() - t0
        c_pt = np.clip(c_pt, 0.0, None)
        preds, nnd, dis, cost = retrieve_prefix_forecasts(sales, entities, oi, args.kmax, k_grid)
        feats, _names = SE.gate_features(sales, oi, nnd, dis, c_q), None
        rec = {
            "origin": np.array([oi]),
            "rmsse_backbone": SE.rmsse_series(c_pt, sales, oi),
            "features": feats,
            "chronos_seconds": np.array([chronos_s]),
            **{f"rmsse_k{k}": SE.rmsse_series(preds[k], sales, oi) for k in k_grid},
            **{f"cost_{k}": np.array([v]) for k, v in cost.items()},
        }
        np.savez_compressed(path, **rec)
        per_origin[oi] = rec

    # --- item 24: the sweep, evaluated at the evaluation origin
    ev = per_origin[o_eval]
    bb = ev["rmsse_backbone"]
    sweep = []
    for k in k_grid:
        r = ev[f"rmsse_k{k}"]
        ok = np.isfinite(r) & np.isfinite(bb)
        sweep.append(
            {
                "k": k,
                "rmsse_mean": float(np.nanmean(r[ok])),
                "win_rate_vs_backbone": float(((bb - r)[ok] > 0).mean()),
                "n": int(ok.sum()),
            }
        )
    best = min(sweep, key=lambda d: d["rmsse_mean"])

    # --- item 9: gate trained on the frozen origins, scored on an unused origin
    import lightgbm as lgb
    from sklearn.metrics import brier_score_loss, roc_auc_score

    spec = {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 50}
    xs, ys = [], []
    for go in gate_origins:
        rec = per_origin[go]
        lab = (rec["rmsse_k20"] < rec["rmsse_backbone"]).astype(int)
        ok = np.isfinite(rec["rmsse_k20"]) & np.isfinite(rec["rmsse_backbone"])
        xs.append(rec["features"][ok])
        ys.append(lab[ok])
    xg, yg = np.vstack(xs), np.concatenate(ys)

    ho = per_origin[holdout_origin]
    ok_h = np.isfinite(ho["rmsse_k20"]) & np.isfinite(ho["rmsse_backbone"])
    y_h = (ho["rmsse_k20"] < ho["rmsse_backbone"]).astype(int)[ok_h]
    x_h = ho["features"][ok_h]

    scores = []
    for seed in (42, 43, 44):
        g = lgb.LGBMClassifier(**spec, verbose=-1, random_state=seed).fit(xg, yg)
        p = g.predict_proba(x_h)[:, 1]
        scores.append(
            {
                "seed": seed,
                "auc": float(roc_auc_score(y_h, p)),
                "accuracy": float(((p > 0.5).astype(int) == y_h).mean()),
                "brier": float(brier_score_loss(y_h, p)),
            }
        )
    base_rate = float(y_h.mean())

    # --- item 19: the M5 cost row
    cost_keys = [k for k in ev if k.startswith("cost_")]
    m5_cost = {k[5:]: float(np.mean([per_origin[o][k][0] for o in origins])) for k in cost_keys}
    m5_cost["chronos_seconds_mean"] = float(
        np.mean([per_origin[o]["chronos_seconds"][0] for o in origins])
    )
    m5_cost["n_series"] = int(n)

    payload = {
        "experiment": "m5-ksweep-gate-cost",
        "dataset": "M5",
        "split": "validation",
        "guard": "test split is consumed and untouched (rule 2)",
        "note": "diagnostic only; the frozen k=20 and the frozen gate are unchanged",
        "eval_origin": int(o_eval),
        "gate_train_origins": gate_origins,
        "gate_holdout_origin": int(holdout_origin),
        "k_sweep": sweep,
        "k_sweep_best": best,
        "deployed_k": 20,
        "gate_out_of_origin": {
            "per_seed": scores,
            "auc_mean": float(np.mean([s["auc"] for s in scores])),
            "accuracy_mean": float(np.mean([s["accuracy"] for s in scores])),
            "brier_mean": float(np.mean([s["brier"] for s in scores])),
            "majority_class_baseline": max(base_rate, 1 - base_rate),
            "positive_rate": base_rate,
            "n_holdout": int(ok_h.sum()),
            "n_train": len(yg),
        },
        "m5_cost": m5_cost,
        "timestamp": datetime.now(UTC).isoformat(),
        "runtime_sec": round(time.time() - started, 2),
        "run_context": RunContext().to_dict(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"m5-val-ksweep-gate-{tag}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path.relative_to(REPO)}\n")

    print(f"{'k':>5s} {'RMSSE':>9s} {'win rate':>10s}")
    for r in sweep:
        mark = "  <- deployed" if r["k"] == 20 else ("  <- best" if r["k"] == best["k"] else "")
        print(f"{r['k']:5d} {r['rmsse_mean']:9.4f} {r['win_rate_vs_backbone']:10.3f}{mark}")

    g = payload["gate_out_of_origin"]
    print(f"\ngate, held-out origin {holdout_origin} (trained on {gate_origins}):")
    print(
        f"  AUC {g['auc_mean']:.4f} | accuracy {g['accuracy_mean']:.4f} "
        f"vs majority {g['majority_class_baseline']:.4f} | Brier {g['brier_mean']:.4f}"
    )
    print(
        f"\nM5 cost: index {m5_cost['index_vectors_mb_float32']:.1f} MB float32, "
        f"{m5_cost['search_latency_ms_per_query']:.3f} ms/query, "
        f"pool {m5_cost['candidate_pool_windows']:.0f} windows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
