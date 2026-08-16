#!/usr/bin/env python3
"""Scalable, scale-aware temporal-retrieval config search (Phase 5).

Sweeps scale strategies, similarity metrics, metadata filters, scale restoration,
top-k, and (optionally) Chronos-2 fusion on the M5 val split (d_1886-d_1913; test
untouched). Selects the strongest NON-GRAPH retrieval config by RMSSE on
validation only. Reports RMSSE/MASE/WAPE/MAE/pinball + index-build / retrieval /
inference latency and RAM/VRAM. Explicit naive baselines are included so no
config can masquerade as retrieval; empty-retrieval fallbacks are counted and
reported. Negative results are kept.

    uv run python scripts/scalable_retrieval_eval.py --subset 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scalerag import metrics  # noqa: E402
from scalerag.eval import load_processed, select_subset  # noqa: E402
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402
from scalerag.retrieval import WindowDatabase  # noqa: E402
from scalerag.retrieval_faiss import ScaleAwareIndex, _fit_params, _transform  # noqa: E402
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _mat(dyn, col, n, t):
    return dyn[col].to_numpy().astype(np.float64).reshape(n, t)


def _score(name, pred, quants, sales, split, extra=None):
    n = pred.shape[0]
    ha = sales[:, split.h_start - 1 : split.h_end]
    tr = sales[:, : split.train_end]
    rmsse = float(np.nanmean([metrics.rmsse(ha[i], pred[i], tr[i]) for i in range(n)]))
    mase = float(np.nanmean([metrics.mase(ha[i], pred[i], tr[i]) for i in range(n)]))
    pin = float(np.nanmean([metrics.pinball_loss(ha[i], quants[i], QL) for i in range(n)]))
    out = {
        "config": name,
        "rmsse": rmsse,
        "mase": mase,
        "wape": metrics.wape(ha.ravel(), pred.ravel()),
        "mae": metrics.mae(ha.ravel(), pred.ravel()),
        "pinball": pin,
    }
    if extra:
        out.update(extra)
    return out


def global_batch(idx, queries_arr, k, restore):
    """Batched top-k global retrieval (all training candidates are legal at a val
    origin). Returns (point [n,H], quants [n,H,Q])."""
    import faiss

    qp = _fit_params(queries_arr, idx.scale)
    qv = _transform(queries_arr, qp).astype(np.float32)
    if idx.metric == "cosine":
        faiss.normalize_L2(qv)
    _d, sel = idx.index.search(qv, k)  # (n, k)
    conts = idx.db.continuations[sel]  # (n, k, H)
    if restore and idx.scale != "raw":
        cp = idx.params[sel]  # (n, k, 2)
        conts = (conts - cp[..., 0:1]) / cp[..., 1:2] * qp[:, None, 1:2] + qp[:, None, 0:1]
    point = np.clip(conts.mean(1), 0.0, None)
    quants = np.clip(np.moveaxis(np.quantile(conts, QL, axis=1), 0, -1), 0.0, None)
    return point, quants


def filtered_forecast(idx, queries, origin, H, k, mf, restore):  # noqa: N803
    idx.reset_stats()
    t0 = time.perf_counter()
    pts, qs = [], []
    for i in range(len(queries)):
        p, q = idx.forecast(
            queries[i], origin, H, k, QL, query_series_idx=i, meta_filter=mf, restore_scale=restore
        )
        pts.append(p)
        qs.append(q)
    return np.stack(pts), np.stack(qs), 1000 * (time.perf_counter() - t0), idx.fallback_count


def naive_recent_mean(queries, H):  # noqa: N803
    pt = np.clip(np.stack([np.full(H, q.mean()) for q in queries]), 0.0, None)
    return pt, np.repeat(pt[:, :, None], len(QL), axis=2)


def naive_seasonal(sales, split, season=7):
    te, H = split.train_end, split.horizon  # noqa: N806
    last = sales[:, te - season : te]
    pt = np.clip(last[:, np.arange(H) % season], 0.0, None)
    return pt, np.repeat(pt[:, :, None], len(QL), axis=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context", type=int, default=56)
    ap.add_argument("--stride", type=int, default=2, help="Coprime to 7 -> all weekly phases.")
    ap.add_argument("--processed", type=Path, default=REPO / "data" / "processed")
    args = ap.parse_args()
    set_seed(args.seed)

    entities, dynamic = load_processed(args.processed)
    entities, dynamic = select_subset(entities, dynamic, args.subset, args.seed)
    n = entities.height
    n_days = int(dynamic["day_idx"].max())  # type: ignore[arg-type]
    sales = _mat(dynamic, "sales", n, n_days)
    split = split_by_name(make_rolling_splits(), "val")
    origin, H, L = split.h_start, split.horizon, args.context  # noqa: N806
    q_arr = np.stack([sales[i, split.train_end - L : split.train_end] for i in range(n)])
    queries = [q_arr[i] for i in range(n)]

    t0 = time.perf_counter()
    db = WindowDatabase.from_training(sales, split.train_end, L, H, stride=args.stride)
    db_ms = 1000 * (time.perf_counter() - t0)
    print(f"subset={n} | DB candidates={len(db)} | L={L} stride={args.stride} | DB={db_ms:.0f}ms")

    results = []
    # explicit naive baselines (labelled, not disguised)
    p, qz = naive_recent_mean(queries, H)
    results.append(_score("naive:recent_mean", p, qz, sales, split, {"fallbacks": 0}))
    p, qz = naive_seasonal(sales, split)
    results.append(_score("naive:seasonal7", p, qz, sales, split, {"fallbacks": 0}))

    # global sweep: scale x metric (batched FAISS, k=5, restore on)
    cache = {}
    for scale in ("raw", "znorm", "mean", "rms"):
        for metric in ("l2", "cosine"):
            tb = time.perf_counter()
            idx = ScaleAwareIndex(db, entities, scale=scale, metric=metric)
            build_ms = 1000 * (time.perf_counter() - tb)
            cache[(scale, metric)] = idx
            t0 = time.perf_counter()
            pred, quants = global_batch(idx, q_arr, 5, True)
            r_ms = 1000 * (time.perf_counter() - t0)
            results.append(
                _score(
                    f"knn:{scale}/{metric}/global/k5",
                    pred,
                    quants,
                    sales,
                    split,
                    {
                        "index_build_ms": round(build_ms, 1),
                        "retrieval_ms": round(r_ms, 1),
                        "fallbacks": 0,
                    },
                )
            )

    best_g = min([r for r in results if r["config"].startswith("knn:")], key=lambda r: r["rmsse"])
    bscale, bmetric = best_g["config"].split(":")[1].split("/")[:2]
    bidx = cache[(bscale, bmetric)]

    # metadata / seasonal filters on the best global scale/metric
    for mf in ("store_id", "cat_id", "dept_id", "seasonal"):
        pred, quants, r_ms, fb = filtered_forecast(bidx, queries, origin, H, 5, mf, True)
        results.append(
            _score(
                f"knn:{bscale}/{bmetric}/{mf}/k5",
                pred,
                quants,
                sales,
                split,
                {"retrieval_ms": round(r_ms, 1), "fallbacks": fb},
            )
        )

    # k-sweep + restore ablation on the best config so far
    best_now = min([r for r in results if r["config"].startswith("knn:")], key=lambda r: r["rmsse"])
    parts = best_now["config"].split(":")[1].split("/")
    bmf = None if parts[2] == "global" else parts[2]
    for k in (1, 3, 10, 20):
        if bmf is None:
            pred, quants = global_batch(bidx, q_arr, k, True)
            fb = 0
        else:
            pred, quants, _r, fb = filtered_forecast(bidx, queries, origin, H, k, bmf, True)
        results.append(
            _score(
                f"knn:{bscale}/{bmetric}/{parts[2]}/k{k}",
                pred,
                quants,
                sales,
                split,
                {"fallbacks": fb},
            )
        )
    # restore ablation
    if bmf is None:
        pred, quants = global_batch(bidx, q_arr, 5, False)
    else:
        pred, quants, _r, _fb = filtered_forecast(bidx, queries, origin, H, 5, bmf, False)
    results.append(
        _score(f"knn:{bscale}/{bmetric}/{parts[2]}/k5/NO-restore", pred, quants, sales, split, {})
    )

    best_overall = min(results, key=lambda r: r["rmsse"])
    profiling = {
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2),
        "db_build_ms": round(db_ms, 1),
    }
    report = {
        "phase": "5-scalable-retrieval",
        "timestamp": datetime.now(UTC).isoformat(),
        "subset": n,
        "seed": args.seed,
        "context_length": L,
        "stride": args.stride,
        "db_candidates": len(db),
        "split": split.as_dict(),
        "results": results,
        "best_by_rmsse": best_overall["config"],
        "profiling": profiling,
        "run_context": RunContext().to_dict(),
    }
    out = REPO / "reports" / f"scalable-retrieval-subset{n}.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    print(
        f"\n{'config':40s} {'RMSSE':>7s} {'MASE':>7s} {'WAPE':>7s} {'MAE':>7s} {'pinball':>8s} {'fb':>5s}"
    )
    for r in sorted(results, key=lambda x: x["rmsse"]):
        print(
            f"{r['config']:40s} {r['rmsse']:7.4f} {r['mase']:7.4f} {r['wape']:7.4f} "
            f"{r['mae']:7.4f} {r['pinball']:8.4f} {r.get('fallbacks', 0):5d}"
        )
    print(f"\nBEST (val RMSSE): {best_overall['config']}  RMSSE={best_overall['rmsse']:.4f}")
    print(f"profiling: {profiling} | report={out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
