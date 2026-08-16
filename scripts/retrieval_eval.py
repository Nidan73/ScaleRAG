#!/usr/bin/env python3
"""Chronos-2 + temporal-retrieval evaluation on a declared M5 subset (Phase 4).

Methods (task 8): random / seasonal / euclidean k-NN retrieval, target-only
Chronos-2, and Chronos-2 + retrieved context (late fusion). Also runs
Chronos-2 + random/seasonal context for the task-13 comparison.

Evaluates on the val split (d_1886-d_1913); test (d_1914-d_1941) is untouched.
Reports WRMSSE/MASE/RMSSE/WAPE/MAE/pinball + retrieval & inference latency, VRAM,
peak RAM, and checkpoint size. Declared subset first (task 5).

    uv run python scripts/retrieval_eval.py --subset 100 --seed 42
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

from scalerag import hierarchy, metrics  # noqa: E402
from scalerag.eval import load_processed, select_subset  # noqa: E402
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402
from scalerag.retrieval import (  # noqa: E402
    EuclideanRetriever,
    RandomRetriever,
    SeasonalRetriever,
    WindowDatabase,
)
from scalerag.retrieval_forecast import knn_forecast, late_fusion  # noqa: E402
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CONTEXT_L = 56  # retrieval query/candidate window (8 weeks)
TOPK = 5


def _matrix(dyn, col, n, t):
    return dyn[col].to_numpy().astype(np.float64).reshape(n, t)


def _score(name, pred, quants, entities, sales, split, weights, retr_ms, infer_ms):
    n = pred.shape[0]
    ha = sales[:, split.h_start - 1 : split.h_end]
    tr = sales[:, : split.train_end]
    rmsse_i = np.array([metrics.rmsse(ha[i], pred[i], tr[i]) for i in range(n)])
    mase_i = np.array([metrics.mase(ha[i], pred[i], tr[i]) for i in range(n)])
    pin_i = np.array([metrics.pinball_loss(ha[i], quants[i], QL) for i in range(n)])
    wrmsse, _ = hierarchy.wrmsse(entities, tr, ha, pred, weights)
    return {
        "method": name,
        "mae": metrics.mae(ha.ravel(), pred.ravel()),
        "wape": metrics.wape(ha.ravel(), pred.ravel()),
        "mase_mean": float(np.nanmean(mase_i)),
        "rmsse_mean": float(np.nanmean(rmsse_i)),
        "wrmsse": wrmsse,
        "pinball_mean": float(np.nanmean(pin_i)),
        "retrieval_ms": round(retr_ms, 1),
        "inference_ms": round(infer_ms, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=int, default=100, help="Declared small subset of series.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--processed", type=Path, default=REPO / "data" / "processed")
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--alpha", type=float, default=0.5, help="Fusion weight on Chronos-2.")
    args = ap.parse_args()
    set_seed(args.seed)

    entities, dynamic = load_processed(args.processed)
    entities, dynamic = select_subset(entities, dynamic, args.subset, args.seed)
    n = entities.height
    n_days = int(dynamic["day_idx"].max())  # type: ignore[arg-type]
    sales = _matrix(dynamic, "sales", n, n_days)
    price = _matrix(dynamic, "sell_price", n, n_days)

    split = split_by_name(make_rolling_splits(), "val")
    origin = split.h_start  # 1886
    H = split.horizon  # noqa: N806 — horizon, standard TS notation
    weights = hierarchy.dollar_weights(
        sales[:, split.train_end - 28 : split.train_end],
        price[:, split.train_end - 28 : split.train_end],
    )

    # --- leakage-safe window DB from TRAINING only ---
    db = WindowDatabase.from_training(sales, split.train_end, CONTEXT_L, H, stride=7)
    print(f"subset={n} series | DB candidates={len(db)} | context_L={CONTEXT_L} k={args.topk}")

    queries = [sales[i, split.train_end - CONTEXT_L : split.train_end] for i in range(n)]

    results = []

    # --- retrieval-only k-NN methods ---
    retrievers = {
        "random": RandomRetriever(seed=args.seed),
        "seasonal": SeasonalRetriever(season=7),
        "euclidean": EuclideanRetriever(),
    }
    knn_out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, r in retrievers.items():
        t0 = time.perf_counter()
        pts, qs = [], []
        for i in range(n):
            p, q = knn_forecast(db, r, queries[i], origin, H, args.topk, i, QL)
            pts.append(p)
            qs.append(q)
        retr_ms = 1000 * (time.perf_counter() - t0)
        pred, quants = np.stack(pts), np.stack(qs)
        knn_out[name] = (pred, quants)
        results.append(
            _score(f"{name}_knn", pred, quants, entities, sales, split, weights, retr_ms, 0.0)
        )

    # --- frozen Chronos-2 target-only ---
    from scalerag.tsfm.chronos2 import Chronos2Forecaster

    fc = Chronos2Forecaster()
    contexts = [sales[i, : split.train_end] for i in range(n)]
    t0 = time.perf_counter()
    c_point, c_quants = fc.forecast(contexts, H, QL)
    infer_ms = 1000 * (time.perf_counter() - t0)
    c_point = np.clip(c_point, 0.0, None)
    results.append(
        _score("chronos2_target", c_point, c_quants, entities, sales, split, weights, 0.0, infer_ms)
    )
    vram = fc.cuda_peak_gib()

    # --- Chronos-2 + retrieved context (late fusion); euclidean is primary ---
    for name in ("random", "seasonal", "euclidean"):
        r_point, r_quants = knn_out[name]
        f_point, f_quants = late_fusion(c_point, c_quants, r_point, r_quants, alpha=args.alpha)
        results.append(
            _score(
                f"chronos2_{name}",
                f_point,
                f_quants,
                entities,
                sales,
                split,
                weights,
                0.0,
                infer_ms,
            )
        )

    peak_rss = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)
    ckpt_mb = round(
        sum(f.stat().st_size for f in (REPO / ".hf_cache").rglob("*.safetensors")) / 1e6, 1
    )

    report = {
        "phase": "4-chronos2-retrieval",
        "timestamp": datetime.now(UTC).isoformat(),
        "subset": n,
        "seed": args.seed,
        "split": split.as_dict(),
        "context_length": CONTEXT_L,
        "top_k": args.topk,
        "fusion_alpha": args.alpha,
        "db_candidates": len(db),
        "results": results,
        "profiling": {
            "gpu_vram_gib": round(vram, 3),
            "peak_rss_gib": peak_rss,
            "checkpoint_mb": ckpt_mb,
            "chronos_inference_ms": round(infer_ms, 1),
        },
        "run_context": RunContext().to_dict(),
    }
    out = REPO / "reports" / f"retrieval-eval-subset{n}.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    print(
        f"\n{'method':22s} {'WRMSSE':>8s} {'RMSSE':>7s} {'MASE':>7s} {'WAPE':>7s} {'MAE':>7s} {'pinball':>8s} {'retr_ms':>8s}"
    )
    for r in sorted(results, key=lambda x: x["wrmsse"]):
        print(
            f"{r['method']:22s} {r['wrmsse']:8.4f} {r['rmsse_mean']:7.4f} {r['mase_mean']:7.4f} "
            f"{r['wape']:7.4f} {r['mae']:7.4f} {r['pinball_mean']:8.4f} {r['retrieval_ms']:8.1f}"
        )
    print(
        f"\nVRAM={vram:.3f} GiB | peak RAM={peak_rss} GiB | ckpt={ckpt_mb} MB | "
        f"chronos inference={infer_ms:.0f}ms | report={out.relative_to(REPO)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
