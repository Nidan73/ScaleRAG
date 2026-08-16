#!/usr/bin/env python3
"""ScaleRAG-TS headline confirmation on Favorita (Phase 9, increment 2).

Confirms the M5 verdict cross-dataset and supplies criterion-2's Favorita side:
chronos-2 target-only vs scale-aware retrieval vs learned gated fusion vs
recent-mean, on the validated 5,000-series Favorita subset. Paired bootstrap CIs;
gate trained on historical origins; final window untouched.

    uv run python scripts/scalerag_favorita.py --gate-seeds 42 43 44
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scalerag import fusion, metrics  # noqa: E402
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402
from scalerag.retrieval import WindowDatabase  # noqa: E402
from scalerag.retrieval_faiss import ScaleAwareIndex, restore_continuation  # noqa: E402
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

QL = [0.1, 0.3, 0.5, 0.7, 0.9]
L, H, K = 56, 28, 20
PROC = REPO / "data" / "processed" / "favorita"


def retrieval(sales, entities, o, queries, scale="mean", mf="cat_id"):
    db = WindowDatabase.from_training(sales, o, L, H, stride=7)
    idx = ScaleAwareIndex(db, entities, scale=scale, metric="l2")
    n = len(queries)
    pt = np.zeros((n, H))
    qt = np.zeros((n, H, len(QL)))
    nnd = np.zeros(n)
    dis = np.zeros(n)
    for i in range(n):
        ids, dists, qp = idx.search(queries[i], o + 1, K, query_series_idx=i, meta_filter=mf)
        if ids.size == 0:
            b = float(queries[i].mean())
            pt[i], qt[i], nnd[i], dis[i] = b, b, 1e6, 0.0
            continue
        conts = np.stack(
            [
                restore_continuation(idx.db.continuations[ids][j], idx.params[ids[j]], qp)
                for j in range(len(ids))
            ]
        )
        conts = np.clip(conts, 0.0, None)
        pt[i] = conts.mean(0)
        qt[i] = np.quantile(conts, QL, axis=0).T
        nnd[i] = float(dists.min())
        dis[i] = float(conts.std(0).mean())
    return pt, qt, nnd, dis


def gate_feats(sales, o, nnd, dis, chr_q):
    ctx = sales[:, o - L : o]
    mean, std = ctx.mean(1), ctx.std(1)
    unc = (chr_q[:, :, QL.index(0.9)] - chr_q[:, :, QL.index(0.1)]).mean(1)
    return np.stack([nnd, dis, (ctx == 0).mean(1), np.log(mean + 1), unc, std / (mean + 1e-6)], 1)


def rmsse_i(pred, sales, o):
    ha, tr = sales[:, o : o + H], sales[:, :o]
    return np.array([metrics.rmsse(ha[i], pred[i], tr[i]) for i in range(pred.shape[0])])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    set_seed(42)
    import lightgbm as lgb

    from scalerag.tsfm.chronos2 import Chronos2Forecaster

    ent = pl.read_parquet(PROC / "entities.parquet").sort("id")
    dyn = pl.read_parquet(PROC / "dynamic.parquet").sort(["id", "day_idx"])
    n = ent.height
    n_days = int(dyn["day_idx"].max())
    sales = dyn["sales"].to_numpy().astype(np.float64).reshape(n, n_days)
    # compatibility view for the M5-shaped retrieval index (family->cat, class->dept)
    ent_c = ent.select(
        "id",
        "item_id",
        pl.col("family_id").alias("cat_id"),
        pl.col("class_id").alias("dept_id"),
        "store_id",
        "state_id",
    )
    splits = make_rolling_splits(last_labeled_day=n_days)
    o = split_by_name(splits, "val").train_end
    gorigins = [
        split_by_name(splits, "val_m2").train_end,
        split_by_name(splits, "val_m1").train_end,
    ]

    fc = Chronos2Forecaster()

    def chronos(origin):
        pt, q = fc.forecast([sales[i, :origin] for i in range(n)], H, QL)
        return np.clip(pt, 0.0, None), q

    Xg, yg = [], []
    for go in gorigins:
        cg, cgq = chronos(go)
        rg, _, ng, dg = retrieval(sales, ent_c, go, [sales[i, go - L : go] for i in range(n)])
        act = sales[:, go : go + H]
        Xg.append(gate_feats(sales, go, ng, dg, cgq))
        yg.append(
            (np.sqrt(((act - rg) ** 2).mean(1)) < np.sqrt(((act - cg) ** 2).mean(1))).astype(int)
        )
    Xg = np.vstack(Xg)
    yg = np.concatenate(yg)

    c_pt, c_q = chronos(o)
    r_pt, r_q, nnd, dis = retrieval(sales, ent_c, o, [sales[i, o - L : o] for i in range(n)])
    Xe = gate_feats(sales, o, nnd, dis, c_q)
    alphas = np.mean(
        [
            lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=15,
                learning_rate=0.05,
                min_child_samples=50,
                verbose=-1,
                random_state=s,
            )
            .fit(Xg, yg)
            .predict_proba(Xe)[:, 1]
            for s in args.gate_seeds
        ],
        0,
    )
    f_pt, _fq = fusion.fuse(c_pt, c_q, r_pt, r_q, alphas)
    rm_pt = np.clip(np.stack([np.full(H, sales[i, o - L : o].mean()) for i in range(n)]), 0, None)

    ri = {
        "recent_mean": rmsse_i(rm_pt, sales, o),
        "chronos2_target": rmsse_i(c_pt, sales, o),
        "retrieval_scaleaware": rmsse_i(r_pt, sales, o),
        "ScaleRAG_gated": rmsse_i(f_pt, sales, o),
    }
    rows = {m: float(np.nanmean(v)) for m, v in ri.items()}
    strongest = min([m for m in rows if m != "ScaleRAG_gated"], key=lambda m: rows[m])
    boot = {
        "vs_chronos": fusion.paired_bootstrap_rel_improvement(
            ri["chronos2_target"], ri["ScaleRAG_gated"]
        ),
        "vs_strongest": fusion.paired_bootstrap_rel_improvement(
            ri[strongest], ri["ScaleRAG_gated"]
        ),
    }
    report = {
        "phase": "9-scalerag-favorita",
        "dataset": "Favorita",
        "timestamp": datetime.now(UTC).isoformat(),
        "n_series": n,
        "eval_origin": o,
        "gate_seeds": args.gate_seeds,
        "rmsse": rows,
        "strongest_baseline": strongest,
        "bootstrap": boot,
        "gpu_vram_gib": round(fc.cuda_peak_gib(), 3),
        "run_context": RunContext().to_dict(),
    }
    (REPO / "reports" / "scalerag-favorita.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    for m in sorted(rows, key=lambda m: rows[m]):
        print(f"{m:24s} RMSSE={rows[m]:.4f}")
    print(
        f"\nScaleRAG vs chronos: {boot['vs_chronos']['rel_improvement']:+.2%} "
        f"[{boot['vs_chronos']['ci95_low']:+.2%},{boot['vs_chronos']['ci95_high']:+.2%}]"
    )
    print(
        f"ScaleRAG vs strongest({strongest}): {boot['vs_strongest']['rel_improvement']:+.2%} "
        f"[{boot['vs_strongest']['ci95_low']:+.2%},{boot['vs_strongest']['ci95_high']:+.2%}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
