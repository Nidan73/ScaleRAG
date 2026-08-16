#!/usr/bin/env python3
"""ScaleRAG-TS decisive core comparison (Phase 9, increment 1).

The gating question: does scale-aware retrieval + a learned gated fusion beat
target-only Chronos-2 (and recent-mean / raw retrieval / frozen Phase 5 retrieval)
by the pre-registered margin? Paired-bootstrap 95% CIs over series; gate trained on
HISTORICAL origins only; eval on val (d_1886-1913); test untouched. 3 gate seeds.

    uv run python scripts/scalerag_eval.py --subset 1000 --seed 42
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scalerag import fusion, metrics  # noqa: E402
from scalerag.eval import load_processed, select_subset  # noqa: E402
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402
from scalerag.retrieval import WindowDatabase  # noqa: E402
from scalerag.retrieval_faiss import (  # noqa: E402
    ScaleAwareIndex,
    restore_continuation,
)
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
L, H, K = 56, 28, 20


def retrieval_all(sales, entities, o, queries, scale="mean", mf="cat_id"):
    """Per-series scale-restored retrieval forecast at origin o (train_end=o).
    Returns point[n,H], quants[n,H,Q], nn_dist[n], disagreement[n]."""
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
            base = float(queries[i].mean())
            pt[i] = base
            qt[i] = base
            nnd[i] = 1e6
            dis[i] = 0.0
            continue
        conts = idx.db.continuations[ids]
        if scale != "raw":
            conts = np.stack(
                [restore_continuation(conts[j], idx.params[ids[j]], qp) for j in range(len(ids))]
            )
        conts = np.clip(conts, 0.0, None)
        pt[i] = conts.mean(0)
        qt[i] = np.quantile(conts, QL, axis=0).T
        nnd[i] = float(dists.min()) if dists.size else 1e6
        dis[i] = float(conts.std(0).mean())
    return pt, qt, nnd, dis


def rmsse_series(pred, sales, o_eval):
    ha, tr = sales[:, o_eval : o_eval + H], sales[:, :o_eval]
    return np.array([metrics.rmsse(ha[i], pred[i], tr[i]) for i in range(pred.shape[0])])


def pinball_series(quants, sales, o_eval):
    ha = sales[:, o_eval : o_eval + H]
    return np.array([metrics.pinball_loss(ha[i], quants[i], QL) for i in range(quants.shape[0])])


def coverage80(quants, sales, o_eval):
    ha = sales[:, o_eval : o_eval + H]
    lo, hi = quants[:, :, QL.index(0.1)], quants[:, :, QL.index(0.9)]
    return float(np.mean((ha >= lo) & (ha <= hi)))


def gate_features(sales, o, nnd, dis, chr_q):
    ctx = sales[:, o - L : o]
    mean = ctx.mean(1)
    std = ctx.std(1)
    unc = (chr_q[:, :, QL.index(0.9)] - chr_q[:, :, QL.index(0.1)]).mean(1)
    return np.stack(
        [
            nnd,
            dis,
            (ctx == 0).mean(1),
            np.log(mean + 1.0),
            unc,
            std / (mean + 1e-6),
        ],
        axis=1,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate-seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    set_seed(args.seed)
    import lightgbm as lgb

    from scalerag.tsfm.chronos2 import Chronos2Forecaster

    entities, dynamic = load_processed(REPO / "data" / "processed")
    entities, dynamic = select_subset(entities, dynamic, args.subset, args.seed)
    n = entities.height
    n_days = int(dynamic["day_idx"].max())
    sales = dynamic["sales"].to_numpy().astype(np.float64).reshape(n, n_days)
    splits = make_rolling_splits(last_labeled_day=n_days)
    o_eval = split_by_name(splits, "val").train_end
    gate_origins = [
        split_by_name(splits, "val_m2").train_end,
        split_by_name(splits, "val_m1").train_end,
    ]

    fc = Chronos2Forecaster()

    def chronos_at(o):
        t0 = time.perf_counter()
        pt, q = fc.forecast([sales[i, :o] for i in range(n)], H, QL)
        return np.clip(pt, 0.0, None), q, 1000 * (time.perf_counter() - t0)

    # --- gate training data from historical origins ---
    Xg, yg = [], []
    for o in gate_origins:
        c_pt, c_q, _ = chronos_at(o)
        r_pt, _rq, nnd, dis = retrieval_all(
            sales, entities, o, [sales[i, o - L : o] for i in range(n)]
        )
        actual = sales[:, o : o + H]
        loss_c = np.sqrt(((actual - c_pt) ** 2).mean(1))
        loss_r = np.sqrt(((actual - r_pt) ** 2).mean(1))
        Xg.append(gate_features(sales, o, nnd, dis, c_q))
        yg.append((loss_r < loss_c).astype(int))
    Xg = np.vstack(Xg)
    yg = np.concatenate(yg)

    # --- eval at val ---
    queries = [sales[i, o_eval - L : o_eval] for i in range(n)]
    c_pt, c_q, chr_ms = chronos_at(o_eval)
    r_pt, r_q, nnd, dis = retrieval_all(sales, entities, o_eval, queries, scale="mean")
    raw_pt, raw_q, _, _ = retrieval_all(sales, entities, o_eval, queries, scale="raw")
    rm_pt = np.clip(
        np.stack([np.full(H, sales[i, o_eval - L : o_eval].mean()) for i in range(n)]), 0, None
    )
    rm_q = np.repeat(rm_pt[:, :, None], len(QL), axis=2)
    Xe = gate_features(sales, o_eval, nnd, dis, c_q)

    methods = {
        "recent_mean": (rm_pt, rm_q),
        "chronos2_target": (c_pt, c_q),
        "retrieval_raw": (raw_pt, raw_q),
        "retrieval_scaleaware(Phase5)": (r_pt, r_q),
    }
    # proposed: gated fusion, averaged over gate seeds
    fused_pt_seeds, fused_q_seeds = [], []
    for gs in args.gate_seeds:
        gate = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=50,
            verbose=-1,
            random_state=gs,
        )
        gate.fit(Xg, yg)
        alpha = gate.predict_proba(Xe)[:, 1]  # P(retrieval better) -> retrieval weight
        fp, fq = fusion.fuse(c_pt, c_q, r_pt, r_q, alpha)
        fused_pt_seeds.append(fp)
        fused_q_seeds.append(fq)
    methods["ScaleRAG(gated_fusion)"] = (np.mean(fused_pt_seeds, 0), np.mean(fused_q_seeds, 0))

    # scores + per-series RMSSE for bootstrap
    rmsse_i = {m: rmsse_series(p, sales, o_eval) for m, (p, _q) in methods.items()}
    rows = {}
    for m, (_p, q) in methods.items():
        ri = rmsse_i[m]
        rows[m] = {
            "rmsse": float(np.nanmean(ri)),
            "pinball": float(np.nanmean(pinball_series(q, sales, o_eval))),
            "coverage80": coverage80(q, sales, o_eval),
        }

    # pre-registered criterion 1: proposed vs strongest matched baseline (chronos target)
    base = "chronos2_target"
    boot_vs_chronos = fusion.paired_bootstrap_rel_improvement(
        rmsse_i[base], rmsse_i["ScaleRAG(gated_fusion)"]
    )
    # also vs the strongest baseline overall by RMSSE
    strongest = min(
        [m for m in methods if m != "ScaleRAG(gated_fusion)"], key=lambda m: rows[m]["rmsse"]
    )
    boot_vs_best = fusion.paired_bootstrap_rel_improvement(
        rmsse_i[strongest], rmsse_i["ScaleRAG(gated_fusion)"]
    )

    crit1 = boot_vs_best["rel_improvement"] >= 0.03 and boot_vs_best["ci95_low"] > 0

    # pre-registered criterion 3: >=7% on predefined sparse/cold-start slices
    tr_e = sales[:, :o_eval]
    zf, vol, nz = (tr_e == 0).mean(1), tr_e.mean(1), (tr_e != 0).sum(1)
    slices = {
        "intermittent(z>0.8)": zf > 0.8,
        "low_volume(<median)": vol < np.median(vol),
        "reduced_history(<100nz)": nz < 100,
    }
    slice_boot, crit3 = {}, False
    prop = "ScaleRAG(gated_fusion)"
    for sn, sm in slices.items():
        if sm.sum() < 20:
            continue
        si = np.flatnonzero(sm)
        vs_best = fusion.paired_bootstrap_rel_improvement(rmsse_i[strongest][si], rmsse_i[prop][si])
        vs_chr = fusion.paired_bootstrap_rel_improvement(rmsse_i[base][si], rmsse_i[prop][si])
        slice_boot[sn] = {"n": int(sm.sum()), "vs_strongest": vs_best, "vs_chronos": vs_chr}
        if vs_best["rel_improvement"] >= 0.07 and vs_best["ci95_low"] > 0:
            crit3 = True

    report = {
        "phase": "9-scalerag-core",
        "timestamp": datetime.now(UTC).isoformat(),
        "subset": n,
        "seed": args.seed,
        "gate_seeds": args.gate_seeds,
        "eval_origin": o_eval,
        "gate_origins": gate_origins,
        "methods": rows,
        "strongest_baseline": strongest,
        "bootstrap_vs_chronos_target": boot_vs_chronos,
        "bootstrap_vs_strongest_baseline": boot_vs_best,
        "criterion1_3pct_rmsse_ci_excl_zero": bool(crit1),
        "slice_bootstrap": slice_boot,
        "criterion3_7pct_slice_ci_excl_zero": bool(crit3),
        "chronos_inference_ms": round(chr_ms),
        "gpu_vram_gib": round(fc.cuda_peak_gib(), 3),
        "run_context": RunContext().to_dict(),
    }
    (REPO / "reports" / f"scalerag-m5-subset{n}.json").write_text(
        json.dumps(report, indent=2, default=str)
    )

    print(f"\n{'method':32s} {'RMSSE':>7s} {'pinball':>8s} {'cov80':>6s}")
    for m in sorted(rows, key=lambda m: rows[m]["rmsse"]):
        print(
            f"{m:32s} {rows[m]['rmsse']:7.4f} {rows[m]['pinball']:8.4f} {rows[m]['coverage80']:6.3f}"
        )
    print(f"\nstrongest baseline: {strongest} (RMSSE {rows[strongest]['rmsse']:.4f})")
    print(
        f"ScaleRAG vs chronos_target: rel_impr={boot_vs_chronos['rel_improvement']:+.3%} "
        f"CI95=[{boot_vs_chronos['ci95_low']:+.3%},{boot_vs_chronos['ci95_high']:+.3%}]"
    )
    print(
        f"ScaleRAG vs strongest({strongest}): rel_impr={boot_vs_best['rel_improvement']:+.3%} "
        f"CI95=[{boot_vs_best['ci95_low']:+.3%},{boot_vs_best['ci95_high']:+.3%}]"
    )
    print("\nslices (ScaleRAG rel-impr vs strongest / vs chronos):")
    for sn, d in slice_boot.items():
        b, c = d["vs_strongest"], d["vs_chronos"]
        print(
            f"  {sn:26s} n={d['n']:4d}  vs_best={b['rel_improvement']:+.2%} "
            f"[{b['ci95_low']:+.2%},{b['ci95_high']:+.2%}]  vs_chr={c['rel_improvement']:+.2%}"
        )
    print(f"\nCRITERION 1 (>=3% over strongest, CI excl 0): {crit1}")
    print(f"CRITERION 3 (>=7% on a slice over strongest, CI excl 0): {crit3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
