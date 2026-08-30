#!/usr/bin/env python3
"""Referee item 3: in-context conditioning as the competitor to Stage 4.

The referee's request, verbatim:

    "Pass the k retrieved neighbours to Chronos-2 as additional in-context series
     or covariates and compare that against your convex blend. This is the natural
     competitor to Stage 4, it costs no trainable parameters, and it tests whether
     the frozen backbone can use the analogues better than an external weighted
     average can. ... Either way it belongs in Table V."

They exempt Chronos-Bolt explicitly (*"That statement is true of Chronos and
Chronos-Bolt. It is not true of Chronos-2"*), so there is no ETTm2 arm to build.
This is M5 with the frozen Chronos-2, on **validation** origins, because the M5
test split is consumed (rule 2).

Route. ``Chronos2Pipeline.predict`` treats a 2-D element of shape
``(n_variates, history_length)`` as a multivariate group, sharing information
across variates through group attention. Row 0 is the target; rows 1..k are the
retrieved analogue contexts, **NaN-padded on the left to the target's own
history length** so the target's input is byte-identical to the target-only arm.
The only change between arms is the added variates. Verified before writing this
script: adding neighbours changes the target's forecast, and the change depends
on which neighbours are added, so the arm is not vacuous. The covariate route is
not used -- ``chronos2/preprocess.py`` raises on a covariate length mismatch, and
the referee's "or" permits the series route.

Controls. The corpus documents that undiscriminated context expansion degrades
accuracy (irrelevant neighbours falling below univariate baselines, attention
entropy inflation). A positive in-context result is therefore uninterpretable
without separating "the analogues are informative" from "multivariate mode is
just different". Every k is run twice: once with the retrieved neighbours and
once with the same number of **random** series drawn from the same legal
candidate pool.

k is swept over {1, 5, 20} for the same reason: 20 is the deployed retrieval
budget, and the failure mode is documented to depend on how much context is
added.

Nothing here selects a configuration. The frozen system is unchanged whatever
this returns (rules 9, 12).

Usage (fish):
    uv run python scripts/m5_incontext_baseline_run.py --subset 1000 --origins 5
    uv run python scripts/m5_incontext_baseline_run.py --subset 1000 --origins 5 --resume
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
from scalerag.fusion import paired_bootstrap_rel_improvement as pb  # noqa: E402
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

OUT = REPO / "reports" / "incontext-baseline"
CKPT = OUT / "ckpt"
KS: tuple[int, ...] = (1, 5, 20)
BLEND_W = 0.5  # the parameter-free convex blend of Table V


def _group(target_hist: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """Target row 0, neighbours left-NaN-padded to the target's history length."""
    t = target_hist.shape[0]
    pad = np.full((nb.shape[0], t), np.nan)
    pad[:, -nb.shape[1] :] = nb
    return np.vstack([target_hist[None, :], pad])


def _forecast_groups(fc, groups: list[np.ndarray], horizon: int, batch: int) -> np.ndarray:
    """Target-variate point forecast for each group (the pipeline's predictive mean)."""
    out = np.empty((len(groups), horizon))
    for s in range(0, len(groups), batch):
        chunk = groups[s : s + batch]
        _q, mean = fc.pipe.predict_quantiles(
            chunk, prediction_length=horizon, quantile_levels=[0.5]
        )
        for j, m in enumerate(mean):
            out[s + j] = np.asarray(m.float().cpu())[0]  # row 0 = the target variate
    return np.clip(out, 0.0, None)


def run_origin(sales: np.ndarray, entities, oi: int, fc, rng, batch: int) -> dict:
    n = sales.shape[0]
    hist = [sales[i, :oi] for i in range(n)]
    queries = [sales[i, oi - SE.L : oi] for i in range(n)]

    backbone = np.clip(fc.forecast(hist, SE.H, SE.QL)[0], 0.0, None)
    r_pt, _q, _nnd, _dis, arts = SE.retrieval_all(
        sales, entities, oi, queries, scale="mean", return_artifacts=True
    )
    db = SE.WindowDatabase.from_training(sales, oi, SE.L, SE.H, stride=7)
    legal = np.flatnonzero(db.legal_mask(oi + 1))

    rec: dict[str, np.ndarray] = {
        "origin": np.array([oi]),
        "zero_fraction": (sales[:, oi - SE.L : oi] == 0).mean(axis=1),
        "rmsse_backbone": SE.rmsse_series(backbone, sales, oi),
        "rmsse_retrieval": SE.rmsse_series(r_pt, sales, oi),
        "rmsse_blend": SE.rmsse_series((1 - BLEND_W) * backbone + BLEND_W * r_pt, sales, oi),
    }
    for k in KS:
        for kind in ("retrieved", "random"):
            groups = []
            for i in range(n):
                ids = arts[i]["ids"][:k]
                if kind == "random":
                    ids = rng.choice(legal, size=min(k, legal.size), replace=False)
                if ids.size == 0:  # no legal candidate: fall back to target-only
                    groups.append(hist[i][None, :])
                    continue
                groups.append(_group(hist[i], db.contexts[ids]))
            pred = _forecast_groups(fc, groups, SE.H, batch)
            rec[f"rmsse_incontext_{kind}_{k}"] = SE.rmsse_series(pred, sales, oi)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--origins", type=int, default=5)
    ap.add_argument("--stride", type=int, default=28)
    ap.add_argument("--batch", type=int, default=64, help="groups per Chronos-2 call")
    ap.add_argument("--resume", action="store_true")
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
    origins = [o_eval - i * args.stride for i in range(args.origins)]
    if max(o + SE.H for o in origins) > test_start:
        raise SystemExit("refusing: an evaluation window reaches the consumed test split")
    if min(origins) - SE.L < 0:
        raise SystemExit(f"origin {min(origins)} has no room for a length-{SE.L} context")

    CKPT.mkdir(parents=True, exist_ok=True)
    tag = f"s{args.subset}-seed{args.seed}"
    fc = None
    records: list[dict[str, np.ndarray]] = []
    bar = tqdm(origins, desc="item 3: in-context", unit="origin", dynamic_ncols=True)
    for oi in bar:
        path = CKPT / f"{tag}-origin{oi}.npz"
        if args.resume and path.exists():
            with np.load(path) as z:
                records.append({k: z[k] for k in z.files})
            bar.set_postfix_str(f"origin {oi}: cached")
            continue
        if fc is None:
            fc = Chronos2Forecaster()
        rec = run_origin(sales, entities, oi, fc, np.random.default_rng(args.seed + oi), args.batch)
        np.savez_compressed(path, **rec)
        records.append(rec)
        bar.set_postfix_str(f"origin {oi}: backbone {rec['rmsse_backbone'].mean():.4f}")
    bar.close()

    cat = {k: np.concatenate([r[k] for r in records]) for k in records[0] if k != "origin"}
    arms = [c for c in cat if c.startswith("rmsse_")]
    ok = np.ones(cat["rmsse_backbone"].shape, dtype=bool)
    for a in arms:
        ok &= np.isfinite(cat[a])

    rows = []
    for a in arms:
        row = {
            "arm": a.replace("rmsse_", ""),
            "rmsse": float(cat[a][ok].mean()),
            "vs_backbone": pb(cat["rmsse_backbone"][ok], cat[a][ok], n_boot=2000, seed=0),
            "vs_blend": pb(cat["rmsse_blend"][ok], cat[a][ok], n_boot=2000, seed=0),
        }
        rows.append(row)
    rows.sort(key=lambda r: r["rmsse"])

    payload = {
        "experiment": "m5-incontext-baseline",
        "referee_item": "3",
        "dataset": "M5",
        "split": "validation",
        "guard": "test split d_1914-d_1941 is consumed and untouched (rule 2)",
        "route": (
            "retrieved contexts passed as extra variates of a 2-D Chronos-2 input; "
            "neighbours left-NaN-padded so the target's own input is identical to "
            "the target-only arm"
        ),
        "control": "same k drawn at random from the same legal candidate pool",
        "n_origins": len(origins),
        "origins": origins,
        "n_scored": int(ok.sum()),
        "k_sweep": list(KS),
        "blend_weight": BLEND_W,
        "context_length_L": SE.L,
        "horizon_H": SE.H,
        "top_k_retrieval": SE.K,
        "arms": rows,
        "runtime_sec": round(time.time() - started, 1),
        "timestamp": datetime.now(UTC).isoformat(),
        "run_context": RunContext().to_dict(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"m5-incontext-{tag}-{len(origins)}origins.json"
    dest.write_text(json.dumps(payload, indent=2))

    print(
        f"\nwrote {dest.relative_to(REPO)}   ({payload['runtime_sec']}s, {int(ok.sum())} series)\n"
    )
    print(f"{'arm':>26} {'RMSSE':>8}  {'vs backbone':>22}  {'vs blend':>22}")
    print("-" * 84)
    for r in rows:
        b, v = r["vs_backbone"], r["vs_blend"]
        print(
            f"{r['arm']:>26} {r['rmsse']:>8.4f}  "
            f"{100 * b['rel_improvement']:>+7.2f}% [{100 * b['ci95_low']:>+6.2f},{100 * b['ci95_high']:>+6.2f}]  "
            f"{100 * v['rel_improvement']:>+7.2f}% [{100 * v['ci95_low']:>+6.2f},{100 * v['ci95_high']:>+6.2f}]"
        )
    print("\nThe referee's question: does any in-context arm beat 'blend'?")
    print("The control matters: if 'random' tracks 'retrieved', group attention is")
    print("responding to extra context, not to analogue quality.")


if __name__ == "__main__":
    main()
