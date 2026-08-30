#!/usr/bin/env python3
"""Referee item 6(b): re-run the sparse M5 bins under the RMS divisor and Eq. (3).

The referee asked for exactly this and it was never done:

    "Re-run the sparsest two bins with the RMS divisor and with the full rule (3),
     on validation origins only. You already have both implemented."

Unit 2 (``scale_identifiability_run.py``) answered the *other* two diagnostics by
holding the retrieved set fixed and swapping the restoration scale. That cannot
answer this one: ``rms`` and ``znorm`` change ``_transform``, so they change the
index and therefore which neighbours come back. This script re-runs retrieval in
full under each rule.

It matters more than it did when the referee wrote it. On ETTm2 validation the
full rule Eq. (3) beats the deployed Eq. (4) by 5.8% fused and turns a loss to
the backbone into a gain (``docs/ettm2-scale-operator-selection.md``), and the
deployed rule degenerates on near-zero contexts -- which is precisely the sparse
M5 bin. So this run forks the paper's headline:

  * if Eq. (3) rescues the sparse end, the operator rather than the regime
    explains both arms of the inverted U, and C1 becomes a normalisation result;
  * if it does not, C1 has survived the strongest challenge available to it.

Either way it is reported, never selected from. The frozen configuration stays
Eq. (4) at k=20 regardless of what comes back (rules 9, 12).

**Validation origins only.** The consumed M5 test split is refused.

Usage (fish):
    uv run python scripts/m5_sparse_scale_rule_run.py --subset 1000 --origins 50
    uv run python scripts/m5_sparse_scale_rule_run.py --subset 1000 --origins 50 --resume
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

import scalerag_eval as SE  # noqa: E402, N812  (sibling holds the frozen constants)
from scalerag import scale_operators as so  # noqa: E402
from scalerag.eval import load_processed, select_subset  # noqa: E402
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402
from scalerag.sparse_regime import FIXED_BIN_EDGES, bin_index  # noqa: E402
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

OUT = REPO / "reports" / "sparse-scale-rule"
CKPT = OUT / "ckpt"
LOCK = REPO / "M5_TEST_CONSUMED.lock"

# Eq. (4) is the deployed rule; `rms` is the divisor-only variant the referee named;
# `znorm` is the full rule Eq. (3). All three already exist in `_fit_params`.
RULES: tuple[str, ...] = ("mean", "rms", "znorm")


def run_origin(sales: np.ndarray, entities, oi: int, fc) -> dict[str, np.ndarray]:
    """One origin: the backbone once, then a full retrieval under each scale rule."""
    n = sales.shape[0]
    queries = [sales[i, oi - SE.L : oi] for i in range(n)]

    c_pt = np.clip(fc.forecast([sales[i, :oi] for i in range(n)], SE.H, SE.QL)[0], 0.0, None)
    ctx = sales[:, oi - SE.L : oi]
    out: dict[str, np.ndarray] = {
        "origin": np.array([oi]),
        "zero_fraction": (ctx == 0).mean(axis=1),
        "degenerate": so.degenerate_mask(ctx),
        "never_launched": sales[:, :oi].sum(axis=1) == 0,
        "rmsse_backbone": SE.rmsse_series(c_pt, sales, oi),
    }
    for rule in RULES:
        # Full re-retrieval: the scale rule changes the index, not just restoration.
        r_pt = SE.retrieval_all(sales, entities, oi, queries, scale=rule)[0]
        out[f"rmsse_{rule}"] = SE.rmsse_series(r_pt, sales, oi)
        out[f"rmsse_fused_{rule}"] = SE.rmsse_series(0.5 * c_pt + 0.5 * r_pt, sales, oi)
    return out


def _finite(*arrays: np.ndarray) -> np.ndarray:
    m = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def summarise(rec: dict[str, np.ndarray]) -> list[dict]:
    """Per fixed zero-fraction bin, each rule's win rate and effect size.

    Every rule is scored on ONE common population -- the series where the backbone
    and all three rules produce a finite RMSSE -- so the columns are comparable.
    Scoring each rule on its own survivors is how a weaker rule can look stronger.
    """
    bb = rec["rmsse_backbone"]
    ok = _finite(bb, *(rec[f"rmsse_{r}"] for r in RULES)) & ~rec["never_launched"]
    idx = bin_index(rec["zero_fraction"])
    rows: list[dict] = []
    for b in range(FIXED_BIN_EDGES.size - 1):
        m = ok & (idx == b)
        if not m.any():
            continue
        row: dict = {
            "lo": float(FIXED_BIN_EDGES[b]),
            "hi": float(FIXED_BIN_EDGES[b + 1]),
            "n": int(m.sum()),
            "degenerate_share": float(rec["degenerate"][m].mean()),
        }
        for rule in RULES:
            u = bb[m] - rec[f"rmsse_{rule}"][m]  # positive = retrieval better
            f = bb[m] - rec[f"rmsse_fused_{rule}"][m]
            row[rule] = {
                "win_rate": float((u > 0).mean()),
                "mean_delta": float(u.mean()),
                "median_delta": float(np.median(u)),
                "fused_win_rate": float((f > 0).mean()),
                "fused_mean_delta": float(f.mean()),
            }
        rows.append(row)
    return rows


def pooled(records: list[dict[str, np.ndarray]]) -> list[dict]:
    cat = {k: np.concatenate([r[k] for r in records]) for k in records[0] if k != "origin"}
    return summarise(cat)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--origins", type=int, default=50)
    ap.add_argument("--stride", type=int, default=28, help="days between origins; default is H")
    ap.add_argument("--resume", action="store_true", help="reuse finished origins")
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
        raise SystemExit(f"refusing: eval origin {o_eval} reaches the consumed test split")
    origins = [o_eval - i * args.stride for i in range(args.origins)]
    if min(origins) - SE.L < 0:
        raise SystemExit(f"origin {min(origins)} has no room for a length-{SE.L} context")
    if max(o + SE.H for o in origins) > test_start:
        raise SystemExit("refusing: an evaluation window reaches the consumed test split")

    CKPT.mkdir(parents=True, exist_ok=True)
    tag = f"s{args.subset}-seed{args.seed}"
    fc = None
    records: list[dict[str, np.ndarray]] = []
    bar = tqdm(origins, desc="item 6(b): scale rules", unit="origin", dynamic_ncols=True)
    for oi in bar:
        path = CKPT / f"{tag}-origin{oi}.npz"
        if args.resume and path.exists():
            with np.load(path) as z:
                records.append({k: z[k] for k in z.files})
            bar.set_postfix_str(f"origin {oi}: cached")
            continue
        if fc is None:  # lazily loaded so a fully cached resume needs no GPU
            fc = Chronos2Forecaster()
        rec = run_origin(sales, entities, oi, fc)
        np.savez_compressed(path, **rec)
        records.append(rec)
        u = rec["rmsse_backbone"] - rec["rmsse_mean"]
        k = np.isfinite(u)
        bar.set_postfix_str(f"origin {oi}: Eq.(4) win {float((u[k] > 0).mean()):.3f}")
    bar.close()

    rows = pooled(records)
    payload = {
        "experiment": "m5-sparse-scale-rule",
        "referee_item": "6(b)",
        "dataset": "M5",
        "split": "validation",
        "guard": "test split d_1914-d_1941 is consumed and untouched (rule 2)",
        "note": (
            "Full re-retrieval under each scale rule -- rms and znorm change the "
            "index, so the retrieved set is not shared. Reported, never selected "
            "from; the frozen configuration stays Eq. (4) at k=20 (rules 9, 12)."
        ),
        "rules": {"mean": "Eq. (4), deployed", "rms": "divisor only", "znorm": "Eq. (3), full"},
        "n_origins": len(origins),
        "origins": origins,
        "stride": args.stride,
        "subset_draw": {
            "n_requested": args.subset,
            "n_selected": int(n),
            "seed": args.seed,
            "method": "scalerag.eval.select_subset",
        },
        "context_length_L": SE.L,
        "horizon_H": SE.H,
        "top_k": SE.K,
        "population": "one common population: finite RMSSE under all rules, never-launched excluded",
        "bins": rows,
        "runtime_sec": round(time.time() - started, 1),
        "timestamp": datetime.now(UTC).isoformat(),
        "run_context": RunContext().to_dict(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"m5-sparse-scale-rule-{tag}-{len(origins)}origins.json"
    dest.write_text(json.dumps(payload, indent=2))

    print(f"\nwrote {dest.relative_to(REPO)}   ({payload['runtime_sec']}s)\n")
    hdr = f"{'bin':>12} {'n':>6} {'deg':>6} " + " ".join(f"{r:>18}" for r in RULES)
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        cells = " ".join(f"{row[r]['win_rate']:>8.3f}/{row[r]['mean_delta']:>+9.4f}" for r in RULES)
        print(
            f"[{row['lo']:.1f},{row['hi']:.1f}] {row['n']:>6} "
            f"{row['degenerate_share']:>6.3f} {cells}"
        )
    print("\n(win rate / mean delta-RMSSE; positive delta = retrieval beats the backbone)")
    print("The referee's question is the last two rows: does Eq. (3) rescue the sparse end?")


if __name__ == "__main__":
    main()
