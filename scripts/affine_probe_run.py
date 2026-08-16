#!/usr/bin/env python3
"""Controlled synthetic affine probe: invariance and equivariance, measured.

Answers the "this is just denormalisation" critique with a causal experiment
rather than an argument. A query is constructed as an exact affine image
``x' = a*x + b`` of a known donor motif, so both the correct analogue and the
correct continuation are known in closed form and each stage can be scored on its
own:

* **retrieval** is scored by top-1 hit rate against the known donor motif
  (invariance);
* **reconstruction** is scored by scale-free error against the known continuation
  ``a*y + b`` (equivariance).

Crossing retrieval space with reconstruction rule separates the two. Reported
across several seeds with a paired bootstrap CI on each contrast (rules 6, 8, 10).
This is synthetic data only: it touches no M5 or Favorita split, opens none of the
blocked Phase-11B datasets, and makes no claim about end-to-end accuracy.

    uv run python scripts/affine_probe_run.py
    uv run python scripts/affine_probe_run.py --seeds 10 --n-queries 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scalerag.affine_probe import AffineCondition, run_affine_grid  # noqa: E402
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402

OUT = REPO / "reports" / "affine-probe"

CONDITIONS = [
    AffineCondition("raw", False),
    AffineCondition("znorm", False),
    AffineCondition("znorm", True),
    AffineCondition("mean", True),
    AffineCondition("rms", True),
]
A_VALUES = [1.0, 2.0, 10.0, 100.0]
B_VALUES = [0.0, 50.0, 200.0]
N_BOOT = 2000


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Percentile CI for ``mean(a - b)`` over paired per-seed observations.

    Pairing is by seed: both conditions see the identical panel and queries, so the
    difference isolates the condition rather than the draw.
    """
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must align, got {a.shape} and {b.shape}")
    rng = np.random.default_rng(seed)
    diff = a - b
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5, help="number of independent panels")
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--n-motifs", type=int, default=12)
    ap.add_argument("--rows-per-motif", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=20260726)
    args = ap.parse_args()

    if args.seeds < 2:
        raise SystemExit("need >= 2 seeds: a single run is not a result (rule 6)")

    set_seed(args.base_seed)
    started = time.time()
    seeds = [args.base_seed + i for i in range(args.seeds)]

    # (condition, a, b) -> per-seed metric vectors
    hit: dict[tuple[str, float, float], list[float]] = {}
    nmse: dict[tuple[str, float, float], list[float]] = {}

    for seed in seeds:
        grid = run_affine_grid(
            CONDITIONS,
            A_VALUES,
            B_VALUES,
            n_motifs=args.n_motifs,
            rows_per_motif=args.rows_per_motif,
            n_queries=args.n_queries,
            noise=args.noise,
            k=args.k,
            seed=seed,
        )
        for cell in grid.cells:
            key = (cell.condition, cell.a, cell.b)
            hit.setdefault(key, []).append(cell.hit_rate)
            nmse.setdefault(key, []).append(cell.nmse)

    cells = [
        {
            "condition": cond,
            "a": a,
            "b": b,
            "hit_rate_mean": float(np.mean(hit[(cond, a, b)])),
            "hit_rate_sd": float(np.std(hit[(cond, a, b)], ddof=1)),
            "nmse_mean": float(np.mean(nmse[(cond, a, b)])),
            "nmse_sd": float(np.std(nmse[(cond, a, b)], ddof=1)),
        }
        for (cond, a, b) in sorted(hit, key=lambda t: (t[0], t[1], t[2]))
    ]

    # Contrasts, each stated as a directional prediction the probe can falsify.
    transformed = [(a, b) for a in A_VALUES for b in B_VALUES if (a, b) != (1.0, 0.0)]

    def pooled(metric: dict[tuple[str, float, float], list[float]], cond: str) -> np.ndarray:
        return np.array([np.mean(metric[(cond, a, b)]) for a, b in transformed])

    d_hit, lo_hit, hi_hit = paired_bootstrap_ci(
        pooled(hit, "znorm"), pooled(hit, "raw"), N_BOOT, args.base_seed
    )
    d_nmse, lo_nmse, hi_nmse = paired_bootstrap_ci(
        pooled(nmse, "znorm+restore"), pooled(nmse, "znorm"), N_BOOT, args.base_seed + 1
    )

    # Equivariance is an exactness claim, so it is checked as a spread, not a mean.
    def spread(cond: str, keys: list[tuple[float, float]]) -> float:
        return float(np.ptp([np.mean(nmse[(cond, a, b)]) for a, b in keys]))

    scale_only = [(a, 0.0) for a in A_VALUES]
    equivariance = {
        "znorm+restore_nmse_spread_full_grid": spread(
            "znorm+restore", [(a, b) for a in A_VALUES for b in B_VALUES]
        ),
        "mean+restore_nmse_spread_scale_only": spread("mean+restore", scale_only),
        "mean+restore_nmse_spread_full_grid": spread(
            "mean+restore", [(a, b) for a in A_VALUES for b in B_VALUES]
        ),
        "rms+restore_nmse_spread_scale_only": spread("rms+restore", scale_only),
    }

    payload = {
        "experiment": "controlled-synthetic-affine-probe",
        "purpose": (
            "Separate invariance of retrieval from equivariance of reconstruction "
            "under a known affine transform x' = a*x + b."
        ),
        "data": "synthetic only — no M5, Favorita, or Phase-11B dataset is touched",
        "config": {
            "seeds": seeds,
            "a_values": A_VALUES,
            "b_values": B_VALUES,
            "conditions": [c.name for c in CONDITIONS],
            "n_motifs": args.n_motifs,
            "rows_per_motif": args.rows_per_motif,
            "n_queries": args.n_queries,
            "noise": args.noise,
            "k": args.k,
            "n_bootstrap": N_BOOT,
            "chance_hit_rate": 1.0 / args.n_motifs,
        },
        "cells": cells,
        "contrasts": {
            "invariance_znorm_minus_raw_hit_rate": {
                "delta": d_hit,
                "ci95": [lo_hit, hi_hit],
                "pooled_over": "all transformed (a,b) cells, excluding the a=1,b=0 control",
            },
            "restoration_effect_nmse_znorm_restore_minus_znorm": {
                "delta": d_nmse,
                "ci95": [lo_nmse, hi_nmse],
                "note": "negative favours restoration (lower error)",
            },
        },
        "equivariance": equivariance,
        "runtime_sec": round(time.time() - started, 2),
        "run_context": RunContext().to_dict(),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "affine-probe-results.json"
    path.write_text(json.dumps(payload, indent=2))

    print(f"{'condition':16s} {'a':>7s} {'b':>7s} {'hit':>13s} {'nmse':>13s}")
    last = None
    for c in cells:
        if last and c["condition"] != last:
            print()
        last = c["condition"]
        print(
            f"{c['condition']:16s} {c['a']:7.1f} {c['b']:7.1f} "
            f"{c['hit_rate_mean']:8.3f}±{c['hit_rate_sd']:<4.2f} "
            f"{c['nmse_mean']:9.3g}±{c['nmse_sd']:<.1g}"
        )
    print(f"\nchance hit rate: {1.0 / args.n_motifs:.3f}   seeds: {len(seeds)}")
    print(f"invariance  znorm - raw hit rate: {d_hit:+.3f}  CI95 [{lo_hit:+.3f}, {hi_hit:+.3f}]")
    print(
        f"restoration znorm+restore - znorm nmse: {d_nmse:+.4g}  "
        f"CI95 [{lo_nmse:+.4g}, {hi_nmse:+.4g}]"
    )
    for key, val in equivariance.items():
        print(f"  {key}: {val:.3g}")
    print(f"\nwrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
