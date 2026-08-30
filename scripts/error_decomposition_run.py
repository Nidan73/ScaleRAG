#!/usr/bin/env python3
"""Why a +85.4% retrieval gain does not become a forecasting gain (ETTm2).

Post-hoc attribution over the forecasts Phase 11A already stored. This reads
`reports/phase11a/scalerag_native_ettm2_{split}_preds.npz` and recomputes nothing:
no model runs, no retrieval, no dataset is opened, and no configuration is
selected. The frozen config (scale=mean, k=20, weight=0.25) is read from
`docs/scalerag-native-frozen-config.json` and used as-is.

The optimal fusion weight is printed as a **diagnostic**. Adopting it would be
tuning on an evaluation split, which rules 9 and 12 forbid; the frozen weight
stays frozen.

    uv run python scripts/error_decomposition_run.py
    uv run python scripts/error_decomposition_run.py --split val
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

from scalerag.error_decomposition import (  # noqa: E402
    decompose_errors,
    paired_bootstrap_mean_diff,
)
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402

PRED_DIR = REPO / "reports" / "phase11a"
OUT = REPO / "reports" / "error-decomposition"
FROZEN = REPO / "docs" / "scalerag-native-frozen-config.json"
N_BOOT = 2000
SEED = 20260726


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    set_seed(SEED)
    started = time.time()

    cfg = json.loads(FROZEN.read_text())
    scale, k, weight = cfg["scale_strategy"], cfg["top_k"], cfg["fusion_weight"]
    path = PRED_DIR / f"scalerag_native_ettm2_{args.split}_preds.npz"
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO)} — regenerate with scripts/scalerag_native_ettm2.py"
        )

    npz = np.load(path)
    raw_key, res_key = f"raw_{scale}_{k}", f"res_{scale}_{k}"
    for key in ("trues", "chronos", raw_key, res_key):
        if key not in npz.files:
            raise SystemExit(f"{path.name} lacks '{key}'; stored keys: {sorted(npz.files)}")

    dec = decompose_errors(
        truth=npz["trues"],
        backbone=npz["chronos"],
        retrieval_raw=npz[raw_key],
        retrieval_restored=npz[res_key],
        fusion_weight=weight,
    )
    pw = dec.per_window

    contrasts = {}
    for label, (a, b) in {
        "restoration_gain_raw_minus_restored": ("raw_retrieval", "restored_retrieval"),
        "residual_scale_error_restored_minus_shape": ("restored_retrieval", "shape_floor"),
        "fusion_effect_fused_minus_backbone": ("fused", "backbone"),
        "retrieval_deficit_restored_minus_backbone": ("restored_retrieval", "backbone"),
        # Referee item 4: both branches under the SAME oracle correction. A positive
        # delta means the analogues carry the better shape; negative means the
        # backbone does. The asymmetric comparison cannot answer this.
        "symmetric_shape_backbone_minus_retrieval": ("backbone_shape_floor", "shape_floor"),
    }.items():
        delta, lo, hi = paired_bootstrap_mean_diff(pw[a], pw[b], args.n_boot, SEED)
        contrasts[label] = {
            "delta_mse": delta,
            "ci95": [lo, hi],
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        }

    # Share of windows where the retrieval branch actually beats the backbone.
    better = float(np.mean(pw["restored_retrieval"] < pw["backbone"]))
    shape_better = float(np.mean(pw["shape_floor"] < pw["backbone_shape_floor"]))

    payload = {
        "experiment": "retrieval-to-forecasting-error-decomposition",
        "split": args.split,
        "source": str(path.relative_to(REPO)),
        "note": (
            "Post-hoc attribution of stored Phase-11A forecasts. No model is run, no "
            "dataset opened, no configuration selected. The optimal fusion weight is "
            "reported as a diagnostic and is NOT adopted (rules 9, 12)."
        ),
        "frozen_config": {"scale": scale, "k": k, "fusion_weight": weight},
        "decomposition": dec.to_dict(),
        "contrasts": contrasts,
        "windows_where_retrieval_beats_backbone": better,
        "windows_where_retrieval_shape_beats_backbone_shape": shape_better,
        "n_bootstrap": args.n_boot,
        "runtime_sec": round(time.time() - started, 2),
        "run_context": RunContext().to_dict(),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"ettm2-{args.split}-error-decomposition.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print(
        f"ETTm2 {args.split}: {dec.n_windows:,} windows x H={dec.horizon}, "
        f"frozen scale={scale} k={k} w={weight}\n"
    )
    print(f"  {'raw retrieval':<34s} MSE {dec.e_raw:9.4f}")
    print(f"  {'restored retrieval':<34s} MSE {dec.e_res:9.4f}")
    print(f"  {'  of which pure shape (oracle)':<34s} MSE {dec.e_shape:9.4f}")
    print(f"  {'backbone (frozen Chronos-Bolt)':<34s} MSE {dec.e_backbone:9.4f}")
    print(f"  {'  of which pure shape (oracle)':<34s} MSE {dec.e_backbone_shape:9.4f}")
    print(f"  {'fused (shipped)':<34s} MSE {dec.e_fused:9.4f}\n")
    print(f"  scale error removed by restoration : {dec.scale_error_removed:9.4f}")
    print(f"  scale error still remaining        : {dec.scale_error_remaining:9.4f}")
    print(f"  shape share of restored error      : {dec.shape_fraction_of_restored:9.1%}")
    print(f"  restored retrieval / backbone      : {dec.retrieval_backbone_ratio:9.2f}x")
    print(f"  fusion penalty vs best branch      : {dec.fusion_penalty:+9.5f}")
    print(f"  windows where retrieval wins       : {better:9.1%}")
    print(f"  ASYMMETRIC ratio (NOT a claim)     : {dec.asymmetric_shape_ratio:9.2f}x")
    print(f"  SYMMETRIC shape ratio              : {dec.symmetric_shape_ratio:9.2f}x")
    print(f"  windows retrieval shape wins       : {shape_better:9.1%}")
    print(f"  optimal weight (DIAGNOSTIC ONLY)   : {dec.optimal_weight:9.4f}\n")
    for label, c in contrasts.items():
        flag = "CI excludes 0" if c["excludes_zero"] else "CI includes 0"
        print(
            f"  {label:<44s} {c['delta_mse']:+9.4f}  "
            f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  {flag}"
        )
    print(f"\nwrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
