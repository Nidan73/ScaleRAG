#!/usr/bin/env python3
"""Part-4 config selection + Part-5 mechanism analysis for ScaleRAG native (Phase 11A).

Selection is on the **ETTm2 validation split only**: over the pre-registered grid
(scale {mean,rms} x k {5,10,20} x weight {0.25,0.50,0.75}) of
``scalerag_restored_fixed_fusion``, pick the lowest-val-MSE config among those whose val
MAE is within 2% of the grid-best (lowest) val MAE — the pre-registered MSE-primary /
MAE-safeguard rule. No learned gate, no test peeking.

Mechanism checks are on the **ETTm2 test split** (still the dev dataset — the four
Phase-11B datasets are never opened here), for the single frozen config:

* A  no-restoration vs restoration        (raw vs restored retrieval)
* B  retrieval-only vs fixed fusion        (restored vs fused)
* C  ScaleRAG vs target-only Chronos-Bolt   (fused vs backbone)
* D  ScaleRAG vs official TS-RAG            (fused vs ARM)

Each reports aggregate MSE/MAE, per-series (per-variable) errors + win/loss counts, and a
paired bootstrap 95% CI for the relative MSE improvement. The bootstrap unit is the
**window** (n=80199), matching TS-RAG's own ``boot_res`` convention; overlapping stride-1
windows are autocorrelated, so the CI is reported as an approximate (window-level) interval
(threats-to-validity).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scalerag.fusion import paired_bootstrap_rel_improvement

OUT = Path(__file__).resolve().parents[1] / "reports/phase11a"
DOCS = Path(__file__).resolve().parents[1] / "docs"
SCALES = ("mean", "rms")
MAE_SAFEGUARD = 0.02  # selected config MAE must be within 2% of grid-best MAE


def _win_mse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return ((pred - true) ** 2).mean(axis=1)  # (N,) per-window MSE


def _win_mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.abs(pred - true).mean(axis=1)


def select_config(val: dict) -> dict:
    grid = val["scalerag_restored_fixed_fusion"]
    min_mae = min(g["mae"] for g in grid)
    eligible = [g for g in grid if g["mae"] <= (1.0 + MAE_SAFEGUARD) * min_mae]
    sel = min(eligible, key=lambda g: (g["mse"], SCALES.index(g["scale"]), g["k"], g["weight"]))
    return {
        "scale": sel["scale"],
        "k": sel["k"],
        "weight": sel["weight"],
        "val_mse": sel["mse"],
        "val_mae": sel["mae"],
        "grid_best_mae": min_mae,
        "selected_mae_regression_vs_best": (sel["mae"] - min_mae) / min_mae,
        "n_eligible": len(eligible),
        "rule": "min val MSE s.t. val MAE <= 1.02 * grid-best val MAE",
    }


def _mechanism(name: str, base: np.ndarray, method: np.ndarray, true: np.ndarray) -> dict:
    lb, lm = _win_mse(base, true), _win_mse(method, true)
    boot = paired_bootstrap_rel_improvement(lb, lm, n_boot=2000, seed=0)
    return {
        "comparison": name,
        "baseline_mse": float(lb.mean()),
        "method_mse": float(lm.mean()),
        "baseline_mae": float(_win_mae(base, true).mean()),
        "method_mae": float(_win_mae(method, true).mean()),
        "rel_mse_improvement": boot["rel_improvement"],
        "ci95_low": boot["ci95_low"],
        "ci95_high": boot["ci95_high"],
        "excludes_zero": boot["excludes_zero"],
        "n_windows": boot["n"],
    }


def _per_series(
    base: np.ndarray, method: np.ndarray, true: np.ndarray, var_of: np.ndarray, names: list[str]
) -> dict:
    rows, wins = [], 0
    for v, nm in enumerate(names):
        m = var_of == v
        bm = float(_win_mse(base[m], true[m]).mean())
        mm = float(_win_mse(method[m], true[m]).mean())
        win = mm < bm
        wins += int(win)
        rows.append({"var": nm, "baseline_mse": bm, "method_mse": mm, "win": win})
    return {"per_series": rows, "wins": wins, "n_series": len(names)}


def main() -> None:
    val = json.loads((OUT / "scalerag_native_ettm2_val.json").read_text())
    sel = select_config(val)
    scale, k, weight = sel["scale"], sel["k"], sel["weight"]
    print("SELECTED (val):", sel)

    frozen = {
        "phase": "11A",
        "method": "scalerag_restored_fixed_fusion",
        "backbone": "amazon/chronos-bolt-base (frozen)",
        "context_length": 512,
        "horizon": 64,
        "scale_strategy": scale,
        "top_k": k,
        "fusion_weight": weight,
        "candidate_pool": "train-only (t_r + H <= 34560), exact scale-aware kNN",
        "learned_gate": False,
        "selection_split": "ETTm2 validation",
        "selection_rule": sel["rule"],
        "val_mse": sel["val_mse"],
        "val_mae": sel["val_mae"],
        "frozen_before_test_datasets": True,
    }
    (DOCS / "scalerag-native-frozen-config.json").write_text(json.dumps(frozen, indent=2))

    report: dict = {"selected_config": sel, "frozen_config": frozen, "val_grid": val}

    test_json = OUT / "scalerag_native_ettm2_test.json"
    preds_npz = OUT / "scalerag_native_ettm2_test_preds.npz"
    tsrag_npz = OUT / "repro_tsrag_official_ettm2.npz"
    if test_json.exists() and preds_npz.exists():
        test = json.loads(test_json.read_text())
        d = np.load(preds_npz)
        true = d["trues"].astype(np.float64)
        var_of = d["var_of"]
        names = test["variables"]
        chronos = d["chronos"].astype(np.float64)
        raw = d[f"raw_{scale}_{k}"].astype(np.float64)
        res = d[f"res_{scale}_{k}"].astype(np.float64)
        fused = (1.0 - weight) * chronos + weight * res
        tsrag = np.load(tsrag_npz)["preds"].astype(np.float64)

        mechs = {
            "A_restoration": _mechanism("restored_vs_raw", raw, res, true),
            "B_fusion": _mechanism("fused_vs_restored_only", res, fused, true),
            "C_vs_target": _mechanism("fused_vs_chronos_target", chronos, fused, true),
            "D_vs_tsrag": _mechanism("fused_vs_tsrag_official", tsrag, fused, true),
        }
        series = {
            "A_restoration": _per_series(raw, res, true, var_of, names),
            "C_vs_target": _per_series(chronos, fused, true, var_of, names),
            "D_vs_tsrag": _per_series(tsrag, fused, true, var_of, names),
        }
        report["test_mechanisms"] = mechs
        report["test_per_series"] = series
        report["test_diag"] = test["invalid_scale_diag"]

        print("\n=== TEST mechanism checks (frozen config) ===")
        for key, mrow in mechs.items():
            print(
                f"{key:16} {mrow['comparison']:24} "
                f"base={mrow['baseline_mse']:.5f} meth={mrow['method_mse']:.5f} "
                f"rel={mrow['rel_mse_improvement'] * 100:+.2f}% "
                f"CI[{mrow['ci95_low'] * 100:+.2f},{mrow['ci95_high'] * 100:+.2f}] "
                f"excl0={mrow['excludes_zero']}"
            )
        # Decision-gate conditions 2 & 3 (2/4/5 elsewhere)
        cond2 = (
            mechs["A_restoration"]["rel_mse_improvement"] > 0
            and mechs["A_restoration"]["excludes_zero"]
        )
        cond3 = mechs["C_vs_target"]["rel_mse_improvement"] > 0
        report["decision_gate"] = {
            "cond2_restored_beats_raw_significant": bool(cond2),
            "cond3_fusion_beats_target_mse": bool(cond3),
            "cond3_fusion_beats_target_significant": bool(mechs["C_vs_target"]["excludes_zero"]),
        }
        print(f"\ncond2 (restored>raw, CI excl 0): {cond2}")
        print(f"cond3 (fusion>target on MSE):    {cond3}")

    (DOCS / "scalerag-native-dev-results.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {DOCS / 'scalerag-native-dev-results.json'}")
    print(f"wrote {DOCS / 'scalerag-native-frozen-config.json'}")


if __name__ == "__main__":
    main()
