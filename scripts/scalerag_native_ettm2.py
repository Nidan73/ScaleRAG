#!/usr/bin/env python3
"""ScaleRAG native-protocol dev driver on ETTm2 (Phase 11A · Parts 3-5).

Runs the frozen ScaleRAG scale-aware retriever (:mod:`scalerag.native`)
over the canonical channel-major ETTm2 windows and produces, per split, the five
protocol methods:

* ``chronos_bolt_target``            — frozen backbone 0.5-quantile (captured separately)
* ``tsrag_official``                 — official ARM (Part-2 reproduction npz, test only)
* ``scalerag_raw_retrieval``         — mean of top-k continuations, **no** restoration
* ``scalerag_restored_retrieval``    — mean of top-k continuations, scale-restored
* ``scalerag_restored_fixed_fusion`` — ``(1-w)·chronos + w·restored`` (no learned gate)

Grid: scale {mean, rms} x top-k {5, 10, 20} x weight {0.25, 0.50, 0.75}.

Retrieval is over a **strictly train-only** candidate pool (``t_r + H <= 34560``) — the
same (context-512, continuation-64) ETTm2 pairs TS-RAG's KB is built from, filtered to the
leakage-legal subset (rule 5 / rule 3). This is a deliberately *stricter* leakage posture
than TS-RAG's full-series KB and is reported as a fairness caveat, never as an advantage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # ettm2_data (sibling)

import ettm2_data as E  # noqa: N812  (canonical helper; `E` matches sibling scripts)
from scalerag.native import NativeScaleRetriever, fixed_fusion

SCALES: tuple[str, ...] = ("mean", "rms")
KS: tuple[int, ...] = (5, 10, 20)
WEIGHTS: tuple[float, ...] = (0.25, 0.50, 0.75)
KMAX: int = max(KS)
OUT = Path(__file__).resolve().parents[1] / "reports/phase11a"


def _mse_mae(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    return float(((pred - true) ** 2).mean()), float(np.abs(pred - true).mean())


def run(split: str, save_preds: bool) -> dict:
    z, names = E.load_normalized()
    w = E.build_windows(z, split)
    trues, origins, var_of, contexts = w["trues"], w["origins"], w["var_of"], w["contexts"]
    n_var = z.shape[1]

    # canonical alignment guard: the isolated-env Chronos capture must share our row order
    cz = np.load(OUT / f"chronos_target_ettm2_{split}.npz")
    if not np.allclose(cz["trues"].astype(np.float64), trues, atol=1e-4):
        raise SystemExit(f"chronos npz trues misaligned with canonical build ({split})")
    chronos = cz["preds"].astype(np.float64)

    raw: dict[tuple[str, int], np.ndarray] = {}
    res: dict[tuple[str, int], np.ndarray] = {}
    diag: dict[str, dict[str, int]] = {}
    t0 = time.time()
    for scale in SCALES:
        r_buf = {k: np.empty_like(trues) for k in KS}
        s_buf = {k: np.empty_like(trues) for k in KS}
        fb = iq = ic = 0
        for v in range(n_var):
            m = var_of == v
            retr = NativeScaleRetriever(z[:, v], E.TRAIN_END, scale, E.L, E.H)
            tk = retr.retrieve(contexts[m], origins[m], KMAX)
            for k in KS:
                r_buf[k][m] = retr.forecast_from_topk(tk, contexts[m], restore=False, k=k).point
                o = retr.forecast_from_topk(tk, contexts[m], restore=True, k=k)
                s_buf[k][m] = o.point
                if k == KMAX:
                    fb += o.fallback_count
                    iq += o.invalid_query_scale
                    ic += o.invalid_cand_restore
        for k in KS:
            raw[(scale, k)] = r_buf[k]
            res[(scale, k)] = s_buf[k]
        diag[scale] = {"fallback": fb, "invalid_query_scale": iq, "invalid_cand_restore": ic}
    retrieval_seconds = time.time() - t0

    c_mse, c_mae = _mse_mae(chronos, trues)
    results: dict = {
        "split": split,
        "n_windows": int(trues.shape[0]),
        "n_var": n_var,
        "variables": names,
        "retrieval_seconds": retrieval_seconds,
        "chronos_bolt_target": {"mse": c_mse, "mae": c_mae},
        "scalerag_raw_retrieval": [],
        "scalerag_restored_retrieval": [],
        "scalerag_restored_fixed_fusion": [],
        "invalid_scale_diag": diag,
    }
    for scale in SCALES:
        for k in KS:
            rm, ra = _mse_mae(raw[(scale, k)], trues)
            sm, sa = _mse_mae(res[(scale, k)], trues)
            results["scalerag_raw_retrieval"].append({"scale": scale, "k": k, "mse": rm, "mae": ra})
            results["scalerag_restored_retrieval"].append(
                {"scale": scale, "k": k, "mse": sm, "mae": sa}
            )
            for wgt in WEIGHTS:
                fm, fa = _mse_mae(fixed_fusion(chronos, res[(scale, k)], wgt), trues)
                results["scalerag_restored_fixed_fusion"].append(
                    {"scale": scale, "k": k, "weight": wgt, "mse": fm, "mae": fa}
                )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"scalerag_native_ettm2_{split}.json").write_text(json.dumps(results, indent=2))

    if save_preds:
        bundle: dict[str, np.ndarray] = {
            "trues": trues.astype(np.float32),
            "origins": origins,
            "var_of": var_of,
            "chronos": chronos.astype(np.float32),
        }
        for scale in SCALES:
            for k in KS:
                bundle[f"raw_{scale}_{k}"] = raw[(scale, k)].astype(np.float32)
                bundle[f"res_{scale}_{k}"] = res[(scale, k)].astype(np.float32)
        np.savez_compressed(
            OUT / f"scalerag_native_ettm2_{split}_preds.npz",
            **bundle,  # type: ignore[arg-type]  # numpy stub types **kwds as bool
        )

    print(f"[{split}] chronos mse={c_mse:.5f} mae={c_mae:.5f} | retrieval {retrieval_seconds:.1f}s")
    for row in results["scalerag_restored_fixed_fusion"]:
        print(
            f"  fusion scale={row['scale']:<4} k={row['k']:<2} w={row['weight']:.2f}"
            f"  mse={row['mse']:.5f} mae={row['mae']:.5f}"
        )
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--save-preds", action="store_true", help="write per-window preds npz")
    a = ap.parse_args()
    run(a.split, a.save_preds)


if __name__ == "__main__":
    main()
