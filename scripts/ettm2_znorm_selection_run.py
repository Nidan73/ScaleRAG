#!/usr/bin/env python3
"""Eq. (3) against Eq. (4) on the ETTm2 validation grid — the pre-registered
selection step of ``docs/znorm-preregistration.md``.

The manuscript already centres Eq. (3), the full mean/RMS rule (``znorm`` in the
code): both propositions are proved about it, two figures draw it, and it is
called the canonical form of Stage 1. Every reported number, however, comes from
Eq. (4), the scale-only rule (``mean``). The paper states in
``sec:sicn`` that Eq. (3) "was never run on that panel". This script closes
that gap on validation, which is exactly the selection step the pre-registration
names — nothing here is a new operator.

**Motivation sharpened by the code-versus-text audit**
(``docs/code-versus-text-audit.md``, finding B): on ETTm2 the Eq. (4) divisor is
the *signed* window mean, negative for 65.4% of validation queries, so 26.4% of
retrieved pairs are restored through a negative ratio and emerge sign-inverted.
Eq. (3) divides by a non-negative RMS deviation and cannot do this. The script
therefore records the sign-mismatch rate per scale alongside the accuracy grid,
so the mechanism is measured rather than asserted.

**VALIDATION ONLY.** ETTm2 test was consumed once by the Phase-11A decision
gate. Choosing a scale rule after seeing its test attribution and then rescoring
on test is exactly the violation rules 2, 9 and 12 forbid, so ``--split test``
is refused unconditionally.

Grid: scale {mean, rms, znorm} x k {5, 10, 20} x w {0.25, 0.50, 0.75} — the
frozen grid, so the znorm column is like-for-like against the recorded
selection. k is deliberately not extended here; that is referee item 24's
question and mixing the two would confound them.

Resumable: work is checkpointed per (scale, variable); ``--resume`` reloads
finished pairs and continues.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))  # ettm2_data (sibling)

import ettm2_data as E  # noqa: N812  (canonical helper; `E` matches sibling scripts)
from scalerag.native import NativeScaleRetriever, fixed_fusion
from scalerag.reproducibility import RunContext, set_seed

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports/ettm2-znorm"
PHASE11A = REPO / "reports/phase11a"

# Eq. (4) = "mean", Eq. (3) = "znorm". "rms" is the divisor-only variant already
# reported on validation, kept so the three-way comparison is on one population.
SCALES: tuple[str, ...] = ("mean", "rms", "znorm")
KS: tuple[int, ...] = (5, 10, 20)
WEIGHTS: tuple[float, ...] = (0.25, 0.50, 0.75)
KMAX: int = max(KS)

# Post-tie-break-fix anchors from reports/phase11a/scalerag_native_ettm2_val.json.
# If this harness cannot reproduce them the comparison is meaningless, so it aborts.
ANCHORS: dict[str, float] = {
    "restored_mean_k20": 0.23308555217392715,
    "fusion_mean_k20_w0.25": 0.11084341389714716,
}
ANCHOR_TOL = 1e-9


def _mse_mae(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    return float(((pred - true) ** 2).mean()), float(np.abs(pred - true).mean())


def _ckpt_path(ckpt_dir: Path, scale: str, var: int) -> Path:
    return ckpt_dir / f"val-{scale}-v{var}.npz"


def _run_pair(
    z: np.ndarray, contexts: np.ndarray, origins: np.ndarray, var: int, scale: str
) -> dict[str, np.ndarray]:
    """Retrieve once for one (scale, variable) pair and derive every k from it."""
    retr = NativeScaleRetriever(z[:, var], E.TRAIN_END, scale, E.L, E.H)
    tk = retr.retrieve(contexts, origins, KMAX)

    out: dict[str, np.ndarray] = {}
    for k in KS:
        out[f"raw_{k}"] = retr.forecast_from_topk(tk, contexts, restore=False, k=k).point
        o = retr.forecast_from_topk(tk, contexts, restore=True, k=k)
        out[f"res_{k}"] = o.point
        if k == KMAX:
            out["diag"] = np.array(
                [o.fallback_count, o.invalid_query_scale, o.invalid_cand_restore],
                dtype=np.int64,
            )

    # finding-B mechanism check: how often is a retrieved candidate's scale signed
    # opposite to the query's, so restoration multiplies through a negative ratio?
    if tk.valid_idx.size:
        q_sign = np.sign(tk.qparams[tk.valid_idx, 1])
        c_sign = np.sign(retr.params[tk.ids, 1])
        mism = c_sign != q_sign[:, None]
        out["sign"] = np.array([int(mism.sum()), int(mism.size)], dtype=np.int64)
    else:
        out["sign"] = np.array([0, 0], dtype=np.int64)
    return out


def run(resume: bool) -> dict:
    set_seed(42)
    z, names = E.load_normalized()
    w = E.build_windows(z, "val")
    trues, origins, var_of, contexts = w["trues"], w["origins"], w["var_of"], w["contexts"]
    n_var = z.shape[1]

    cz = np.load(PHASE11A / "chronos_target_ettm2_val.npz")
    if not np.allclose(cz["trues"].astype(np.float64), trues, atol=1e-4):
        raise SystemExit("chronos npz trues misaligned with the canonical build (val)")
    chronos = cz["preds"].astype(np.float64)

    ckpt_dir = OUT / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    raw: dict[tuple[str, int], np.ndarray] = {
        (s, k): np.empty_like(trues) for s in SCALES for k in KS
    }
    res: dict[tuple[str, int], np.ndarray] = {
        (s, k): np.empty_like(trues) for s in SCALES for k in KS
    }
    diag = {s: np.zeros(3, dtype=np.int64) for s in SCALES}
    sign = {s: np.zeros(2, dtype=np.int64) for s in SCALES}

    pairs = [(s, v) for s in SCALES for v in range(n_var)]
    t0 = time.time()
    bar = tqdm(pairs, desc="ETTm2 Eq.(3) vs Eq.(4)", unit="pair", dynamic_ncols=True)
    for scale, var in bar:
        bar.set_postfix_str(f"{scale}/{names[var]}")
        p = _ckpt_path(ckpt_dir, scale, var)
        if resume and p.exists():
            d = dict(np.load(p))
            bar.write(f"  resume: reusing {p.name}")
        else:
            m = var_of == var
            d = _run_pair(z, contexts[m], origins[m], var, scale)
            np.savez_compressed(p, **d)
        m = var_of == var
        for k in KS:
            raw[(scale, k)][m] = d[f"raw_{k}"]
            res[(scale, k)][m] = d[f"res_{k}"]
        diag[scale] += d["diag"]
        sign[scale] += d["sign"]
    bar.close()
    runtime = time.time() - t0

    c_mse, c_mae = _mse_mae(chronos, trues)
    results: dict = {
        "experiment": "ettm2-znorm-selection",
        "question": "Eq. (3) (znorm) vs the deployed Eq. (4) (mean) on the ETTm2 validation grid",
        "preregistration": "docs/znorm-preregistration.md",
        "split": "val",
        "guard": "test refused unconditionally; ETTm2 test was consumed in Phase 11A",
        "n_windows": int(trues.shape[0]),
        "n_var": n_var,
        "variables": names,
        "grid": {"scales": list(SCALES), "k": list(KS), "weights": list(WEIGHTS)},
        "context_length_L": E.L,
        "horizon_H": E.H,
        "scale_eps": 1e-2,
        "runtime_sec": round(runtime, 2),
        "chronos_bolt_target": {"mse": c_mse, "mae": c_mae},
        "raw_retrieval": [],
        "restored_retrieval": [],
        "restored_fixed_fusion": [],
        "invalid_scale_diag": {
            s: {
                "fallback": int(diag[s][0]),
                "invalid_query_scale": int(diag[s][1]),
                "invalid_cand_restore": int(diag[s][2]),
            }
            for s in SCALES
        },
        "sign_mismatch": {
            s: {
                "mismatched_pairs": int(sign[s][0]),
                "total_pairs": int(sign[s][1]),
                "share": (float(sign[s][0] / sign[s][1]) if sign[s][1] else None),
                "note": "share of retrieved (query, candidate) pairs whose scale "
                "denominators carry opposite signs, so restoration multiplies the "
                "analogue by a negative ratio (audit finding B)",
            }
            for s in SCALES
        },
        "run_context": RunContext().to_dict(),
    }

    for scale in SCALES:
        for k in KS:
            rm, ra = _mse_mae(raw[(scale, k)], trues)
            sm, sa = _mse_mae(res[(scale, k)], trues)
            results["raw_retrieval"].append({"scale": scale, "k": k, "mse": rm, "mae": ra})
            results["restored_retrieval"].append({"scale": scale, "k": k, "mse": sm, "mae": sa})
            for wgt in WEIGHTS:
                fm, fa = _mse_mae(fixed_fusion(chronos, res[(scale, k)], wgt), trues)
                results["restored_fixed_fusion"].append(
                    {"scale": scale, "k": k, "weight": wgt, "mse": fm, "mae": fa}
                )

    # The recorded Phase-11A validation numbers must come back exactly, or this
    # harness is not measuring the same thing and the comparison is void.
    got = {
        "restored_mean_k20": next(
            r["mse"] for r in results["restored_retrieval"] if r["scale"] == "mean" and r["k"] == 20
        ),
        "fusion_mean_k20_w0.25": next(
            r["mse"]
            for r in results["restored_fixed_fusion"]
            if r["scale"] == "mean" and r["k"] == 20 and r["weight"] == 0.25
        ),
    }
    bad = {n: (got[n], v) for n, v in ANCHORS.items() if abs(got[n] - v) > ANCHOR_TOL}
    if bad:
        raise SystemExit(
            "anchor mismatch against reports/phase11a/scalerag_native_ettm2_val.json: "
            + "; ".join(f"{n}: got {g!r}, recorded {e!r}" for n, (g, e) in bad.items())
        )
    results["anchor_check"] = {"status": "reproduced", "anchors": ANCHORS}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ettm2-znorm-selection-val.json").write_text(json.dumps(results, indent=2))
    return results


def report(results: dict) -> None:
    c = results["chronos_bolt_target"]["mse"]
    print(f"\nfrozen Chronos-Bolt backbone   mse={c:.5f}")
    print("\nrestored retrieval (lower is better):")
    for r in results["restored_retrieval"]:
        if r["k"] == 20:
            print(f"  {r['scale']:<6} k=20  mse={r['mse']:.5f}  mae={r['mae']:.5f}")
    print("\nfused, w=0.25 (the deployed weight):")
    best = None
    for r in results["restored_fixed_fusion"]:
        if r["k"] == 20 and r["weight"] == 0.25:
            d = 100.0 * (c - r["mse"]) / c
            print(f"  {r['scale']:<6} k=20  mse={r['mse']:.5f}  vs backbone {d:+.3f}%")
    for r in results["restored_fixed_fusion"]:
        if best is None or r["mse"] < best["mse"]:
            best = r
    assert best is not None
    print(
        f"\nbest cell on the whole grid: scale={best['scale']} k={best['k']} "
        f"w={best['weight']:.2f}  mse={best['mse']:.5f} "
        f"({100.0 * (c - best['mse']) / c:+.3f}% vs backbone)"
    )
    print("\nsign-mismatch in the retrieved set (audit finding B):")
    for s, d in results["sign_mismatch"].items():
        share = "n/a" if d["share"] is None else f"{100 * d['share']:.2f}%"
        print(f"  {s:<6} {share:>7} of {d['total_pairs']:,} retrieved pairs")
    print("\ninvalid-scale diagnostics:")
    for s, d in results["invalid_scale_diag"].items():
        print(
            f"  {s:<6} fallback queries={d['fallback']:,} "
            f"skipped candidate restorations={d['invalid_cand_restore']:,}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="validation only; test is refused (ETTm2 test was consumed in Phase 11A)",
    )
    ap.add_argument("--resume", action="store_true", help="reuse finished (scale, variable) pairs")
    a = ap.parse_args()

    if a.split == "test":
        raise SystemExit(
            "refusing to run on the ETTm2 test split. It was consumed once by the "
            "Phase-11A decision gate; selecting a scale rule on validation and then "
            "rescoring it on test is test-driven tuning (rules 2, 9, 12). This "
            "experiment is a validation selection step and has no test arm."
        )

    report(run(a.resume))


if __name__ == "__main__":
    main()
