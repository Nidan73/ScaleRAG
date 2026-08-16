#!/usr/bin/env python3
"""Does the learned fusion gate transfer between datasets?

TS-RAG trains its Adaptive Retrieval Mixer on a multi-domain corpus and applies it
zero-shot to unseen benchmarks, so cross-dataset transfer is table stakes in this
literature rather than a differentiator. ScaleRAG's gate has only ever been fitted
and evaluated on one dataset at a time. This tests the missing cell directly: train
the gate on M5, apply it unchanged to Favorita, and the reverse.

Four arms per evaluation dataset, all sharing one retrieval and one backbone run:

* ``chronos_only``   -- the frozen backbone, no retrieval
* ``fixed_0.5``      -- an untrained convex blend, the floor a gate must clear
* ``gate_in_domain`` -- gate fitted on this dataset's own historical origins
* ``gate_transfer``  -- gate fitted on the *other* dataset, applied unchanged

Both gates are fitted on historical origins only (rule 5) and evaluated on the
validation window. The consumed M5 test split is never read (rule 2), and no
configuration is selected from the outcome (rules 9, 12).

    uv run python scripts/gate_transfer_run.py --subset 1000 --favorita-subset 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import scalerag_eval as SE  # noqa: E402,N812 — frozen M5 retrieval + gate features
import scalerag_favorita as SF  # noqa: E402,N812 — frozen Favorita retrieval
from scalerag import fusion  # noqa: E402
from scalerag.eval import load_processed, select_subset  # noqa: E402
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

OUT = REPO / "reports" / "gate-transfer"
QL = SE.QL  # use one quantile grid for both datasets so the features are commensurable


class Panel:
    """A dataset reduced to what the gate experiment needs."""

    def __init__(
        self,
        name: str,
        sales: np.ndarray,
        entities: pl.DataFrame,
        o_eval: int,
        gate_origins: list[int],
        meta_filter: str,
    ) -> None:
        self.name = name
        self.sales = sales
        self.entities = entities
        self.o_eval = o_eval
        self.gate_origins = gate_origins
        self.meta_filter = meta_filter
        self.x_gate: np.ndarray | None = None
        self.y_gate: np.ndarray | None = None
        self.x_eval: np.ndarray | None = None
        self.chronos_pt: np.ndarray | None = None
        self.retr_pt: np.ndarray | None = None


def _forecast(fc, sales: np.ndarray, o: int) -> tuple[np.ndarray, np.ndarray]:
    pt, q = fc.forecast([sales[i, :o] for i in range(sales.shape[0])], SE.H, QL)
    return np.clip(pt, 0.0, None), q


def build(panel: Panel, fc) -> None:
    """Populate gate-training and evaluation matrices for one dataset."""
    n = panel.sales.shape[0]
    xs, ys = [], []
    for o in panel.gate_origins:
        c_pt, c_q = _forecast(fc, panel.sales, o)
        r_pt, _q, nnd, dis = SE.retrieval_all(
            panel.sales,
            panel.entities,
            o,
            [panel.sales[i, o - SE.L : o] for i in range(n)],
            mf=panel.meta_filter,
        )
        actual = panel.sales[:, o : o + SE.H]
        loss_c = np.sqrt(((actual - c_pt) ** 2).mean(1))
        loss_r = np.sqrt(((actual - r_pt) ** 2).mean(1))
        xs.append(SE.gate_features(panel.sales, o, nnd, dis, c_q))
        ys.append((loss_r < loss_c).astype(int))
    panel.x_gate, panel.y_gate = np.vstack(xs), np.concatenate(ys)

    o = panel.o_eval
    c_pt, c_q = _forecast(fc, panel.sales, o)
    r_pt, _q, nnd, dis = SE.retrieval_all(
        panel.sales,
        panel.entities,
        o,
        [panel.sales[i, o - SE.L : o] for i in range(n)],
        mf=panel.meta_filter,
    )
    panel.chronos_pt, panel.retr_pt = c_pt, r_pt
    panel.x_eval = SE.gate_features(panel.sales, o, nnd, dis, c_q)


def evaluate(panel: Panel, source: Panel | None, seeds: list[int]) -> dict[str, object]:
    """RMSSE of a gate trained on ``source`` (or in-domain when None) applied here."""
    import lightgbm as lgb

    train = panel if source is None else source
    assert train.x_gate is not None and train.y_gate is not None
    assert panel.x_eval is not None and panel.chronos_pt is not None
    per_seed = []
    for s in seeds:
        gate = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=50,
            verbose=-1,
            random_state=s,
        )
        gate.fit(train.x_gate, train.y_gate)
        w = gate.predict_proba(panel.x_eval)[:, 1][:, None]
        fused = np.clip((1 - w) * panel.chronos_pt + w * panel.retr_pt, 0.0, None)
        per_seed.append(SE.rmsse_series(fused, panel.sales, panel.o_eval))
    stacked = np.stack(per_seed)  # (n_seeds, n_series)
    return {
        "rmsse_mean": float(np.nanmean(stacked)),
        "rmsse_sd_over_seeds": float(np.std([np.nanmean(r) for r in stacked], ddof=1)),
        "per_series": np.nanmean(stacked, axis=0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=int, default=1000, help="M5 series")
    ap.add_argument("--favorita-subset", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate-seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    set_seed(args.seed)
    started = time.time()
    from scalerag.tsfm.chronos2 import Chronos2Forecaster

    # ---- M5 -----------------------------------------------------------------
    ents, dyn = load_processed(REPO / "data" / "processed")
    ents, dyn = select_subset(ents, dyn, args.subset, args.seed)
    n_days = int(dyn["day_idx"].max())
    m5_sales = dyn["sales"].to_numpy().astype(np.float64).reshape(ents.height, n_days)
    sp = make_rolling_splits(last_labeled_day=n_days)
    if split_by_name(sp, "val").train_end >= split_by_name(sp, "test").train_end:
        raise SystemExit("refusing to run: eval origin reaches the consumed M5 test split")
    m5 = Panel(
        "M5",
        m5_sales,
        ents,
        split_by_name(sp, "val").train_end,
        [split_by_name(sp, "val_m2").train_end, split_by_name(sp, "val_m1").train_end],
        "cat_id",
    )

    # ---- Favorita -----------------------------------------------------------
    f_ents = pl.read_parquet(SF.PROC / "entities.parquet").sort("id")
    f_dyn = pl.read_parquet(SF.PROC / "dynamic.parquet").sort(["id", "day_idx"])
    keep_ids = f_ents["id"].head(args.favorita_subset)
    f_ents = f_ents.filter(pl.col("id").is_in(keep_ids))
    f_dyn = f_dyn.filter(pl.col("id").is_in(keep_ids))
    f_days = int(f_dyn["day_idx"].max())
    f_sales = f_dyn["sales"].to_numpy().astype(np.float64).reshape(f_ents.height, f_days)
    # The retrieval index is M5-shaped, so present Favorita metadata under the names
    # scalerag_favorita.py already uses: family -> cat_id, class -> dept_id.
    f_view = f_ents.select(
        "id",
        "item_id",
        pl.col("family_id").alias("cat_id"),
        pl.col("class_id").alias("dept_id"),
        "store_id",
        "state_id",
    )
    f_sp = make_rolling_splits(last_labeled_day=f_days)
    fav = Panel(
        "Favorita",
        f_sales,
        f_view,
        split_by_name(f_sp, "val").train_end,
        [
            split_by_name(f_sp, "val_m2").train_end,
            split_by_name(f_sp, "val_m1").train_end,
        ],
        "cat_id",
    )

    fc = Chronos2Forecaster()
    for panel in (m5, fav):
        print(
            f"building {panel.name}: {panel.sales.shape[0]} series, "
            f"eval origin {panel.o_eval}, gate origins {panel.gate_origins}"
        )
        build(panel, fc)

    results: dict[str, object] = {}
    for target, other in ((m5, fav), (fav, m5)):
        assert target.chronos_pt is not None and target.retr_pt is not None
        fixed = np.clip(0.5 * target.chronos_pt + 0.5 * target.retr_pt, 0.0, None)
        base = {
            "chronos_only": SE.rmsse_series(target.chronos_pt, target.sales, target.o_eval),
            "retrieval_only": SE.rmsse_series(target.retr_pt, target.sales, target.o_eval),
            "fixed_0.5": SE.rmsse_series(fixed, target.sales, target.o_eval),
        }
        in_dom = evaluate(target, None, args.gate_seeds)
        trans = evaluate(target, other, args.gate_seeds)

        arms = {k: float(np.nanmean(v)) for k, v in base.items()}
        arms["gate_in_domain"] = in_dom["rmsse_mean"]
        arms["gate_transfer"] = trans["rmsse_mean"]

        # Transfer is judged against the in-domain gate and against untrained fusion.
        vs_in = fusion.paired_bootstrap_rel_improvement(in_dom["per_series"], trans["per_series"])
        vs_fixed = fusion.paired_bootstrap_rel_improvement(base["fixed_0.5"], trans["per_series"])
        results[target.name] = {
            "n_series": int(target.sales.shape[0]),
            "eval_origin": int(target.o_eval),
            "gate_trained_on": other.name,
            "rmsse": arms,
            "gate_seed_sd": {
                "in_domain": in_dom["rmsse_sd_over_seeds"],
                "transfer": trans["rmsse_sd_over_seeds"],
            },
            "transfer_vs_in_domain_rel_improvement": vs_in,
            "transfer_vs_fixed_0.5_rel_improvement": vs_fixed,
        }

    payload = {
        "experiment": "cross-dataset-gate-transfer",
        "question": "does a fusion gate fitted on one dataset work unchanged on another?",
        "guard": "validation windows only; consumed M5 test split untouched (rules 2, 9, 12)",
        "gate_features": list(fusion.GATE_FEATURES),
        "gate_seeds": args.gate_seeds,
        "seed": args.seed,
        "results": results,
        "timestamp": datetime.now(UTC).isoformat(),
        "runtime_sec": round(time.time() - started, 2),
        "run_context": RunContext().to_dict(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gate-transfer.json"
    path.write_text(json.dumps(payload, indent=2, default=float))

    for name, r in results.items():
        rr = r["rmsse"]  # type: ignore[index]
        print(f"\n=== evaluated on {name} (gate transferred from {r['gate_trained_on']}) ===")  # type: ignore[index]
        for arm in (
            "chronos_only",
            "retrieval_only",
            "fixed_0.5",
            "gate_in_domain",
            "gate_transfer",
        ):
            print(f"  {arm:18s} RMSSE {rr[arm]:.5f}")
        for label in (
            "transfer_vs_in_domain_rel_improvement",
            "transfer_vs_fixed_0.5_rel_improvement",
        ):
            c = r[label]  # type: ignore[index]
            flag = "CI excludes 0" if c["excludes_zero"] else "CI includes 0"
            print(
                f"  {label:42s} {100 * c['rel_improvement']:+7.2f}%  "
                f"[{100 * c['ci95_low']:+.2f}%, {100 * c['ci95_high']:+.2f}%]  {flag}"
            )
    print(f"\nwrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
