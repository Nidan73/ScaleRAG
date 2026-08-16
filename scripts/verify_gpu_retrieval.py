#!/usr/bin/env python3
"""Parity gate: GPU exact retriever == frozen numpy retriever (Phase 10).

Proves ``retrieval_gpu.retrieve_scaleaware_gpu`` reproduces the frozen
``retrieval_faiss.ScaleAwareIndex(scale='mean', metric='l2')`` searched with
``meta_filter='cat_id'`` on the 1,000-series validation subset, BEFORE the GPU
path is used for the single locked full-panel test. Checks:

1. ``build_windows`` == ``WindowDatabase.from_training`` (contexts/conts/t_r/idx).
2. per-series retrieval point forecast is bit-close (identical retrieved sets).
3. per-series retrieval RMSSE matches the frozen tables value ``mean_scaling``
   = 0.7424561064 to <1e-6.

    uv run python scripts/verify_gpu_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scalerag import metrics  # noqa: E402
from scalerag.eval import load_processed, select_subset  # noqa: E402
from scalerag.reproducibility import set_seed  # noqa: E402
from scalerag.retrieval import WindowDatabase  # noqa: E402
from scalerag.retrieval_faiss import ScaleAwareIndex, restore_continuation  # noqa: E402
from scalerag.retrieval_gpu import build_windows, retrieve_scaleaware_gpu  # noqa: E402
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

QL = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
L, H, K, STRIDE = 56, 28, 20, 7
FROZEN_MEAN_SCALING_RMSSE = 0.7424561064366584  # docs/scalerag-final-tables.json


def frozen_retrieval(sales, entities, o, queries):
    """Exact replica of scalerag_matrix.retrieval(mean, restore, cat_id) point path."""
    db = WindowDatabase.from_training(sales, o, L, H, stride=STRIDE)
    idx = ScaleAwareIndex(db, entities, scale="mean", metric="l2")
    n = len(queries)
    pt = np.zeros((n, H))
    for i in range(n):
        ids, _dists, qp = idx.search(queries[i], o + 1, K, query_series_idx=i, meta_filter="cat_id")
        if ids.size == 0:
            pt[i] = float(queries[i].mean())
            continue
        conts = idx.db.continuations[ids]
        conts = np.stack(
            [restore_continuation(conts[j], idx.params[ids[j]], qp) for j in range(len(ids))]
        )
        pt[i] = np.clip(conts, 0.0, None).mean(0)
    return pt


def main() -> int:
    set_seed(42)
    entities, dynamic = load_processed(REPO / "data" / "processed")
    entities, dynamic = select_subset(entities, dynamic, 1000, 42)
    n = entities.height
    n_days = int(dynamic["day_idx"].max())
    sales = dynamic["sales"].to_numpy().astype(np.float64).reshape(n, n_days)
    o = split_by_name(make_rolling_splits(last_labeled_day=n_days), "val").train_end
    queries = np.stack([sales[i, o - L : o] for i in range(n)])
    cat_codes = np.unique(entities["cat_id"].to_numpy(), return_inverse=True)[1].astype(np.int64)

    # --- check 1: window construction identical ---
    ctx_g, cont_g, sidx_g, tr_g = build_windows(sales, o, L, H, STRIDE)
    db_ref = WindowDatabase.from_training(sales, o, L, H, stride=STRIDE)
    w_ok = (
        np.array_equal(ctx_g, db_ref.contexts)
        and np.array_equal(cont_g, db_ref.continuations)
        and np.array_equal(sidx_g, db_ref.series_idx)
        and np.array_equal(tr_g, db_ref.t_r)
    )
    print(f"[1] build_windows == WindowDatabase: {w_ok}  (Nc={ctx_g.shape[0]})")

    # --- checks 2-3: frozen vs gpu retrieval forecast ---
    ref_pt = frozen_retrieval(sales, entities, o, queries)
    res = retrieve_scaleaware_gpu(
        sales, cat_codes, o, o + 1, queries, cat_codes, QL, k=K, device="cuda"
    )
    tr, ha = sales[:, :o], sales[:, o : o + H]
    rmsse_ref = np.array([metrics.rmsse(ha[i], ref_pt[i], tr[i]) for i in range(n)])
    rmsse_gpu = np.array([metrics.rmsse(ha[i], res.point[i], tr[i]) for i in range(n)])

    max_pt_diff = float(np.abs(ref_pt - res.point).max())
    rel_ref, rel_gpu = float(np.nanmean(rmsse_ref)), float(np.nanmean(rmsse_gpu))
    per_series_max = float(np.nanmax(np.abs(rmsse_ref - rmsse_gpu)))

    print(f"[2] max |point_ref - point_gpu|       : {max_pt_diff:.3e}")
    print(f"[3] retrieval RMSSE  ref={rel_ref:.10f}  gpu={rel_gpu:.10f}")
    print(f"    frozen table mean_scaling        : {FROZEN_MEAN_SCALING_RMSSE:.10f}")
    print(f"    max per-series |RMSSE_ref-gpu|   : {per_series_max:.3e}")

    ref_matches_frozen = abs(rel_ref - FROZEN_MEAN_SCALING_RMSSE) < 1e-6
    gpu_matches_ref = abs(rel_gpu - rel_ref) < 1e-6 and max_pt_diff < 1e-6
    ok = w_ok and ref_matches_frozen and gpu_matches_ref
    print(
        f"\nPARITY {'PASS' if ok else 'FAIL'}  "
        f"(windows={w_ok}, ref==frozen_table={ref_matches_frozen}, gpu==ref={gpu_matches_ref})"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
