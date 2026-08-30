#!/usr/bin/env python3
"""ETTm2 retrieval cost on the same device as the adapter it is compared against.

Referee item 19: Table X reports ScaleRAG's index as CPU-resident at 0.607 ms per
window against TS-RAG's 0.450 ms GPU adapter, and Fig. 11 calls the result
Pareto-dominated. A CPU-versus-GPU latency comparison cannot support that. This
re-measures the identical retrieval with the candidate matrix on the GPU, using
``scalerag_native.topk_exact_torch`` -- the same distance identity, the same
ascending order, the same tie-break by candidate index, verified in
``tests/unit/test_scalerag_native.py`` to select identical candidates.

It also corrects the storage figure. Table X reports 1,096.2 MB, which is the
float64 candidate matrix. FAISS ``IndexFlatL2`` and the native retriever both
operate in float32, so the index is half that; the float64 array is a retained
intermediate, not index storage. Both are reported here.

Nothing is selected or tuned; this is a measurement of the frozen configuration.

Usage (fish):
    uv run python scripts/ettm2_cost_run.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import ettm2_data as E  # noqa: E402,N812 -- matches the convention in scripts/
from scalerag.native import (  # noqa: E402
    NativeScaleRetriever,
    topk_exact,
    topk_exact_torch,
)
from scalerag.reproducibility import RunContext, set_seed  # noqa: E402

OUT = REPO / "reports" / "ettm2-cost"


def time_path(fn, queries, cands, k, repeats: int) -> tuple[float, np.ndarray]:
    fn(queries[:8], cands, k)  # warm-up, excluded from the timing
    best = float("inf")
    ids = None
    for _ in range(repeats):
        t0 = time.time()
        ids, _d = fn(queries, cands, k)
        best = min(best, time.time() - t0)
    return best, ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--n-queries", type=int, default=2048, help="queries per timing pass")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--var", type=int, default=0, help="which channel's pool to measure")
    args = ap.parse_args()

    set_seed(42)
    started = time.time()

    z, names = E.load_normalized()
    built = E.build_windows(z, args.split)
    contexts = built["contexts"]
    var_of = built["var_of"]
    m = var_of == args.var
    queries = contexts[m][: args.n_queries]

    retr = NativeScaleRetriever(z[:, args.var], E.TRAIN_END, "mean", E.L, E.H)
    # `vecs` is the transformed candidate pool the retriever searches. It is held in
    # float64, which is the source of Table X's overstated storage figure.
    cand = retr.vecs

    cpu_s, ids_cpu = time_path(
        lambda q, c, k: topk_exact(q, c, k), queries, cand, args.k, args.repeats
    )
    gpu_s, ids_gpu = time_path(
        lambda q, c, k: topk_exact_torch(q, c, k, device="cuda"),
        queries,
        cand,
        args.k,
        args.repeats,
    )
    identical = bool(np.array_equal(ids_cpu, ids_gpu))
    if not identical:
        raise SystemExit(
            "GPU retrieval selected different candidates from the CPU path; the cost "
            "comparison would not be measuring the same retrieval"
        )

    nq = queries.shape[0]
    payload = {
        "experiment": "ettm2-retrieval-cost-same-device",
        "split": args.split,
        "channel": names[args.var],
        "k": args.k,
        "n_queries_timed": int(nq),
        "candidate_pool_windows": int(cand.shape[0]),
        "gpu_matches_cpu_exactly": identical,
        "latency_ms_per_query": {
            "cpu": 1000.0 * cpu_s / nq,
            "gpu": 1000.0 * gpu_s / nq,
            "speedup": cpu_s / gpu_s if gpu_s > 0 else float("nan"),
        },
        "storage_mb": {
            "index_vectors_float32": cand.astype(np.float32).nbytes / 1e6,
            "index_vectors_float64_as_reported": cand.astype(np.float64).nbytes / 1e6,
            "note": (
                "Table X's 1,096.2 MB is the float64 candidate matrix. The searched "
                "index is float32, i.e. half. Both are given so the cost claim can be "
                "restated on the right one."
            ),
        },
        "timestamp": datetime.now(UTC).isoformat(),
        "runtime_sec": round(time.time() - started, 2),
        "run_context": RunContext().to_dict(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"ettm2-{args.split}-cost.json"
    path.write_text(json.dumps(payload, indent=2))

    lat = payload["latency_ms_per_query"]
    print(f"\nwrote {path.relative_to(REPO)}\n")
    print(f"pool {cand.shape[0]:,} windows x {cand.shape[1]}, k={args.k}, {nq:,} queries")
    print(f"  CPU  {lat['cpu']:.4f} ms/query")
    print(f"  GPU  {lat['gpu']:.4f} ms/query   ({lat['speedup']:.1f}x)")
    print(f"  identical candidates: {identical}")
    st = payload["storage_mb"]
    print(
        f"  index float32 {st['index_vectors_float32']:.1f} MB "
        f"(reported as {st['index_vectors_float64_as_reported']:.1f} MB float64)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
