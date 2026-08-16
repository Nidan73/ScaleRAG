#!/usr/bin/env python3
"""Chronos-2 GPU compatibility smoke test (Phase 4, task 2).

Verifies the official Chronos-2 checkpoint loads and runs on the RTX 5070 Ti
(Blackwell/sm_120) in bfloat16, on a tiny synthetic batch and a small real-M5
subset, BEFORE any full-scale evaluation. Reports checkpoint size, load time,
VRAM, and inference latency. Read-only w.r.t. data.

    HF_HOME=.hf_cache uv run python scripts/chronos2_smoke.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Project-local, gitignored HF cache unless the caller already set one.
os.environ.setdefault("HF_HOME", str(REPO / ".hf_cache"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

MODEL = "amazon/chronos-2"
H = 28


def _vram_gib() -> float:
    return torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0


def main() -> int:
    assert torch.cuda.is_available(), "CUDA not available — Chronos-2 GPU smoke test requires a GPU"
    dev = "cuda"
    print(f"HF_HOME={os.environ['HF_HOME']}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from chronos import BaseChronosPipeline

    t0 = time.perf_counter()
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map=dev, torch_dtype=torch.bfloat16)
    load_s = time.perf_counter() - t0
    print(f"loaded {MODEL} in {load_s:.1f}s (bfloat16, {dev})")

    # checkpoint size on disk
    cache = Path(os.environ["HF_HOME"])
    ckpt_bytes = sum(f.stat().st_size for f in cache.rglob("*.safetensors"))
    print(f"checkpoint on disk: {ckpt_bytes / 1e6:.1f} MB")

    # --- tiny synthetic batch ---
    rng = np.random.default_rng(0)
    t = np.arange(200)
    synth = [np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.2, 200) for _ in range(8)]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    q, mean = pipe.predict_quantiles(synth, prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
    torch.cuda.synchronize()
    synth_ms = 1000 * (time.perf_counter() - t0)
    print(
        f"\n[synthetic] batch=8  quantiles shape={tuple(q[0].shape)}  mean shape={tuple(mean[0].shape)}"
    )
    print(f"[synthetic] latency={synth_ms:.0f}ms  peak VRAM={_vram_gib():.2f} GiB")
    assert len(mean) == 8 and mean[0].shape[-1] == H
    assert torch.isfinite(mean[0]).all()

    # --- small real-M5 subset (if ingested) ---
    proc = REPO / "data" / "processed" / "dynamic.parquet"
    if proc.exists():
        import polars as pl

        ids = (
            pl.scan_parquet(proc).select("id").unique().sort("id").head(8).collect()["id"].to_list()
        )
        ctx = []
        for sid in ids:
            s = (
                pl.scan_parquet(proc)
                .filter((pl.col("id") == sid) & (pl.col("day_idx") <= 1885))
                .sort("day_idx")
                .select("sales")
                .collect()["sales"]
                .to_numpy()
                .astype(np.float64)
            )
            ctx.append(s)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        pm = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])[1]
        torch.cuda.synchronize()
        real_ms = 1000 * (time.perf_counter() - t0)
        print(f"\n[real-M5] 8 series, context<=d_1885  forecast shape={tuple(pm[0].shape)}")
        print(f"[real-M5] latency={real_ms:.0f}ms  peak VRAM={_vram_gib():.2f} GiB")
        assert pm[0].shape[-1] == H and torch.isfinite(pm[0]).all()
    else:
        print("\n[real-M5] skipped — data/processed/dynamic.parquet not present")

    print("\nSMOKE TEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
