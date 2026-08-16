#!/usr/bin/env python3
"""Shared ETTm2 data helper for Phase 11A (dependency-free: numpy + pandas only).

Both the isolated-env Chronos capture and the main-env ScaleRAG driver import this so
that windows are built in one identical canonical order and normalised with one
identical train-only scaler — decoupling per-window alignment from TS-RAG's
``data_provider`` (which shuffles the val split).

Canonical window order is **channel-major**: variable ``v`` outer, window start ``s``
inner — the same order TS-RAG's unshuffled *test* loader yields, so the reproduction's
per-window arrays line up too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

L: int = 512  # context length (standard TS notation)
H: int = 64  # forecast horizon
TRAIN_END: int = 12 * 30 * 24 * 4  # 34560 (15-min ETTm convention)
_Q: int = 4 * 30 * 24 * 4  # 11520 (val and test block length)

ETTM2_CSV = Path(__file__).resolve().parents[1] / "external/ts-rag/datasets/ETT-small/ETTm2.csv"


def borders(split: str) -> tuple[int, int]:
    """(border1, border2) for a split, matching TS-RAG's ``Dataset_ETT_minute``."""
    b1s = [0, TRAIN_END - L, TRAIN_END + _Q - L]
    b2s = [TRAIN_END, TRAIN_END + _Q, TRAIN_END + 2 * _Q]
    m = {"train": 0, "val": 1, "test": 2}[split]
    return b1s[m], b2s[m]


def load_normalized(csv: Path = ETTM2_CSV) -> tuple[np.ndarray, list[str]]:
    """Return train-only-`StandardScaler`-normalised series ``Z`` (T, n_var) and var names.

    Scaler statistics use the training region ``[:TRAIN_END]`` only (population std,
    ddof=0 — matching sklearn ``StandardScaler``)."""
    df = pd.read_csv(csv)
    variables = list(df.columns[1:])
    x = df[variables].to_numpy(np.float64)
    mu = x[:TRAIN_END].mean(axis=0)
    sd = x[:TRAIN_END].std(axis=0)  # ddof=0
    z = (x - mu) / sd
    return z, variables


def build_windows(z: np.ndarray, split: str) -> dict[str, Any]:
    """Canonical channel-major windows for a split.

    Returns dict with ``contexts`` (N, L), ``trues`` (N, H), ``origins`` (N,) global
    forecast origins, ``var_of`` (N,) variable index, and scalar ``tot`` per-variable count.
    """
    b1, b2 = borders(split)
    data = z[b1:b2]
    tot = len(data) - L - H + 1
    if tot <= 0:
        raise ValueError(f"split {split!r} too short: {len(data)} < L+H")
    n_var = z.shape[1]
    contexts = np.empty((tot * n_var, L), dtype=np.float64)
    trues = np.empty((tot * n_var, H), dtype=np.float64)
    origins = np.empty(tot * n_var, dtype=np.int64)
    var_of = np.empty(tot * n_var, dtype=np.int64)
    row = 0
    for v in range(n_var):
        for s in range(tot):
            contexts[row] = data[s : s + L, v]
            trues[row] = data[s + L : s + L + H, v]
            origins[row] = b1 + s + L
            var_of[row] = v
            row += 1
    return {
        "contexts": contexts,
        "trues": trues,
        "origins": origins,
        "var_of": var_of,
        "tot": np.int64(tot),
    }
