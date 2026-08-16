"""Frozen Chronos-2 forecaster (Phase 4, tasks 1, 3, 4).

Thin wrapper over the official ``chronos-forecasting`` Chronos-2 pipeline. The
model is used **frozen** (inference only). Supports:
- target-only forecasting (task 3),
- known-future covariates via Chronos-2's covariate API, guarded so only
  genuinely-known-future covariates are used (task 4).

Point forecast = the pipeline's predictive **mean** (minimises squared error, so
appropriate for RMSSE/WRMSSE). Quantiles are returned for pinball loss.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scalerag.leakage import assert_no_future_covariates

_REPO = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(_REPO / ".hf_cache"))

DEFAULT_MODEL = "amazon/chronos-2"
DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class Chronos2Forecaster:
    """Frozen Chronos-2. Loads once; reuse across methods."""

    def __init__(
        self, model: str = DEFAULT_MODEL, device: str = "cuda", dtype: str = "bfloat16"
    ) -> None:
        import torch
        from chronos import BaseChronosPipeline

        self.device = device
        torch_dtype = getattr(torch, dtype)
        self.pipe = BaseChronosPipeline.from_pretrained(
            model, device_map=device, torch_dtype=torch_dtype
        )
        self._torch = torch

    def _to_np(self, tensors) -> np.ndarray:
        return np.stack([t[0].float().cpu().numpy() for t in tensors])

    def forecast(
        self,
        contexts: Sequence[np.ndarray],
        horizon: int,
        quantile_levels: list[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Target-only forecast. Returns (point [n, H], quantiles [n, H, Q])."""
        ql = quantile_levels or DEFAULT_QUANTILES
        q, mean = self.pipe.predict_quantiles(
            list(contexts), prediction_length=horizon, quantile_levels=ql
        )
        point = self._to_np(mean)  # (n, H)
        quants = np.stack([t[0].float().cpu().numpy() for t in q])  # (n, H, Q)
        return point, quants

    def forecast_with_covariates(
        self,
        contexts: Sequence[np.ndarray],
        past_covariates: Sequence[dict[str, np.ndarray]],
        future_covariates: Sequence[dict[str, np.ndarray]],
        horizon: int,
        quantile_levels: list[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecast with known-future covariates (task 4).

        Every future-covariate name must also appear in past covariates (i.e. it
        is a genuinely known-future series) — enforced by
        ``assert_no_future_covariates`` (raises LeakageViolation otherwise).
        """
        ql = quantile_levels or DEFAULT_QUANTILES
        inputs = []
        for ctx, pc, fc in zip(contexts, past_covariates, future_covariates, strict=True):
            assert_no_future_covariates(list(fc.keys()), list(pc.keys()))
            inputs.append({"target": ctx, "past_covariates": pc, "future_covariates": fc})
        q, mean = self.pipe.predict_quantiles(inputs, prediction_length=horizon, quantile_levels=ql)
        point = self._to_np(mean)
        quants = np.stack([t[0].float().cpu().numpy() for t in q])
        return point, quants

    def cuda_peak_gib(self) -> float:
        if self.device == "cuda" and self._torch.cuda.is_available():
            return self._torch.cuda.max_memory_allocated() / (1024**3)
        return 0.0
