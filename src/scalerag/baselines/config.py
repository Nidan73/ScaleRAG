"""Typed baseline experiment configuration (Phase 2, task 11).

One YAML file = one reproducible baseline run definition. Validated with
``extra='forbid'`` so unknown keys are errors, not silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class M5BaselineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Experiment name (also the report filename stem).")
    baseline: Literal["seasonal_naive", "lightgbm"]
    processed_dir: str = Field("data/processed", description="Ingested Parquet directory.")
    last_labeled_day: int = Field(1941, description="Public label horizon (M5 = 1941).")
    split: str = Field("test", description="Rolling split name to evaluate.")
    n_earlier_val: int = Field(2, ge=2, description="Number of earlier validation origins.")
    subset: int | None = Field(None, description="Small declared series subset (None = all).")
    seed: int = Field(42, description="Primary RNG seed.")
    seeds: list[int] | None = Field(
        None, description="Multiple seeds for dispersion (rule 6); overrides `seed` when set."
    )
    params: dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters.")

    def seed_list(self) -> list[int]:
        return self.seeds if self.seeds else [self.seed]


def load_baseline_config(path: str | Path) -> M5BaselineConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Baseline config not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return M5BaselineConfig.model_validate(raw)
