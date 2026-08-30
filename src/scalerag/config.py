"""Typed configuration loading for GraphRoute-TS.

Experiment configuration is YAML on disk, validated into pydantic models so that
every run has an explicit, serialisable, reproducible record (see the
reproducibility rules in CLAUDE.md). Keep this module dependency-light: it must
import without torch or any ML stack present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SplitConfig(BaseModel):
    """Chronological split boundaries. All splits are time-ordered — never
    random — to avoid temporal leakage (CLAUDE.md research rule 1)."""

    model_config = ConfigDict(extra="forbid")

    train_end: str = Field(..., description="Inclusive last timestamp of training data.")
    val_end: str = Field(..., description="Inclusive last timestamp of validation data.")
    # test runs from val_end (exclusive) to the end of the series.
    horizon: int = Field(..., gt=0, description="Forecast horizon H (steps).")


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(..., description="Dataset identifier, e.g. 'm5'.")
    freq: str = Field("D", description="Sampling frequency (pandas offset alias).")
    target: str = Field("sales", description="Target column name.")


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Human-readable experiment name.")
    seed: int = Field(42, description="Global RNG seed for reproducibility.")
    data: DataConfig
    split: SplitConfig
    model: dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters.")


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment config.

    Raises pydantic.ValidationError on malformed configs and FileNotFoundError
    if the path does not exist — fail loudly, never silently default.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}: {path}")
    return ExperimentConfig.model_validate(raw)
