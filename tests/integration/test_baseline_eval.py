"""End-to-end baseline evaluation on synthetic data (Phase 2, tasks 7-10)."""

from __future__ import annotations

import math

import pytest

from scalerag.data.m5_ingest import ingest_m5
from scalerag.data.synthetic import generate_m5
from scalerag.eval import evaluate
from scalerag.splits import make_rolling_splits, split_by_name

N = 120  # small day range; canonical split logic, smaller numbers


@pytest.fixture
def processed(tmp_path):
    raw = generate_m5(tmp_path / "raw", n_days=N, seed=5)
    out = tmp_path / "processed"
    ingest_m5(raw.root, out, n_days=N)
    return out


def _split():
    return split_by_name(make_rolling_splits(last_labeled_day=N), "test")  # train_end=92


@pytest.mark.integration
def test_seasonal_naive_end_to_end(processed) -> None:
    rep = evaluate(processed, _split(), "seasonal_naive")
    m = rep["metrics"]
    assert rep["n_series"] == 24
    assert len(rep["wrmsse_per_level"]) == 12
    assert math.isfinite(m["wrmsse"]) and m["wrmsse"] > 0
    assert math.isfinite(m["mae"]) and m["mae"] >= 0
    assert rep["run_context"]["torch_version"]  # repro fingerprint captured


@pytest.mark.integration
def test_lightgbm_end_to_end(processed) -> None:
    rep = evaluate(processed, _split(), "lightgbm", params={"n_estimators": 40})
    m = rep["metrics"]
    assert math.isfinite(m["wrmsse"]) and m["wrmsse"] > 0
    assert math.isfinite(m["rmsse_mean"])
    assert rep["model"]["kind"] == "lightgbm"


@pytest.mark.integration
def test_determinism_same_seed(processed) -> None:
    a = evaluate(processed, _split(), "lightgbm", params={"n_estimators": 40}, seed=7)
    b = evaluate(processed, _split(), "lightgbm", params={"n_estimators": 40}, seed=7)
    assert a["metrics"]["wrmsse"] == pytest.approx(b["metrics"]["wrmsse"])
