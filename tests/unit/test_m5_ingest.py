"""Ingestion + schema + synthetic-fixture tests (Phase 2, tasks 1-3)."""

from __future__ import annotations

import polars as pl
import pytest

from scalerag.data import m5_schema as sch
from scalerag.data.m5_ingest import ingest_m5
from scalerag.data.synthetic import generate_m5

N = 120  # small day range for fast tests


@pytest.fixture
def raw(tmp_path):
    return generate_m5(tmp_path / "raw", n_days=N, seed=1)


@pytest.mark.unit
def test_day_index_roundtrip() -> None:
    assert sch.day_index("d_1941") == 1941
    assert sch.day_label(1914) == "d_1914"
    with pytest.raises(sch.SchemaError):
        sch.day_index("day_5")


@pytest.mark.unit
def test_synthetic_files_validate(raw) -> None:
    sch.validate_all(raw, n_days=N)  # must not raise


@pytest.mark.unit
def test_validate_rejects_wrong_day_range(raw) -> None:
    # Default expects d_1941; synthetic has d_120 → must fail loudly.
    with pytest.raises(sch.SchemaError):
        sch.validate_all(raw, n_days=sch.N_DAYS_EVAL)


@pytest.mark.unit
def test_ingest_entities_and_dynamic(tmp_path, raw) -> None:
    out = tmp_path / "processed"
    summary = ingest_m5(raw.root, out, n_days=N)
    entities = pl.read_parquet(out / "entities.parquet")
    dynamic = pl.read_parquet(out / "dynamic.parquet")

    # 2 states x 2 stores x 2 depts x 3 items = 24 series
    assert summary["n_series"] == 24
    assert entities.height == 24
    assert set(entities.columns) == set(sch.ENTITY_COLS)
    assert dynamic.height == 24 * N == summary["n_rows"]
    # entities are stable (unique per id); no day columns leaked in
    assert entities["id"].n_unique() == 24
    assert "sales" not in entities.columns


@pytest.mark.unit
def test_ingest_is_idempotent(tmp_path, raw) -> None:
    out = tmp_path / "processed"
    first = ingest_m5(raw.root, out, n_days=N)
    second = ingest_m5(raw.root, out, n_days=N)
    assert first["skipped"] is False
    assert second["skipped"] is True
    assert second["fingerprint"] == first["fingerprint"]


@pytest.mark.unit
def test_snap_matches_state(tmp_path, raw) -> None:
    out = tmp_path / "processed"
    ingest_m5(raw.root, out, n_days=N)
    dynamic = pl.read_parquet(out / "dynamic.parquet")
    # snap column is 0/1 and present for every row
    assert dynamic["snap"].is_in([0, 1]).all()
    assert dynamic["snap"].null_count() == 0


@pytest.mark.unit
def test_missing_prices_present(tmp_path, raw) -> None:
    # synthetic withholds early-week prices → some null sell_price rows exist
    out = tmp_path / "processed"
    summary = ingest_m5(raw.root, out, n_days=N)
    assert summary["missing_price_rows"] > 0
