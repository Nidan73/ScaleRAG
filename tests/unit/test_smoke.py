"""Environment smoke tests (fast, offline).

Covers setup-phase checks 1-10: package import, torch, CUDA, GPU tensor round-trip,
mixed-precision reporting, Polars Parquet I/O, DuckDB over Parquet, a tiny LightGBM
fit, and the config loader. GPU-dependent checks skip cleanly when CUDA is absent
so the suite is green on CPU-only machines while still exercising the GPU here.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_package_imports() -> None:
    import scalerag

    assert scalerag.__version__


@pytest.mark.unit
def test_torch_imports() -> None:
    torch = pytest.importorskip("torch")
    assert torch.__version__


@pytest.mark.gpu
def test_cuda_visible() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")
    assert torch.cuda.device_count() >= 1


@pytest.mark.gpu
def test_tensor_roundtrip_gpu() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")
    x = torch.arange(16, dtype=torch.float32)
    y = (x.to("cuda") * 2).cpu()
    torch.cuda.synchronize()
    assert torch.equal(y, x * 2)


@pytest.mark.gpu
def test_mixed_precision_reported() -> None:
    """Report AMP capability without assuming it works."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")
    # Capability probes only — we assert they report, not that AMP is used.
    assert isinstance(torch.cuda.is_bf16_supported(), bool)
    fp16_ok = torch.cuda.get_device_capability(0)[0] >= 7
    assert isinstance(fp16_ok, bool)
    # autocast context must at least be constructible.
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
        pass


@pytest.mark.unit
def test_polars_parquet_roundtrip(tmp_path) -> None:
    import polars as pl

    df = pl.DataFrame({"id": [1, 2, 3], "sales": [0.0, 1.5, 2.0]})
    path = tmp_path / "tiny.parquet"
    df.write_parquet(path)
    back = pl.read_parquet(path)
    assert back.shape == (3, 2)
    assert back["sales"].sum() == 3.5


@pytest.mark.unit
def test_duckdb_queries_parquet(tmp_path) -> None:
    import duckdb
    import polars as pl

    path = tmp_path / "tiny.parquet"
    pl.DataFrame({"id": [1, 2, 3], "sales": [10, 20, 30]}).write_parquet(path)
    total = duckdb.sql(f"SELECT sum(sales) AS s FROM '{path}'").fetchone()[0]
    assert total == 60


@pytest.mark.unit
def test_lightgbm_tiny_fit() -> None:
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    y = x[:, 0] * 2.0 + rng.normal(scale=0.1, size=200)
    model = lgb.LGBMRegressor(n_estimators=10, num_leaves=7, verbose=-1)
    model.fit(x, y)
    preds = model.predict(x)
    assert preds.shape == (200,)
    assert np.isfinite(preds).all()


@pytest.mark.unit
def test_config_loader(tmp_path) -> None:
    from scalerag.config import ExperimentConfig, load_config

    cfg_text = (
        "name: t\nseed: 7\n"
        "data:\n  dataset: m5\n  freq: D\n  target: sales\n"
        "split:\n  train_end: '2016-03-27'\n  val_end: '2016-04-24'\n  horizon: 28\n"
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg_text)
    cfg = load_config(path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.seed == 7
    assert cfg.split.horizon == 28
