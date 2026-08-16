"""M5 data ingestion and validation for GraphRoute-TS.

Leakage-safe, idempotent CSV→Parquet ingestion that separates stable entities
from dynamic features. All modules here are offline and dependency-light
(Polars/PyArrow only) — no torch, no network.
"""

from __future__ import annotations
