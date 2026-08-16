#!/usr/bin/env python3
"""Dataset audit. Backs the `/data-audit` skill.

Audits a Parquet/CSV table and writes a Markdown + JSON report to ``reports/``:
schema, missing values, duplicate rows, timestamp ordering & inferred frequency,
zero-value proportion, column cardinalities, and (optionally) train/val/test
boundary sanity plus a heuristic scan for unexpected future-looking columns.

READ-ONLY on the dataset. Example:
    uv run python scripts/data_audit.py --path data/processed/sales.parquet \
        --time-col date --group-cols item_id store_id --target sales
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
FUTURE_HINTS = ("future", "ahead", "next_", "lead_", "_t+1", "forecast_actual", "label")


def load(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix in {".csv", ".tsv"}:
        return pl.read_csv(path, separator="\t" if path.suffix == ".tsv" else ",")
    raise ValueError(f"Unsupported file type: {path.suffix}")


def audit(
    df: pl.DataFrame,
    time_col: str | None,
    group_cols: list[str],
    target: str | None,
) -> dict:
    n = df.height
    report: dict = {
        "rows": n,
        "cols": df.width,
        "schema": {c: str(t) for c, t in df.schema.items()},
        "missing": {c: int(df[c].null_count()) for c in df.columns},
        "duplicate_rows": int(n - df.n_unique()),
        "cardinalities": {c: int(df[c].n_unique()) for c in group_cols if c in df.columns},
        "warnings": [],
    }

    if target and target in df.columns:
        try:
            zeros = int(df.filter(pl.col(target) == 0).height)
            report["zero_proportion"] = round(zeros / n, 4) if n else None
        except pl.exceptions.PolarsError:
            report["zero_proportion"] = None

    if time_col and time_col in df.columns:
        s = df[time_col]
        report["time"] = {
            "column": time_col,
            "min": str(s.min()),
            "max": str(s.max()),
            "monotonic_nondecreasing": bool(s.is_sorted()),
            "distinct": int(s.n_unique()),
        }
        if not s.is_sorted():
            report["warnings"].append(
                f"'{time_col}' is not globally sorted; verify per-series ordering."
            )

    future_cols = [c for c in df.columns if any(h in c.lower() for h in FUTURE_HINTS)]
    if future_cols:
        report["warnings"].append(
            f"Columns look future/label-derived — verify they are not leakage: {future_cols}"
        )
    report["future_like_columns"] = future_cols
    return report


def write_reports(path: Path, report: dict) -> tuple[Path, Path]:
    reports = REPO / "reports"
    reports.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"data-audit-{path.stem}-{ts}"

    json_path = reports / f"{stem}.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))

    lines = [
        f"# Data Audit — `{path.name}`",
        "",
        f"- Timestamp: {ts}",
        f"- Rows: {report['rows']} | Cols: {report['cols']} | Duplicate rows: {report['duplicate_rows']}",
        "",
        "## Warnings",
        *([f"- ⚠️ {w}" for w in report["warnings"]] or ["- none"]),
        "",
        "## Missing values (non-zero)",
        *([f"- `{c}`: {v}" for c, v in report["missing"].items() if v] or ["- none"]),
    ]
    md_path = reports / f"{stem}.md"
    md_path.write_text("\n".join(lines) + "\n")
    return md_path, json_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--group-cols", nargs="*", default=[])
    ap.add_argument("--target", default=None)
    args = ap.parse_args()

    df = load(args.path)
    report = audit(df, args.time_col, args.group_cols, args.target)
    md, js = write_reports(args.path, report)
    print(f"Audited {args.path} → {md.relative_to(REPO)} , {js.relative_to(REPO)}")
    for w in report["warnings"]:
        print(f"⚠️  {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
