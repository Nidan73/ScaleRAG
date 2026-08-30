#!/usr/bin/env python3
"""Declared, small baseline runner. Backs the `/baseline-run` skill.

Runs EXACTLY ONE small, explicitly-declared baseline configuration. Before doing
anything it prints the full declaration (model, dataset subset, split, metrics,
seed(s), expected outputs, approximate compute). It REFUSES to launch an
undeclared or full-scale run (CLAUDE.md research rule 12).

    uv run python scripts/baseline_run.py --config configs/<cfg>.yaml --dry-run
    uv run python scripts/baseline_run.py --config configs/<cfg>.yaml --confirm
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scalerag.baselines.config import M5BaselineConfig, load_baseline_config  # noqa: E402
from scalerag.eval import evaluate  # noqa: E402
from scalerag.splits import make_rolling_splits, split_by_name  # noqa: E402

# Guardrails: refuse anything that smells like an undeclared full-scale run.
MAX_SUBSET_SERIES = 200


def _split_for(cfg: M5BaselineConfig):
    splits = make_rolling_splits(
        last_labeled_day=cfg.last_labeled_day, n_earlier_val=cfg.n_earlier_val
    )
    return split_by_name(splits, cfg.split)


def declare(cfg: M5BaselineConfig, split) -> None:
    seeds = cfg.seed_list()
    print("=" * 70)
    print("BASELINE RUN — DECLARATION (rule 12)")
    print("=" * 70)
    print(f"  experiment      : {cfg.name}")
    print(f"  model           : {cfg.baseline}  params={cfg.params or '{}'}")
    print(f"  processed data  : {cfg.processed_dir}")
    print(
        f"  dataset subset  : {cfg.subset if cfg.subset is not None else 'ALL'} series "
        f"(cap {MAX_SUBSET_SERIES})"
    )
    print(
        f"  split           : {split.name}  train<=d_{split.train_end}  "
        f"forecast d_{split.h_start}-d_{split.h_end}  (chronological)"
    )
    print("  metrics         : MAE, WAPE, MASE, RMSSE, WRMSSE (official-style)")
    print(f"  seed(s)         : {seeds}")
    print(f"  expected output : reports/baseline-{cfg.name}.json , reports/baseline-{cfg.name}.md")
    if cfg.subset is None:
        n_runs = len(seeds)
        print(
            f"  approx compute  : FULL real M5 (~30.5k series) x {n_runs} seed(s); "
            "CPU-only; ~1-4 min/seed; peak RAM up to ~25 GiB (eager pipeline)"
        )
    else:
        print(f"  approx compute  : {cfg.subset}-series subset; CPU-only; < 1 min; < 4 GiB RAM")
    print("=" * 70)


def _refuse(msg: str) -> int:
    print(f"Refusing: {msg}", file=sys.stderr)
    return 3


def run(cfg: M5BaselineConfig, split) -> dict:
    seeds = cfg.seed_list()
    t0 = time.perf_counter()
    per_seed = []
    for seed in seeds:
        rep = evaluate(
            cfg.processed_dir, split, cfg.baseline, subset=cfg.subset, seed=seed, params=cfg.params
        )
        per_seed.append(rep)
    runtime = time.perf_counter() - t0
    peak_rss_gib = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)

    # Aggregate metrics across seeds (mean + stdev for dispersion; rule 6).
    keys = per_seed[0]["metrics"].keys()
    agg = {}
    for k in keys:
        vals = [r["metrics"][k] for r in per_seed]
        agg[k] = {
            "mean": statistics.fmean(vals),
            "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    return {
        "name": cfg.name,
        "baseline": cfg.baseline,
        "timestamp": datetime.now(UTC).isoformat(),
        "seeds": seeds,
        "split": split.as_dict(),
        "subset": cfg.subset,
        "n_series": per_seed[0]["n_series"],
        "metrics_aggregated": agg,
        "runtime_sec": round(runtime, 2),
        "peak_rss_gib": peak_rss_gib,
        "run_context": per_seed[0]["run_context"],
        "wrmsse_per_level": per_seed[0]["wrmsse_per_level"],
    }


def write_reports(result: dict) -> tuple[Path, Path]:
    reports = REPO / "reports"
    reports.mkdir(exist_ok=True)
    stem = f"baseline-{result['name']}"
    json_path = reports / f"{stem}.json"
    json_path.write_text(json.dumps(result, indent=2, default=str))

    m = result["metrics_aggregated"]
    lines = [
        f"# Baseline Report — {result['name']}",
        "",
        f"- Model: `{result['baseline']}` | split: `{result['split']['name']}` "
        f"(forecast d_{result['split']['h_start']}-d_{result['split']['h_end']})",
        f"- Series: {result['n_series']} | seeds: {result['seeds']} | "
        f"runtime: {result['runtime_sec']}s | peak RSS: {result.get('peak_rss_gib', '?')} GiB",
        f"- Git commit: `{result['run_context'].get('git_commit', '?')}`",
        "",
        "| Metric | Mean | Stdev |",
        "|--------|------|-------|",
        *[f"| {k} | {v['mean']:.5f} | {v['stdev']:.5f} |" for k, v in m.items()],
    ]
    md_path = reports / f"{stem}.md"
    md_path.write_text("\n".join(lines) + "\n")
    return md_path, json_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="Declared baseline config YAML.")
    ap.add_argument("--dry-run", action="store_true", help="Print declaration and exit.")
    ap.add_argument("--confirm", action="store_true", help="Required to actually run.")
    ap.add_argument(
        "--allow-full",
        action="store_true",
        help="Explicitly permit an ALL-series run (still capped by good sense).",
    )
    args = ap.parse_args()

    cfg = load_baseline_config(args.config)
    split = _split_for(cfg)

    if cfg.subset is not None and cfg.subset > MAX_SUBSET_SERIES:
        return _refuse(
            f"subset={cfg.subset} exceeds cap {MAX_SUBSET_SERIES}. Declare + approve a large run."
        )
    if cfg.subset is None and not args.allow_full:
        return _refuse(
            "subset is ALL but --allow-full not given. Declare a small subset or pass --allow-full."
        )

    declare(cfg, split)

    if args.dry_run or not args.confirm:
        print("\n[dry-run] Declaration only. Pass --confirm to execute.")
        return 0

    processed = Path(cfg.processed_dir)
    if not (processed / "dynamic.parquet").exists():
        return _refuse(
            f"no processed data at {processed}. Run scripts/make_synthetic.py (or ingest real M5) first."
        )

    result = run(cfg, split)
    md, js = write_reports(result)
    print("\n--- RESULT (mean over seeds) ---")
    for k, v in result["metrics_aggregated"].items():
        print(f"  {k:12} {v['mean']:.5f}  (± {v['stdev']:.5f})")
    print(f"\nReports: {md.relative_to(REPO)} , {js.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
