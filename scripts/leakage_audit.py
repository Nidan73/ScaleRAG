#!/usr/bin/env python3
"""Leakage / split-integrity audit. Backs the `/leakage-audit` skill.

Given a split specification (JSON or CLI), run the invariants in
``scalerag.leakage`` and FAIL LOUDLY on any violation. Writes a Markdown
and a JSON report to ``reports/`` and exits non-zero if any check fails.

Example:
    uv run python scripts/leakage_audit.py --demo
    uv run python scripts/leakage_audit.py --spec configs/split_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scalerag import leakage  # noqa: E402


def run_checks(spec: dict) -> list[dict]:
    """Run each declared check, capturing pass/fail without stopping early."""
    checks: list[dict] = []

    def record(name: str, fn) -> None:
        try:
            fn()
            checks.append({"name": name, "ok": True, "detail": "ok"})
        except leakage.LeakageViolation as exc:
            checks.append({"name": name, "ok": False, "detail": str(exc)})

    if "split" in spec:
        s = spec["split"]
        record(
            "chronological_split",
            lambda: leakage.assert_chronological_split(
                s["train_end"], s["val_end"], s["test_start"]
            ),
        )
    if "retrieval" in spec:
        r = spec["retrieval"]
        record(
            "retrieval_horizon",
            lambda: leakage.assert_retrieval_horizon(r["t_r"], r["horizon"], r["origin"]),
        )
    if "windows" in spec:
        w = spec["windows"]
        record(
            "window_overlap",
            lambda: leakage.assert_no_window_overlap(
                [tuple(x) for x in w["train"]], [tuple(x) for x in w["eval"]]
            ),
        )
    if "features" in spec:
        f = spec["features"]
        record(
            "target_not_in_features",
            lambda: leakage.assert_target_not_in_features(f["columns"], f["target"]),
        )
    return checks


DEMO_INVALID_SPEC = {
    # Deliberately invalid: val_end precedes train_end (rule 1 violation) and a
    # retrieval window reaches past the forecast origin (rule 3 violation).
    "split": {"train_end": 100, "val_end": 90, "test_start": 120},
    "retrieval": {"t_r": 95, "horizon": 28, "origin": 110},
    "windows": {"train": [[0, 100]], "eval": [[95, 120]]},
    "features": {"columns": ["price", "sales"], "target": "sales"},
}


def write_reports(spec: dict, checks: list[dict]) -> tuple[Path, Path]:
    reports = REPO / "reports"
    reports.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    passed = all(c["ok"] for c in checks)

    json_path = reports / f"leakage-audit-{ts}.json"
    json_path.write_text(json.dumps({"pass": passed, "spec": spec, "checks": checks}, indent=2))

    lines = [
        "# Leakage Audit",
        "",
        f"- Timestamp: {ts}",
        f"- Result: {'PASS' if passed else 'FAIL'}",
        "",
        "| Check | Result | Detail |",
        "|-------|--------|--------|",
    ]
    for c in checks:
        lines.append(f"| {c['name']} | {'PASS' if c['ok'] else 'FAIL'} | {c['detail']} |")
    md_path = reports / f"leakage-audit-{ts}.md"
    md_path.write_text("\n".join(lines) + "\n")
    return md_path, json_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, help="JSON split specification.")
    ap.add_argument("--demo", action="store_true", help="Run the invalid-split demo.")
    ap.add_argument("--no-report", action="store_true", help="Skip writing report files.")
    args = ap.parse_args()

    if args.demo:
        spec = DEMO_INVALID_SPEC
    elif args.spec:
        spec = json.loads(args.spec.read_text())
    else:
        ap.error("provide --spec PATH or --demo")

    checks = run_checks(spec)
    passed = all(c["ok"] for c in checks)

    for c in checks:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    if not args.no_report:
        md, js = write_reports(spec, checks)
        print(f"\nReports: {md.relative_to(REPO)} , {js.relative_to(REPO)}")
    print("RESULT:", "PASS" if passed else "FAIL — leakage detected")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
