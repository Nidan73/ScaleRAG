#!/usr/bin/env python3
"""Environment health check for GraphRoute-TS. READ-ONLY: never modifies the env.

Backs the `/environment-check` skill. Emits a pass/fail table and exits non-zero
if any hard check fails. Optional `--json` for machine-readable output.

Run with: `uv run python scripts/environment_check.py`
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REQUIRED_DIRS = [
    "src/scalerag",
    "configs",
    "data/raw",
    "data/interim",
    "data/processed",
    "tests/unit",
    "tests/leakage",
    "artifacts",
    "logs",
    "reports",
]


def _check(name: str, ok: bool, detail: str, hard: bool = True) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail, "hard": hard}


def gather() -> list[dict]:
    results: list[dict] = []

    # Active virtual environment
    venv = os.environ.get("VIRTUAL_ENV")
    in_project_venv = bool(venv) and Path(venv).resolve() == (REPO / ".venv").resolve()
    results.append(_check("active venv", in_project_venv, venv or "no VIRTUAL_ENV set", hard=False))

    # Python version
    py = sys.version_info
    results.append(_check("python 3.11", py[:2] == (3, 11), f"{py.major}.{py.minor}.{py.micro}"))

    # Dependency lock present & consistency (best-effort, non-mutating)
    lock = REPO / "uv.lock"
    if shutil.which("uv") and lock.exists():
        proc = subprocess.run(["uv", "lock", "--check"], cwd=REPO, capture_output=True, text=True)
        results.append(
            _check(
                "uv.lock consistent",
                proc.returncode == 0,
                "in sync" if proc.returncode == 0 else proc.stderr.strip()[:200],
                hard=False,
            )
        )
    else:
        results.append(_check("uv.lock present", lock.exists(), str(lock), hard=False))

    # PyTorch + CUDA + GPU
    try:
        import torch

        results.append(_check("torch import", True, torch.__version__))
        cuda_ok = torch.cuda.is_available()
        results.append(
            _check(
                "cuda available",
                cuda_ok,
                f"torch.version.cuda={getattr(torch.version, 'cuda', None)}",
                hard=False,
            )
        )
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            gib = props.total_memory / (1024**3)
            results.append(
                _check(
                    "gpu",
                    True,
                    f"{props.name} | sm_{props.major}{props.minor} | {gib:.1f} GiB",
                    hard=False,
                )
            )
    except ImportError as exc:
        results.append(_check("torch import", False, f"not installed ({exc})", hard=False))

    # Required directories
    missing = [d for d in REQUIRED_DIRS if not (REPO / d).is_dir()]
    results.append(
        _check("required dirs", not missing, "all present" if not missing else f"missing {missing}")
    )

    # Git working tree
    if shutil.which("git") and (REPO / ".git").exists():
        proc = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True
        )
        dirty = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        results.append(
            _check("git tree", True, "clean" if not dirty else f"{len(dirty)} changes", hard=False)
        )
    else:
        results.append(_check("git repo", False, "not a git repo", hard=False))

    # Disk space on repo filesystem
    usage = shutil.disk_usage(REPO)
    free_gib = usage.free / (1024**3)
    results.append(_check("disk space", free_gib > 20, f"{free_gib:.0f} GiB free", hard=False))

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="Emit JSON.")
    args = ap.parse_args()

    results = gather()
    hard_fail = any(not r["ok"] and r["hard"] for r in results)

    if args.json:
        print(json.dumps({"pass": not hard_fail, "checks": results}, indent=2))
    else:
        for r in results:
            mark = "PASS" if r["ok"] else ("FAIL" if r["hard"] else "warn")
            print(f"[{mark:>4}] {r['name']:<20} {r['detail']}")
        print("\nRESULT:", "PASS" if not hard_fail else "FAIL (hard check failed)")

    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
