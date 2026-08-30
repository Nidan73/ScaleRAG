"""Reproducibility helpers: seeding and run-context capture.

Every experiment must record seeds, versions, config, git commit, runtime and
hardware (CLAUDE.md research rule 10). This module centralises seeding and a
best-effort environment fingerprint. It imports torch lazily so it stays usable
in environments where the ML stack is not installed.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs.

    ``deterministic`` requests deterministic cuDNN kernels where supported. It
    does not guarantee bit-exactness across hardware — record the seed and
    environment fingerprint alongside results.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class RunContext:
    """Best-effort fingerprint of an experiment run for the experiment log."""

    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    git_commit: str | None = field(default_factory=_git_commit)
    torch_version: str | None = None
    cuda_available: bool | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None

    def __post_init__(self) -> None:
        try:
            import torch

            self.torch_version = torch.__version__
            self.cuda_available = torch.cuda.is_available()
            self.cuda_version = getattr(torch.version, "cuda", None)
            if self.cuda_available:
                self.gpu_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
