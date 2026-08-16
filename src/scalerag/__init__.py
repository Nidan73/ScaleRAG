"""ScaleRAG: scale-aware retrieval augmentation for time-series foundation models.

A frozen forecasting backbone is augmented with a parameter-free retrieval
branch. Four stages: scale-invariant context normalization, exact shape-space
k-NN under a horizon guard, closed-form scale restoration, and a convex fusion
whose weight is either a constant or a learned gate.

Stages 1-3 add no trainable parameters. The backbone is never updated.

Layout
------
``retrieval`` / ``retrieval_faiss`` / ``retrieval_gpu``
    Exact shape-space k-NN. The three backends return identical candidates;
    the GPU path is verified bit-for-bit against the NumPy reference.
``retrieval_forecast``
    Retrieved continuations to a forecast, with scale restoration.
``fusion``
    Convex blend of backbone and retrieval branch, plus the paired bootstrap.
``native``
    Standalone adapter for dense continuous panels (ETTm2).
``tsfm.chronos2``
    Frozen backbone wrapper.
``affine_probe`` / ``error_decomposition`` / ``regime``
    The three diagnostics: invariance and equivariance under a controlled
    affine shift, the shape/magnitude split of the residual, and the per-series
    regime profile of retrieval utility.
``leakage`` / ``splits``
    Chronological splitting and the horizon guard, enforced by tests.

The evaluation protocol these modules must satisfy is stated in
``docs/research-rules.md``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
