# Reproducibility Policy

Every result must be reproducible from recorded inputs. This policy operationalises
`research-rules.md` rules 6 and 10.

## Record for every experiment

- **Seed(s)** — set via `graphroute_ts.reproducibility.set_seed`.
- **Package versions** — the committed `uv.lock` is authoritative; also capture
  `torch.__version__` and `torch.version.cuda`.
- **Config file** — the exact YAML (validated by `graphroute_ts.config`).
- **Git commit** — `git rev-parse HEAD`; the tree must be clean (no uncommitted diff).
- **Runtime** — wall-clock duration and peak memory / VRAM.
- **Hardware** — GPU name + compute capability, driver, CPU, RAM.

`graphroute_ts.reproducibility.RunContext` captures a best-effort fingerprint
(python, platform, git commit, torch, CUDA, GPU) — persist it alongside metrics.

## Determinism

- Seed Python, NumPy, and Torch (CPU + CUDA).
- Request deterministic cuDNN where practical; note that bit-exactness is **not**
  guaranteed across hardware/driver versions — record the environment, don't assume.
- No nondeterministic ordering (e.g. set iteration) in data pipelines.

## Metrics & claims

- **No single-run headline numbers.** Report central tendency and dispersion across
  multiple seeds (rule 6).
- Statistical claims require a named test or confidence interval (rule 8).
- Aggregation level (per-series vs pooled) and weighting are stated explicitly.

## Environment reproducibility

- `uv.lock` is committed and never hand-edited. Recreate with
  `uv sync --extra ml --extra retrieval --extra tsfm`.
- Python is pinned by `.python-version`; the PyTorch CUDA index is pinned in
  `pyproject.toml`.

## Data reproducibility

- Datasets are not committed; record dataset name, version/snapshot, and the
  preprocessing commit that produced `data/processed/*`.
- All fitted transforms are fit on **train only** and persisted for reuse.

## Prohibited

Editing evaluation code to improve a number (rule 9); swapping a failed model
silently (rule 7); reporting results from an unclean/uncommitted tree.
