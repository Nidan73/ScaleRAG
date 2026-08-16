.PHONY: help sync fmt lint type test leakage unit check verify synth figures clean

help:
	@echo "sync      install the environment from the lockfile"
	@echo "check     fmt + lint + type + unit + leakage  (the gate)"
	@echo "test      full pytest suite"
	@echo "unit      unit tests only"
	@echo "leakage   temporal-leakage / split-integrity guards"
	@echo "verify    environment + GPU report"
	@echo "synth     build the offline synthetic M5 fixture"
	@echo "figures   regenerate the paper figures"

sync:
	uv sync --extra ml --extra retrieval --extra tsfm --extra viz

fmt:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

lint:
	uv run ruff format --check src tests scripts
	uv run ruff check src tests scripts

type:
	uv run mypy

unit:
	uv run pytest -m unit -q

leakage:
	uv run pytest -m leakage -q

test:
	uv run pytest -q

check: lint type unit leakage
	@echo "OK"

verify:
	uv run python scripts/environment_check.py

synth:
	uv run python scripts/make_synthetic.py --days 1941 \
		--raw data/raw_synth --processed data/processed

figures:
	uv run python scripts/make_phase11a_figures.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
