PY := .venv/bin/python
FCDA := .venv/bin/fcda

.PHONY: help setup smoke test lint data train report demo clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the 3.12 venv and install everything
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python -e ".[app,dev]"

smoke:  ## End-to-end pipeline on synthetic data, no network, <2 min
	$(PY) -m fcda.cli smoke

test:  ## Unit tests (leakage, splits, augmentation isolation, model shapes)
	.venv/bin/pytest tests/ -q

lint:  ## Ruff
	.venv/bin/ruff check src tests

data:  ## Fetch the smallest real tier
	$(PY) -m fcda.cli download --tier T1_tiny

train:  ## Progressive tiered training of all five hybrids
	$(PY) -m fcda.cli train --progressive

report:  ## Regenerate figures, tables and the results section
	$(PY) -m fcda.cli report

demo:  ## Launch the Streamlit app
	.venv/bin/streamlit run app/streamlit_app.py

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
