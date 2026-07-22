.PHONY: install format lint typecheck clean

PYTHON ?= .venv/bin/python
RUFF ?= .venv/bin/ruff
MYPY ?= .venv/bin/mypy

install:
	uv venv .venv --python 3.11
	uv pip install -e packages/mwrl -e packages/mwrl-maxsat -e packages/mwrl-circuits -e packages/mwrl-suzuki --python .venv/bin/python

format:
	$(RUFF) format .
	$(RUFF) check --fix .

lint:
	$(RUFF) check .

typecheck:
	$(MYPY) packages/mwrl/src packages/mwrl-maxsat/src packages/mwrl-circuits/src packages/mwrl-suzuki/src

clean:
	rm -rf runs .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
