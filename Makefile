.PHONY: install install-dev test lint format bench example example-dev clean

PY ?= python

install:
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check agent_tool_router tests examples benchmarks

format:
	$(PY) -m ruff format agent_tool_router tests examples benchmarks

bench:
	$(PY) benchmarks/run.py

example:
	$(PY) examples/customer_support_bot.py

example-dev:
	$(PY) examples/dev_assistant.py

clean:
	rm -rf build dist .pytest_cache .ruff_cache *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
