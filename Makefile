.PHONY: install test lint demo bench cost clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/

lint:
	ruff check .

# Eight tools, twenty queries, offline (HashingEncoder + the EchoBackend stub judge).
demo:
	python examples/customer_support_bot.py

# Routing and refusal accuracy on the bundled SYNTHETIC corpus (test half, offline).
bench:
	python benchmarks/run.py

# Illustrative cost arithmetic from stated assumptions (not a measurement).
cost:
	python benchmarks/cost_model.py

clean:
	rm -rf .pytest_cache .ruff_cache *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
