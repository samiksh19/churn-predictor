.PHONY: install test lint format typecheck run clean all

all: install lint typecheck test

install:
	pip install -e ".[dev]"

test:
	pytest

test-fast:
	pytest -x -q --no-cov

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy src tests

run:
	uvicorn churn_predictor.api.main:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov dist build
