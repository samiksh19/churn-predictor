# CLAUDE.md — Churn Predictor

## Project Overview

Academic project (NEU Big Data) — an end-to-end customer churn prediction
pipeline with a FastAPI inference service backed by XGBoost.

**Stack:** Python 3.11+, pandas, scikit-learn, XGBoost, FastAPI, Pydantic, pytest

---

## Coding Standards

### Type hints
- All functions and methods must have complete type annotations.
- `mypy --strict` must pass with zero errors on `src/`.
- Never use `Any` unless there is no alternative; add a comment explaining why.

### Logging
- Never use `print()`. Use Python's `logging` module everywhere.
- In library modules (`data/`, `features/`, `models/`) get a module-level logger:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  ```
- In `api/main.py` configure logging at app startup; do not call
  `logging.basicConfig` inside library code.

### General style
- Line length: 88 characters (ruff default).
- Imports sorted by ruff (`I` rule set).
- `ruff check` and `ruff format` must pass before committing.

---

## Architecture

### Layer order (strict — no reverse imports)
```
api  →  models  →  features  →  data
```
Upper layers may import from lower ones; lower layers must never import from
upper ones.

### Data validation
- Use **Pydantic models** at all API boundaries (request bodies, response
  schemas, and app config).
- Do not pass raw dicts across module boundaries — use typed dataclasses or
  Pydantic models.

### No hardcoded values
- File paths, model paths, and host/port config belong in environment variables
  or a `.env` file (never committed).
- Use `pydantic-settings` `BaseSettings` for typed env-var config if the
  config surface grows.

---

## Tests

- Framework: **pytest** with fixtures. No `unittest.TestCase` classes.
- Scope: **happy-path coverage** — test the main success case for each public
  function/endpoint. Edge cases only where a bug has already been found.
- Mirror `src/` layout under `tests/`:
  ```
  tests/data/        → tests for churn_predictor.data
  tests/features/    → tests for churn_predictor.features
  tests/models/      → tests for churn_predictor.models
  tests/api/         → tests for churn_predictor.api
  ```
- Run with `make test` (includes coverage report).
- Do not mock internal functions — prefer `tmp_path` fixtures and small
  real DataFrames.

---

## Common Commands

```bash
make install     # pip install -e ".[dev]"
make test        # pytest with coverage
make lint        # ruff check src tests
make format      # ruff format + ruff --fix
make typecheck   # mypy src
make run         # uvicorn on :8000
```

---

## What Claude Should Never Do

- Use `print()` — always `logger.*`.
- Hardcode file paths or model paths.
- Skip type annotations on new functions.
- Import upward through the layer stack (e.g., `data` importing from `models`).
- Create new files when editing an existing one is sufficient.
- Add docstrings, comments, or type hints to code that was not changed.
