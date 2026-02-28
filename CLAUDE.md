# CLAUDE.md — Churn Predictor

## Project Overview

Academic project (NEU Big Data) — an end-to-end customer churn prediction
pipeline with a FastAPI inference service backed by XGBoost.

**Stack:** Python 3.11+, pandas, scikit-learn, XGBoost, FastAPI, Pydantic, pytest

---

## Coding Standards

### Type hints
- All functions and methods must have complete type annotations.
- `mypy --strict` must pass with zero errors on **both** `src/` and `tests/`.
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
- Imports sorted by ruff (`I` rule set). Use `from collections.abc` not
  `from typing` for `Generator`, `AsyncGenerator`, etc. (ruff UP035).
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

### Environment variables
| Variable        | Default      | Used by                              |
|-----------------|--------------|--------------------------------------|
| `CHURN_RAW_DIR` | `data/raw/`  | `data/loader.py` — raw CSV location  |
| `CHURN_MODEL_DIR` | `models/`  | `models/trainer.py`, `api/main.py`   |

---

## Model Artefacts

Two files must always be **saved together** after training and **loaded
together** at API startup. Both live in `CHURN_MODEL_DIR`:

| File                    | Written by                        | Loaded by          |
|-------------------------|-----------------------------------|--------------------|
| `churn_model.joblib`    | `ChurnTrainer.save_model()`       | `ChurnTrainer.load_model()` |
| `churn_engineer.joblib` | `joblib.dump(engineer, path)`     | `joblib.load(path)` |

The API enters **degraded mode** (HTTP 503 on prediction endpoints) if either
file is missing at startup.

### Training vs inference — critical distinction
- **Training:** call `FeatureEngineer.fit_transform(X_train)` — fits scalers
  and encoders on training data, then transforms it.
- **Inference:** call `FeatureEngineer.transform(X)` — applies the already-
  fitted pipeline. **Never call `fit_transform()` on prediction data**; doing
  so re-fits the scaler on a single row, producing garbage output.

---

## API Conventions

### Validated enum fields on `CustomerFeatures`
```python
contract:         "Month-to-month" | "One year" | "Two year"
internet_service: "DSL" | "Fiber optic" | "No"
```
Adding a new categorical field requires updating both the Pydantic validator
and the feature engineering pipeline.

### Decision threshold
`_THRESHOLD = 0.5` in `api/main.py`. Returned in every `PredictionResponse`
so callers know which boundary was applied.

### Batch endpoint limits
`POST /predict/batch` accepts 1–1 000 customers per request (`min_length=1`,
`max_length=1000` on `BatchRequest.customers`).

---

## Tests

- Framework: **pytest** with fixtures. No `unittest.TestCase` classes.
- Target: **branch coverage** — every conditional branch should be exercised.
  Current baseline: **102 tests, 99% coverage** (`make test`).
- Mirror `src/` layout under `tests/`:
  ```
  tests/data/        → tests for churn_predictor.data
  tests/features/    → tests for churn_predictor.features
  tests/models/      → tests for churn_predictor.models
  tests/api/         → tests for churn_predictor.api
  ```

### Shared fixtures (`tests/conftest.py`)
| Fixture                  | Scope   | Provides                                              |
|--------------------------|---------|-------------------------------------------------------|
| `synthetic_raw_df`       | session | 400-row raw DataFrame with camelCase columns & signal |
| `tiny_Xy`                | session | 200-row pre-encoded `(X, y)` for fast trainer tests   |
| `fitted_trainer_session` | session | Trained `ChurnTrainer` (n_splits=2, n_estimators=50)  |
| `engineered_splits`      | session | `(X_train, X_test, y_train, y_test)` after clean→engineer→split |

Use **session-scoped** fixtures for anything that trains a model or runs the
full pipeline — XGBoost training is slow and should not repeat per test.

### Path-dependent tests
Use `monkeypatch.setenv()` + `importlib.reload()` to override `CHURN_MODEL_DIR`
or `CHURN_RAW_DIR` and re-read the module-level `Path(os.environ.get(...))`:
```python
def test_something(tmp_path, monkeypatch):
    monkeypatch.setenv("CHURN_MODEL_DIR", str(tmp_path))
    import importlib
    import churn_predictor.models.trainer as m
    importlib.reload(m)
    # now m._MODEL_DIR points to tmp_path
```

### Mocking the FastAPI lifespan
`patch` is acceptable — and necessary — for injecting mock models before the
ASGI lifespan runs. Use `unittest.mock.patch` on `ChurnTrainer` and
`joblib.load`, then use `TestClient(app)` as a context manager:
```python
with (
    patch("churn_predictor.api.main.ChurnTrainer") as mock_cls,
    patch("churn_predictor.api.main.joblib.load", return_value=mock_engineer),
):
    mock_cls.load_model.return_value = mock_model
    with TestClient(app) as client:
        yield client
```
For all other code, prefer `tmp_path` fixtures and real DataFrames over mocks.

---

## Common Commands

```bash
make install     # pip install -e ".[dev]"
make test        # pytest with coverage
make lint        # ruff check src tests
make format      # ruff format + ruff --fix
make typecheck   # mypy src tests
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
- Call `engineer.fit_transform()` at inference — use `engineer.transform()`.
- Add a new categorical API field without updating the Pydantic validator
  **and** the feature engineering pipeline.
