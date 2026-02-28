# Churn Predictor

End-to-end customer churn prediction pipeline with a FastAPI inference service.
Built with Python 3.11+, XGBoost, scikit-learn, and FastAPI.

> **Academic project** — NEU Big Data (SEM 3)

---

## Project Structure

```
churn-predictor/
├── src/churn_predictor/
│   ├── data/          # CSV loading & cleaning
│   ├── features/      # sklearn Pipeline + custom transformers
│   ├── models/        # XGBoost trainer with cross-validation
│   └── api/           # FastAPI prediction service
├── tests/             # pytest suite mirroring src/
├── pyproject.toml
└── Makefile
```

---

## Installation

**Requirements:** Python 3.11+

```bash
# Clone and install in editable mode with dev dependencies
pip install -e ".[dev]"
```

Or use the Makefile:

```bash
make install
```

---

## Usage

### 1. Prepare data

Place your churn CSV in `data/raw/churn.csv`.
The file must contain these columns (camelCase or snake_case are both supported):

| Column | Description |
|---|---|
| `customerID` | Unique customer identifier |
| `tenure` | Months as a customer |
| `MonthlyCharges` | Current monthly bill |
| `TotalCharges` | Cumulative charges to date |
| `Contract` | `Month-to-month`, `One year`, or `Two year` |
| `PaymentMethod` | Payment type |
| `InternetService` | `DSL`, `Fiber optic`, or `No` |
| `Churn` | `Yes` / `No` target label |

Override the data directory with an environment variable:

```bash
export CHURN_RAW_DIR=/path/to/your/data
```

### 2. Train a model

```python
from churn_predictor.data.loader import load_raw_data, clean_data
from churn_predictor.features.engineering import FeatureEngineer
from churn_predictor.models.trainer import ChurnTrainer
from sklearn.model_selection import train_test_split

# Load and clean
df = clean_data(load_raw_data())

feature_cols = [
    "tenure", "monthly_charges", "total_charges",
    "contract", "payment_method", "internet_service",
]
X, y = df[feature_cols], df["churn"]

# Engineer features (fit on train only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
engineer = FeatureEngineer()
X_train = engineer.fit_transform(X_train)
X_test  = engineer.transform(X_test)

# Train with 5-fold cross-validation
trainer = ChurnTrainer(n_splits=5, n_estimators=200)
cv_metrics = trainer.train(X_train, y_train)
print(cv_metrics)
# {'cv_roc_auc_mean': 0.84, 'cv_roc_auc_std': 0.02}

# Evaluate on held-out test set
metrics = trainer.evaluate(X_test, y_test)
print(metrics.as_dict())
# {'precision': 0.81, 'recall': 0.74, 'f1': 0.77, 'roc_auc': 0.86, ...}

# Save the model
trainer.save_model("churn_v1.joblib")
```

Override the model directory:

```bash
export CHURN_MODEL_DIR=/path/to/models
```

### 3. Start the API

```bash
make run
# or
uvicorn churn_predictor.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API loads the model from `CHURN_MODEL_DIR/churn_model.joblib` on startup.

#### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + model status |
| `POST` | `/predict` | Single customer prediction |
| `POST` | `/predict/batch` | Batch predictions |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

#### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "monthly_charges": 79.5,
    "total_charges": 954.0,
    "contract": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic"
  }'
```

```json
{
  "churn_probability": 0.713,
  "churn_prediction": true
}
```

#### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {"tenure": 48, "monthly_charges": 45.0, "total_charges": 2160.0,
       "contract": "Two year", "payment_method": "Bank transfer", "internet_service": "DSL"},
      {"tenure": 2, "monthly_charges": 95.0, "total_charges": 190.0,
       "contract": "Month-to-month", "payment_method": "Electronic check", "internet_service": "Fiber optic"}
    ]
  }'
```

---

## Development

```bash
make test        # pytest with coverage
make lint        # ruff check src tests
make format      # ruff format + auto-fix
make typecheck   # mypy --strict
make clean       # remove __pycache__, .coverage, dist
```

### Run just the end-to-end pipeline test

```bash
pytest tests/test_pipeline.py -s -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CHURN_RAW_DIR` | `data/raw` | Directory containing raw CSV files |
| `CHURN_MODEL_DIR` | `models` | Directory for saving/loading model artefacts |
