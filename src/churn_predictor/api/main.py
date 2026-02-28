"""FastAPI prediction service for customer churn."""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from churn_predictor import __version__
from churn_predictor.features.engineering import FeatureEngineer
from churn_predictor.models.trainer import ChurnTrainer

logger = logging.getLogger(__name__)

_THRESHOLD: float = 0.5
_ENGINEER_FILENAME: str = "churn_engineer.joblib"

_VALID_CONTRACTS: frozenset[str] = frozenset(
    {"Month-to-month", "One year", "Two year"}
)
_VALID_INTERNET: frozenset[str] = frozenset({"DSL", "Fiber optic", "No"})


# ---------------------------------------------------------------------------
# Pydantic v2 schemas
# ---------------------------------------------------------------------------


class CustomerFeatures(BaseModel):
    """Input features describing a single customer."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "tenure": 12,
                "monthly_charges": 79.5,
                "total_charges": 954.0,
                "contract": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber optic",
            }
        },
    )

    tenure: float = Field(..., ge=0.0, description="Months as a customer")
    monthly_charges: float = Field(..., ge=0.0, description="Current monthly bill (USD)")
    total_charges: float = Field(..., ge=0.0, description="Cumulative charges to date (USD)")
    contract: str = Field(
        ..., description="Month-to-month | One year | Two year"
    )
    payment_method: str = Field(..., description="Customer payment method")
    internet_service: str = Field(..., description="DSL | Fiber optic | No")

    @field_validator("contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        """Reject unrecognised contract types."""
        if v not in _VALID_CONTRACTS:
            raise ValueError(
                f"contract must be one of {sorted(_VALID_CONTRACTS)}, got '{v}'"
            )
        return v

    @field_validator("internet_service")
    @classmethod
    def validate_internet_service(cls, v: str) -> str:
        """Reject unrecognised internet service types."""
        if v not in _VALID_INTERNET:
            raise ValueError(
                f"internet_service must be one of {sorted(_VALID_INTERNET)}, got '{v}'"
            )
        return v


class PredictionResponse(BaseModel):
    """Churn prediction for a single customer."""

    model_config = ConfigDict(frozen=True)

    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Estimated churn probability [0, 1]"
    )
    churn_prediction: bool = Field(
        ..., description="True when churn_probability >= threshold"
    )
    threshold: float = Field(
        default=_THRESHOLD, description="Decision threshold applied"
    )


class BatchRequest(BaseModel):
    """Request body for bulk scoring."""

    customers: list[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Customers to score (1–1000 per request)",
    )


class BatchResponse(BaseModel):
    """Batch prediction output."""

    predictions: list[PredictionResponse]
    count: int = Field(..., description="Number of predictions returned")


class HealthResponse(BaseModel):
    """API liveness and readiness status."""

    status: str = Field(..., description="'ok' when model is loaded, 'degraded' otherwise")
    model_loaded: bool
    version: str


# ---------------------------------------------------------------------------
# Application state and lifespan
# ---------------------------------------------------------------------------

_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the trained model and fitted engineer on startup.

    Both artefacts are expected in ``CHURN_MODEL_DIR`` (default: ``models/``):

    - ``churn_model.joblib`` — written by :meth:`~ChurnTrainer.save_model`.
    - ``churn_engineer.joblib`` — written by :func:`joblib.dump` at train time.

    If either file is missing the API starts in **degraded** mode and returns
    HTTP 503 on prediction endpoints.
    """
    model_dir = Path(os.environ.get("CHURN_MODEL_DIR", "models"))
    engineer_path = model_dir / _ENGINEER_FILENAME

    loaded = False
    try:
        model = ChurnTrainer.load_model()
        engineer: FeatureEngineer = joblib.load(engineer_path)
        _state["model"] = model
        _state["engineer"] = engineer
        loaded = True
        logger.info("Model and engineer loaded from %s", model_dir)
    except FileNotFoundError as exc:
        logger.warning("Model artefacts not found — API in degraded mode: %s", exc)
        _state["model"] = None
        _state["engineer"] = None

    _state["model_loaded"] = loaded
    yield
    _state.clear()
    logger.info("App shutdown — state cleared.")


app = FastAPI(
    title="Churn Predictor API",
    description="Real-time customer churn prediction powered by XGBoost.",
    version=__version__,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_model_and_engineer() -> tuple[ChurnTrainer, FeatureEngineer]:
    """Return loaded model and engineer or raise HTTP 503."""
    model = _state.get("model")
    engineer = _state.get("engineer")
    if model is None or engineer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Train and save a model first.",
        )
    return model, engineer  # type: ignore[return-value]


def _score(rows: pd.DataFrame, threshold: float = _THRESHOLD) -> list[PredictionResponse]:
    """Run inference on *rows* and return a prediction per row.

    Args:
        rows: Raw customer feature DataFrame (one row per customer).
        threshold: Decision boundary applied to churn probability.

    Returns:
        One :class:`PredictionResponse` per input row.
    """
    model, engineer = _get_model_and_engineer()
    transformed = engineer.transform(rows)
    probs: list[float] = model.predict_proba(transformed).tolist()
    return [
        PredictionResponse(
            churn_probability=p,
            churn_prediction=p >= threshold,
            threshold=threshold,
        )
        for p in probs
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness check",
    tags=["ops"],
)
def health() -> HealthResponse:
    """Return API version, status, and whether a trained model is loaded."""
    loaded = bool(_state.get("model_loaded", False))
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        version=__version__,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict churn for a single customer",
    tags=["prediction"],
)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Score one customer and return their churn probability and decision.

    Returns HTTP 503 when no trained model is loaded.
    """
    row = pd.DataFrame([customer.model_dump()])
    return _score(row)[0]


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict churn for up to 1 000 customers",
    tags=["prediction"],
)
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Score a list of customers in a single API call.

    Returns HTTP 503 when no trained model is loaded.
    """
    rows = pd.DataFrame([c.model_dump() for c in request.customers])
    predictions = _score(rows)
    return BatchResponse(predictions=predictions, count=len(predictions))
