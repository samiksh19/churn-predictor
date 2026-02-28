"""Tests for churn_predictor.api.main — health, predict, batch, validation."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from churn_predictor.api.main import _THRESHOLD, app

# ---------------------------------------------------------------------------
# Shared payload
# ---------------------------------------------------------------------------

VALID_PAYLOAD: dict[str, object] = {
    "tenure": 12.0,
    "monthly_charges": 70.0,
    "total_charges": 840.0,
    "contract": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
}

BATCH_PAYLOAD: dict[str, object] = {
    "customers": [VALID_PAYLOAD, VALID_PAYLOAD, VALID_PAYLOAD],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """TestClient with no model loaded (degraded mode)."""
    return TestClient(app)


@pytest.fixture()
def client_with_model() -> Generator[TestClient, None, None]:
    """TestClient with a mock model and engineer injected via lifespan.

    Patches ``ChurnTrainer.load_model`` and ``joblib.load`` before the ASGI
    lifespan runs so ``_state`` is populated with mock objects.
    """
    mock_model = MagicMock()
    # Return a probability proportional to the number of rows
    mock_model.predict_proba.side_effect = lambda df: np.full(len(df), 0.72)

    mock_engineer = MagicMock()
    # Pass the DataFrame through unchanged — the model is also mocked
    mock_engineer.transform.side_effect = lambda df: df

    with (
        patch("churn_predictor.api.main.ChurnTrainer") as mock_cls,
        patch("churn_predictor.api.main.joblib.load", return_value=mock_engineer),
    ):
        mock_cls.load_model.return_value = mock_model
        with TestClient(app) as client:
            yield client


@pytest.fixture()
def client_low_prob() -> Generator[TestClient, None, None]:
    """TestClient whose mock model always returns probability 0.2 (no churn)."""
    mock_model = MagicMock()
    mock_model.predict_proba.side_effect = lambda df: np.full(len(df), 0.2)

    mock_engineer = MagicMock()
    mock_engineer.transform.side_effect = lambda df: df

    with (
        patch("churn_predictor.api.main.ChurnTrainer") as mock_cls,
        patch("churn_predictor.api.main.joblib.load", return_value=mock_engineer),
    ):
        mock_cls.load_model.return_value = mock_model
        with TestClient(app) as client:
            yield client


# ---------------------------------------------------------------------------
# /health — degraded mode (no model)
# ---------------------------------------------------------------------------


def test_health_returns_200(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_health_status_degraded_without_model(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_health_has_version_field(client: TestClient) -> None:
    body = client.get("/health").json()
    assert "version" in body
    assert isinstance(body["version"], str)


# ---------------------------------------------------------------------------
# /health — with model loaded
# ---------------------------------------------------------------------------


def test_health_status_ok_with_model(client_with_model: TestClient) -> None:
    body = client_with_model.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_version_matches_package(client_with_model: TestClient) -> None:
    from churn_predictor import __version__

    body = client_with_model.get("/health").json()
    assert body["version"] == __version__


# ---------------------------------------------------------------------------
# /predict — degraded mode (503)
# ---------------------------------------------------------------------------


def test_predict_503_without_model(client: TestClient) -> None:
    assert client.post("/predict", json=VALID_PAYLOAD).status_code == 503


def test_predict_batch_503_without_model(client: TestClient) -> None:
    assert client.post("/predict/batch", json=BATCH_PAYLOAD).status_code == 503


# ---------------------------------------------------------------------------
# /predict — validation errors (422) — no model required
# ---------------------------------------------------------------------------


def test_predict_invalid_payload_type_returns_422(client: TestClient) -> None:
    response = client.post("/predict", json={"tenure": "not-a-number"})
    assert response.status_code == 422


def test_predict_negative_tenure_returns_422(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "tenure": -1.0}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_negative_monthly_charges_returns_422(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "monthly_charges": -5.0}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_invalid_contract_returns_422(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "contract": "Biannual"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_invalid_internet_service_returns_422(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "internet_service": "5G"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_missing_required_field_returns_422(client: TestClient) -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_batch_empty_customers_returns_422(client: TestClient) -> None:
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


# ---------------------------------------------------------------------------
# /predict — happy path (with model)
# ---------------------------------------------------------------------------


def test_predict_returns_200(client_with_model: TestClient) -> None:
    assert client_with_model.post("/predict", json=VALID_PAYLOAD).status_code == 200


def test_predict_response_has_required_fields(client_with_model: TestClient) -> None:
    body = client_with_model.post("/predict", json=VALID_PAYLOAD).json()
    assert "churn_probability" in body
    assert "churn_prediction" in body
    assert "threshold" in body


def test_predict_probability_in_unit_interval(client_with_model: TestClient) -> None:
    body = client_with_model.post("/predict", json=VALID_PAYLOAD).json()
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_high_prob_yields_churn_true(client_with_model: TestClient) -> None:
    # mock returns 0.72 which is >= 0.5
    body = client_with_model.post("/predict", json=VALID_PAYLOAD).json()
    assert body["churn_prediction"] is True


def test_predict_low_prob_yields_churn_false(client_low_prob: TestClient) -> None:
    # mock returns 0.2 which is < 0.5
    body = client_low_prob.post("/predict", json=VALID_PAYLOAD).json()
    assert body["churn_prediction"] is False


def test_predict_threshold_matches_default(client_with_model: TestClient) -> None:
    body = client_with_model.post("/predict", json=VALID_PAYLOAD).json()
    assert body["threshold"] == _THRESHOLD


def test_predict_strips_whitespace_from_strings(client_with_model: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "contract": "  Month-to-month  "}
    # Should pass validation after stripping (model_config str_strip_whitespace=True)
    assert client_with_model.post("/predict", json=payload).status_code == 200


# ---------------------------------------------------------------------------
# /predict/batch — happy path (with model)
# ---------------------------------------------------------------------------


def test_predict_batch_returns_200(client_with_model: TestClient) -> None:
    assert client_with_model.post("/predict/batch", json=BATCH_PAYLOAD).status_code == 200


def test_predict_batch_count_matches_input(client_with_model: TestClient) -> None:
    body = client_with_model.post("/predict/batch", json=BATCH_PAYLOAD).json()
    assert body["count"] == len(BATCH_PAYLOAD["customers"])  # type: ignore[arg-type]
    assert len(body["predictions"]) == body["count"]


def test_predict_batch_each_prediction_has_fields(client_with_model: TestClient) -> None:
    body = client_with_model.post("/predict/batch", json=BATCH_PAYLOAD).json()
    for pred in body["predictions"]:
        assert "churn_probability" in pred
        assert "churn_prediction" in pred
        assert "threshold" in pred


def test_predict_batch_single_customer(client_with_model: TestClient) -> None:
    payload = {"customers": [VALID_PAYLOAD]}
    body = client_with_model.post("/predict/batch", json=payload).json()
    assert body["count"] == 1


def test_predict_batch_all_contracts_accepted(client_with_model: TestClient) -> None:
    contracts = ["Month-to-month", "One year", "Two year"]
    customers = [{**VALID_PAYLOAD, "contract": c} for c in contracts]
    body = client_with_model.post("/predict/batch", json={"customers": customers}).json()
    assert body["count"] == 3


def test_predict_batch_all_internet_services_accepted(
    client_with_model: TestClient,
) -> None:
    services = ["DSL", "Fiber optic", "No"]
    customers = [{**VALID_PAYLOAD, "internet_service": s} for s in services]
    body = client_with_model.post("/predict/batch", json={"customers": customers}).json()
    assert body["count"] == 3
