"""Shared pytest fixtures for the churn-predictor test suite.

Session-scoped fixtures generate synthetic data once and reuse it across
all test modules, keeping the suite fast while providing realistic signal.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from churn_predictor.data.loader import clean_data
from churn_predictor.features.engineering import FeatureEngineer
from churn_predictor.models.trainer import ChurnTrainer

# Feature columns passed to the model (excludes customerID and target).
_FEATURE_COLS: list[str] = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract",
    "payment_method",
    "internet_service",
]


@pytest.fixture(scope="session")
def synthetic_raw_df() -> pd.DataFrame:
    """400-row synthetic churn DataFrame with a realistic churn signal.

    Schema mirrors the Telco churn CSV (capitalised column names, string
    TotalCharges, Yes/No Churn) so it exercises the full ``clean_data``
    path.  Churn probability is driven by a logistic function of tenure,
    monthly charges, and contract type, giving the model genuine signal
    to learn from.

    Returns:
        Raw DataFrame ready to be passed to :func:`~churn_predictor.data.loader.clean_data`.
    """
    rng = np.random.default_rng(42)
    n = 400

    tenure = rng.uniform(1, 72, n)
    monthly_charges = rng.uniform(20, 120, n)
    total_charges = (tenure * monthly_charges + rng.normal(0, 50, n)).clip(100)

    contracts = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.55, 0.25, 0.20],
    )
    payment_methods = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n,
    )
    internet_service = rng.choice(
        ["Fiber optic", "DSL", "No"],
        size=n,
        p=[0.44, 0.34, 0.22],
    )

    # Logistic churn signal: short tenure + M2M contract → higher risk.
    logit = (
        -0.8
        - 0.05 * (tenure / 10)
        + 0.03 * (monthly_charges / 10)
        + 0.80 * (contracts == "Month-to-month").astype(float)
        - 0.50 * (contracts == "Two year").astype(float)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, churn_prob, n)

    return pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n)],
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            # Store as string to simulate raw CSV (exercises pd.to_numeric path).
            "TotalCharges": total_charges.round(2).astype(str),
            "Contract": contracts,
            "PaymentMethod": payment_methods,
            "InternetService": internet_service,
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )


@pytest.fixture(scope="session")
def tiny_Xy() -> tuple[pd.DataFrame, pd.Series]:
    """Tiny pre-encoded feature matrix for fast trainer unit tests.

    Uses 200 rows with a seeded random signal so tests are deterministic.
    The data is already numerically encoded — suitable for passing directly
    to :class:`~churn_predictor.models.trainer.ChurnTrainer`.

    Returns:
        ``(X, y)`` tuple with 200 samples and 6 numeric features.
    """

    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame(
        {
            "tenure": rng.uniform(0, 72, n),
            "monthly_charges": rng.uniform(20, 120, n),
            "total_charges": rng.uniform(100, 8000, n),
            "contract": rng.integers(0, 3, n).astype(float),
            "payment_method": rng.integers(0, 4, n).astype(float),
            "internet_service": rng.integers(0, 3, n).astype(float),
        }
    )
    y = pd.Series(rng.integers(0, 2, n), name="churn")
    return X, y


@pytest.fixture(scope="session")
def fitted_trainer_session(
    tiny_Xy: tuple[pd.DataFrame, pd.Series],
) -> ChurnTrainer:
    """Session-scoped fitted :class:`~churn_predictor.models.trainer.ChurnTrainer`.

    Trained once per session on :func:`tiny_Xy` and shared across all
    read-only model tests to avoid redundant training overhead.

    Returns:
        A fitted :class:`~churn_predictor.models.trainer.ChurnTrainer`.
    """
    X, y = tiny_Xy
    trainer = ChurnTrainer(n_splits=2, n_estimators=50)
    trainer.train(X, y)
    return trainer


@pytest.fixture(scope="session")
def engineered_splits(
    synthetic_raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Run clean → feature-engineer → train/test split on synthetic data.

    The :class:`~churn_predictor.features.engineering.FeatureEngineer`
    is fitted on the training portion only to avoid data leakage.

    Args:
        synthetic_raw_df: Raw fixture produced by :func:`synthetic_raw_df`.

    Returns:
        ``(X_train, X_test, y_train, y_test)`` — feature matrices and
        target series ready for :class:`~churn_predictor.models.trainer.ChurnTrainer`.
    """
    df = clean_data(synthetic_raw_df)

    X_raw = df[_FEATURE_COLS]
    y = df["churn"].reset_index(drop=True)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    engineer = FeatureEngineer()
    X_train = engineer.fit_transform(X_train_raw.reset_index(drop=True))
    X_test = engineer.transform(X_test_raw.reset_index(drop=True))

    return X_train, X_test, y_train, y_test
