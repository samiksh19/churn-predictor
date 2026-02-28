"""Tests for churn_predictor.features.engineering — transformers and pipeline."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn_predictor.features.engineering import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    CategoricalEncoder,
    DerivedFeatureTransformer,
    FeatureEngineer,
    NumericScaler,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Standard 3-row feature DataFrame covering all column groups."""
    return pd.DataFrame(
        {
            "tenure": [12.0, 24.0, 6.0],
            "monthly_charges": [70.0, 50.0, 90.0],
            "total_charges": [840.0, 1200.0, 540.0],
            "contract": ["Month-to-month", "One year", "Two year"],
            "payment_method": ["Electronic check", "Bank transfer", "Credit card"],
            "internet_service": ["Fiber optic", "DSL", "No"],
            "churn": [1, 0, 1],
        }
    )


@pytest.fixture()
def sample_df_with_support(sample_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_df.copy()
    df["support_calls"] = [3.0, 1.0, 5.0]
    return df


# ---------------------------------------------------------------------------
# DerivedFeatureTransformer
# ---------------------------------------------------------------------------


def test_derived_fit_returns_self(sample_df: pd.DataFrame) -> None:
    t = DerivedFeatureTransformer()
    result = t.fit(sample_df)
    assert result is t


def test_derived_adds_tenure_months(sample_df: pd.DataFrame) -> None:
    result = DerivedFeatureTransformer().fit_transform(sample_df)
    assert "tenure_months" in result.columns
    pd.testing.assert_series_equal(
        result["tenure_months"], sample_df["tenure"], check_names=False
    )


def test_derived_adds_avg_monthly_spend(sample_df: pd.DataFrame) -> None:
    result = DerivedFeatureTransformer().fit_transform(sample_df)
    assert "avg_monthly_spend" in result.columns
    expected = sample_df["total_charges"] / (sample_df["tenure"] + 1)
    pd.testing.assert_series_equal(
        result["avg_monthly_spend"], expected, check_names=False
    )


def test_derived_adds_support_calls_ratio_when_present(
    sample_df_with_support: pd.DataFrame,
) -> None:
    result = DerivedFeatureTransformer().fit_transform(sample_df_with_support)
    assert "support_calls_ratio" in result.columns
    expected = (
        sample_df_with_support["support_calls"]
        / (sample_df_with_support["tenure"] + 1)
    )
    pd.testing.assert_series_equal(
        result["support_calls_ratio"], expected, check_names=False
    )


def test_derived_skips_support_calls_ratio_when_absent(sample_df: pd.DataFrame) -> None:
    result = DerivedFeatureTransformer().fit_transform(sample_df)
    assert "support_calls_ratio" not in result.columns


def test_derived_no_tenure_column() -> None:
    """Branch: outer ``if 'tenure' in X.columns`` evaluates to False."""
    df = pd.DataFrame({"monthly_charges": [50.0, 70.0], "total_charges": [600.0, 840.0]})
    result = DerivedFeatureTransformer().fit_transform(df)
    assert "tenure_months" not in result.columns
    assert "avg_monthly_spend" not in result.columns
    # Original columns preserved
    assert list(result.columns) == ["monthly_charges", "total_charges"]


def test_derived_tenure_without_total_charges() -> None:
    """Branch: inner ``if 'total_charges' in X.columns`` evaluates to False."""
    df = pd.DataFrame({"tenure": [12.0, 24.0], "monthly_charges": [50.0, 70.0]})
    result = DerivedFeatureTransformer().fit_transform(df)
    assert "tenure_months" in result.columns
    assert "avg_monthly_spend" not in result.columns


def test_derived_does_not_mutate_input(sample_df: pd.DataFrame) -> None:
    original_cols = list(sample_df.columns)
    DerivedFeatureTransformer().fit_transform(sample_df)
    assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# CategoricalEncoder
# ---------------------------------------------------------------------------


def test_categorical_encoder_fit_returns_self(sample_df: pd.DataFrame) -> None:
    enc = CategoricalEncoder()
    assert enc.fit(sample_df) is enc


def test_categorical_encoder_produces_numeric(sample_df: pd.DataFrame) -> None:
    result = CategoricalEncoder().fit_transform(sample_df)
    for col in CATEGORICAL_COLS:
        assert pd.api.types.is_numeric_dtype(result[col]), f"{col} not numeric"


def test_categorical_encoder_is_deterministic(sample_df: pd.DataFrame) -> None:
    enc = CategoricalEncoder()
    enc.fit(sample_df)
    pd.testing.assert_frame_equal(enc.transform(sample_df), enc.transform(sample_df))


def test_categorical_encoder_skips_absent_fit_column() -> None:
    """Columns in ``cols`` absent from X are silently skipped during fit."""
    df = pd.DataFrame({"contract": ["Month-to-month", "One year"]})
    enc = CategoricalEncoder()
    enc.fit(df)
    assert "contract" in enc.encoders_
    assert "payment_method" not in enc.encoders_


def test_categorical_encoder_transform_skips_absent_column() -> None:
    """Branch: ``if col in X.columns`` is False during transform — col skipped."""
    train = pd.DataFrame(
        {
            "contract": ["Month-to-month", "One year", "Two year"],
            "payment_method": ["Electronic check", "Bank transfer", "Credit card"],
            "internet_service": ["Fiber optic", "DSL", "No"],
        }
    )
    enc = CategoricalEncoder()
    enc.fit(train)

    # Transform on DataFrame that is missing payment_method
    test_df = train.drop(columns=["payment_method"])
    result = enc.transform(test_df)

    assert "payment_method" not in result.columns
    assert pd.api.types.is_numeric_dtype(result["contract"])
    assert pd.api.types.is_numeric_dtype(result["internet_service"])


def test_categorical_encoder_does_not_mutate_input(sample_df: pd.DataFrame) -> None:
    original = sample_df["contract"].tolist()
    enc = CategoricalEncoder()
    enc.fit_transform(sample_df)
    assert sample_df["contract"].tolist() == original


# ---------------------------------------------------------------------------
# NumericScaler
# ---------------------------------------------------------------------------


def test_numeric_scaler_fit_returns_self(sample_df: pd.DataFrame) -> None:
    present = [c for c in NUMERIC_COLS if c in sample_df.columns]
    scaler = NumericScaler(cols=present)
    assert scaler.fit(sample_df) is scaler


def test_numeric_scaler_zero_mean(sample_df: pd.DataFrame) -> None:
    present = [c for c in NUMERIC_COLS if c in sample_df.columns]
    scaler = NumericScaler(cols=present)
    result = scaler.fit_transform(sample_df)
    for col in present:
        assert abs(result[col].mean()) < 1e-9, f"{col} mean not ~0"


def test_numeric_scaler_no_matching_columns() -> None:
    """Branch: ``if self.present_cols_:`` evaluates to False — nothing scaled."""
    df = pd.DataFrame({"some_other_col": [1.0, 2.0, 3.0]})
    scaler = NumericScaler(cols=["tenure", "monthly_charges"])
    result = scaler.fit_transform(df)
    # Data should pass through unchanged
    assert list(result.columns) == ["some_other_col"]
    pd.testing.assert_series_equal(result["some_other_col"], df["some_other_col"])


def test_numeric_scaler_does_not_mutate_input(sample_df: pd.DataFrame) -> None:
    original_tenure = sample_df["tenure"].tolist()
    present = [c for c in NUMERIC_COLS if c in sample_df.columns]
    NumericScaler(cols=present).fit_transform(sample_df)
    assert sample_df["tenure"].tolist() == original_tenure


# ---------------------------------------------------------------------------
# FeatureEngineer — pipeline façade
# ---------------------------------------------------------------------------


def test_fit_transform_returns_dataframe(sample_df: pd.DataFrame) -> None:
    result = FeatureEngineer().fit_transform(sample_df)
    assert isinstance(result, pd.DataFrame)


def test_fit_transform_adds_derived_columns(sample_df: pd.DataFrame) -> None:
    result = FeatureEngineer().fit_transform(sample_df)
    assert "tenure_months" in result.columns
    assert "avg_monthly_spend" in result.columns


def test_categorical_columns_numeric_after_pipeline(sample_df: pd.DataFrame) -> None:
    result = FeatureEngineer().fit_transform(sample_df)
    for col in CATEGORICAL_COLS:
        assert pd.api.types.is_numeric_dtype(result[col]), f"{col} not numeric"


def test_transform_before_fit_raises(sample_df: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="fit_transform"):
        FeatureEngineer().transform(sample_df)


def test_transform_output_matches_fit_transform(sample_df: pd.DataFrame) -> None:
    eng = FeatureEngineer()
    train_result = eng.fit_transform(sample_df)
    test_result = eng.transform(sample_df)
    pd.testing.assert_frame_equal(train_result, test_result)


def test_transform_is_idempotent(sample_df: pd.DataFrame) -> None:
    eng = FeatureEngineer()
    eng.fit_transform(sample_df)
    pd.testing.assert_frame_equal(eng.transform(sample_df), eng.transform(sample_df))


def test_fit_transform_does_not_mutate_input(sample_df: pd.DataFrame) -> None:
    original = sample_df["contract"].tolist()
    FeatureEngineer().fit_transform(sample_df)
    assert sample_df["contract"].tolist() == original


def test_get_pipeline_returns_sklearn_pipeline() -> None:
    assert isinstance(FeatureEngineer().get_pipeline(), Pipeline)


def test_get_feature_names_covers_all_cols() -> None:
    names = FeatureEngineer().get_feature_names()
    assert set(names) == set(CATEGORICAL_COLS + NUMERIC_COLS)


def test_pipeline_steps_are_named_correctly() -> None:
    pipe = FeatureEngineer().get_pipeline()
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["derived", "categorical", "numeric"]
