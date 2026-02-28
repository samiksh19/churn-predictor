"""Feature engineering using sklearn-compatible Transformers and Pipeline."""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

CATEGORICAL_COLS: list[str] = ["contract", "payment_method", "internet_service"]
NUMERIC_COLS: list[str] = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "tenure_months",
    "avg_monthly_spend",
    "support_calls_ratio",
]
TARGET_COL: str = "churn"


# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------


class DerivedFeatureTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Create derived features from raw billing and usage columns.

    Derived columns added:

    - ``tenure_months``: explicit alias of ``tenure``, confirming the unit
      is months.
    - ``avg_monthly_spend``: ``total_charges / (tenure + 1)`` — smoothed
      average spend per active month.
    - ``support_calls_ratio``: ``support_calls / (tenure + 1)`` — call
      frequency per active month.  Only added when a ``support_calls``
      column is present in the input.

    This transformer is stateless: ``fit`` is a no-op and it can be
    applied to any DataFrame that contains a ``tenure`` column.
    """

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "DerivedFeatureTransformer":
        """No-op fit; returns self.

        Args:
            X: Input feature DataFrame.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns to a copy of *X*.

        Args:
            X: Input feature DataFrame with at least a ``tenure`` column.

        Returns:
            DataFrame with additional derived feature columns appended.
        """
        X = X.copy()

        if "tenure" in X.columns:
            tenure_safe = X["tenure"] + 1  # avoid division by zero

            X["tenure_months"] = X["tenure"]
            logger.debug("Added derived feature: tenure_months")

            if "total_charges" in X.columns:
                X["avg_monthly_spend"] = X["total_charges"] / tenure_safe
                logger.debug("Added derived feature: avg_monthly_spend")

            if "support_calls" in X.columns:
                X["support_calls_ratio"] = X["support_calls"] / tenure_safe
                logger.debug("Added derived feature: support_calls_ratio")

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Label-encode a fixed list of categorical columns.

    Each column gets its own :class:`~sklearn.preprocessing.LabelEncoder`
    fitted independently during ``fit``.

    Args:
        cols: Column names to encode.  Columns absent from the DataFrame
            are silently skipped.  Defaults to :data:`CATEGORICAL_COLS`.
    """

    def __init__(self, cols: list[str] | None = None) -> None:
        self.cols = cols or CATEGORICAL_COLS

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "CategoricalEncoder":
        """Fit one LabelEncoder per column found in *X*.

        Args:
            X: Training feature DataFrame.
            y: Ignored.

        Returns:
            self
        """
        self.encoders_: dict[str, LabelEncoder] = {}
        for col in self.cols:
            if col in X.columns:
                enc = LabelEncoder()
                enc.fit(X[col].astype(str))
                self.encoders_[col] = enc
                logger.debug("Fitted LabelEncoder for column '%s'", col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders to a copy of *X*.

        Args:
            X: Feature DataFrame with categorical columns.

        Returns:
            DataFrame with categorical columns replaced by integer codes.
        """
        X = X.copy()
        for col, enc in self.encoders_.items():
            if col in X.columns:
                X[col] = enc.transform(X[col].astype(str))
        return X


class NumericScaler(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Standard-scale a fixed list of numeric columns.

    Uses :class:`~sklearn.preprocessing.StandardScaler` (zero mean,
    unit variance) fitted on columns that are present in the training
    DataFrame.  Columns listed in *cols* but absent from the DataFrame
    are silently skipped.

    Args:
        cols: Numeric column names to scale.
            Defaults to :data:`NUMERIC_COLS`.
    """

    def __init__(self, cols: list[str] | None = None) -> None:
        self.cols = cols or NUMERIC_COLS

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "NumericScaler":
        """Fit the StandardScaler on numeric columns present in *X*.

        Args:
            X: Training feature DataFrame.
            y: Ignored.

        Returns:
            self
        """
        self.present_cols_: list[str] = [c for c in self.cols if c in X.columns]
        self.scaler_: StandardScaler = StandardScaler()
        if self.present_cols_:
            self.scaler_.fit(X[self.present_cols_])
            logger.debug("Fitted StandardScaler on columns: %s", self.present_cols_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to numeric columns in a copy of *X*.

        Args:
            X: Feature DataFrame.

        Returns:
            DataFrame with numeric columns standardised in-place.
        """
        X = X.copy()
        if self.present_cols_:
            scaled: np.ndarray = self.scaler_.transform(X[self.present_cols_])
            X[self.present_cols_] = scaled
        return X


# ---------------------------------------------------------------------------
# Public façade
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """End-to-end feature engineering pipeline backed by sklearn.

    Composes three transformers in sequence:

    1. :class:`DerivedFeatureTransformer` — adds computed columns.
    2. :class:`CategoricalEncoder` — label-encodes categoricals.
    3. :class:`NumericScaler` — standard-scales numeric columns.

    Usage::

        engineer = FeatureEngineer()
        X_train = engineer.fit_transform(train_df)
        X_test  = engineer.transform(test_df)
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline = Pipeline(
            steps=[
                ("derived", DerivedFeatureTransformer()),
                ("categorical", CategoricalEncoder()),
                ("numeric", NumericScaler()),
            ]
        )
        self._fitted: bool = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the pipeline on *df* and return the transformed copy.

        Args:
            df: Raw feature DataFrame.  May include the target column;
                it is passed through unchanged.

        Returns:
            Transformed DataFrame with derived, encoded, and scaled columns.
        """
        result: pd.DataFrame = self._pipeline.fit_transform(df)
        self._fitted = True
        logger.info(
            "FeatureEngineer fitted: %d rows, %d columns", len(result), len(result.columns)
        )
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted pipeline to new data.

        Args:
            df: Raw feature DataFrame with the same schema as the
                training data passed to :meth:`fit_transform`.

        Returns:
            Transformed DataFrame.

        Raises:
            RuntimeError: If called before :meth:`fit_transform`.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform on training data before transform.")
        result: pd.DataFrame = self._pipeline.transform(df)
        return result

    def get_feature_names(self) -> list[str]:
        """Return the ordered list of output feature column names.

        Returns:
            Categorical columns followed by numeric (including derived) columns.
        """
        return CATEGORICAL_COLS + NUMERIC_COLS

    def get_pipeline(self) -> Pipeline:
        """Return the underlying sklearn Pipeline for inspection or export.

        Returns:
            The fitted (or unfitted) :class:`~sklearn.pipeline.Pipeline`.
        """
        return self._pipeline
