"""CSV loading, schema validation, and cleaning for churn datasets."""

import logging
import os
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_RAW_DIR = Path(os.environ.get("CHURN_RAW_DIR", "data/raw"))


def _to_snake_case(col: str) -> str:
    """Convert a camelCase or space-separated column name to snake_case.

    Examples::

        "TotalCharges"   → "total_charges"
        "customerID"     → "customer_id"
        "Internet Service" → "internet_service"
    """
    col = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", col)
    col = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col)
    return col.lower().replace(" ", "_")

REQUIRED_COLUMNS: set[str] = {
    "customer_id",
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract",
    "payment_method",
    "internet_service",
    "churn",
}


def load_raw_data(filename: str = "churn.csv") -> pd.DataFrame:
    """Load raw churn data from the ``data/raw/`` directory.

    The directory is resolved from the ``CHURN_RAW_DIR`` environment variable,
    falling back to ``data/raw`` relative to the current working directory.

    Args:
        filename: Name of the CSV file inside the raw data directory.
            Defaults to ``"churn.csv"``.

    Returns:
        Raw DataFrame exactly as stored on disk — no transformations applied.

    Raises:
        FileNotFoundError: If the resolved file path does not exist.
    """
    path = _RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw churn DataFrame in place and return it.

    Transformations applied (in order):

    1. Column names are lowercased and spaces replaced with underscores.
    2. ``total_charges`` is coerced to float; non-numeric values become NaN.
    3. Rows where ``total_charges`` is NaN are dropped (typically new customers
       with no billing history).
    4. ``churn`` is mapped from ``"Yes"``/``"No"`` strings to ``1``/``0``
       integers.  Already-numeric values are left unchanged.
    5. ``tenure``, ``monthly_charges``, and ``total_charges`` are cast to
       ``float64`` for consistency.

    Args:
        df: Raw DataFrame as returned by :func:`load_raw_data` or
            :func:`load_csv`.

    Returns:
        A cleaned copy of the input DataFrame.
    """
    df = df.copy()

    # 1. Normalise column names (handles camelCase and space-separated names)
    df.columns = [_to_snake_case(c) for c in df.columns]

    # 2. Coerce total_charges to numeric (raw data often stores it as string)
    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    # 3. Drop rows with no billing history
    before = len(df)
    df = df.dropna(subset=["total_charges"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d row(s) with missing total_charges", dropped)

    # 4. Encode churn target
    if "churn" in df.columns:
        df["churn"] = df["churn"].map({"Yes": 1, "No": 0}).fillna(df["churn"])
        df["churn"] = df["churn"].astype(int)

    # 5. Ensure numeric dtypes for model-relevant columns
    for col in ("tenure", "monthly_charges", "total_charges"):
        if col in df.columns:
            df[col] = df[col].astype("float64")

    logger.info("Cleaned data: %d rows remaining", len(df))
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        Raw DataFrame with all columns from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def validate_schema(
    df: pd.DataFrame,
    required: set[str] | None = None,
) -> pd.DataFrame:
    """Check that a DataFrame contains the expected columns.

    Args:
        df: Input DataFrame to validate.
        required: Column names that must be present. Defaults to
            REQUIRED_COLUMNS.

    Returns:
        The original DataFrame unchanged if validation passes.

    Raises:
        ValueError: If any required columns are missing.
    """
    expected = required if required is not None else REQUIRED_COLUMNS
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")
    return df
