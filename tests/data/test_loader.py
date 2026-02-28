"""Tests for churn_predictor.data.loader — loading, cleaning, validation."""

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from churn_predictor.data.loader import (
    REQUIRED_COLUMNS,
    _to_snake_case,
    clean_data,
    load_csv,
    validate_schema,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

RAW_CSV = textwrap.dedent("""\
    customerID,tenure,MonthlyCharges,TotalCharges,Contract,PaymentMethod,InternetService,Churn
    C001,12,70.0,840.0,Month-to-month,Electronic check,Fiber optic,Yes
    C002,24,50.0,1200.0,One year,Bank transfer,DSL,No
    C003,0,30.0, ,Month-to-month,Credit card,No,No
""")


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    """Small CSV with already-normalised snake_case column names."""
    content = textwrap.dedent("""\
        customer_id,tenure,monthly_charges,total_charges,contract,payment_method,internet_service,churn
        C001,12,70.0,840.0,Month-to-month,Electronic check,Fiber optic,1
        C002,24,50.0,1200.0,One year,Bank transfer,DSL,0
    """)
    p = tmp_path / "customers.csv"
    p.write_text(content)
    return p


@pytest.fixture()
def raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Temp directory containing a camelCase raw CSV; patches CHURN_RAW_DIR."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "churn.csv").write_text(RAW_CSV)
    monkeypatch.setenv("CHURN_RAW_DIR", str(raw))
    import importlib

    import churn_predictor.data.loader as m

    importlib.reload(m)
    return raw


# ---------------------------------------------------------------------------
# _to_snake_case
# ---------------------------------------------------------------------------


def test_to_snake_case_camel_case() -> None:
    assert _to_snake_case("TotalCharges") == "total_charges"
    assert _to_snake_case("MonthlyCharges") == "monthly_charges"
    assert _to_snake_case("InternetService") == "internet_service"
    assert _to_snake_case("PaymentMethod") == "payment_method"


def test_to_snake_case_space_separated() -> None:
    assert _to_snake_case("Internet Service") == "internet_service"
    assert _to_snake_case("Payment Method") == "payment_method"


def test_to_snake_case_customer_id() -> None:
    assert _to_snake_case("customerID") == "customer_id"


def test_to_snake_case_already_snake() -> None:
    assert _to_snake_case("tenure") == "tenure"
    assert _to_snake_case("total_charges") == "total_charges"


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


def test_load_csv_returns_dataframe(csv_file: Path) -> None:
    df = load_csv(csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_load_csv_accepts_string_path(csv_file: Path) -> None:
    df = load_csv(str(csv_file))
    assert isinstance(df, pd.DataFrame)


def test_load_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------


def test_validate_schema_passes(csv_file: Path) -> None:
    df = load_csv(csv_file)
    result = validate_schema(df)
    assert result is df


def test_validate_schema_raises_on_missing_columns() -> None:
    df = pd.DataFrame({"customer_id": [1], "tenure": [6]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(df)


def test_validate_schema_custom_required() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = validate_schema(df, required={"a", "b"})
    assert set(result.columns) == {"a", "b"}


def test_validate_schema_default_uses_required_columns() -> None:
    df = pd.DataFrame({col: [1] for col in REQUIRED_COLUMNS})
    assert validate_schema(df) is df


# ---------------------------------------------------------------------------
# load_raw_data
# ---------------------------------------------------------------------------


def test_load_raw_data_returns_dataframe(raw_dir: Path) -> None:
    from churn_predictor.data.loader import load_raw_data as _load

    df = _load()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_load_raw_data_row_count(raw_dir: Path) -> None:
    from churn_predictor.data.loader import load_raw_data as _load

    df = _load()
    assert df.shape[1] == 8  # customerID + 6 features + Churn


def test_load_raw_data_missing_file_raises(raw_dir: Path) -> None:
    from churn_predictor.data.loader import load_raw_data as _load

    with pytest.raises(FileNotFoundError):
        _load("nonexistent.csv")


# ---------------------------------------------------------------------------
# clean_data — column normalisation
# ---------------------------------------------------------------------------


def test_clean_data_normalises_camel_case_columns(raw_dir: Path) -> None:
    from churn_predictor.data.loader import load_raw_data as _load

    df = _load()
    cleaned = clean_data(df)
    assert "total_charges" in cleaned.columns
    assert "monthly_charges" in cleaned.columns
    assert "internet_service" in cleaned.columns
    # Originals should be gone
    assert "TotalCharges" not in cleaned.columns
    assert "MonthlyCharges" not in cleaned.columns


# ---------------------------------------------------------------------------
# clean_data — missing value handling
# ---------------------------------------------------------------------------


def test_clean_data_drops_blank_total_charges(raw_dir: Path) -> None:
    from churn_predictor.data.loader import load_raw_data as _load

    df = _load()
    cleaned = clean_data(df)
    # C003 has a blank TotalCharges → dropped
    assert len(cleaned) == 2


def test_clean_data_no_rows_dropped_when_all_valid() -> None:
    """Branch: ``if dropped:`` evaluates to False — no warning issued."""
    df = pd.DataFrame(
        {
            "total_charges": [100.0, 200.0, 300.0],
            "tenure": [6.0, 12.0, 24.0],
            "monthly_charges": [50.0, 70.0, 90.0],
            "churn": ["Yes", "No", "Yes"],
        }
    )
    cleaned = clean_data(df)
    assert len(cleaned) == 3


# ---------------------------------------------------------------------------
# clean_data — churn encoding
# ---------------------------------------------------------------------------


def test_clean_data_encodes_yes_no_churn(raw_dir: Path) -> None:
    from churn_predictor.data.loader import load_raw_data as _load

    df = _load()
    cleaned = clean_data(df)
    assert cleaned["churn"].dtype == int
    assert set(cleaned["churn"].unique()).issubset({0, 1})


def test_clean_data_numeric_churn_passthrough() -> None:
    """Branch: churn already int — fillna restores values, astype(int) succeeds."""
    df = pd.DataFrame(
        {
            "total_charges": [100.0, 200.0],
            "tenure": [6.0, 12.0],
            "monthly_charges": [50.0, 70.0],
            "churn": [1, 0],
        }
    )
    cleaned = clean_data(df)
    assert list(cleaned["churn"]) == [1, 0]


def test_clean_data_without_churn_column() -> None:
    """Branch: ``if 'churn' in df.columns:`` evaluates to False."""
    df = pd.DataFrame(
        {
            "total_charges": [100.0, 200.0],
            "tenure": [6.0, 12.0],
            "monthly_charges": [50.0, 70.0],
        }
    )
    cleaned = clean_data(df)
    assert "churn" not in cleaned.columns
    assert len(cleaned) == 2


# ---------------------------------------------------------------------------
# clean_data — dtype casting
# ---------------------------------------------------------------------------


def test_clean_data_casts_numeric_columns_to_float64() -> None:
    df = pd.DataFrame(
        {
            "total_charges": ["100", "200"],
            "tenure": [6, 12],
            "monthly_charges": [50, 70],
            "churn": ["Yes", "No"],
        }
    )
    cleaned = clean_data(df)
    assert cleaned["tenure"].dtype == "float64"
    assert cleaned["monthly_charges"].dtype == "float64"
    assert cleaned["total_charges"].dtype == "float64"


def test_clean_data_skips_cast_for_absent_numeric_columns() -> None:
    """Branch: ``if col in df.columns:`` is False for some numeric cols."""
    df = pd.DataFrame(
        {
            "total_charges": [100.0, 200.0],
            "churn": ["Yes", "No"],
            # tenure and monthly_charges deliberately absent
        }
    )
    cleaned = clean_data(df)
    assert "tenure" not in cleaned.columns
    assert "monthly_charges" not in cleaned.columns
    assert len(cleaned) == 2


# ---------------------------------------------------------------------------
# clean_data — immutability
# ---------------------------------------------------------------------------


def test_clean_data_does_not_mutate_input(raw_dir: Path) -> None:
    from churn_predictor.data.loader import load_raw_data as _load

    df = _load()
    original_len = len(df)
    original_cols = list(df.columns)
    clean_data(df)
    assert len(df) == original_len
    assert list(df.columns) == original_cols
