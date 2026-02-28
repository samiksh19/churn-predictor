"""Training entry point for the churn prediction pipeline.

Runs the full training workflow:
  1. Load and clean raw data from ``CHURN_RAW_DIR``.
  2. Split into 80 / 20 train / test sets (stratified).
  3. Fit the feature engineering pipeline on the training split.
  4. Train an XGBoost classifier with 5-fold cross-validation.
  5. Evaluate on the held-out test set.
  6. Save both artefacts to ``CHURN_MODEL_DIR``.

Usage::

    python -m churn_predictor.train
    # or, after pip install -e .
    churn-train

Environment variables:
    CHURN_RAW_DIR:   directory containing ``churn.csv`` (default: ``data/raw``).
    CHURN_MODEL_DIR: directory for saved artefacts  (default: ``models``).
"""

import logging
import os
import sys
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from churn_predictor.data.loader import clean_data, load_raw_data
from churn_predictor.features.engineering import FeatureEngineer
from churn_predictor.models.trainer import ChurnTrainer

_FEATURE_COLS: list[str] = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract",
    "payment_method",
    "internet_service",
]

_ENGINEER_FILENAME: str = "churn_engineer.joblib"


def _setup_logging() -> None:
    """Configure root logger to write INFO+ to stdout with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def main() -> None:
    """Run the end-to-end training pipeline and save artefacts."""
    _setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=== Churn Predictor — Training Pipeline ===")

    # 1. Load and clean
    df = clean_data(load_raw_data())

    # 2. Split features / target
    X = df[_FEATURE_COLS]
    y = df["churn"]
    logger.info("Dataset: %d rows, %d features", len(X), len(_FEATURE_COLS))

    # 3. Train / test split (stratified to preserve churn ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Split: %d train / %d test", len(X_train), len(X_test))

    # 4. Feature engineering — fit on train only to avoid data leakage
    engineer = FeatureEngineer()
    X_train_eng = engineer.fit_transform(X_train.reset_index(drop=True))
    X_test_eng = engineer.transform(X_test.reset_index(drop=True))

    # 5. Train with cross-validation then full refit
    trainer = ChurnTrainer()
    trainer.train(X_train_eng, y_train)

    # 6. Evaluate on held-out test set
    metrics = trainer.evaluate(X_test_eng, y_test)

    # 7. Save model
    model_path = trainer.save_model()

    # 8. Save engineer (must be persisted alongside the model for the API)
    model_dir = Path(os.environ.get("CHURN_MODEL_DIR", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    engineer_path = model_dir / _ENGINEER_FILENAME
    joblib.dump(engineer, engineer_path)
    logger.info("Engineer saved to %s", engineer_path.resolve())

    # 9. Summary
    logger.info("=== Results ===")
    logger.info("  CV ROC-AUC : %.4f ± %.4f", metrics.cv_roc_auc_mean, metrics.cv_roc_auc_std)
    logger.info("  Test AUC   : %.4f", metrics.roc_auc)
    logger.info("  Precision  : %.4f", metrics.precision)
    logger.info("  Recall     : %.4f", metrics.recall)
    logger.info("  F1         : %.4f", metrics.f1)
    logger.info("Artefacts written to %s", model_path.parent)


if __name__ == "__main__":
    main()
