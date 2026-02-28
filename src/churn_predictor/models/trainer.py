"""XGBoost-based churn trainer with cross-validation, evaluation, and persistence."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(os.environ.get("CHURN_MODEL_DIR", "models"))

# Default XGBoost hyper-parameters — override via constructor kwargs.
_DEFAULTS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


@dataclass
class EvalMetrics:
    """Holds evaluation metrics for a single model assessment.

    Attributes:
        precision: Fraction of predicted churners that actually churned.
        recall: Fraction of actual churners correctly identified.
        f1: Harmonic mean of precision and recall.
        roc_auc: Area under the ROC curve.
        cv_roc_auc_mean: Mean cross-validation ROC-AUC (set after CV training).
        cv_roc_auc_std: Std of cross-validation ROC-AUC (set after CV training).
    """

    precision: float
    recall: float
    f1: float
    roc_auc: float
    cv_roc_auc_mean: float = field(default=float("nan"))
    cv_roc_auc_std: float = field(default=float("nan"))

    def as_dict(self) -> dict[str, float]:
        """Return all metrics as a plain dictionary.

        Returns:
            Mapping of metric name to float value.
        """
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "cv_roc_auc_mean": self.cv_roc_auc_mean,
            "cv_roc_auc_std": self.cv_roc_auc_std,
        }


class ChurnTrainer:
    """Train, evaluate, and persist an XGBoost churn classifier.

    Wraps :class:`~xgboost.XGBClassifier` with:

    - Stratified k-fold cross-validation during :meth:`train`.
    - A dedicated :meth:`evaluate` method returning :class:`EvalMetrics`.
    - :meth:`save_model` / :meth:`load_model` backed by ``joblib``.

    The model artefact directory is read from the ``CHURN_MODEL_DIR``
    environment variable, falling back to ``models/`` relative to the
    working directory.

    Args:
        n_splits: Number of folds for stratified cross-validation.
        **kwargs: XGBoost hyper-parameters that override the defaults.
            Supported keys: ``n_estimators``, ``max_depth``,
            ``learning_rate``, ``subsample``, ``colsample_bytree``.

    Example::

        trainer = ChurnTrainer(n_splits=5, n_estimators=300)
        metrics = trainer.train(X_train, y_train)
        eval_metrics = trainer.evaluate(X_test, y_test)
        trainer.save_model("churn_v1.joblib")
    """

    def __init__(self, n_splits: int = 5, **kwargs: Any) -> None:
        self.n_splits = n_splits
        params = {**_DEFAULTS, **kwargs}
        self._model = XGBClassifier(**params)
        self._fitted: bool = False
        self._cv_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Fit the classifier with stratified cross-validation.

        Cross-validation is run first to collect generalisation estimates,
        then the model is re-fitted on the full training set so it is
        ready for :meth:`predict` and :meth:`evaluate`.

        Args:
            X: Feature matrix (already engineered and scaled).
            y: Binary target series — ``1`` for churned, ``0`` for retained.

        Returns:
            Dictionary with ``cv_roc_auc_mean`` and ``cv_roc_auc_std``
            computed across the *n_splits* folds.
        """
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        logger.info("Running %d-fold cross-validation …", self.n_splits)
        cv_results = cross_validate(
            self._model,
            X,
            y,
            cv=cv,
            scoring="roc_auc",
            return_train_score=False,
            n_jobs=-1,
        )
        mean_auc = float(np.mean(cv_results["test_score"]))
        std_auc = float(np.std(cv_results["test_score"]))
        self._cv_metrics = {"cv_roc_auc_mean": mean_auc, "cv_roc_auc_std": std_auc}
        logger.info(
            "CV ROC-AUC: %.4f ± %.4f",
            mean_auc,
            std_auc,
        )

        logger.info("Fitting on full training set (%d rows) …", len(X))
        self._model.fit(X, y, verbose=False)
        self._fitted = True
        logger.info("Training complete.")

        return self._cv_metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> EvalMetrics:
        """Compute precision, recall, F1, and ROC-AUC on a held-out set.

        Args:
            X: Feature matrix for the evaluation split.
            y: True binary labels for the evaluation split.

        Returns:
            :class:`EvalMetrics` populated with all four metrics plus any
            cross-validation stats stored from the last :meth:`train` call.

        Raises:
            RuntimeError: If called before :meth:`train`.
        """
        if not self._fitted:
            raise RuntimeError("Call train() before evaluate().")

        y_pred = self._model.predict(X)
        y_proba: np.ndarray = self._model.predict_proba(X)[:, 1]

        metrics = EvalMetrics(
            precision=float(precision_score(y, y_pred, zero_division=0)),
            recall=float(recall_score(y, y_pred, zero_division=0)),
            f1=float(f1_score(y, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y, y_proba)),
            cv_roc_auc_mean=self._cv_metrics.get("cv_roc_auc_mean", float("nan")),
            cv_roc_auc_std=self._cv_metrics.get("cv_roc_auc_std", float("nan")),
        )

        logger.info(
            "Evaluation — precision: %.4f | recall: %.4f | F1: %.4f | AUC: %.4f",
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.roc_auc,
        )
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary class predictions (0 or 1).

        Args:
            X: Feature matrix.

        Returns:
            1-D integer array of predicted labels.

        Raises:
            RuntimeError: If called before :meth:`train`.
        """
        if not self._fitted:
            raise RuntimeError("Call train() before predict().")
        result: np.ndarray = self._model.predict(X)
        return result

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return churn probability for each row.

        Args:
            X: Feature matrix.

        Returns:
            1-D float array of churn probabilities in ``[0, 1]``.

        Raises:
            RuntimeError: If called before :meth:`train`.
        """
        if not self._fitted:
            raise RuntimeError("Call train() before predict_proba().")
        result: np.ndarray = self._model.predict_proba(X)[:, 1]
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, filename: str = "churn_model.joblib") -> Path:
        """Persist the fitted model under ``CHURN_MODEL_DIR``.

        The directory is created if it does not exist.

        Args:
            filename: File name for the saved artefact.
                Defaults to ``"churn_model.joblib"``.

        Returns:
            Absolute path to the saved file.

        Raises:
            RuntimeError: If called before :meth:`train`.
        """
        if not self._fitted:
            raise RuntimeError("Call train() before save_model().")

        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = _MODEL_DIR / filename
        joblib.dump(self._model, path)
        logger.info("Model saved to %s", path)
        return path.resolve()

    @classmethod
    def load_model(cls, filename: str = "churn_model.joblib") -> "ChurnTrainer":
        """Load a previously saved model from ``CHURN_MODEL_DIR``.

        Args:
            filename: File name of the saved artefact inside
                ``CHURN_MODEL_DIR``.  Defaults to
                ``"churn_model.joblib"``.

        Returns:
            A :class:`ChurnTrainer` instance with ``_fitted`` set to
            ``True``, ready for :meth:`predict` and :meth:`evaluate`.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        path = _MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        instance = cls()
        instance._model = joblib.load(path)
        instance._fitted = True
        logger.info("Model loaded from %s", path)
        return instance
