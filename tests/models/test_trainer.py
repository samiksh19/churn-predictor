"""Tests for churn_predictor.models.trainer — train, evaluate, predict, persist."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_predictor.models.trainer import ChurnTrainer, EvalMetrics

# ---------------------------------------------------------------------------
# Local fixtures (function-scoped for isolation)
# ---------------------------------------------------------------------------


@pytest.fixture()
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """300-row pre-encoded feature matrix with a seeded random target."""
    rng = np.random.default_rng(0)
    n = 300
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


@pytest.fixture()
def fitted_trainer(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> ChurnTrainer:
    """Freshly trained ChurnTrainer (fast: n_splits=2, n_estimators=50)."""
    X, y = training_data
    trainer = ChurnTrainer(n_splits=2, n_estimators=50)
    trainer.train(X, y)
    return trainer


# ---------------------------------------------------------------------------
# ChurnTrainer.__init__
# ---------------------------------------------------------------------------


def test_default_init_sets_n_splits() -> None:
    trainer = ChurnTrainer()
    assert trainer.n_splits == 5


def test_custom_hyperparams_accepted() -> None:
    trainer = ChurnTrainer(n_splits=3, n_estimators=100, max_depth=4)
    assert trainer.n_splits == 3


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_returns_cv_metrics(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = training_data
    metrics = ChurnTrainer(n_splits=2, n_estimators=50).train(X, y)
    assert set(metrics.keys()) == {"cv_roc_auc_mean", "cv_roc_auc_std"}


def test_train_cv_roc_auc_in_unit_interval(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = training_data
    metrics = ChurnTrainer(n_splits=2, n_estimators=50).train(X, y)
    assert 0.0 <= metrics["cv_roc_auc_mean"] <= 1.0
    assert metrics["cv_roc_auc_std"] >= 0.0


def test_train_marks_trainer_as_fitted(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = training_data
    trainer = ChurnTrainer(n_splits=2, n_estimators=50)
    assert not trainer._fitted
    trainer.train(X, y)
    assert trainer._fitted


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_eval_metrics_instance(
    fitted_trainer: ChurnTrainer,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = training_data
    assert isinstance(fitted_trainer.evaluate(X, y), EvalMetrics)


def test_evaluate_all_metrics_in_unit_interval(
    fitted_trainer: ChurnTrainer,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = training_data
    m = fitted_trainer.evaluate(X, y)
    for name, value in m.as_dict().items():
        if not np.isnan(value):
            assert 0.0 <= value <= 1.0, f"{name} = {value} out of [0, 1]"


def test_evaluate_cv_fields_populated_from_train(
    fitted_trainer: ChurnTrainer,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = training_data
    m = fitted_trainer.evaluate(X, y)
    assert not np.isnan(m.cv_roc_auc_mean)
    assert not np.isnan(m.cv_roc_auc_std)


def test_evaluate_before_train_raises() -> None:
    trainer = ChurnTrainer()
    with pytest.raises(RuntimeError, match="train"):
        trainer.evaluate(pd.DataFrame({"a": [1]}), pd.Series([0]))


def test_eval_metrics_as_dict_has_all_keys(
    fitted_trainer: ChurnTrainer,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = training_data
    keys = fitted_trainer.evaluate(X, y).as_dict().keys()
    assert keys == {"precision", "recall", "f1", "roc_auc", "cv_roc_auc_mean", "cv_roc_auc_std"}


# ---------------------------------------------------------------------------
# EvalMetrics dataclass
# ---------------------------------------------------------------------------


def test_eval_metrics_as_dict_values_match_fields() -> None:
    m = EvalMetrics(precision=0.8, recall=0.7, f1=0.75, roc_auc=0.85)
    d = m.as_dict()
    assert d["precision"] == 0.8
    assert d["recall"] == 0.7
    assert d["f1"] == 0.75
    assert d["roc_auc"] == 0.85


def test_eval_metrics_cv_defaults_to_nan() -> None:
    m = EvalMetrics(precision=0.8, recall=0.7, f1=0.75, roc_auc=0.85)
    assert np.isnan(m.cv_roc_auc_mean)
    assert np.isnan(m.cv_roc_auc_std)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_returns_correct_shape(
    fitted_trainer: ChurnTrainer,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = training_data
    preds = fitted_trainer.predict(X)
    assert preds.shape == (len(X),)


def test_predict_returns_binary_labels(
    fitted_trainer: ChurnTrainer,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = training_data
    preds = fitted_trainer.predict(X)
    assert set(preds).issubset({0, 1})


def test_predict_before_train_raises() -> None:
    """Branch: ``if not self._fitted`` guard in predict."""
    trainer = ChurnTrainer()
    with pytest.raises(RuntimeError, match="train"):
        trainer.predict(pd.DataFrame({"a": [1, 2]}))


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------


def test_predict_proba_in_unit_interval(
    fitted_trainer: ChurnTrainer,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = training_data
    probs = fitted_trainer.predict_proba(X)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_predict_proba_shape(
    fitted_trainer: ChurnTrainer,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = training_data
    assert fitted_trainer.predict_proba(X).shape == (len(X),)


def test_predict_proba_before_train_raises() -> None:
    """Branch: ``if not self._fitted`` guard in predict_proba."""
    trainer = ChurnTrainer()
    with pytest.raises(RuntimeError, match="train"):
        trainer.predict_proba(pd.DataFrame({"a": [1, 2]}))


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------


def test_save_model_before_train_raises() -> None:
    with pytest.raises(RuntimeError, match="train"):
        ChurnTrainer().save_model()


def test_save_model_returns_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    monkeypatch.setenv("CHURN_MODEL_DIR", str(tmp_path))
    import importlib

    import churn_predictor.models.trainer as m

    importlib.reload(m)

    X, y = training_data
    trainer = m.ChurnTrainer(n_splits=2, n_estimators=10)
    trainer.train(X, y)
    saved_path = trainer.save_model("test.joblib")

    assert isinstance(saved_path, Path)
    assert saved_path.is_absolute()
    assert saved_path.exists()


def test_save_and_load_predictions_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    monkeypatch.setenv("CHURN_MODEL_DIR", str(tmp_path))
    import importlib

    import churn_predictor.models.trainer as m

    importlib.reload(m)

    X, y = training_data
    trainer = m.ChurnTrainer(n_splits=2, n_estimators=50)
    trainer.train(X, y)
    trainer.save_model("churn.joblib")

    loaded = m.ChurnTrainer.load_model("churn.joblib")
    np.testing.assert_array_equal(trainer.predict(X), loaded.predict(X))
    np.testing.assert_array_almost_equal(
        trainer.predict_proba(X), loaded.predict_proba(X)
    )


def test_load_model_sets_fitted_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    monkeypatch.setenv("CHURN_MODEL_DIR", str(tmp_path))
    import importlib

    import churn_predictor.models.trainer as m

    importlib.reload(m)

    X, y = training_data
    trainer = m.ChurnTrainer(n_splits=2, n_estimators=50)
    trainer.train(X, y)
    trainer.save_model("check.joblib")

    loaded = m.ChurnTrainer.load_model("check.joblib")
    assert loaded._fitted is True


def test_load_missing_model_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHURN_MODEL_DIR", str(tmp_path))
    import importlib

    import churn_predictor.models.trainer as m

    importlib.reload(m)

    with pytest.raises(FileNotFoundError):
        m.ChurnTrainer.load_model("nonexistent.joblib")
