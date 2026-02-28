"""End-to-end integration test for the full churn prediction pipeline.

Runs: synthetic data → clean → engineer → train (CV) → evaluate,
then prints a formatted metrics report.  Execute with::

    pytest tests/test_pipeline.py -s -v
"""

import pandas as pd

from churn_predictor.models.trainer import ChurnTrainer, EvalMetrics


def _metrics_report(
    n_train: int,
    n_test: int,
    n_splits: int,
    cv_mean: float,
    cv_std: float,
    m: EvalMetrics,
) -> str:
    bar = "=" * 52
    sep = "-" * 52
    lines = [
        "",
        bar,
        "   CHURN PREDICTOR — END-TO-END EVALUATION",
        bar,
        f"   Training rows  : {n_train}",
        f"   Test rows      : {n_test}",
        f"   CV folds       : {n_splits}",
        sep,
        f"   CV ROC-AUC     : {cv_mean:.4f} ± {cv_std:.4f}",
        sep,
        f"   Precision      : {m.precision:.4f}",
        f"   Recall         : {m.recall:.4f}",
        f"   F1 Score       : {m.f1:.4f}",
        f"   ROC-AUC        : {m.roc_auc:.4f}",
        bar,
    ]
    return "\n".join(lines)


def test_full_pipeline_metrics(
    engineered_splits: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """Train on the session fixture and report all evaluation metrics."""
    X_train, X_test, y_train, y_test = engineered_splits

    n_splits = 3
    trainer = ChurnTrainer(n_splits=n_splits, n_estimators=150)
    cv_metrics = trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)

    report = _metrics_report(
        n_train=len(X_train),
        n_test=len(X_test),
        n_splits=n_splits,
        cv_mean=cv_metrics["cv_roc_auc_mean"],
        cv_std=cv_metrics["cv_roc_auc_std"],
        m=metrics,
    )
    print(report)

    # Sanity assertions — random baseline is 0.5 for AUC.
    assert metrics.roc_auc > 0.5, f"AUC below random baseline: {metrics.roc_auc:.4f}"
    assert metrics.f1 > 0.0, "F1 should be non-zero with a learnable signal"
    assert metrics.precision >= 0.0
    assert metrics.recall >= 0.0
