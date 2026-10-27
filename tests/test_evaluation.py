"""Tests for metrics, threshold selection and evaluation plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation import (
    binary_metrics,
    cross_validate_model,
    find_best_threshold,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    threshold_sweep,
)
from src.preprocessing import build_pipeline


def test_binary_metrics_match_a_hand_computed_confusion_matrix() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.6, 0.7, 0.2, 0.4, 0.8, 0.9])

    metrics = binary_metrics(y_true, y_proba, threshold=0.5)

    # Predictions: 0, 0, 1, 1, 0, 0, 1, 1
    assert metrics["confusion_matrix"] == {
        "true_negative": 2,
        "false_positive": 2,
        "false_negative": 2,
        "true_positive": 2,
    }
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_proba))
    assert metrics["pr_auc"] == pytest.approx(average_precision_score(y_true, y_proba))
    assert metrics["support"] == {"non_default": 4, "default": 4}


def test_find_best_threshold_separates_the_classes() -> None:
    y_true = np.array([0] * 50 + [1] * 50)
    y_proba = np.concatenate(
        [np.linspace(0.0, 0.49, 50), np.linspace(0.51, 1.0, 50)]
    )

    result = find_best_threshold(y_true, y_proba)

    assert 0.49 < result["threshold"] <= 0.51
    assert result["f1"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


def test_threshold_sweep_returns_valid_scores() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])

    sweep = threshold_sweep(y_true, y_proba, steps=21)

    assert len(sweep) == 21
    for column in ["precision", "recall", "f1"]:
        assert sweep[column].between(0, 1).all()


def test_cross_validate_model_returns_every_metric_and_fold() -> None:
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(
        {
            "person_age": rng.integers(21, 60, n),
            "person_income": rng.integers(20_000, 150_000, n),
            "person_home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
            "person_emp_length": rng.uniform(0, 30, n),
            "loan_intent": rng.choice(
                ["PERSONAL", "MEDICAL", "EDUCATION", "VENTURE"], n
            ),
            "loan_grade": rng.choice(list("ABCDE"), n),
            "loan_amnt": rng.integers(1_000, 30_000, n),
            "loan_int_rate": rng.uniform(5, 20, n),
            "loan_percent_income": rng.uniform(0.05, 0.6, n),
            "cb_person_default_on_file": rng.choice(["Y", "N"], n),
            "cb_person_cred_hist_length": rng.integers(1, 20, n),
        }
    )
    y = pd.Series(np.tile([0, 1], n // 2))

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    scores = cross_validate_model(pipeline, X, y, cv=3)

    for metric in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]:
        assert len(scores[metric]["folds"]) == 3
        assert 0 <= scores[metric]["mean"] <= 1
    assert scores["fit_time_seconds"] >= 0


def test_evaluation_plots_are_written(tmp_path) -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    y_proba = np.array([0.05, 0.2, 0.4, 0.7, 0.8, 0.9, 0.6, 0.3, 0.55, 0.1])
    metrics = binary_metrics(y_true, y_proba, threshold=0.5)

    paths = [
        plot_confusion_matrix(metrics, tmp_path / "confusion.png"),
        plot_roc_curve(y_true, y_proba, tmp_path / "roc.png"),
        plot_precision_recall_curve(y_true, y_proba, tmp_path / "pr.png"),
    ]
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 1_000
