"""Tests for the training orchestration and saved artifacts."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS
from train import _select_best, run_training


def make_synthetic_raw(n: int = 100) -> pd.DataFrame:
    """Create a small, valid dataset with a learnable signal and both classes."""
    rng = np.random.default_rng(7)
    grades = list("ABCDEFG")
    rows = []
    for _ in range(n):
        age = int(rng.integers(21, 60))
        income = int(rng.integers(30_000, 150_000))
        loan = int(rng.integers(2_000, 30_000))
        percent = loan / income
        grade_index = min(6, int(percent * 12))
        grade = grades[grade_index]
        int_rate = 5 + grade_index * 2 + float(rng.normal(0, 0.5))
        employment = float(rng.uniform(0, min(30, age - 18)))
        if rng.random() < 0.08:
            int_rate = np.nan
        if rng.random() < 0.05:
            employment = np.nan
        probability = min(0.95, 0.05 + 0.75 * percent + 0.04 * grade_index)
        default = int(rng.random() < probability)
        rows.append(
            {
                "person_age": age,
                "person_income": income,
                "person_home_ownership": rng.choice(
                    ["RENT", "OWN", "MORTGAGE"]
                ),
                "person_emp_length": employment,
                "loan_intent": rng.choice(
                    ["PERSONAL", "MEDICAL", "EDUCATION", "VENTURE"]
                ),
                "loan_grade": grade,
                "loan_amnt": loan,
                "loan_int_rate": int_rate,
                "loan_status": default,
                "loan_percent_income": percent,
                "cb_person_default_on_file": rng.choice(["Y", "N"]),
                "cb_person_cred_hist_length": int(rng.integers(1, 20)),
            }
        )
    return pd.DataFrame(rows)


def test_select_best_uses_roc_auc_then_f1() -> None:
    table = pd.DataFrame(
        {
            "variant": ["a", "b", "c"],
            "roc_auc_mean": [0.90, 0.90, 0.80],
            "f1_mean": [0.70, 0.75, 0.99],
        }
    )
    assert _select_best(table)["variant"] == "b"


def test_quick_training_writes_loadable_artifacts(tmp_path) -> None:
    data_path = tmp_path / "synthetic.csv"
    make_synthetic_raw().to_csv(data_path, index=False)

    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    metrics_path = tmp_path / "metrics.json"
    report_path = tmp_path / "report.md"
    figures_dir = tmp_path / "figures"

    summary = run_training(
        data_path=data_path,
        model_path=model_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        report_path=report_path,
        figures_dir=figures_dir,
        random_state=7,
        cv_splits=3,
        rf_n_iter=2,
        quick=True,
    )

    assert model_path.exists()
    assert metadata_path.exists()
    assert metrics_path.exists()
    assert report_path.exists()
    assert (figures_dir / "09_model_comparison.png").exists()
    assert (figures_dir / "10_confusion_matrix.png").exists()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["selected_model"] in {
        "logistic_regression",
        "decision_tree",
        "random_forest",
    }
    assert 0 < metadata["decision_threshold"] < 1
    assert 0 <= metadata["test_metrics"]["roc_auc"] <= 1
    assert metadata["features"] == FEATURE_COLUMNS
    assert len(metadata["encoded_features"]) >= 10
    assert all("loan_status" not in name for name in metadata["encoded_features"])

    loaded = joblib.load(model_path)
    raw = pd.read_csv(data_path)
    probabilities = loaded.predict_proba(raw[FEATURE_COLUMNS].head(5))
    assert probabilities.shape == (5, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    report = report_path.read_text()
    assert "## 5. Test-set results" in report
    assert "not used for selection" in report
    assert summary["selected_model"] == metadata["selected_model"]
