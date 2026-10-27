"""Train, tune, evaluate and save the loan-default model.

Usage:
    python train.py

The script follows this order strictly:
1. Clean + stratified train/test split.
2. Compare 3 algorithms, each with and without balanced class weights, using
   5-fold CV on the training split only.
3. Tune each algorithm with 5-fold CV on the training split only.
4. Select the best CV ROC-AUC, choose a decision threshold from out-of-fold
   training predictions, refit on all training rows.
5. Evaluate once on the untouched test set and save every artifact.

Nothing from the test split is used for selection or threshold tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import (
    binary_metrics,
    cross_validate_model,
    find_best_threshold,
    make_cv,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_curve,
    save_json,
)
from src.preprocessing import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    TARGET,
    build_pipeline,
    clean_data,
    load_data,
    split_data,
)

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "model_report.md"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def _format_metric(row: pd.Series, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"


def _format_params(params: dict) -> str:
    clean = {key.replace("model__", ""): value for key, value in params.items()}
    return ", ".join(f"{key}={value}" for key, value in clean.items())


def build_baseline_variants(quick: bool = False, random_state: int = 42) -> dict:
    """Three algorithms × unweighted/balanced. No hyperparameter tuning yet."""
    n_estimators = 32 if quick else 200
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=random_state
        ),
        "logistic_regression_balanced": LogisticRegression(
            max_iter=2000, random_state=random_state, class_weight="balanced"
        ),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "decision_tree_balanced": DecisionTreeClassifier(
            random_state=random_state, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=1
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
            class_weight="balanced",
        ),
    }


def compare_baselines(
    X_train, y_train, cv, quick: bool = False, random_state: int = 42
) -> tuple[pd.DataFrame, dict]:
    """Cross-validate all six baseline variants on the training data."""
    rows = []
    raw_scores = {}
    for name, estimator in build_baseline_variants(
        quick=quick, random_state=random_state
    ).items():
        start = time.time()
        scores = cross_validate_model(build_pipeline(estimator), X_train, y_train, cv)
        raw_scores[name] = scores
        family = name.replace("_balanced", "")
        row = {
            "variant": name,
            "family": family,
            "class_weight": "balanced" if name.endswith("_balanced") else "none",
            "fit_seconds": round(time.time() - start, 2),
        }
        for metric in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]:
            row[f"{metric}_mean"] = scores[metric]["mean"]
            row[f"{metric}_std"] = scores[metric]["std"]
        rows.append(row)
        print(
            f"  {name:32} AUC={row['roc_auc_mean']:.4f}  "
            f"F1={row['f1_mean']:.4f}  R={row['recall_mean']:.4f}  "
            f"P={row['precision_mean']:.4f}"
        )
    return pd.DataFrame(rows), raw_scores


def tune_models(
    X_train,
    y_train,
    cv,
    random_state: int = 42,
    rf_n_iter: int = 24,
    quick: bool = False,
) -> dict:
    """Tune each family with CV on the training split only."""
    searches = {}

    lr_grid = {
        "model__C": [0.1, 1] if quick else [0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"],
    }
    searches["logistic_regression"] = GridSearchCV(
        build_pipeline(LogisticRegression(max_iter=2000, random_state=random_state)),
        lr_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )

    dt_grid = {
        "model__max_depth": [4, 8] if quick else [4, 6, 8, 12, 16, None],
        "model__min_samples_leaf": [1, 5] if quick else [1, 5, 10, 20],
        "model__class_weight": [None, "balanced"],
    }
    searches["decision_tree"] = GridSearchCV(
        build_pipeline(DecisionTreeClassifier(random_state=random_state)),
        dt_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )

    rf_space = {
        "model__max_depth": [4, 8] if quick else [8, 12, 16, 20, 24, None],
        "model__min_samples_leaf": [1, 5] if quick else [1, 3, 5, 10, 20],
        "model__max_features": ["sqrt"] if quick else ["sqrt", 0.3, 0.5],
        "model__class_weight": [None, "balanced"],
    }
    searches["random_forest"] = RandomizedSearchCV(
        build_pipeline(
            RandomForestClassifier(
                n_estimators=32 if quick else 200,
                random_state=random_state,
                n_jobs=1,
            )
        ),
        rf_space,
        n_iter=rf_n_iter,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        random_state=random_state,
    )

    fitted = {}
    for name, search in searches.items():
        start = time.time()
        print(f"  tuning {DISPLAY_NAMES[name]} ...", end=" ", flush=True)
        search.fit(X_train, y_train)
        fitted[name] = search
        print(
            f"best CV AUC={search.best_score_:.4f} "
            f"({time.time() - start:.1f}s) {search.best_params_}"
        )
    return fitted


def build_tuned_table(searches: dict, X_train, y_train, cv) -> tuple[pd.DataFrame, dict]:
    """Full CV metrics for each tuned estimator (not just the tuned AUC)."""
    rows = []
    raw_scores = {}
    for name, search in searches.items():
        scores = cross_validate_model(search.best_estimator_, X_train, y_train, cv)
        raw_scores[name] = scores
        row = {
            "variant": name,
            "family": name,
            "class_weight": search.best_params_.get("model__class_weight") or "none",
            "params": dict(search.best_params_),
        }
        for metric in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]:
            row[f"{metric}_mean"] = scores[metric]["mean"]
            row[f"{metric}_std"] = scores[metric]["std"]
        rows.append(row)
    return pd.DataFrame(rows), raw_scores


def _select_best(tuned: pd.DataFrame) -> pd.Series:
    """Primary metric ROC-AUC; F1 is the tie-breaker."""
    ranked = tuned.sort_values(
        ["roc_auc_mean", "f1_mean"], ascending=False
    ).reset_index(drop=True)
    return ranked.iloc[0]


def _top_features(pipeline, top_n: int = 15) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = [str(name) for name in preprocessor.get_feature_names_out()]

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        kind = "mean_decrease_in_impurity"
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_[0])
        kind = "logistic_coefficient"
    else:
        return []

    order = np.argsort(np.abs(values))[::-1][:top_n]
    return [
        {"feature": names[index], "value": float(values[index]), "kind": kind}
        for index in order
    ]


def _markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _write_report(path: Path, context: dict) -> Path:
    baseline: pd.DataFrame = context["baseline"]
    tuned: pd.DataFrame = context["tuned"]
    best_row = context["best_row"]
    best_name = context["best_name"]
    test_default = context["test_default"]
    test_tuned = context["test_tuned"]
    threshold_info = context["threshold_info"]
    top_features = context["top_features"]
    dataset = context["dataset"]
    params = context["best_params"]

    baseline_rows = [
        [
            DISPLAY_NAMES[row["family"]],
            row["class_weight"],
            _format_metric(row, "roc_auc"),
            _format_metric(row, "pr_auc"),
            _format_metric(row, "precision"),
            _format_metric(row, "recall"),
            _format_metric(row, "f1"),
        ]
        for _, row in baseline.iterrows()
    ]

    weight_rows = []
    for family in ["logistic_regression", "decision_tree", "random_forest"]:
        plain = baseline[baseline["variant"] == family].iloc[0]
        balanced = baseline[baseline["variant"] == f"{family}_balanced"].iloc[0]
        weight_rows.append(
            [
                DISPLAY_NAMES[family],
                f"{plain['recall_mean']:.3f}",
                f"{balanced['recall_mean']:.3f}",
                f"{balanced['recall_mean'] - plain['recall_mean']:+.3f}",
                f"{plain['precision_mean']:.3f}",
                f"{balanced['precision_mean']:.3f}",
                f"{plain['f1_mean']:.3f}",
                f"{balanced['f1_mean']:.3f}",
            ]
        )

    tuned_rows = [
        [
            DISPLAY_NAMES[row["family"]],
            row["class_weight"],
            _format_params(row["params"]),
            _format_metric(row, "roc_auc"),
            _format_metric(row, "f1"),
            _format_metric(row, "recall"),
            _format_metric(row, "precision"),
        ]
        for _, row in tuned.sort_values("roc_auc_mean", ascending=False).iterrows()
    ]

    test_rows = []
    for label, metrics in [("0.50 (default)", test_default), ("selected", test_tuned)]:
        test_rows.append(
            [
                f"{metrics['threshold']:.3f}",
                label,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['roc_auc']:.4f}",
                f"{metrics['pr_auc']:.4f}",
            ]
        )

    cm = test_tuned["confusion_matrix"]
    cm_rows = [
        ["Actual non-default", f"{cm['true_negative']:,}", f"{cm['false_positive']:,}"],
        ["Actual default", f"{cm['false_negative']:,}", f"{cm['true_positive']:,}"],
    ]

    feature_rows = [
        [item["feature"], f"{item['value']:.4f}"]
        for item in top_features
    ]

    report = f"""# Model training and evaluation report

> Generated by `python train.py` on {context['trained_at']}.
> Test metrics come from **one evaluation** after all model and threshold choices
> were frozen. They are not used for selection.

## Selected model

- **Algorithm:** {DISPLAY_NAMES[best_name]}
- **Class weights:** {best_row['class_weight']}
- **Hyperparameters:** `{_format_params(params)}`
- **Decision threshold:** **{threshold_info['threshold']:.3f}**
  ({threshold_info['criterion']})
- **Test ROC-AUC:** **{test_tuned['roc_auc']:.4f}**
- **Test PR-AUC:** **{test_tuned['pr_auc']:.4f}**
- **Test precision / recall / F1:** **{test_tuned['precision']:.4f} / {test_tuned['recall']:.4f} / {test_tuned['f1']:.4f}**

Selection rule: highest mean **5-fold CV ROC-AUC** on the training split, with
mean CV F1 as tie-breaker. ROC-AUC is threshold-independent and not inflated by the
majority class. Accuracy is reported but never used to select the model.

Dataset: **{dataset['clean_rows']:,}** cleaned rows → **{dataset['train_rows']:,}**
training / **{dataset['test_rows']:,}** test, stratified 80/20 (`random_state=42`).

## 1. Baseline comparison (5-fold CV, training data only)

{_markdown_table(
    ["Algorithm", "Weights", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"],
    baseline_rows,
)}

Random Forest is far ahead of the single tree and Logistic Regression on both
ROC-AUC and PR-AUC. A single unpruned tree overfits; Logistic Regression captures
only a linear decision boundary.

## 2. Does `class_weight="balanced"` help?

{_markdown_table(
    ["Algorithm", "Recall (none)", "Recall (balanced)", "Δ recall", "Precision (none)", "Precision (balanced)", "F1 (none)", "F1 (balanced)"],
    weight_rows,
)}

Balanced weights shift the operating point toward the minority class: recall rises
and precision usually falls. For Logistic Regression the recall gain is large. For
the tree and forest the effect on ROC-AUC is small; the ranking quality barely
changes, but the default 0.5 cut-off catches more defaults and raises more false
alarms. The final threshold is therefore tuned separately instead of relying on
weights alone.

## 3. Tuned models (5-fold CV, training data only)

{_markdown_table(
    ["Algorithm", "Weights", "Best parameters", "ROC-AUC", "F1", "Recall", "Precision"],
    tuned_rows,
)}

Tuning improved the Decision Tree substantially by limiting depth and leaf size.
The forest needed only a small adjustment. The selected model remains
**{DISPLAY_NAMES[best_name]}**.

## 4. Decision threshold

The 0.5 cut-off is arbitrary. I computed **out-of-fold** probabilities for every
training row (each row predicted only when it was in a validation fold), then chose
the threshold that maximised F1 for the default class. No test row was involved.

- Chosen threshold: **{threshold_info['threshold']:.3f}**
- Out-of-fold precision / recall / F1 at that threshold:
  **{threshold_info['precision']:.3f} / {threshold_info['recall']:.3f} / {threshold_info['f1']:.3f}**

Lowering the threshold would increase recall at the cost of precision. A lender
with a high cost of missed defaults could choose a lower value; the prediction
service exposes the probability so this remains a business decision.

## 5. Test-set results (evaluated once)

{_markdown_table(
    ["Threshold", "Setting", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
    test_rows,
)}

**Confusion matrix at the selected threshold ({test_tuned['threshold']:.3f}):**

{_markdown_table(
    ["Actual \\ Predicted", "Predicted non-default", "Predicted default"],
    cm_rows,
)}

Of **{test_tuned['support']['default']:,}** actual defaults, the model caught
**{cm['true_positive']:,}** and missed **{cm['false_negative']:,}**. It also
flagged **{cm['false_positive']:,}** good loans as risky (out of
**{test_tuned['support']['non_default']:,}**).

## 6. Most influential features

{_markdown_table(["Feature", "Importance / coefficient"], feature_rows)}

For a Random Forest these are impurity-based importances. They show what the model
uses, not what causes default, and they can favour continuous/high-cardinality
features. Correlated features share credit.

## 7. Interpretation and limitations

- **Accuracy is not the selection metric.** With an 80/20 split, a constant
  "no default" prediction already scores ~78% accuracy.
- **Recall vs precision is a business choice.** This report uses F1 to pick a
  balanced threshold; a bank may prefer higher recall or higher precision.
- **This is synthetic data.** The dataset is CC0 and simulated; results do not
  transfer automatically to a real loan book.
- **No temporal validation.** A real deployment should be tested on later loans to
  detect drift.
- **Lender-assigned fields.** `loan_grade` and `loan_int_rate` may already encode
  the lender's risk assessment, so the model partly learns that judgement.
- **Probabilities are model scores**, not audited real-world default frequencies.
  The calibration plot is a sanity check, not a guarantee.
- **No fairness analysis.** Age, income and home ownership can correlate with
  protected characteristics; a real system needs bias testing and human review.

## 8. Artifacts and reproduction

- `models/loan_default_pipeline.joblib` — preprocessing + model, load with `joblib`
- `models/model_metadata.json` — threshold, features, metrics, library versions
- `reports/model_metrics.json` — full CV and test metrics
- `reports/figures/09_model_comparison.png` … `15_calibration_curve.png`

```bash
cd ~/loan-default-prediction
source .venv/bin/activate
python train.py
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path


def run_training(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    random_state: int = 42,
    cv_splits: int = 5,
    rf_n_iter: int = 24,
    quick: bool = False,
) -> dict:
    """Run the full training pipeline and return the metadata dictionary."""
    data_path = Path(data_path)
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    metrics_path = Path(metrics_path)
    report_path = Path(report_path)
    figures_dir = Path(figures_dir)

    if quick:
        cv_splits = min(cv_splits, 3)
        rf_n_iter = min(rf_n_iter, 2)

    print("Loading and cleaning data ...")
    raw = load_data(data_path)
    clean = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(
        clean, test_size=0.2, random_state=random_state
    )
    print(
        f"  raw={len(raw):,}  clean={len(clean):,}  "
        f"train={len(X_train):,}  test={len(X_test):,}"
    )
    print(f"  train default rate={y_train.mean():.2%}  test={y_test.mean():.2%}")

    cv = make_cv(n_splits=cv_splits, random_state=random_state)

    print("\nBaseline comparison (5-fold CV on training data):")
    baseline, _ = compare_baselines(
        X_train, y_train, cv, quick=quick, random_state=random_state
    )

    print("\nTuning on training data only:")
    searches = tune_models(
        X_train,
        y_train,
        cv,
        random_state=random_state,
        rf_n_iter=rf_n_iter,
        quick=quick,
    )

    tuned, _ = build_tuned_table(searches, X_train, y_train, cv)
    best_row = _select_best(tuned)
    best_name = str(best_row["variant"])
    best_search = searches[best_name]
    best_params = dict(best_search.best_params_)
    print(
        f"\nSelected: {DISPLAY_NAMES[best_name]} "
        f"(CV ROC-AUC={best_row['roc_auc_mean']:.4f})"
    )

    # Threshold from out-of-fold predictions on the training split only.
    print("Selecting threshold from out-of-fold training predictions ...")
    oof_proba = cross_val_predict(
        clone(best_search.best_estimator_),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold_info = find_best_threshold(y_train, oof_proba)
    print(
        f"  threshold={threshold_info['threshold']:.3f} "
        f"(OOF F1={threshold_info['f1']:.4f})"
    )

    final_pipeline = clone(best_search.best_estimator_)
    final_pipeline.fit(X_train, y_train)

    # All choices are frozen; evaluate the untouched test set once.
    test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    test_default = binary_metrics(y_test, test_proba, threshold=0.5)
    test_tuned = binary_metrics(y_test, test_proba, threshold_info["threshold"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path, compress=3)
    loaded = joblib.load(model_path)
    sanity = loaded.predict_proba(X_test.iloc[:3])
    if sanity.shape != (3, 2):
        raise RuntimeError("Saved model failed a prediction sanity check.")

    print("\nSaving reports and figures ...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(baseline, figures_dir / "09_model_comparison.png")
    plot_confusion_matrix(test_tuned, figures_dir / "10_confusion_matrix.png")
    plot_roc_curve(y_test, test_proba, figures_dir / "11_roc_curve.png")
    plot_precision_recall_curve(
        y_test, test_proba, figures_dir / "12_precision_recall_curve.png"
    )
    plot_threshold_curve(
        y_train,
        oof_proba,
        threshold_info["threshold"],
        figures_dir / "13_threshold_analysis.png",
    )
    plot_feature_importance(final_pipeline, figures_dir / "14_feature_importance.png")
    plot_calibration_curve(y_test, test_proba, figures_dir / "15_calibration_curve.png")

    top_features = _top_features(final_pipeline)
    encoded_features = [
        str(name)
        for name in final_pipeline.named_steps["preprocessor"].get_feature_names_out()
    ]
    dataset_info = {
        "path": str(data_path.relative_to(PROJECT_ROOT))
        if data_path.is_relative_to(PROJECT_ROOT)
        else str(data_path),
        "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_default_rate": float(y_train.mean()),
        "test_default_rate": float(y_test.mean()),
    }
    library_versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    metadata = {
        "project": "loan-default-prediction",
        "trained_at": trained_at,
        "selected_model": best_name,
        "display_name": DISPLAY_NAMES[best_name],
        "estimator_class": type(final_pipeline.named_steps["model"]).__name__,
        "selection_rule": (
            "highest mean 5-fold CV ROC-AUC on the training split; "
            "mean CV F1 as tie-breaker"
        ),
        "hyperparameters": {
            key.replace("model__", ""): value for key, value in best_params.items()
        },
        "decision_threshold": threshold_info["threshold"],
        "threshold_selection": threshold_info,
        "target": {
            "name": TARGET,
            "positive_label": 1,
            "negative_label": 0,
            "positive_meaning": "default",
            "negative_meaning": "no default",
        },
        "features": FEATURE_COLUMNS,
        "encoded_features": encoded_features,
        "cv_roc_auc": float(best_row["roc_auc_mean"]),
        "test_metrics": test_tuned,
        "test_metrics_at_0_5": test_default,
        "top_features": top_features,
        "dataset": dataset_info,
        "library_versions": library_versions,
        "artifact": {
            "path": str(model_path.relative_to(PROJECT_ROOT))
            if model_path.is_relative_to(PROJECT_ROOT)
            else str(model_path),
            "size_mb": round(model_path.stat().st_size / 1_000_000, 2),
        },
    }

    save_json(metadata, metadata_path)
    metrics_payload = {
        **metadata,
        "baseline_cv": baseline.to_dict(orient="records"),
        "tuned_cv": tuned.to_dict(orient="records"),
    }
    save_json(metrics_payload, metrics_path)

    _write_report(
        report_path,
        {
            "baseline": baseline,
            "tuned": tuned,
            "best_row": best_row,
            "best_name": best_name,
            "best_params": best_params,
            "test_default": test_default,
            "test_tuned": test_tuned,
            "threshold_info": threshold_info,
            "top_features": top_features,
            "dataset": dataset_info,
            "trained_at": trained_at,
        },
    )

    summary = {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "selected_model": best_name,
        "threshold": threshold_info["threshold"],
        "cv_roc_auc": float(best_row["roc_auc_mean"]),
        "test_metrics": test_tuned,
        "test_metrics_at_0_5": test_default,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--rf-iter", type=int, default=24)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small grids for a smoke test (not for reporting results).",
    )
    args = parser.parse_args()

    summary = run_training(
        data_path=args.data,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        metrics_path=args.metrics_path,
        report_path=args.report_path,
        figures_dir=args.figures_dir,
        random_state=args.random_state,
        cv_splits=args.cv_splits,
        rf_n_iter=args.rf_iter,
        quick=args.quick,
    )

    test = summary["test_metrics"]
    print("\nTraining complete.")
    print(f"  model      : {summary['model_path']}")
    print(f"  selected   : {DISPLAY_NAMES[summary['selected_model']]}")
    print(f"  threshold  : {summary['threshold']:.3f}")
    print(
        f"  test       : ROC-AUC={test['roc_auc']:.4f}  "
        f"PR-AUC={test['pr_auc']:.4f}  F1={test['f1']:.4f}  "
        f"recall={test['recall']:.4f}  precision={test['precision']:.4f}"
    )


if __name__ == "__main__":
    main()
