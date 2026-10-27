"""Model evaluation helpers: cross-validation, threshold tuning and plots.

Everything here is model-agnostic. The training script passes in already-built
scikit-learn pipelines so evaluation and preprocessing stay separated.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

RANDOM_STATE = 42
SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "accuracy": "accuracy",
}

FAMILY_COLORS = {
    "logistic_regression": "#4C72B0",
    "decision_tree": "#DD8452",
    "random_forest": "#55A868",
}

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150})


def make_cv(n_splits: int = 5, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """Stratified folds keep the ~22% default rate in every fold."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_model(pipeline, X, y, cv=None) -> dict:
    """Return mean/std/fold scores for every metric in ``SCORING``."""
    cv = cv if cv is not None else make_cv()
    result = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
        error_score="raise",
        return_train_score=False,
    )
    output = {}
    for metric in SCORING:
        values = result[f"test_{metric}"]
        output[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "folds": [float(value) for value in values],
        }
    output["fit_time_seconds"] = float(np.mean(result["fit_time"]))
    return output


def binary_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Compute threshold-dependent and threshold-independent metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "support": {
            "non_default": int((y_true == 0).sum()),
            "default": int((y_true == 1).sum()),
        },
    }


def threshold_sweep(y_true, y_proba, low: float = 0.01, high: float = 0.99, steps: int = 197) -> pd.DataFrame:
    """Precision, recall and F1 at each candidate threshold."""
    thresholds = np.linspace(low, high, steps)
    rows = []
    for threshold in thresholds:
        y_pred = (np.asarray(y_proba) >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def find_best_threshold(y_true, y_proba) -> dict:
    """Pick the F1-maximising threshold on out-of-fold predictions.

    Using out-of-fold predictions means the threshold is selected without
    touching the test set. Ties go to the higher threshold (more conservative).
    """
    sweep = threshold_sweep(y_true, y_proba)
    best_f1 = sweep["f1"].max()
    candidates = sweep[np.isclose(sweep["f1"], best_f1)]
    best = candidates.iloc[-1]
    return {
        "threshold": float(best["threshold"]),
        "f1": float(best["f1"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "criterion": "max F1 on out-of-fold training predictions",
    }


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(results: pd.DataFrame, path: str | Path) -> Path:
    """Four panels: ROC-AUC, F1, recall and precision for each baseline."""
    metrics = [
        ("roc_auc_mean", "ROC-AUC"),
        ("f1_mean", "F1 (default class)"),
        ("recall_mean", "Recall (default class)"),
        ("precision_mean", "Precision (default class)"),
    ]
    data = results.copy()
    data["label"] = data.apply(
        lambda row: row["family"].replace("_", " ").title()
        + (" + balanced" if row["class_weight"] == "balanced" else ""),
        axis=1,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (column, title) in zip(axes.ravel(), metrics):
        ordered = data.sort_values(column, ascending=True)
        colors = [
            to_rgba(
                FAMILY_COLORS[row["family"]],
                0.55 if row["class_weight"] == "balanced" else 0.95,
            )
            for _, row in ordered.iterrows()
        ]
        bars = ax.barh(ordered["label"], ordered[column], color=colors)
        for bar, value in zip(bars, ordered[column]):
            ax.text(
                value + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=8,
            )
        ax.set_xlim(0, 1.05)
        ax.set_title(title)
        ax.set_xlabel("mean 5-fold CV score")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(
        "Baseline model comparison — 5-fold stratified CV on training data\n"
        "(lighter bars = class_weight='balanced')",
        fontsize=12,
    )
    return _save(fig, path)


def plot_confusion_matrix(metrics: dict, path: str | Path) -> Path:
    cm = metrics["confusion_matrix"]
    matrix = np.array(
        [
            [cm["true_negative"], cm["false_positive"]],
            [cm["false_negative"], cm["true_positive"]],
        ]
    )
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = matrix / np.maximum(row_totals, 1)
    annotations = np.array(
        [
            [f"{matrix[i, j]:,}\n({percentages[i, j]:.1%})" for j in range(2)]
            for i in range(2)
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted non-default", "Predicted default"],
        yticklabels=["Actual non-default", "Actual default"],
        ax=ax,
        annot_kws={"fontsize": 10},
    )
    ax.set_title(
        f"Confusion matrix — threshold {metrics['threshold']:.3f}\n"
        f"F1={metrics['f1']:.3f}, recall={metrics['recall']:.3f}, "
        f"precision={metrics['precision']:.3f}"
    )
    ax.set_ylabel("Actual label")
    ax.set_xlabel("Predicted label")
    return _save(fig, path)


def plot_roc_curve(y_true, y_proba, path: str | Path) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#C44E52", linewidth=2, label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — test set")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return _save(fig, path)


def plot_precision_recall_curve(y_true, y_proba, path: str | Path) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#4C72B0", linewidth=2, label=f"Model (AP = {ap:.3f})")
    ax.axhline(baseline, linestyle="--", color="grey", label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall (default class)")
    ax.set_ylabel("Precision (default class)")
    ax.set_title("Precision–recall curve — test set")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return _save(fig, path)


def plot_threshold_curve(y_true, y_proba, chosen_threshold: float, path: str | Path) -> Path:
    sweep = threshold_sweep(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sweep["threshold"], sweep["precision"], label="Precision", color="#4C72B0")
    ax.plot(sweep["threshold"], sweep["recall"], label="Recall", color="#C44E52")
    ax.plot(sweep["threshold"], sweep["f1"], label="F1", color="#55A868")
    ax.axvline(
        chosen_threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Chosen = {chosen_threshold:.3f}",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold trade-off on out-of-fold training predictions")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return _save(fig, path)


def _pretty_feature_name(name: str) -> str:
    name = name.split("__", 1)[-1]
    if name.startswith("missingindicator_"):
        return name.replace("missingindicator_", "") + " (was missing)"
    return name.replace("_", " ")


def plot_feature_importance(pipeline, path: str | Path, top_n: int = 15) -> Path:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = [_pretty_feature_name(n) for n in preprocessor.get_feature_names_out()]

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        order = np.argsort(values)[-top_n:]
        colors = ["#55A868"] * len(order)
        xlabel = "Mean decrease in impurity (RF importance)"
        title = "Random Forest feature importance"
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_[0])
        order = np.argsort(np.abs(values))[-top_n:]
        colors = ["#C44E52" if values[i] > 0 else "#4C72B0" for i in order]
        xlabel = "Coefficient (positive → default)"
        title = "Logistic Regression coefficients"
    else:
        raise ValueError("Model exposes neither feature_importances_ nor coef_")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [names[i] for i in order],
        [values[i] for i in order],
        color=colors,
    )
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title} (top {top_n})")
    return _save(fig, path)


def plot_calibration_curve(y_true, y_proba, path: str | Path, n_bins: int = 10) -> Path:
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="quantile"
    )
    brier = brier_score_loss(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        color="#4C72B0",
        label=f"Model (Brier = {brier:.3f})",
    )
    ax.set_xlabel("Predicted default probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration curve — test set")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return _save(fig, path)


def save_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            indent=2,
            default=lambda value: value.item()
            if hasattr(value, "item")
            else str(value),
        )
    return path
