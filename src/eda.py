"""Exploratory data analysis for the Credit Risk Dataset.

Running ``python -m src.eda`` regenerates every chart in ``reports/figures``
plus ``reports/eda_summary.json`` and ``reports/eda_findings.md``.

All numbers in the generated report come from the dataset; nothing is hardcoded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe for CLI runs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.preprocessing import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    MAX_PLAUSIBLE_AGE,
    MAX_PLAUSIBLE_EMPLOYMENT_YEARS,
    NUMERIC_FEATURES,
    TARGET,
    clean_data,
    cleaning_report,
    load_data,
    split_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.titlesize": 11})

DEFAULT_COLORS = {0: "#4C72B0", 1: "#C44E52"}


def _save(fig: plt.Figure, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_distribution(df: pd.DataFrame) -> Path:
    counts = df[TARGET].value_counts().sort_index()
    labels = ["No default (0)", "Default (1)"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(
        labels,
        [counts.get(0, 0), counts.get(1, 0)],
        color=[DEFAULT_COLORS[0], DEFAULT_COLORS[1]],
    )
    total = len(df)
    for bar, value in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n({value / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_ylabel("Number of loans")
    ax.set_title(
        f"Target distribution — {counts.get(1, 0) / total:.1%} of loans defaulted"
    )
    return _save(fig, "01_target_distribution.png")


def plot_missing_values(df: pd.DataFrame) -> Path:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    else:
        bars = ax.barh(missing.index, missing.values, color="#DD8452")
        for bar, value in zip(bars, missing.values):
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f" {value:,} ({value / len(df):.1%})",
                va="center",
                fontsize=9,
            )
        ax.set_xlabel("Missing rows")
        ax.set_xlim(0, missing.max() * 1.3)
        ax.set_title("Missing values by column")
    return _save(fig, "02_missing_values.png")


def plot_numeric_distributions(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    axes = axes.ravel()
    for ax, column in zip(axes, NUMERIC_FEATURES):
        data = df[column].dropna()
        sns.histplot(data, bins=40, ax=ax, color="#4C72B0", edgecolor="white")
        if column == "person_income":
            ax.set_xscale("log")
            ax.set_title(f"{column}\n(log x-axis, skew={data.skew():.1f})", fontsize=9)
        else:
            ax.set_title(f"{column}\n(skew={data.skew():.1f})", fontsize=9)
        ax.axvline(data.median(), color="#C44E52", linestyle="--", linewidth=1)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(
        "Numeric feature distributions (dashed line = median)", fontsize=13, y=1.02
    )
    return _save(fig, "03_numeric_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    columns = NUMERIC_FEATURES + [TARGET]
    corr = df[columns].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Spearman correlation (robust to skew)")
    return _save(fig, "04_correlation_heatmap.png")


def plot_default_rate_by_category(df: pd.DataFrame) -> Path:
    overall = df[TARGET].mean()
    specs = [
        (
            "loan_grade",
            list("ABCDEFG"),
            "Loan grade (A = best, G = worst)",
        ),
        (
            "person_home_ownership",
            ["OWN", "MORTGAGE", "RENT", "OTHER"],
            "Home ownership",
        ),
        (
            "loan_intent",
            df.groupby("loan_intent")[TARGET].mean().sort_values().index.tolist(),
            "Loan purpose",
        ),
        (
            "cb_person_default_on_file",
            ["N", "Y"],
            "Prior default on credit file",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (column, order, title) in zip(axes.ravel(), specs):
        stats = (
            df.groupby(column, observed=True)[TARGET]
            .agg(["mean", "count"])
            .reindex(order)
            .dropna()
        )
        bars = ax.bar(
            stats.index.astype(str),
            stats["mean"],
            color="#4C72B0",
        )
        for bar, (_, row) in zip(bars, stats.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                row["mean"],
                f"{row['mean']:.1%}\n(n={int(row['count']):,})",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.axhline(overall, color="#C44E52", linestyle="--", linewidth=1)
        ax.set_ylim(0, max(stats["mean"].max() * 1.25, overall * 1.3))
        ax.set_ylabel("Default rate")
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(
        f"Default rate by applicant/loan category (dashed line = overall {overall:.1%})",
        fontsize=13,
        y=1.0,
    )
    return _save(fig, "05_default_rate_by_category.png")


def plot_numeric_by_target(df: pd.DataFrame) -> Path:
    specs = [
        ("loan_int_rate", "Interest rate (%)", False),
        ("loan_percent_income", "Loan amount / income", False),
        ("person_income", "Annual income (log scale)", True),
        ("person_age", "Age (years)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (column, label, log_scale) in zip(axes.ravel(), specs):
        sns.boxplot(
            data=df,
            x=TARGET,
            y=column,
            hue=TARGET,
            palette=DEFAULT_COLORS,
            legend=False,
            ax=ax,
        )
        if log_scale:
            ax.set_yscale("log")
        ax.set_xticks([0, 1], ["No default", "Default"])
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_title(column)
    fig.suptitle("Numerical features by observed outcome", fontsize=13, y=1.0)
    return _save(fig, "06_numeric_by_target.png")


def plot_percent_income_redundancy(df: pd.DataFrame) -> Path:
    calc = df["loan_amnt"] / df["person_income"]
    diff = calc - df["loan_percent_income"]
    within = float((diff.abs() <= 0.01).mean())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].scatter(calc, df["loan_percent_income"], s=5, alpha=0.25, color="#4C72B0")
    lim = [0, max(float(calc.quantile(0.999)), 0.85)]
    axes[0].plot(lim, lim, color="#C44E52", linestyle="--", linewidth=1.2, label="y = x")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("Computed: loan_amnt / person_income")
    axes[0].set_ylabel("Provided: loan_percent_income")
    axes[0].set_title(
        f"Correlation = {calc.corr(df['loan_percent_income']):.4f}\n"
        f"{within:.1%} of rows agree within 0.01"
    )
    axes[0].legend()

    clipped = diff.clip(-0.05, 0.05)
    sns.histplot(clipped, bins=60, ax=axes[1], color="#DD8452")
    axes[1].axvline(0, color="#C44E52", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Provided − computed (clipped to ±0.05)")
    axes[1].set_ylabel("Rows")
    axes[1].set_title(f"Max absolute difference = {diff.abs().max():.4f}")
    fig.suptitle(
        "Redundancy check: loan_percent_income is derived from loan_amnt / income",
        fontsize=13,
        y=1.03,
    )
    return _save(fig, "07_percent_income_redundancy.png")


def plot_outliers(df: pd.DataFrame) -> Path:
    specs = [
        ("person_age", "Age (years)", False, MAX_PLAUSIBLE_AGE),
        (
            "person_emp_length",
            "Employment length (years)",
            False,
            MAX_PLAUSIBLE_EMPLOYMENT_YEARS,
        ),
        ("person_income", "Annual income (log scale)", True, None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (column, label, log_scale, threshold) in zip(axes, specs):
        sns.boxplot(y=df[column].dropna(), ax=ax, color="#4C72B0")
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylabel(label)
        title = column
        if threshold is not None:
            title += f"\n>{threshold} treated as error"
        ax.set_title(title, fontsize=10)
    fig.suptitle("Outlier inspection (values are kept unless physically impossible)", fontsize=13)
    return _save(fig, "08_outliers.png")


def _build_summary(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    report = cleaning_report(raw)
    missing = raw.isna().sum()
    target_counts = clean[TARGET].value_counts().sort_index()

    numeric_summary = {
        column: {
            "mean": float(clean[column].mean()),
            "median": float(clean[column].median()),
            "skew": float(clean[column].skew()),
            "min": float(clean[column].min()),
            "max": float(clean[column].max()),
        }
        for column in NUMERIC_FEATURES
    }

    correlations = (
        clean[NUMERIC_FEATURES]
        .corrwith(clean[TARGET], method="spearman")
        .sort_values(key=abs, ascending=False)
    )

    categorical_rates = {}
    for column in [
        "loan_grade",
        "person_home_ownership",
        "loan_intent",
        "cb_person_default_on_file",
    ]:
        stats = clean.groupby(column, observed=True)[TARGET].agg(["mean", "count"])
        categorical_rates[column] = {
            str(index): {"default_rate": float(row["mean"]), "count": int(row["count"])}
            for index, row in stats.iterrows()
        }

    missingness = {}
    for column in ["person_emp_length", "loan_int_rate"]:
        grouped = clean.groupby(clean[column].isna())[TARGET].mean()
        missingness[column] = {
            "default_rate_when_present": float(grouped.get(False, float("nan"))),
            "default_rate_when_missing": float(grouped.get(True, float("nan"))),
        }

    calc_percent_income = clean["loan_amnt"] / clean["person_income"]
    difference = (calc_percent_income - clean["loan_percent_income"]).abs()

    X_train, X_test, y_train, y_test = split_data(clean)

    return {
        "dataset": {
            "rows_raw": int(len(raw)),
            "columns": int(raw.shape[1]),
            "columns_list": list(raw.columns),
            "missing_values": {
                key: int(value) for key, value in missing[missing > 0].items()
            },
            "duplicate_rows": report["duplicate_rows"],
            "target_counts": {
                "non_default_0": int(target_counts.get(0, 0)),
                "default_1": int(target_counts.get(1, 0)),
            },
            "default_rate": float(clean[TARGET].mean()),
        },
        "cleaning": {
            **report,
            "rows_after_cleaning": int(len(clean)),
            "duplicates_removed": int(report["duplicate_rows"]),
            "rows_removed_total": int(len(raw) - len(clean)),
        },
        "numeric_summary": numeric_summary,
        "spearman_correlation_with_target": {
            key: float(value) for key, value in correlations.items()
        },
        "categorical_default_rates": categorical_rates,
        "missingness_vs_target": missingness,
        "redundancy": {
            "pearson_correlation": float(
                calc_percent_income.corr(clean["loan_percent_income"])
            ),
            "max_absolute_difference": float(difference.max()),
            "share_within_001": float((difference <= 0.01).mean()),
        },
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_default_rate": float(y_train.mean()),
            "test_default_rate": float(y_test.mean()),
        },
    }


def _write_findings(summary: dict) -> Path:
    dataset = summary["dataset"]
    cleaning = summary["cleaning"]
    correlations = summary["spearman_correlation_with_target"]
    rates = summary["categorical_default_rates"]
    redundancy = summary["redundancy"]
    split = summary["split"]

    grade_rows = "\n".join(
        f"| {grade} | {stats['default_rate']:.1%} | {stats['count']:,} |"
        for grade, stats in rates["loan_grade"].items()
    )
    home_rows = "\n".join(
        f"| {home} | {stats['default_rate']:.1%} | {stats['count']:,} |"
        for home, stats in rates["person_home_ownership"].items()
    )
    intent_rows = "\n".join(
        f"| {intent} | {stats['default_rate']:.1%} | {stats['count']:,} |"
        for intent, stats in sorted(
            rates["loan_intent"].items(), key=lambda item: item[1]["default_rate"]
        )
    )
    correlation_rows = "\n".join(
        f"| `{feature}` | {value:+.3f} |" for feature, value in correlations.items()
    )

    text = f"""# EDA findings — Credit Risk Dataset

> Generated by `python -m src.eda`. All values come from the dataset.
> These are **associations, not proof of causation**.

## 1. Dataset and target

- Raw shape: **{dataset['rows_raw']:,} rows × {dataset['columns']} columns**
- After cleaning: **{cleaning['rows_after_cleaning']:,} rows**
- Duplicate rows removed: **{cleaning['duplicate_rows']:,}**
- Non-default (0): **{dataset['target_counts']['non_default_0']:,}**
- Default (1): **{dataset['target_counts']['default_1']:,}** (**{dataset['default_rate']:.1%}**)

A model that always predicts "no default" would score **{1 - dataset['default_rate']:.1%}**
accuracy while catching **zero** defaulters. That is why accuracy alone is not
the selection metric for this project.

## 2. Data quality

| Issue | Count | Decision |
|---|---:|---|
| Exact duplicate rows | {cleaning['duplicate_rows']:,} | Removed (prevents train/test leakage) |
| `person_age` > {MAX_PLAUSIBLE_AGE} | {cleaning['impossible_age']:,} | Set to `NaN`, imputed later |
| `person_emp_length` > {MAX_PLAUSIBLE_EMPLOYMENT_YEARS} | {cleaning['impossible_employment_length']:,} | Set to `NaN`, imputed later |
| Employment longer than age | {cleaning['employment_longer_than_life']:,} | Set to `NaN`, imputed later |
| Missing `person_emp_length` | {cleaning['missing_person_emp_length']:,} | Median imputation + missing indicator |
| Missing `loan_int_rate` | {cleaning['missing_loan_int_rate']:,} | Median imputation + missing indicator |

Rows are **not deleted** for invalid values: the rest of the row is still useful,
and the imputer learns from training data only.

## 3. Strongest numerical associations with default

Spearman correlation (rank-based, so income outliers do not dominate):

| Feature | ρ with `loan_status` |
|---|---:|
{correlation_rows}

Correlation only measures monotonic association. It does not tell us that a
feature causes default, and it cannot capture interactions between features.

## 4. Default rate by category

**Loan grade**

| Grade | Default rate | Loans |
|---|---:|---:|
{grade_rows}

**Home ownership**

| Ownership | Default rate | Loans |
|---|---:|---:|
{home_rows}

**Loan purpose**

| Purpose | Default rate | Loans |
|---|---:|---:|
{intent_rows}

Grade shows a monotonic pattern (A → G), which is why `loan_grade` is encoded
as an **ordinal** feature rather than one-hot encoded.

## 5. Missingness is informative

| Column | Default rate when present | Default rate when missing |
|---|---:|---:|
| `person_emp_length` | {summary['missingness_vs_target']['person_emp_length']['default_rate_when_present']:.1%} | {summary['missingness_vs_target']['person_emp_length']['default_rate_when_missing']:.1%} |
| `loan_int_rate` | {summary['missingness_vs_target']['loan_int_rate']['default_rate_when_present']:.1%} | {summary['missingness_vs_target']['loan_int_rate']['default_rate_when_missing']:.1%} |

Because missing employment length is associated with a higher default rate, the
pipeline keeps a missing-value indicator rather than only filling the gap. The
{cleaning['impossible_age']} impossible ages become missing during cleaning, so
`person_age` also receives an indicator column (a small, harmless addition).

## 6. Redundancy: `loan_percent_income`

- Pearson correlation with `loan_amnt / person_income`: **{redundancy['pearson_correlation']:.4f}**
- Maximum absolute difference: **{redundancy['max_absolute_difference']:.4f}**
- Share of rows within 0.01: **{redundancy['share_within_001']:.1%}**

The provided column is almost identical to the computed ratio, so the project
keeps the provided column and does **not** add a duplicate feature. The small
differences are likely rounding or an income figure that changed after pricing;
they are not modelled separately.

## 7. Train/test split

- Training rows: **{split['train_rows']:,}** ({split['train_default_rate']:.2%} default)
- Test rows: **{split['test_rows']:,}** ({split['test_default_rate']:.2%} default)
- Stratified 80/20 split, `random_state=42`

All imputers, scalers and encoders are fitted **inside** the pipeline on the
training split only. The test set is untouched until final evaluation.

## 8. Charts

1. `01_target_distribution.png`
2. `02_missing_values.png`
3. `03_numeric_distributions.png`
4. `04_correlation_heatmap.png`
5. `05_default_rate_by_category.png`
6. `06_numeric_by_target.png`
7. `07_percent_income_redundancy.png`
8. `08_outliers.png`
"""
    path = REPORTS_DIR / "eda_findings.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_eda(
    df_raw: pd.DataFrame | None = None,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> dict:
    """Run the full EDA, save charts and reports, and return the summary dict."""
    if df_raw is None:
        df_raw = load_data(data_path)
    df = clean_data(df_raw)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_target_distribution(df)
    plot_missing_values(df_raw)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(df)
    plot_default_rate_by_category(df)
    plot_numeric_by_target(df)
    plot_percent_income_redundancy(df)
    plot_outliers(df)

    summary = _build_summary(df_raw, df)
    with (REPORTS_DIR / "eda_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            default=lambda value: value.item()
            if hasattr(value, "item")
            else str(value),
        )
    _write_findings(summary)
    return summary


def main() -> None:
    summary = run_eda()
    print("EDA complete.")
    print(f"  raw rows : {summary['dataset']['rows_raw']:,}")
    print(f"  cleaned  : {summary['cleaning']['rows_after_cleaning']:,}")
    print(f"  defaults : {summary['dataset']['default_rate']:.2%}")
    print(f"  figures  : {FIGURES_DIR}")
    print(f"  report   : {REPORTS_DIR / 'eda_findings.md'}")


if __name__ == "__main__":
    main()
