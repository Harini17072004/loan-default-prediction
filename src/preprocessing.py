"""Data loading, cleaning and leakage-safe preprocessing.

The golden rule in this module: anything that *learns* from data (medians,
most-frequent categories, scaling statistics, encoders) lives inside the
scikit-learn ``Pipeline``. Only row-wise, stateless cleaning happens before the
train/test split.

Run ``python -m src.eda`` to regenerate the exploratory charts.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "loan_default_pipeline.joblib"

TARGET = "loan_status"
POSITIVE_LABEL = 1  # 1 = defaulted

# --- Feature groups -------------------------------------------------------
# Income and loan amount are heavily right-skewed (income skew ~33), so they
# get a log1p transform. The remaining numeric columns feed the standard branch.
LOG_FEATURES = ["person_income", "loan_amnt"]
NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
NUMERIC_BASE_FEATURES = [c for c in NUMERIC_FEATURES if c not in LOG_FEATURES]

# loan_grade is ordinal: A (lowest risk) ... G (highest risk).
ORDINAL_FEATURES = ["loan_grade"]
ORDINAL_CATEGORIES = [["A", "B", "C", "D", "E", "F", "G"]]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + ORDINAL_FEATURES + CATEGORICAL_FEATURES

# --- Cleaning thresholds --------------------------------------------------
# Ages above this are treated as data-entry errors (the raw data contains 123
# and 144). Age 84 and 94 are unusual but plausible, so they are kept. Being
# conservative here avoids throwing away potentially valid applicants.
MAX_PLAUSIBLE_AGE = 100
# Employment length above this is impossible (raw max = 123 years). We do NOT
# use "age - 16" as a rule: 740 rows would be nulled for plausible teenage
# work histories. Only truly impossible cases (employed longer than alive) are
# nulled.
MAX_PLAUSIBLE_EMPLOYMENT_YEARS = 60


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and fail with an actionable message if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download it with `python data/download_data.py` or place "
            "credit_risk_dataset.csv in the data/ directory."
        )

    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(sorted(missing))
        )
    return df


def cleaning_report(df: pd.DataFrame) -> dict[str, int]:
    """Count data-quality problems *before* cleaning.

    Kept separate from :func:`clean_data` so notebooks and reports can show
    exactly how many rows were affected by each rule.
    """
    return {
        "rows": int(len(df)),
        "duplicate_rows": int(df.duplicated().sum()),
        "impossible_age": int((df["person_age"] > MAX_PLAUSIBLE_AGE).sum()),
        "impossible_employment_length": int(
            (df["person_emp_length"] > MAX_PLAUSIBLE_EMPLOYMENT_YEARS).sum()
        ),
        "employment_longer_than_life": int(
            (df["person_emp_length"] > df["person_age"]).sum()
        ),
        "missing_person_emp_length": int(df["person_emp_length"].isna().sum()),
        "missing_loan_int_rate": int(df["loan_int_rate"].isna().sum()),
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply row-wise cleaning that does not learn anything from the data.

    Steps
    -----
    1. Validate the schema and coerce numeric columns.
    2. Replace *impossible* ages / employment lengths with ``NaN`` instead of
       deleting the row. The row still contains useful information, and the
       pipeline's median imputer fills the gap using training data only.
    3. Drop exact duplicate rows. Duplicates are not "more evidence" and could
       appear in both train and test, inflating the test score.
    4. Drop rows with a missing target (none in this dataset, but defensive).

    The input DataFrame is never modified. No target-based statistics and no
    imputation happen here, so calling this before the split cannot leak
    information from the test set.
    """
    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            "Cannot clean data; missing columns: " + ", ".join(sorted(missing))
        )

    out = df.copy()

    for column in NUMERIC_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    age_error = out["person_age"] > MAX_PLAUSIBLE_AGE
    employment_error = (
        out["person_emp_length"] > MAX_PLAUSIBLE_EMPLOYMENT_YEARS
    ) | (out["person_emp_length"] > out["person_age"])
    out.loc[age_error, "person_age"] = np.nan
    out.loc[employment_error, "person_emp_length"] = np.nan

    out = out.drop_duplicates().reset_index(drop=True)

    if out[TARGET].isna().any():
        n_missing = int(out[TARGET].isna().sum())
        warnings.warn(
            f"Dropping {n_missing} rows with a missing target.", stacklevel=2
        )
        out = out.dropna(subset=[TARGET]).reset_index(drop=True)

    out[TARGET] = out[TARGET].astype(int)
    if not set(out[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary (0 = non-default, 1 = default).")

    return out


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return a reproducible, stratified train/test split.

    Stratification keeps the ~22% default rate in both parts, which matters for
    a minority class. The test set is only used once, at final evaluation.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].copy()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing transformer.

    It is intentionally *unfitted* so it can be placed inside a Pipeline and
    fitted on the training split only.
    """
    # Median is robust to the income/loan-amount outliers. add_indicator=True
    # creates a 0/1 column for columns that had missing values in training; this
    # is justified because missing employment length has a 31.7% default rate
    # versus 21.6% when present.
    base_numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    # Log-transform first so extreme incomes stop dominating, then scale.
    log_numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "log",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    grade = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num_base", base_numeric, NUMERIC_BASE_FEATURES),
            ("num_log", log_numeric, LOG_FEATURES),
            ("grade", grade, ORDINAL_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # guarantee loan_status can never sneak in
    )


def build_pipeline(estimator) -> Pipeline:
    """Combine the preprocessor and a classifier into one fitted artifact."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Human-readable names after encoding (useful for reports/coefficients)."""
    return list(preprocessor.get_feature_names_out())
