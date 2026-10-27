"""Tests for cleaning, splitting and the leakage-safe preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.preprocessing import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    MAX_PLAUSIBLE_AGE,
    NUMERIC_BASE_FEATURES,
    TARGET,
    build_pipeline,
    build_preprocessor,
    clean_data,
    cleaning_report,
    get_feature_names,
    load_data,
    split_data,
)

BASE_ROW = {
    "person_age": 30,
    "person_income": 50_000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5.0,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 10_000,
    "loan_int_rate": 10.0,
    "loan_status": 0,
    "loan_percent_income": 0.2,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 5,
}


def make_row(**overrides) -> dict:
    return {**BASE_ROW, **overrides}


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Small synthetic dataset with duplicates, impossible values and 7/7 classes."""
    rows = [
        make_row(),
        make_row(),  # exact duplicate
        make_row(
            person_age=144,
            person_income=6_000_000,
            loan_int_rate=np.nan,
            loan_grade="A",
            loan_status=1,
            cb_person_default_on_file="Y",
        ),
        make_row(person_emp_length=123.0, loan_grade="C", loan_status=0),
        make_row(person_age=22, person_emp_length=30.0, loan_grade="D", loan_status=1),
        make_row(
            person_age=45,
            person_income=120_000,
            person_emp_length=20.0,
            loan_intent="MEDICAL",
            loan_grade="G",
            loan_percent_income=0.3,
            loan_status=1,
            cb_person_default_on_file="Y",
        ),
        make_row(
            person_age=35,
            person_emp_length=np.nan,
            loan_intent="DEBTCONSOLIDATION",
            loan_grade="E",
            loan_status=0,
        ),
        make_row(
            person_age=28,
            loan_int_rate=np.nan,
            loan_intent="EDUCATION",
            loan_grade="F",
            loan_status=1,
        ),
        make_row(
            person_age=50,
            person_income=90_000,
            person_home_ownership="MORTGAGE",
            person_emp_length=25.0,
            loan_intent="HOMEIMPROVEMENT",
            loan_grade="A",
            loan_status=0,
            cb_person_cred_hist_length=20,
        ),
        make_row(
            person_age=33,
            person_income=70_000,
            person_home_ownership="OWN",
            person_emp_length=8.0,
            loan_grade="C",
            loan_status=1,
        ),
        make_row(
            person_age=26,
            person_income=30_000,
            person_emp_length=2.0,
            loan_intent="VENTURE",
            loan_status=0,
        ),
        make_row(
            person_age=41,
            person_income=150_000,
            person_home_ownership="MORTGAGE",
            person_emp_length=15.0,
            loan_grade="D",
            loan_status=1,
        ),
        make_row(
            person_age=24,
            person_income=25_000,
            person_emp_length=1.0,
            loan_intent="EDUCATION",
            loan_int_rate=np.nan,
            loan_status=0,
        ),
        make_row(
            person_age=38,
            person_income=85_000,
            person_home_ownership="OWN",
            person_emp_length=12.0,
            loan_grade="E",
            loan_intent="DEBTCONSOLIDATION",
            loan_status=1,
            cb_person_default_on_file="Y",
        ),
    ]
    return pd.DataFrame(rows)


def test_cleaning_report_counts_real_problems(raw_df: pd.DataFrame) -> None:
    report = cleaning_report(raw_df)
    assert report["rows"] == 14
    assert report["duplicate_rows"] == 1
    assert report["impossible_age"] == 1
    assert report["impossible_employment_length"] == 1
    assert report["employment_longer_than_life"] >= 1


def test_clean_data_removes_duplicates_and_preserves_rows(raw_df: pd.DataFrame) -> None:
    cleaned = clean_data(raw_df)
    assert len(cleaned) == len(raw_df) - 1
    assert cleaned.duplicated().sum() == 0
    # Invalid values become NaN; the row itself is kept.
    assert cleaned["person_age"].isna().sum() == 1
    assert cleaned["person_emp_length"].isna().sum() >= 2


def test_invalid_values_are_nulled_but_valid_extremes_kept(raw_df: pd.DataFrame) -> None:
    cleaned = clean_data(raw_df)
    assert cleaned.loc[cleaned["person_income"] == 6_000_000, "person_age"].isna().all()
    assert cleaned["person_age"].max(skipna=True) <= MAX_PLAUSIBLE_AGE
    assert cleaned["person_emp_length"].max(skipna=True) <= 60
    both_present = cleaned[["person_emp_length", "person_age"]].dropna()
    assert (both_present["person_emp_length"] <= both_present["person_age"]).all()


def test_clean_data_does_not_mutate_the_input(raw_df: pd.DataFrame) -> None:
    original = raw_df.copy(deep=True)
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, original)


def test_clean_data_raises_for_missing_columns(raw_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        clean_data(raw_df.drop(columns=["loan_intent"]))


def test_clean_data_drops_rows_with_missing_target(raw_df: pd.DataFrame) -> None:
    broken = raw_df.copy()
    broken.loc[0, TARGET] = np.nan
    with pytest.warns(UserWarning, match="missing target"):
        cleaned = clean_data(broken)
    # The edited row is no longer an exact duplicate, so only it is dropped.
    assert len(cleaned) == len(raw_df) - 1
    assert cleaned[TARGET].notna().all()


def test_load_data_reports_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_data(tmp_path / "does_not_exist.csv")


def test_load_data_rejects_wrong_schema(raw_df: pd.DataFrame, tmp_path) -> None:
    path = tmp_path / "bad.csv"
    raw_df.drop(columns=["loan_amnt"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="loan_amnt"):
        load_data(path)


def test_split_is_stratified_reproducible_and_excludes_target(
    raw_df: pd.DataFrame,
) -> None:
    cleaned = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(cleaned, test_size=0.25, random_state=7)

    assert TARGET not in X_train.columns
    assert list(X_train.columns) == FEATURE_COLUMNS
    assert len(X_train) + len(X_test) == len(cleaned)
    assert len(X_train.index.intersection(X_test.index)) == 0
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}

    X_train_2, X_test_2, y_train_2, y_test_2 = split_data(
        cleaned, test_size=0.25, random_state=7
    )
    pd.testing.assert_frame_equal(X_train, X_train_2)
    pd.testing.assert_frame_equal(X_test, X_test_2)
    pd.testing.assert_series_equal(y_train, y_train_2)
    pd.testing.assert_series_equal(y_test, y_test_2)


def test_preprocessor_transforms_and_names_features(raw_df: pd.DataFrame) -> None:
    cleaned = clean_data(raw_df)
    X_train, X_test, _, _ = split_data(cleaned, test_size=0.25, random_state=7)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X_train)
    transformed_test = preprocessor.transform(X_test)

    assert transformed.shape[0] == len(X_train)
    assert transformed_test.shape[0] == len(X_test)
    assert transformed.shape[1] == transformed_test.shape[1]
    assert np.isfinite(transformed).all()

    names = get_feature_names(preprocessor)
    assert "num_base__person_age" in names
    assert any(name.startswith("cat__person_home_ownership_") for name in names)
    assert not any(TARGET in name for name in names)


def test_imputation_statistics_come_from_training_data_only() -> None:
    """A huge value in the test set must not influence the learned median/mean."""
    rows = [
        make_row(loan_int_rate=10.0, loan_status=0),
        make_row(loan_int_rate=20.0, loan_status=0),
        make_row(loan_int_rate=np.nan, loan_status=1),
        make_row(loan_int_rate=30.0, loan_status=1),
    ]
    train = pd.DataFrame(rows)[FEATURE_COLUMNS]
    test = pd.DataFrame([make_row(loan_int_rate=np.nan)])[FEATURE_COLUMNS]

    preprocessor = build_preprocessor()
    preprocessor.fit(train)

    index = NUMERIC_BASE_FEATURES.index("loan_int_rate")
    imputer = preprocessor.named_transformers_["num_base"].named_steps["imputer"]
    scaler = preprocessor.named_transformers_["num_base"].named_steps["scaler"]

    # Median of [10, 20, 30] is 20; the test row does not participate.
    assert imputer.statistics_[index] == pytest.approx(20.0)
    assert scaler.mean_[index] == pytest.approx(20.0)

    transformed = preprocessor.transform(test)
    assert np.isfinite(transformed).all()


def test_unseen_categories_are_handled_without_crashing(raw_df: pd.DataFrame) -> None:
    cleaned = clean_data(raw_df)
    X_train = cleaned[FEATURE_COLUMNS]
    preprocessor = build_preprocessor().fit(X_train)

    unseen = pd.DataFrame(
        [
            make_row(
                person_home_ownership="CASTLE",
                loan_intent="TIME_TRAVEL",
                loan_grade="Z",
            )
        ]
    )[FEATURE_COLUMNS]

    transformed = preprocessor.transform(unseen)
    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()


def test_pipeline_trains_and_returns_valid_probabilities(raw_df: pd.DataFrame) -> None:
    cleaned = clean_data(raw_df)
    X_train, X_test, y_train, _ = split_data(cleaned, test_size=0.25, random_state=7)

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)
    assert probabilities.shape == (len(X_test), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert set(pipeline.predict(X_test)).issubset({0, 1})


@pytest.mark.skipif(
    not DEFAULT_DATA_PATH.exists(),
    reason="real dataset not available",
)
def test_real_dataset_cleans_and_splits() -> None:
    raw = load_data()
    cleaned = clean_data(raw)

    assert len(cleaned) < len(raw)
    assert cleaned.duplicated().sum() == 0
    assert set(cleaned[TARGET].unique()) == {0, 1}
    assert cleaned["person_age"].max(skipna=True) <= MAX_PLAUSIBLE_AGE
    assert cleaned["person_emp_length"].max(skipna=True) <= 60

    X_train, X_test, y_train, y_test = split_data(cleaned)
    assert len(X_train) > len(X_test)
    assert TARGET not in X_train.columns
    assert abs(y_train.mean() - y_test.mean()) < 0.01
