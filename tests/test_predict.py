"""Tests for the reusable prediction service and its artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.predict import (
    DEFAULT_METADATA_PATH,
    InvalidApplicantData,
    ModelArtifactError,
    PredictionError,
    PredictionService,
)
from src.preprocessing import DEFAULT_MODEL_PATH

LOW_RISK = {
    "person_age": 35,
    "person_income": 100_000,
    "person_emp_length": 10.0,
    "loan_amnt": 10_000,
    "loan_int_rate": 7.0,
    "loan_percent_income": 0.10,
    "cb_person_cred_hist_length": 10,
    "loan_grade": "A",
    "person_home_ownership": "MORTGAGE",
    "loan_intent": "PERSONAL",
    "cb_person_default_on_file": "N",
}

HIGH_RISK = {
    "person_age": 23,
    "person_income": 30_000,
    "person_emp_length": 1.0,
    "loan_amnt": 20_000,
    "loan_int_rate": 18.0,
    "loan_percent_income": 0.67,
    "cb_person_cred_hist_length": 2,
    "loan_grade": "F",
    "person_home_ownership": "RENT",
    "loan_intent": "MEDICAL",
    "cb_person_default_on_file": "Y",
}


@pytest.fixture(scope="module")
def service() -> PredictionService:
    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_METADATA_PATH.exists():
        pytest.skip("saved model artifacts are not available; run `python train.py`")
    return PredictionService.load()


def test_loads_the_actual_saved_contract(service: PredictionService) -> None:
    metadata = json.loads(DEFAULT_METADATA_PATH.read_text())
    assert service.threshold == pytest.approx(0.62)
    assert service.model_name == "Random Forest"
    assert service.feature_names == tuple(metadata["features"])
    assert service.bundle.categories["loan_grade"] == tuple("ABCDEFG")
    assert service.bundle.categories["person_home_ownership"] == (
        "MORTGAGE",
        "OTHER",
        "OWN",
        "RENT",
    )
    assert service.bundle.categories["loan_intent"] == (
        "DEBTCONSOLIDATION",
        "EDUCATION",
        "HOMEIMPROVEMENT",
        "MEDICAL",
        "PERSONAL",
        "VENTURE",
    )
    assert service.bundle.categories["cb_person_default_on_file"] == ("N", "Y")


def test_valid_prediction_returns_probability_and_class(
    service: PredictionService,
) -> None:
    result = service.predict(LOW_RISK)

    assert 0.0 <= result.default_probability <= 1.0
    assert result.threshold == pytest.approx(service.threshold)
    assert result.model_name == service.model_name
    assert result.predicted_class in {0, 1}
    assert result.predicted_label in {"default", "no default"}
    assert result.predicted_class == int(
        result.default_probability >= result.threshold
    )
    assert result.to_dict()["default_probability"] == pytest.approx(
        result.default_probability
    )


def test_high_risk_applicant_scores_higher_than_low_risk(
    service: PredictionService,
) -> None:
    low = service.predict(LOW_RISK)
    high = service.predict(HIGH_RISK)

    assert low.predicted_class == 0
    assert high.predicted_class == 1
    assert high.default_probability > low.default_probability


def test_optional_imputed_fields_can_be_omitted(service: PredictionService) -> None:
    applicant = dict(LOW_RISK)
    applicant.pop("person_emp_length")
    applicant.pop("loan_int_rate")

    result = service.predict(applicant)
    assert 0.0 <= result.default_probability <= 1.0


def test_missing_required_feature_is_rejected(service: PredictionService) -> None:
    applicant = dict(LOW_RISK)
    applicant.pop("person_income")

    with pytest.raises(InvalidApplicantData, match="person_income"):
        service.predict(applicant)


def test_target_column_cannot_be_used_as_a_feature(
    service: PredictionService,
) -> None:
    applicant = {**LOW_RISK, "loan_status": 1}

    with pytest.raises(InvalidApplicantData, match="target"):
        service.predict(applicant)


def test_unknown_feature_is_rejected(service: PredictionService) -> None:
    applicant = {**LOW_RISK, "favourite_colour": "blue"}

    with pytest.raises(InvalidApplicantData, match="favourite_colour"):
        service.predict(applicant)


def test_invalid_category_is_rejected(service: PredictionService) -> None:
    applicant = {**LOW_RISK, "loan_grade": "Z"}

    with pytest.raises(InvalidApplicantData, match="loan_grade"):
        service.predict(applicant)


def test_categories_are_normalised_to_the_fitted_encoding(
    service: PredictionService,
) -> None:
    applicant = {
        **LOW_RISK,
        "loan_grade": " a ",
        "person_home_ownership": "mortgage",
        "loan_intent": "personal",
    }

    result = service.predict(applicant)
    assert 0.0 <= result.default_probability <= 1.0


def test_non_numeric_value_is_rejected(service: PredictionService) -> None:
    applicant = {**LOW_RISK, "person_income": "not-a-number"}

    with pytest.raises(InvalidApplicantData, match="must be a number"):
        service.predict(applicant)


def test_out_of_range_value_is_rejected(service: PredictionService) -> None:
    applicant = {**LOW_RISK, "person_age": 17}

    with pytest.raises(InvalidApplicantData, match="person_age"):
        service.predict(applicant)


def test_employment_cannot_exceed_age(service: PredictionService) -> None:
    applicant = {**LOW_RISK, "person_age": 25, "person_emp_length": 30}

    with pytest.raises(InvalidApplicantData, match="greater than person_age"):
        service.predict(applicant)


def test_inference_failure_is_wrapped(service: PredictionService, monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("simulated pipeline failure")

    monkeypatch.setattr(service.bundle.pipeline, "predict_proba", explode)

    with pytest.raises(PredictionError, match="inference failed"):
        service.predict(LOW_RISK)


def test_info_returns_metrics_from_metadata(service: PredictionService) -> None:
    metadata = json.loads(DEFAULT_METADATA_PATH.read_text())
    info = service.info()

    assert info["model_name"] == "Random Forest"
    assert info["threshold"] == pytest.approx(metadata["decision_threshold"])
    assert info["metrics"] == metadata["test_metrics"]
    assert info["features"] == metadata["features"]


def test_missing_model_file_has_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError, match="Model artifact not found"):
        PredictionService.load(
            model_path=tmp_path / "does-not-exist.joblib",
            metadata_path=DEFAULT_METADATA_PATH,
        )


def test_missing_metadata_file_has_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError, match="Model metadata not found"):
        PredictionService.load(
            model_path=DEFAULT_MODEL_PATH,
            metadata_path=tmp_path / "does-not-exist.json",
        )


def test_corrupt_model_file_has_a_clear_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.joblib"
    broken.write_bytes(b"this is not a joblib artifact")

    with pytest.raises(ModelArtifactError, match="Could not load model artifact"):
        PredictionService.load(
            model_path=broken, metadata_path=DEFAULT_METADATA_PATH
        )


def test_metadata_must_match_the_pipeline_features(tmp_path: Path) -> None:
    metadata = json.loads(DEFAULT_METADATA_PATH.read_text())
    metadata["features"] = list(reversed(metadata["features"]))
    broken = tmp_path / "metadata.json"
    broken.write_text(json.dumps(metadata))

    with pytest.raises(ModelArtifactError, match="does not match"):
        PredictionService.load(
            model_path=DEFAULT_MODEL_PATH, metadata_path=broken
        )


def test_invalid_threshold_is_rejected(tmp_path: Path) -> None:
    metadata = json.loads(DEFAULT_METADATA_PATH.read_text())
    metadata["decision_threshold"] = 1.5
    broken = tmp_path / "metadata.json"
    broken.write_text(json.dumps(metadata))

    with pytest.raises(ModelArtifactError, match="between 0 and 1"):
        PredictionService.load(
            model_path=DEFAULT_MODEL_PATH, metadata_path=broken
        )
