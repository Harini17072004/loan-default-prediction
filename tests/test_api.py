"""API tests for endpoints, validation, readiness, errors, and CORS."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import CORS_ORIGINS, create_app, get_optional_service
from src.predict import DEFAULT_METADATA_PATH
from src.preprocessing import DEFAULT_MODEL_PATH

VALID_APPLICANT = {
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

HIGH_RISK_APPLICANT = {
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
def app():
    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_METADATA_PATH.exists():
        pytest.skip("saved model artifacts are not available; run `python train.py`")
    return create_app()


@pytest.fixture(scope="module")
def client(app):
    # raise_server_exceptions=False lets us assert the API's 500 response
    # instead of having TestClient re-raise the simulated exception.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_health_reports_ready_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "api": "ok",
        "status": "ok",
        "model_ready": True,
        "model_name": "Random Forest",
        "detail": None,
    }


def test_model_info_returns_saved_metadata(client: TestClient) -> None:
    metadata = json.loads(DEFAULT_METADATA_PATH.read_text())
    response = client.get("/model-info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "Random Forest"
    assert payload["threshold"] == pytest.approx(metadata["decision_threshold"])
    assert payload["metrics"] == metadata["test_metrics"]
    assert payload["features"] == metadata["features"]
    assert payload["target"] == metadata["target"]


def test_predict_low_risk_applicant(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_APPLICANT)

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "predicted_class": 0,
        "predicted_label": "no default",
        "default_probability": pytest.approx(0.0156875, abs=1e-6),
        "threshold": 0.62,
        "model_name": "Random Forest",
    }
    assert payload["predicted_class"] == int(
        payload["default_probability"] >= payload["threshold"]
    )


def test_predict_high_risk_applicant(client: TestClient) -> None:
    response = client.post("/predict", json=HIGH_RISK_APPLICANT)

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] == 1
    assert payload["predicted_label"] == "default"
    assert payload["default_probability"] > 0.62
    assert payload["threshold"] == 0.62
    assert payload["model_name"] == "Random Forest"


def test_optional_imputed_fields_can_be_omitted(client: TestClient) -> None:
    applicant = dict(VALID_APPLICANT)
    applicant.pop("person_emp_length")
    applicant.pop("loan_int_rate")

    response = client.post("/predict", json=applicant)

    assert response.status_code == 200
    assert 0.0 <= response.json()["default_probability"] <= 1.0


def test_blank_optional_fields_are_treated_as_missing(
    client: TestClient,
) -> None:
    response = client.post(
        "/predict",
        json={
            **VALID_APPLICANT,
            "person_emp_length": "",
            "loan_int_rate": " ",
        },
    )

    assert response.status_code == 200
    assert 0.0 <= response.json()["default_probability"] <= 1.0


def test_missing_required_field_is_422(client: TestClient) -> None:
    applicant = dict(VALID_APPLICANT)
    applicant.pop("person_income")

    response = client.post("/predict", json=applicant)

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid request data."
    assert "person_income" in response.text


def test_invalid_category_is_422(client: TestClient) -> None:
    response = client.post(
        "/predict", json={**VALID_APPLICANT, "loan_grade": "Z"}
    )

    assert response.status_code == 422
    assert "loan_grade" in response.text
    assert "Allowed values" in response.text


def test_invalid_numeric_type_is_422(client: TestClient) -> None:
    response = client.post(
        "/predict", json={**VALID_APPLICANT, "person_income": "not-a-number"}
    )

    assert response.status_code == 422
    assert "person_income" in response.text


def test_out_of_range_value_is_422(client: TestClient) -> None:
    response = client.post(
        "/predict", json={**VALID_APPLICANT, "person_age": 17}
    )

    assert response.status_code == 422
    assert "person_age" in response.text


def test_boolean_is_not_accepted_as_a_number(client: TestClient) -> None:
    response = client.post(
        "/predict", json={**VALID_APPLICANT, "loan_percent_income": True}
    )

    assert response.status_code == 422
    assert "loan_percent_income" in response.text
    assert "boolean" in response.text


def test_target_column_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/predict", json={**VALID_APPLICANT, "loan_status": 1}
    )

    assert response.status_code == 422
    assert "loan_status" in response.text
    assert "target column" in response.text


def test_unknown_field_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/predict", json={**VALID_APPLICANT, "favourite_colour": "blue"}
    )

    assert response.status_code == 422
    assert "favourite_colour" in response.text


def test_employment_longer_than_age_is_422(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            **VALID_APPLICANT,
            "person_age": 25,
            "person_emp_length": 30,
        },
    )

    assert response.status_code == 422
    assert "person_emp_length" in response.text


def test_model_unavailability_returns_503(tmp_path: Path) -> None:
    missing_model = tmp_path / "does-not-exist.joblib"
    unavailable_app = create_app(model_path=missing_model)

    with TestClient(unavailable_app, raise_server_exceptions=False) as unavailable:
        health = unavailable.get("/health")
        info = unavailable.get("/model-info")
        prediction = unavailable.post("/predict", json=VALID_APPLICANT)

    assert health.status_code == 503
    assert health.json()["status"] == "degraded"
    assert health.json()["model_ready"] is False
    assert info.status_code == 503
    assert prediction.status_code == 503
    for response in (health, info, prediction):
        assert str(missing_model) not in response.text
        assert "Traceback" not in response.text

    # Body validation still works while the model is down.
    with TestClient(unavailable_app, raise_server_exceptions=False) as unavailable:
        invalid = unavailable.post("/predict", json={"person_age": 35})
    assert invalid.status_code == 422


def test_corrupt_model_artifact_returns_503(tmp_path: Path) -> None:
    broken_model = tmp_path / "broken.joblib"
    broken_model.write_bytes(b"not a joblib artifact")
    unavailable_app = create_app(model_path=broken_model)

    with TestClient(unavailable_app, raise_server_exceptions=False) as unavailable:
        response = unavailable.get("/health")

    assert response.status_code == 503
    assert response.json()["model_ready"] is False
    assert str(broken_model) not in response.text


def test_inference_failure_returns_generic_500(
    client: TestClient, app, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("private failure at /tmp/should-not-leak")

    monkeypatch.setattr(
        app.state.prediction_service.bundle.pipeline,
        "predict_proba",
        explode,
    )

    response = client.post("/predict", json=VALID_APPLICANT)

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            "The model could not complete this prediction. Please try again later."
        )
    }
    assert "/tmp/should-not-leak" not in response.text
    assert "Traceback" not in response.text


def test_unexpected_failure_returns_generic_500(
    client: TestClient, app
) -> None:
    class ExplodingService:
        def predict(self, applicant):
            raise RuntimeError("private /home/harini/should-not-leak")

    app.dependency_overrides[get_optional_service] = lambda: ExplodingService()
    try:
        response = client.post("/predict", json=VALID_APPLICANT)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {
        "detail": "An unexpected server error occurred."
    }
    assert "/home/harini/should-not-leak" not in response.text
    assert "Traceback" not in response.text


@pytest.mark.parametrize("origin", CORS_ORIGINS)
def test_cors_allows_react_development_origins(
    client: TestClient, origin: str
) -> None:
    response = client.get("/health", headers={"Origin": origin})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_cors_preflight_for_predict(client: TestClient) -> None:
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "http://localhost:5173"
    )
    assert "POST" in response.headers["access-control-allow-methods"]


def test_cors_rejects_unlisted_origin(client: TestClient) -> None:
    response = client.get(
        "/health", headers={"Origin": "https://example.com"}
    )

    assert "access-control-allow-origin" not in response.headers


def test_openapi_and_docs_are_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200

    schema = client.get("/openapi.json").json()
    assert {"/health", "/model-info", "/predict"} <= set(schema["paths"])
