"""Pydantic request and response models for the loan-risk API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from src.predict import NUMERIC_BOUNDS

AGE_MIN, AGE_MAX, _ = NUMERIC_BOUNDS["person_age"]
INCOME_MIN, INCOME_MAX, _ = NUMERIC_BOUNDS["person_income"]
EMPLOYMENT_MIN, EMPLOYMENT_MAX, _ = NUMERIC_BOUNDS["person_emp_length"]
LOAN_MIN, LOAN_MAX, _ = NUMERIC_BOUNDS["loan_amnt"]
RATE_MIN, RATE_MAX, _ = NUMERIC_BOUNDS["loan_int_rate"]
INCOME_SHARE_MIN, INCOME_SHARE_MAX, _ = NUMERIC_BOUNDS["loan_percent_income"]
CREDIT_HISTORY_MIN, CREDIT_HISTORY_MAX, _ = NUMERIC_BOUNDS["cb_person_cred_hist_length"]

EXAMPLE_APPLICANT = {
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


class PredictRequest(BaseModel):
    """One applicant, using the exact feature names the saved model expects.

    ``person_emp_length`` and ``loan_int_rate`` are optional because the fitted
    preprocessing pipeline learned imputers and missing-value indicators for
    them. All other fields are required.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={"example": EXAMPLE_APPLICANT},
    )

    person_age: float = Field(
        ...,
        ge=AGE_MIN,
        le=AGE_MAX,
        description="Applicant age in years.",
    )
    person_income: float = Field(
        ...,
        ge=INCOME_MIN,
        le=INCOME_MAX,
        description="Gross annual income. The model applies a log transform.",
    )
    person_emp_length: float | None = Field(
        default=None,
        ge=EMPLOYMENT_MIN,
        le=EMPLOYMENT_MAX,
        description="Years employed. Optional; imputed by the saved pipeline.",
    )
    loan_amnt: float = Field(
        ...,
        ge=LOAN_MIN,
        le=LOAN_MAX,
        description="Requested loan amount.",
    )
    loan_int_rate: float | None = Field(
        default=None,
        ge=RATE_MIN,
        le=RATE_MAX,
        description="Annual interest rate in percent. Optional; imputed.",
    )
    loan_percent_income: float = Field(
        ...,
        ge=INCOME_SHARE_MIN,
        le=INCOME_SHARE_MAX,
        description="Loan amount divided by annual income (0 to 1).",
    )
    cb_person_cred_hist_length: float = Field(
        ...,
        ge=CREDIT_HISTORY_MIN,
        le=CREDIT_HISTORY_MAX,
        description="Length of the applicant's credit history in years.",
    )
    loan_grade: str = Field(
        ...,
        min_length=1,
        description="Lender-assigned grade A (lowest risk) through G (highest).",
    )
    person_home_ownership: str = Field(
        ...,
        min_length=1,
        description="One of MORTGAGE, OWN, RENT, OTHER.",
    )
    loan_intent: str = Field(
        ...,
        min_length=1,
        description=(
            "One of DEBTCONSOLIDATION, EDUCATION, HOMEIMPROVEMENT, "
            "MEDICAL, PERSONAL, VENTURE."
        ),
    )
    cb_person_default_on_file: str = Field(
        ...,
        min_length=1,
        description="Whether a prior default is on file: Y or N.",
    )

    @field_validator(
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        mode="before",
    )
    @classmethod
    def reject_boolean_numeric_values(
        cls, value: Any, info: ValidationInfo
    ) -> Any:
        """Match the service's rejection of ``True``/``False`` as numbers."""
        if isinstance(value, bool):
            raise ValueError(
                f"{info.field_name} must be a number, not a boolean."
            )
        return value

    @field_validator("person_emp_length", "loan_int_rate", mode="before")
    @classmethod
    def blank_optional_field_means_missing(cls, value: Any) -> Any:
        """Match the service's treatment of blank optional values as missing."""
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @model_validator(mode="before")
    @classmethod
    def reject_target_column(cls, data: Any) -> Any:
        """Preserve the prediction service's explicit target-leakage guard."""
        if isinstance(data, dict) and "loan_status" in data:
            raise ValueError(
                "'loan_status' is the target column and cannot be used as a feature."
            )
        return data

    @model_validator(mode="after")
    def employment_cannot_exceed_age(self) -> "PredictRequest":
        if (
            self.person_emp_length is not None
            and self.person_emp_length > self.person_age
        ):
            raise ValueError(
                "person_emp_length cannot be greater than person_age."
            )
        return self


class ConfusionMatrix(BaseModel):
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


class Support(BaseModel):
    non_default: int
    default: int


class ModelMetrics(BaseModel):
    """The verified test metrics stored in ``model_metadata.json``."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: ConfusionMatrix
    support: Support


class TargetInfo(BaseModel):
    name: str
    positive_label: int
    negative_label: int
    positive_meaning: str
    negative_meaning: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_key: str | None = None
    estimator_class: str | None = None
    threshold: float
    trained_at: str | None = None
    metrics: ModelMetrics
    features: list[str]
    target: TargetInfo


class PredictResponse(BaseModel):
    predicted_class: Literal[0, 1]
    predicted_label: Literal["default", "no default"]
    default_probability: float = Field(ge=0.0, le=1.0)
    threshold: float
    model_name: str


class HealthResponse(BaseModel):
    api: Literal["ok"] = "ok"
    status: Literal["ok", "degraded"]
    model_ready: bool
    model_name: str | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    detail: str
    errors: list[dict[str, Any]] | None = None
