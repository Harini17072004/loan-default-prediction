"""Reusable inference service for the saved loan-default pipeline.

This module deliberately does **not** train anything. It loads the existing
``models/loan_default_pipeline.joblib`` artifact and its metadata, validates an
applicant payload against the fitted feature contract, and returns a probability
plus a class prediction at the saved decision threshold.

Typical use::

    from src.predict import get_prediction_service

    service = get_prediction_service()
    result = service.predict({...})
    print(result.default_probability, result.predicted_label)

The service is intentionally separate from FastAPI so it can be tested directly
and reused by scripts or a future batch job.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import DEFAULT_MODEL_PATH, PROJECT_ROOT

DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

logger = logging.getLogger(__name__)

# These are application sanity checks, not training statistics. They prevent
# clearly impossible values from reaching the model while leaving room for
# unusual-but-valid applicants.
NUMERIC_BOUNDS: dict[str, tuple[float, float, str]] = {
    "person_age": (18.0, 100.0, "years"),
    "person_income": (1_000.0, 10_000_000.0, "currency units"),
    "person_emp_length": (0.0, 60.0, "years"),
    "loan_amnt": (100.0, 1_000_000.0, "currency units"),
    "loan_int_rate": (0.0, 100.0, "percent"),
    "loan_percent_income": (0.0, 1.0, "fraction of income"),
    "cb_person_cred_hist_length": (0.0, 80.0, "years"),
}

# These two fields were missing for some training rows and the fitted
# SimpleImputer learned how to handle a missing value for them. They may be
# omitted by a caller, but when supplied they must still be valid numbers.
OPTIONAL_NUMERIC_FEATURES = frozenset({"person_emp_length", "loan_int_rate"})


class PredictionServiceError(Exception):
    """Base class for expected prediction-service failures."""


class ModelArtifactError(PredictionServiceError):
    """The saved model or metadata is missing, corrupt, or inconsistent."""


class InvalidApplicantData(PredictionServiceError):
    """The applicant payload does not match the model's input contract."""


class PredictionError(PredictionServiceError):
    """The fitted pipeline failed while scoring an otherwise valid payload."""


@dataclass(frozen=True)
class ModelBundle:
    """A fitted pipeline plus the contract extracted from it."""

    pipeline: Any
    metadata: Mapping[str, Any]
    feature_names: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    categories: Mapping[str, tuple[str, ...]]
    threshold: float
    model_name: str
    positive_label: int


@dataclass(frozen=True)
class PredictionResult:
    """The two values a caller needs: a probability and a class decision."""

    predicted_class: int
    predicted_label: str
    default_probability: float
    threshold: float
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "default_probability": self.default_probability,
            "threshold": self.threshold,
            "model_name": self.model_name,
        }


def _is_missing(value: Any) -> bool:
    """Return True for None/blank/NaN-like scalar values, but not for lists."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _coerce_number(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise InvalidApplicantData(f"{name} must be a number, not a boolean.")

    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError as exc:
            raise InvalidApplicantData(
                f"{name} must be a number; got {value!r}."
            ) from exc
    else:
        raise InvalidApplicantData(
            f"{name} must be a number; got {type(value).__name__}."
        )

    if not math.isfinite(number):
        raise InvalidApplicantData(f"{name} must be finite; got {number!r}.")
    return number


def _validate_numeric(name: str, value: Any) -> float:
    bounds = NUMERIC_BOUNDS.get(name)
    if bounds is None:
        raise InvalidApplicantData(f"No validation rule is configured for {name}.")

    if _is_missing(value):
        if name in OPTIONAL_NUMERIC_FEATURES:
            return np.nan
        raise InvalidApplicantData(f"{name} is required.")

    number = _coerce_number(name, value)
    low, high, unit = bounds
    if not low <= number <= high:
        raise InvalidApplicantData(
            f"{name} must be between {low:g} and {high:g} {unit}; got {number:g}."
        )
    return number


def _validate_category(name: str, value: Any, allowed: tuple[str, ...]) -> str:
    if _is_missing(value):
        raise InvalidApplicantData(f"{name} is required.")
    if not isinstance(value, str):
        raise InvalidApplicantData(
            f"{name} must be text and one of: {', '.join(allowed)}."
        )

    normalized = value.strip().upper()
    if normalized not in allowed:
        raise InvalidApplicantData(
            f"Invalid value for {name}: {value!r}. Allowed values: "
            + ", ".join(allowed)
            + "."
        )
    return normalized


def _extract_contract(pipeline: Any, metadata: Mapping[str, Any]) -> tuple:
    """Read feature order, categories, threshold and model name from artifacts."""
    if not hasattr(pipeline, "named_steps"):
        raise ModelArtifactError(
            "The model artifact is not a scikit-learn Pipeline."
        )

    preprocessor = pipeline.named_steps.get("preprocessor")
    model = pipeline.named_steps.get("model")
    if preprocessor is None or model is None:
        raise ModelArtifactError(
            "The pipeline must contain 'preprocessor' and 'model' steps."
        )

    pipeline_features = tuple(getattr(preprocessor, "feature_names_in_", ()))
    if not pipeline_features:
        raise ModelArtifactError(
            "The saved preprocessor has no feature_names_in_ contract."
        )

    metadata_features = metadata.get("features")
    if not isinstance(metadata_features, list) or not all(
        isinstance(feature, str) for feature in metadata_features
    ):
        raise ModelArtifactError("metadata.json is missing a valid 'features' list.")
    if tuple(metadata_features) != pipeline_features:
        raise ModelArtifactError(
            "Metadata feature order does not match the fitted pipeline. "
            "Refusing to guess which order is correct."
        )

    numeric: set[str] = set()
    categories: dict[str, tuple[str, ...]] = {}
    for branch_name, transformer, columns in preprocessor.transformers_:
        if branch_name == "remainder":
            continue
        branch_columns = [columns] if isinstance(columns, str) else list(columns)

        if branch_name in {"num_base", "num_log"}:
            numeric.update(branch_columns)
        elif branch_name == "grade":
            encoder = transformer.named_steps.get("ordinal")
            if encoder is None or len(encoder.categories_) != 1:
                raise ModelArtifactError(
                    "Could not read ordinal categories from the fitted pipeline."
                )
            categories[branch_columns[0]] = tuple(
                str(category) for category in encoder.categories_[0]
            )
        elif branch_name == "cat":
            encoder = transformer.named_steps.get("onehot")
            if encoder is None:
                raise ModelArtifactError(
                    "Could not read one-hot categories from the fitted pipeline."
                )
            for column, values in zip(branch_columns, encoder.categories_):
                categories[column] = tuple(str(value) for value in values)
        else:
            raise ModelArtifactError(
                f"Unexpected preprocessor branch {branch_name!r}."
            )

    categorical = set(categories)
    if numeric & categorical:
        raise ModelArtifactError("A feature is classified as both numeric and categorical.")
    if numeric | categorical != set(pipeline_features):
        missing = sorted(set(pipeline_features) - (numeric | categorical))
        raise ModelArtifactError(
            "Could not determine validation rules for features: " + ", ".join(missing)
        )

    threshold = metadata.get("decision_threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ModelArtifactError("metadata.json has no numeric decision_threshold.")
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise ModelArtifactError(
            f"decision_threshold must be strictly between 0 and 1; got {threshold}."
        )

    target = metadata.get("target", {})
    positive_label = target.get("positive_label", 1) if isinstance(target, dict) else 1
    if isinstance(positive_label, bool) or not isinstance(positive_label, int):
        raise ModelArtifactError("target.positive_label must be an integer.")

    classes = list(getattr(model, "classes_", []))
    if positive_label not in classes:
        raise ModelArtifactError(
            f"The fitted model has no class {positive_label}; classes are {classes}."
        )
    if not hasattr(model, "predict_proba"):
        raise ModelArtifactError("The fitted model does not expose predict_proba().")

    try:
        encoded_feature_count = len(preprocessor.get_feature_names_out())
    except Exception as exc:
        raise ModelArtifactError(
            "The saved preprocessor is not fitted or has no output feature names."
        ) from exc
    if getattr(model, "n_features_in_", None) != encoded_feature_count:
        raise ModelArtifactError(
            "The model expects a different number of encoded features than the "
            f"preprocessor produces ({getattr(model, 'n_features_in_', None)} vs "
            f"{encoded_feature_count})."
        )

    model_name = (
        metadata.get("display_name")
        or metadata.get("selected_model")
        or type(model).__name__
    )

    return (
        pipeline_features,
        tuple(feature for feature in pipeline_features if feature in numeric),
        tuple(feature for feature in pipeline_features if feature in categorical),
        categories,
        threshold,
        str(model_name),
        positive_label,
    )


def load_model_bundle(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> ModelBundle:
    """Load and validate the saved pipeline + metadata pair."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise ModelArtifactError(
            f"Model metadata not found at {metadata_path}. Run `python train.py` first."
        )
    if not model_path.exists():
        raise ModelArtifactError(
            f"Model artifact not found at {model_path}. Run `python train.py` first."
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(
            f"Could not read model metadata at {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise ModelArtifactError("Model metadata must be a JSON object.")

    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:  # joblib/pickle can raise many exception types
        raise ModelArtifactError(
            f"Could not load model artifact at {model_path}: {exc}"
        ) from exc

    (
        feature_names,
        numeric_features,
        categorical_features,
        categories,
        threshold,
        model_name,
        positive_label,
    ) = _extract_contract(pipeline, metadata)

    return ModelBundle(
        pipeline=pipeline,
        metadata=metadata,
        feature_names=feature_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        categories=categories,
        threshold=threshold,
        model_name=model_name,
        positive_label=positive_label,
    )


class PredictionService:
    """Validate applicant data and score it with one loaded model bundle."""

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    @classmethod
    def load(
        cls,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
    ) -> "PredictionService":
        return cls(load_model_bundle(model_path, metadata_path))

    @property
    def threshold(self) -> float:
        return self.bundle.threshold

    @property
    def model_name(self) -> str:
        return self.bundle.model_name

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.bundle.feature_names

    def info(self) -> dict[str, Any]:
        """Return actual metadata, not recomputed or invented metrics."""
        metadata = self.bundle.metadata
        return {
            "model_name": self.bundle.model_name,
            "model_key": metadata.get("selected_model"),
            "estimator_class": metadata.get("estimator_class"),
            "threshold": self.bundle.threshold,
            "trained_at": metadata.get("trained_at"),
            "metrics": dict(metadata.get("test_metrics", {})),
            "features": list(self.bundle.feature_names),
            "target": dict(metadata.get("target", {})),
        }

    def _prepare_features(self, applicant: Mapping[str, Any]) -> pd.DataFrame:
        if not isinstance(applicant, Mapping):
            raise InvalidApplicantData(
                "Applicant payload must be a mapping of feature names to values."
            )

        provided = dict(applicant)
        if "loan_status" in provided:
            raise InvalidApplicantData(
                "'loan_status' is the target column and cannot be used as a feature."
            )

        unknown = sorted(set(provided) - set(self.bundle.feature_names))
        if unknown:
            raise InvalidApplicantData(
                "Unknown feature(s): " + ", ".join(unknown)
            )

        required = set(self.bundle.feature_names) - OPTIONAL_NUMERIC_FEATURES
        missing = sorted(required - set(provided))
        if missing:
            raise InvalidApplicantData(
                "Missing required feature(s): " + ", ".join(missing)
            )

        values: dict[str, Any] = {}
        for name in self.bundle.feature_names:
            raw_value = provided.get(name)
            if name in self.bundle.numeric_features:
                values[name] = _validate_numeric(name, raw_value)
            else:
                values[name] = _validate_category(
                    name, raw_value, self.bundle.categories[name]
                )

        age = values.get("person_age")
        employment = values.get("person_emp_length")
        if (
            age is not None
            and not _is_missing(employment)
            and employment is not None
            and employment > age
        ):
            raise InvalidApplicantData(
                "person_emp_length cannot be greater than person_age."
            )

        # Build the frame in the exact order the preprocessor was fitted with.
        frame = pd.DataFrame([values], columns=list(self.bundle.feature_names))
        for name in self.bundle.numeric_features:
            frame[name] = frame[name].astype(float)
        for name in self.bundle.categorical_features:
            frame[name] = frame[name].astype(object)
        return frame

    def predict(self, applicant: Mapping[str, Any]) -> PredictionResult:
        """Validate one applicant and return a probability + class decision."""
        frame = self._prepare_features(applicant)

        try:
            probabilities = self.bundle.pipeline.predict_proba(frame)
        except Exception as exc:
            logger.exception("Model inference failed")
            raise PredictionError(f"Model inference failed: {exc}") from exc

        probabilities = np.asarray(probabilities, dtype=float)
        model = self.bundle.pipeline.named_steps["model"]
        expected_shape = (1, len(model.classes_))
        if probabilities.shape != expected_shape:
            raise PredictionError(
                f"Unexpected predict_proba output shape: {probabilities.shape}."
            )

        classes = list(model.classes_)
        try:
            positive_index = classes.index(self.bundle.positive_label)
        except ValueError as exc:
            raise ModelArtifactError(
                f"Positive label {self.bundle.positive_label} is missing from {classes}."
            ) from exc

        probability = float(probabilities[0, positive_index])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise PredictionError(
                f"Model returned an invalid probability: {probability!r}."
            )

        predicted_class = int(probability >= self.bundle.threshold)
        label = "default" if predicted_class == 1 else "no default"
        return PredictionResult(
            predicted_class=predicted_class,
            predicted_label=label,
            default_probability=probability,
            threshold=self.bundle.threshold,
            model_name=self.bundle.model_name,
        )


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """Load the default artifacts once per process (safe for FastAPI)."""
    return PredictionService.load()


def clear_prediction_cache() -> None:
    """Clear the cached service (useful after retraining or in tests)."""
    get_prediction_service.cache_clear()
