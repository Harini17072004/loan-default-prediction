"""FastAPI application for the saved loan-default prediction model."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.predict import (
    DEFAULT_METADATA_PATH,
    InvalidApplicantData,
    ModelArtifactError,
    PredictionError,
    PredictionService,
    PredictionServiceError,
    get_prediction_service,
)
from src.preprocessing import DEFAULT_MODEL_PATH

from app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

API_TITLE = "Loan Default Prediction API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Serve the saved Random Forest loan-default model. The model returns a "
    "default probability and a class decision at the threshold selected during "
    "training."
)

# These are the development origins used by Vite/React.
CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)

MODEL_UNAVAILABLE_DETAIL = (
    "Model artifacts are unavailable. Check the server logs or retrain the model."
)
GENERIC_INFERENCE_ERROR = (
    "The model could not complete this prediction. Please try again later."
)
GENERIC_ERROR = "An unexpected server error occurred."


def get_service(request: Request) -> PredictionService:
    """Return the model loaded at startup, or a clear 503 if it is unavailable."""
    service = getattr(request.app.state, "prediction_service", None)
    if not isinstance(service, PredictionService):
        raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE_DETAIL)
    return service


def get_optional_service(request: Request) -> PredictionService | None:
    """Return the model if ready, otherwise ``None``.

    ``/predict`` uses this dependency so an invalid request body can still be
    rejected with 422 even when the model is unavailable. The endpoint itself
    returns 503 for a valid body when no model is loaded.
    """
    service = getattr(request.app.state, "prediction_service", None)
    return service if isinstance(service, PredictionService) else None


def create_app(
    *,
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    preload_model: bool = True,
) -> FastAPI:
    """Build the API.

    The default app uses ``get_prediction_service()``, which caches the loaded
    artifact for the life of the process. Tests can pass custom paths or disable
    preloading to exercise unavailable-model behavior.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.prediction_service = None
        app.state.model_error = None

        if preload_model:
            try:
                if model_path is None and metadata_path is None:
                    service = get_prediction_service()
                else:
                    service = PredictionService.load(
                        model_path=model_path or DEFAULT_MODEL_PATH,
                        metadata_path=metadata_path or DEFAULT_METADATA_PATH,
                    )
            except ModelArtifactError as exc:
                # Keep the public response generic; the detailed error stays in
                # server logs and may contain local filesystem paths.
                app.state.model_error = MODEL_UNAVAILABLE_DETAIL
                logger.error("Model artifacts could not be loaded: %s", exc)
            except Exception:
                # A corrupt or unexpected artifact layout should degrade the API
                # rather than crash the process during startup.
                app.state.model_error = MODEL_UNAVAILABLE_DETAIL
                logger.exception("Unexpected error while loading model artifacts")
            else:
                app.state.prediction_service = service
                logger.info("Loaded model: %s", service.model_name)

        yield

        app.state.prediction_service = None

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )

    # These values also make endpoints return 503 instead of "attribute missing"
    # if someone calls the app without running the startup event (for example a
    # misconfigured test client).
    app.state.prediction_service = None
    app.state.model_error = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(CORS_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"message": API_TITLE, "docs": "/docs"}

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
        summary="Check API and model readiness",
        responses={503: {"model": HealthResponse}},
    )
    def health(request: Request):
        service = getattr(request.app.state, "prediction_service", None)
        if not isinstance(service, PredictionService):
            payload = HealthResponse(
                status="degraded",
                model_ready=False,
                detail=(
                    getattr(request.app.state, "model_error", None)
                    or MODEL_UNAVAILABLE_DETAIL
                ),
            )
            return JSONResponse(
                status_code=503,
                content=payload.model_dump(exclude_none=True),
            )

        return HealthResponse(
            status="ok",
            model_ready=True,
            model_name=service.model_name,
        )

    @app.get(
        "/model-info",
        response_model=ModelInfoResponse,
        tags=["model"],
        summary="Get the model name, threshold, and verified test metrics",
        responses={503: {"model": ErrorResponse}},
    )
    def model_info(
        service: PredictionService = Depends(get_service),
    ) -> dict[str, Any]:
        return service.info()

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["prediction"],
        summary="Predict whether an applicant will default",
        responses={
            422: {"model": ErrorResponse, "description": "Invalid applicant input"},
            500: {"model": ErrorResponse, "description": "Inference failed"},
            503: {"model": ErrorResponse, "description": "Model unavailable"},
        },
    )
    def predict(
        payload: PredictRequest,
        service: PredictionService | None = Depends(get_optional_service),
    ) -> PredictResponse:
        if service is None:
            raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE_DETAIL)
        result = service.predict(payload.model_dump())
        return PredictResponse(**result.to_dict())

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.info("Request validation failed for %s", request.url.path)
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Invalid request data.",
                "errors": jsonable_encoder(exc.errors()),
            },
        )

    @app.exception_handler(InvalidApplicantData)
    async def invalid_applicant_handler(
        request: Request, exc: InvalidApplicantData
    ) -> JSONResponse:
        # The service's validation messages describe field names and allowed
        # values, never local paths or Python tracebacks.
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(ModelArtifactError)
    async def model_artifact_handler(
        request: Request, exc: ModelArtifactError
    ) -> JSONResponse:
        logger.error("Model artifact error on %s", request.url.path)
        return JSONResponse(
            status_code=503,
            content={"detail": MODEL_UNAVAILABLE_DETAIL},
        )

    @app.exception_handler(PredictionError)
    async def prediction_error_handler(
        request: Request, exc: PredictionError
    ) -> JSONResponse:
        # The service logs the underlying exception. Do not return it because it
        # can contain implementation details or filesystem paths.
        logger.error("Inference failed on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": GENERIC_INFERENCE_ERROR},
        )

    @app.exception_handler(PredictionServiceError)
    async def service_error_handler(
        request: Request, exc: PredictionServiceError
    ) -> JSONResponse:
        logger.error("Prediction service error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": GENERIC_ERROR},
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": GENERIC_ERROR},
        )

    return app


app = create_app()
