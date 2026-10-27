# Loan Default Prediction API (FastAPI)

This is the backend for the saved loan-default model. It does **not** train or
change the model. At startup it loads the existing artifacts once:

- `models/loan_default_pipeline.joblib` — preprocessing pipeline + Random Forest
- `models/model_metadata.json` — feature contract, threshold, and verified metrics

The API returns a probability and a class decision using the saved threshold
(`0.62`). It does not invent a new threshold or recalculate metrics.

## Files

| File | Purpose |
| --- | --- |
| `app/main.py` | App factory, startup model loading, endpoints, CORS, error handlers |
| `app/schemas.py` | Pydantic request and response models |
| `tests/test_api.py` | TestClient tests for endpoints and API behavior |

The reusable inference logic remains in `src/predict.py`; the API is a thin HTTP
layer around it.

## Install on Debian

From the project root:

```bash
cd ~/loan-default-prediction
.venv/bin/python -m pip install -r requirements.txt
```

If the virtual environment does not exist yet, create it first with
`python3 -m venv .venv`. On Debian, the `python` command may not exist; use
`python3` or the interpreter inside `.venv`.

## Run the backend

```bash
cd ~/loan-default-prediction
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Then open:

- Interactive docs: <http://127.0.0.1:8000/docs>
- Alternative docs: <http://127.0.0.1:8000/redoc>
- OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>

If port `8000` is already in use, choose another port, for example
`--port 8765`, and use that port in the URLs above. Stop the server with
`Ctrl+C`.

The `--reload` flag is for development. It restarts the server when Python files
change. Do not use it as a production process manager.

## Endpoints

### `GET /health`

Reports whether the API is up and whether the model loaded successfully.

```bash
curl http://127.0.0.1:8000/health
```

Ready model:

```json
{
  "api": "ok",
  "status": "ok",
  "model_ready": true,
  "model_name": "Random Forest",
  "detail": null
}
```

If the artifacts cannot be loaded, the endpoint returns HTTP `503` with
`status: "degraded"` and `model_ready: false`.

### `GET /model-info`

Returns the actual model name, decision threshold, feature order, target
metadata, and verified test metrics from `models/model_metadata.json`.

```bash
curl http://127.0.0.1:8000/model-info
```

If the model is unavailable, it returns HTTP `503`.

### `POST /predict`

Accepts one applicant as JSON. The request body uses the same feature names as
the fitted pipeline. `person_emp_length` and `loan_int_rate` are optional because
the pipeline has imputers for them.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "person_age": 35,
    "person_income": 100000,
    "person_emp_length": 10.0,
    "loan_amnt": 10000,
    "loan_int_rate": 7.0,
    "loan_percent_income": 0.10,
    "cb_person_cred_hist_length": 10,
    "loan_grade": "A",
    "person_home_ownership": "MORTGAGE",
    "loan_intent": "PERSONAL",
    "cb_person_default_on_file": "N"
  }'
```

Actual response from the saved model:

```json
{
  "predicted_class": 0,
  "predicted_label": "no default",
  "default_probability": 0.0156875,
  "threshold": 0.62,
  "model_name": "Random Forest"
}
```

`predicted_class` is `1` when `default_probability >= threshold`; otherwise it is
`0`. The probability is a model output, not a guarantee or financial advice.

### Error examples

Missing required field, invalid number, out-of-range value, or extra field:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"person_age": 35}'
```

The API returns HTTP `422` with a short `detail` and field-level `errors` for
Pydantic validation failures. Invalid categories (for example
`{"loan_grade": "Z"}`) are rejected by the prediction service and also return
HTTP `422`.

`loan_status` is the target column and is explicitly rejected:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"loan_status": 1}'
```

Status codes used by the API:

| Code | Meaning |
| --- | --- |
| `200` | Request succeeded |
| `422` | Applicant data is invalid |
| `500` | Unexpected inference or server failure |
| `503` | Model artifacts are unavailable |

Error responses never include Python tracebacks or local filesystem paths.

## Run the tests

```bash
cd ~/loan-default-prediction
.venv/bin/python -m pytest tests/test_api.py -q
.venv/bin/python -m pytest -q
```

The API tests use FastAPI's `TestClient`. They cover successful predictions,
health, model information, missing fields, invalid categories, target-column
rejection, unknown fields, optional imputed fields, model unavailability,
inference failures, CORS, and the generated docs.

## How the code fits together

1. Constants in `app/main.py` define the CORS origins and safe error
   messages.
2. `create_app()` builds the FastAPI app and registers a **lifespan** handler.
3. The lifespan handler calls the cached `get_prediction_service()` once, so one
   process loads the 10.75 MB artifact once instead of on every request.
4. `get_service()` is a FastAPI dependency used by `/model-info`. `/predict`
   uses `get_optional_service()` so invalid JSON can still become a `422` before
   the endpoint decides whether the model is available. A valid body gets `503`
   when startup loading failed.
5. `PredictRequest` validates JSON shape, required fields, numeric ranges, and
   the target/unknown-field guards before the service is called.
6. `PredictionService.predict()` remains the final authority on the fitted
   feature contract and category lists.
7. Exception handlers convert known failures into safe HTTP responses and keep
   implementation details in server logs.

## REST and interview concepts

- **Endpoint / route**: a URL plus HTTP method, such as `POST /predict`.
- **JSON request body**: the applicant data sent by the browser.
- **Pydantic model**: a typed Python class that validates JSON and generates the
  OpenAPI schema shown in `/docs`.
- **Dependency injection**: FastAPI calls `get_service()`/`get_optional_service()`
  and passes the loaded service into the endpoint function; the endpoint does
  not reload the model.
- **Lifespan**: startup/shutdown code. Here it loads the model once and clears
  the app state on shutdown.
- **CORS**: browsers block JavaScript from calling a different origin unless the
  server allows it. The allowed React development origins are
  `http://localhost:5173` and `http://127.0.0.1:5173`.
- **HTTP status codes**: `200` success, `422` invalid input, `500` server/model
  failure, `503` dependency unavailable.
- **Separation of concerns**: ML inference lives in `src/predict.py`; HTTP and
  validation live in `app/`; tests import both.

## Responsible use

This is an educational project trained on historical, public credit-risk data.
It is not a production lending decision system. Real use would require fairness
audits, privacy controls, monitoring, explanation requirements, and human review.
