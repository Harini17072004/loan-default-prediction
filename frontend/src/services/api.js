const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

function normalizeBaseUrl(value) {
  const raw = typeof value === "string" ? value.trim() : "";
  return (raw || DEFAULT_API_BASE_URL).replace(/\/+$/, "");
}

/** Base URL for all API calls. Configure with VITE_API_BASE_URL. */
export const API_BASE_URL = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL);

export class ApiError extends Error {
  constructor(
    message,
    { status = 0, detail = "", errors = [], payload = null } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.errors = Array.isArray(errors) ? errors : [];
    this.payload = payload;
  }
}

async function readResponseBody(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  const text = await response.text();
  return text ? { detail: text } : null;
}

async function request(path, options = {}) {
  const { headers: extraHeaders, ...rest } = options;
  const url = `${API_BASE_URL}${path}`;

  let response;
  try {
    response = await fetch(url, {
      ...rest,
      headers: {
        Accept: "application/json",
        ...(rest.body ? { "Content-Type": "application/json" } : {}),
        ...(extraHeaders || {}),
      },
    });
  } catch (error) {
    throw new ApiError(
      `The prediction API at ${API_BASE_URL} could not be reached. Make sure the FastAPI server is running.`,
      {
        status: 0,
        detail:
          error instanceof Error ? error.message : "Network request failed.",
      },
    );
  }

  const payload = await readResponseBody(response);

  if (!response.ok) {
    const detail =
      typeof payload?.detail === "string"
        ? payload.detail
        : `The API returned HTTP ${response.status}.`;

    throw new ApiError(detail, {
      status: response.status,
      detail,
      errors: Array.isArray(payload?.errors) ? payload.errors : [],
      payload,
    });
  }

  return { data: payload, status: response.status };
}

/**
 * Check API/model readiness.
 *
 * A 503 is an expected "degraded" response when the saved artifact cannot be
 * loaded, so it is returned as data instead of being treated as a crash.
 */
export async function getHealth() {
  try {
    const { data, status } = await request("/health");
    if (!data || typeof data !== "object") {
      throw new ApiError("The health endpoint returned an empty response.", {
        status,
      });
    }
    return { ...data, httpStatus: status };
  } catch (error) {
    if (
      error instanceof ApiError &&
      error.status === 503 &&
      error.payload &&
      typeof error.payload === "object"
    ) {
      return { ...error.payload, httpStatus: 503 };
    }
    throw error;
  }
}

/** Fetch the real saved-model metadata and test metrics. */
export async function getModelInfo() {
  const { data, status } = await request("/model-info");
  if (!data || typeof data !== "object") {
    throw new ApiError("The model information endpoint returned no data.", {
      status,
    });
  }
  return data;
}

/** Submit an applicant profile and return the backend's prediction. */
export async function predict(applicant) {
  const { data, status } = await request("/predict", {
    method: "POST",
    body: JSON.stringify(applicant),
  });

  if (!data || typeof data !== "object") {
    throw new ApiError("The prediction endpoint returned no data.", { status });
  }

  const requiredFields = [
    "predicted_class",
    "predicted_label",
    "default_probability",
    "threshold",
    "model_name",
  ];
  const missing = requiredFields.filter((field) => !(field in data));
  if (missing.length > 0) {
    throw new ApiError(
      "The prediction response did not match the expected API contract.",
      { status, payload: data },
    );
  }

  return data;
}
