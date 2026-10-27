import { ALL_FIELDS, FIELD_BY_NAME } from "../constants/model";
import { formatNumber } from "./format";

function numericValue(value) {
  const raw = value ?? "";
  if (typeof raw === "string" && raw.trim() === "") return null;
  const number = Number(raw);
  return Number.isFinite(number) ? number : null;
}

function rangeLabel(field) {
  const suffix =
    field.unit === "%" ? "%" : field.unit ? ` ${field.unit}` : "";
  return `${formatNumber(field.min)}–${formatNumber(field.max)}${suffix}`;
}

/**
 * Validate the controlled form before sending it to the API.
 *
 * This is deliberately a UX layer only: FastAPI/Pydantic and the prediction
 * service still validate every request they receive.
 */
export function validateForm(form) {
  const errors = {};

  for (const field of ALL_FIELDS) {
    const raw = form[field.name] ?? "";
    const value = typeof raw === "string" ? raw.trim() : String(raw);

    if (field.type === "select") {
      if (!value) {
        errors[field.name] = `${field.label} is required.`;
      } else if (!field.options?.some((option) => option.value === value)) {
        errors[field.name] = "Choose a valid option.";
      }
      continue;
    }

    if (!value) {
      if (!field.optional) {
        errors[field.name] = `${field.label} is required.`;
      }
      continue;
    }

    const number = Number(value);
    if (!Number.isFinite(number)) {
      errors[field.name] = "Enter a valid number.";
      continue;
    }

    if (number < field.min || number > field.max) {
      errors[field.name] = `${field.label} must be between ${rangeLabel(field)}.`;
    }
  }

  const age = numericValue(form.person_age);
  const employment = numericValue(form.person_emp_length);
  if (
    age !== null &&
    employment !== null &&
    employment > age &&
    !errors.person_emp_length
  ) {
    errors.person_emp_length = "Employment length cannot be greater than age.";
  }

  return {
    valid: Object.keys(errors).length === 0,
    errors,
    payload: buildPayload(form),
  };
}

/**
 * Convert controlled string inputs into the API's JSON shape.
 *
 * Blank optional values are omitted rather than sent as null. `loan_status` is
 * intentionally absent: it is the target and must never be a model input.
 */
export function buildPayload(form) {
  const payload = {};

  for (const field of ALL_FIELDS) {
    const raw = form[field.name] ?? "";
    const value = typeof raw === "string" ? raw.trim() : String(raw);

    if (field.type === "number") {
      if (value === "") continue;
      payload[field.name] = Number(value);
    } else if (value) {
      payload[field.name] = value;
    }
  }

  return payload;
}

function cleanApiMessage(message) {
  if (typeof message !== "string" || !message.trim()) {
    return "This value was rejected by the API.";
  }
  return message.replace(/^Value error,\s*/i, "").trim();
}

/**
 * Attach Pydantic `loc` entries such as ["body", "person_age"] to form fields.
 */
export function mapApiFieldErrors(apiErrors = []) {
  const fieldNames = new Set(ALL_FIELDS.map((field) => field.name));
  const mapped = {};

  for (const error of apiErrors) {
    const location = Array.isArray(error?.loc) ? error.loc : [];
    const field = [...location]
      .reverse()
      .find((part) => typeof part === "string" && fieldNames.has(part));

    if (field && !mapped[field]) {
      mapped[field] = cleanApiMessage(error.msg);
    }
  }

  return mapped;
}

/** Human-friendly title for an API or service error. */
export function getErrorTitle(error) {
  switch (error?.status) {
    case 0:
      return "API connection failed";
    case 422:
      return "Check the applicant details";
    case 503:
      return "Prediction service unavailable";
    case 500:
      return "Prediction failed";
    default:
      return "Request failed";
  }
}

/** Concise error message that is safe to show to the user. */
export function getErrorMessage(error) {
  if (!error) return "An unexpected error occurred.";

  if (error.status === 0) {
    return (
      error.message ||
      "The prediction API could not be reached. Make sure the FastAPI server is running."
    );
  }

  if (error.status === 422) {
    return (
      error.detail ||
      "The API rejected one or more values. Please review the highlighted fields."
    );
  }

  if (error.status === 503) {
    return (
      error.detail ||
      "The model is not currently available. Check the backend logs and try again."
    );
  }

  return (
    error.detail ||
    error.message ||
    "The prediction request could not be completed. Please try again."
  );
}

/** Return the first field name from a validation error map. */
export function firstErrorField(errors) {
  return Object.keys(errors)[0] ?? null;
}

export { FIELD_BY_NAME };
