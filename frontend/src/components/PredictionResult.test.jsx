import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import PredictionResult from "./PredictionResult";
import { ApiError } from "../services/api";

const NON_DEFAULT = {
  predicted_class: 0,
  predicted_label: "no default",
  default_probability: 0.0156875,
  threshold: 0.62,
  model_name: "Random Forest",
};

const DEFAULT = {
  predicted_class: 1,
  predicted_label: "default",
  default_probability: 1,
  threshold: 0.62,
  model_name: "Random Forest",
};

function renderResult(overrides = {}) {
  return render(
    <PredictionResult
      prediction={null}
      isSubmitting={false}
      error={null}
      serviceAvailable
      serviceMessage={null}
      {...overrides}
    />,
  );
}

describe("PredictionResult", () => {
  it("renders a non-default prediction with probability and threshold", () => {
    renderResult({ prediction: NON_DEFAULT });

    expect(
      screen.getByRole("heading", { name: "No default predicted" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Lower risk")).toBeInTheDocument();
    expect(screen.getByText("1.57%")).toBeInTheDocument();
    expect(screen.getByText("62.00%")).toBeInTheDocument();
    expect(screen.getByText("Threshold 62.00%")).toBeInTheDocument();
    expect(screen.getByText("Random Forest")).toBeInTheDocument();
    expect(
      screen.getByText(/This is an estimate, not a guarantee/),
    ).toBeInTheDocument();
  });

  it("distinguishes a predicted default", () => {
    renderResult({ prediction: DEFAULT });

    expect(
      screen.getByRole("heading", { name: "Default predicted" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Higher risk")).toBeInTheDocument();
    expect(screen.getByText("100.00%")).toBeInTheDocument();
    expect(
      screen.getByText(/at or above the saved decision threshold/),
    ).toBeInTheDocument();
  });

  it("shows a loading state while the API request is in flight", () => {
    renderResult({ isSubmitting: true });

    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByText("Scoring applicant…")).toBeInTheDocument();
  });

  it("shows API failures without crashing", () => {
    renderResult({
      error: new ApiError("Could not reach the API.", {
        status: 0,
        detail: "connection refused",
      }),
    });

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText(/API connection failed/)).toBeInTheDocument();
  });

  it("shows an awaiting state before a submission", () => {
    renderResult();

    expect(
      screen.getByRole("heading", { name: "Awaiting applicant" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Complete the applicant form/i),
    ).toBeInTheDocument();
  });

  it("explains when the prediction service is unavailable", () => {
    renderResult({
      serviceAvailable: false,
      serviceMessage: "Model artifacts are unavailable.",
    });

    expect(
      screen.getByRole("heading", { name: "Prediction service unavailable" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Model artifacts are unavailable."),
    ).toBeInTheDocument();
  });
});
