import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ApplicantForm from "./ApplicantForm";
import { EMPTY_FORM } from "../constants/model";

function renderForm(overrides = {}) {
  const handlers = {
    onChange: vi.fn(),
    onSubmit: vi.fn(),
    onReset: vi.fn(),
    onUseExample: vi.fn(),
    ...overrides,
  };

  render(
    <ApplicantForm
      form={EMPTY_FORM}
      errors={{}}
      isSubmitting={false}
      disabledReason={null}
      submitError={null}
      {...handlers}
    />,
  );

  return handlers;
}

describe("ApplicantForm", () => {
  it("renders all 11 real fields and no target field", () => {
    renderForm();

    for (const label of [
      "Age",
      "Annual income",
      "Employment length",
      "Home ownership",
      "Loan amount",
      "Interest rate",
      "Loan-to-income ratio",
      "Loan grade",
      "Loan purpose",
      "Credit history length",
      "Prior default on file",
    ]) {
      expect(screen.getByLabelText(label)).toBeInTheDocument();
    }

    expect(screen.getAllByRole("spinbutton")).toHaveLength(7);
    expect(screen.getAllByRole("combobox")).toHaveLength(4);
    expect(screen.queryByLabelText(/loan status/i)).not.toBeInTheDocument();
  });

  it("marks the two imputed fields as optional and uses real categories", () => {
    renderForm();

    expect(screen.getByLabelText("Age")).toBeRequired();
    expect(screen.getByLabelText("Employment length")).not.toBeRequired();
    expect(screen.getByLabelText("Interest rate")).not.toBeRequired();
    expect(screen.getByLabelText("Loan amount")).toBeRequired();

    expect(
      screen.getByRole("option", { name: "A — Lowest risk" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: "Debt consolidation" }),
    ).toHaveAttribute("value", "DEBTCONSOLIDATION");
    expect(screen.getByRole("option", { name: "Yes" })).toHaveAttribute(
      "value",
      "Y",
    );
  });

  it("submits and resets through its callbacks", () => {
    const handlers = renderForm();

    const submitButton = screen.getByRole("button", {
      name: /estimate default risk/i,
    });
    fireEvent.submit(submitButton.closest("form"));
    expect(handlers.onSubmit).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: /reset form/i }));
    expect(handlers.onReset).toHaveBeenCalledTimes(1);

    fireEvent.click(
      screen.getByRole("button", { name: /load sample applicant/i }),
    );
    expect(handlers.onUseExample).toHaveBeenCalledTimes(1);
  });

  it("shows field errors against the correct input", () => {
    renderForm({ errors: { person_age: "Age is required." } });

    expect(screen.getByLabelText("Age")).toHaveAttribute(
      "aria-invalid",
      "true",
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Age is required.");
  });

  it("disables submission while the service is unavailable", () => {
    renderForm({ disabledReason: "Checking the prediction service…" });

    expect(
      screen.getByRole("button", { name: /estimate default risk/i }),
    ).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent(
      "Checking the prediction service",
    );
  });

  it("disables duplicate submissions while scoring", () => {
    renderForm({ isSubmitting: true });

    expect(
      screen.getByRole("button", { name: /scoring applicant/i }),
    ).toBeDisabled();
    expect(screen.getByLabelText("Age")).toBeDisabled();
  });

  it("shows a request-level error when there are no field errors", () => {
    renderForm({
      submitError: new Error("The API is offline."),
    });

    expect(screen.getByRole("alert")).toHaveTextContent(
      "The API is offline.",
    );
  });
});
