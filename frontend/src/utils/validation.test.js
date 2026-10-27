import { describe, expect, it } from "vitest";
import {
  ALL_FIELDS,
  EMPTY_FORM,
  EXAMPLE_FORM,
  OPTIONAL_FIELDS,
  REQUIRED_FIELDS,
} from "../constants/model";
import {
  buildPayload,
  firstErrorField,
  mapApiFieldErrors,
  validateForm,
} from "./validation";

const EXPECTED_FIELDS = [
  "person_age",
  "person_income",
  "person_emp_length",
  "person_home_ownership",
  "loan_amnt",
  "loan_int_rate",
  "loan_percent_income",
  "loan_grade",
  "loan_intent",
  "cb_person_cred_hist_length",
  "cb_person_default_on_file",
];

describe("model form schema", () => {
  it("contains the exact 11 backend features and never the target", () => {
    expect(ALL_FIELDS.map((field) => field.name).sort()).toEqual(
      [...EXPECTED_FIELDS].sort(),
    );
    expect(ALL_FIELDS.some((field) => field.name === "loan_status")).toBe(false);
    expect(REQUIRED_FIELDS).toHaveLength(9);
    expect([...OPTIONAL_FIELDS].sort()).toEqual([
      "loan_int_rate",
      "person_emp_length",
    ]);
  });

  it("uses the real categorical options", () => {
    const byName = Object.fromEntries(ALL_FIELDS.map((field) => [field.name, field]));
    expect(byName.loan_grade.options.map((option) => option.value)).toEqual([
      "A",
      "B",
      "C",
      "D",
      "E",
      "F",
      "G",
    ]);
    expect(
      byName.person_home_ownership.options.map((option) => option.value),
    ).toEqual(["MORTGAGE", "OWN", "RENT", "OTHER"]);
    expect(byName.loan_intent.options.map((option) => option.value)).toEqual([
      "DEBTCONSOLIDATION",
      "EDUCATION",
      "HOMEIMPROVEMENT",
      "MEDICAL",
      "PERSONAL",
      "VENTURE",
    ]);
    expect(
      byName.cb_person_default_on_file.options.map((option) => option.value),
    ).toEqual(["N", "Y"]);
  });

  it("accepts the documented example and builds numeric JSON", () => {
    const result = validateForm(EXAMPLE_FORM);
    expect(result.valid).toBe(true);
    expect(result.errors).toEqual({});
    expect(result.payload).toEqual({
      person_age: 35,
      person_income: 100000,
      person_emp_length: 10,
      loan_amnt: 10000,
      loan_int_rate: 7,
      loan_percent_income: 0.1,
      cb_person_cred_hist_length: 10,
      loan_grade: "A",
      person_home_ownership: "MORTGAGE",
      loan_intent: "PERSONAL",
      cb_person_default_on_file: "N",
    });
  });

  it("omits blank optional fields and never includes loan_status", () => {
    const result = validateForm({
      ...EXAMPLE_FORM,
      person_emp_length: "",
      loan_int_rate: "",
    });
    expect(result.valid).toBe(true);
    expect(result.payload).not.toHaveProperty("person_emp_length");
    expect(result.payload).not.toHaveProperty("loan_int_rate");
    expect(result.payload).not.toHaveProperty("loan_status");
  });

  it("requires every non-optional field", () => {
    const result = validateForm(EMPTY_FORM);
    expect(result.valid).toBe(false);
    for (const field of REQUIRED_FIELDS) {
      expect(result.errors[field]).toBeTruthy();
    }
    expect(result.errors.person_emp_length).toBeUndefined();
    expect(result.errors.loan_int_rate).toBeUndefined();
  });

  it("rejects values outside the backend bounds", () => {
    expect(validateForm({ ...EXAMPLE_FORM, person_age: "17" }).errors).toHaveProperty(
      "person_age",
    );
    expect(
      validateForm({ ...EXAMPLE_FORM, person_income: "999" }).errors,
    ).toHaveProperty("person_income");
    expect(
      validateForm({ ...EXAMPLE_FORM, loan_percent_income: "1.01" }).errors,
    ).toHaveProperty("loan_percent_income");
    expect(
      validateForm({ ...EXAMPLE_FORM, cb_person_cred_hist_length: "81" })
        .errors,
    ).toHaveProperty("cb_person_cred_hist_length");
  });

  it("rejects unavailable categories and employment longer than age", () => {
    const categoryResult = validateForm({
      ...EXAMPLE_FORM,
      loan_grade: "Z",
    });
    expect(categoryResult.valid).toBe(false);
    expect(categoryResult.errors.loan_grade).toContain("valid option");

    const employmentResult = validateForm({
      ...EXAMPLE_FORM,
      person_age: "20",
      person_emp_length: "25",
    });
    expect(employmentResult.valid).toBe(false);
    expect(employmentResult.errors.person_emp_length).toContain(
      "cannot be greater than age",
    );
  });

  it("maps Pydantic field locations back to the form", () => {
    const mapped = mapApiFieldErrors([
      { loc: ["body", "person_age"], msg: "Value error, age is invalid" },
      { loc: ["body", "loan_grade"], msg: "Value error, grade is invalid" },
      { loc: ["body"], msg: "A non-field error" },
    ]);
    expect(mapped).toEqual({
      person_age: "age is invalid",
      loan_grade: "grade is invalid",
    });
    expect(firstErrorField(mapped)).toBe("person_age");
  });

  it("builds payloads without mutating the form", () => {
    const copy = { ...EXAMPLE_FORM };
    const payload = buildPayload(EXAMPLE_FORM);
    expect(EXAMPLE_FORM).toEqual(copy);
    expect(payload.person_age).toBe(35);
    expect(Object.keys(payload)).toHaveLength(11);
  });
});
