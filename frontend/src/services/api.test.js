import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  API_BASE_URL,
  ApiError,
  getHealth,
  getModelInfo,
  predict,
} from "./api";

let fetchMock;

function mockResponse({
  status = 200,
  body = {},
  contentType = "application/json",
} = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: {
      get(name) {
        return name.toLowerCase() === "content-type" ? contentType : null;
      },
    },
    json: async () => body,
    text: async () =>
      typeof body === "string" ? body : JSON.stringify(body),
  };
}

const MODEL_INFO = {
  model_name: "Random Forest",
  estimator_class: "RandomForestClassifier",
  threshold: 0.62,
  metrics: { accuracy: 0.9358 },
};

const PREDICTION = {
  predicted_class: 0,
  predicted_label: "no default",
  default_probability: 0.0156875,
  threshold: 0.62,
  model_name: "Random Forest",
};

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("API client", () => {
  it("uses the documented local base URL", () => {
    expect(API_BASE_URL).toBe("http://127.0.0.1:8000");
  });

  it("fetches real model information", async () => {
    fetchMock.mockResolvedValue(mockResponse({ body: MODEL_INFO }));

    await expect(getModelInfo()).resolves.toEqual(MODEL_INFO);
    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE_URL}/model-info`,
      expect.objectContaining({
        headers: expect.objectContaining({ Accept: "application/json" }),
      }),
    );
  });

  it("posts the applicant payload as JSON", async () => {
    const applicant = {
      person_age: 35,
      person_income: 100000,
      loan_amnt: 10000,
    };
    fetchMock.mockResolvedValue(mockResponse({ body: PREDICTION }));

    await expect(predict(applicant)).resolves.toEqual(PREDICTION);

    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_BASE_URL}/predict`);
    expect(options.method).toBe("POST");
    expect(options.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(options.body)).toEqual(applicant);
  });

  it("turns network failures into a safe ApiError", async () => {
    fetchMock.mockRejectedValue(new Error("connection refused"));

    await expect(getModelInfo()).rejects.toMatchObject({
      name: "ApiError",
      status: 0,
    });
    await expect(getModelInfo()).rejects.toThrow(/could not be reached/i);
  });

  it("surfaces HTTP validation errors and field locations", async () => {
    fetchMock.mockResolvedValue(
      mockResponse({
        status: 422,
        body: {
          detail: "Invalid applicant data.",
          errors: [
            {
              loc: ["body", "person_age"],
              msg: "Input should be greater than or equal to 18",
            },
          ],
        },
      }),
    );

    try {
      await predict({ person_age: 12 });
      throw new Error("Expected predict to reject");
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect(error.status).toBe(422);
      expect(error.detail).toBe("Invalid applicant data.");
      expect(error.errors).toHaveLength(1);
    }
  });

  it("returns a 503 health payload as degraded state", async () => {
    fetchMock.mockResolvedValue(
      mockResponse({
        status: 503,
        body: {
          api: "ok",
          status: "degraded",
          model_ready: false,
          model_name: null,
          detail: "Model artifacts are unavailable.",
        },
      }),
    );

    await expect(getHealth()).resolves.toMatchObject({
      httpStatus: 503,
      model_ready: false,
      status: "degraded",
    });
  });

  it("still rejects unexpected health failures", async () => {
    fetchMock.mockResolvedValue(
      mockResponse({ status: 500, body: { detail: "Server error." } }),
    );

    await expect(getHealth()).rejects.toMatchObject({
      name: "ApiError",
      status: 500,
    });
  });

  it("rejects a prediction response that violates the contract", async () => {
    fetchMock.mockResolvedValue(
      mockResponse({ body: { predicted_class: 0 } }),
    );

    await expect(predict({})).rejects.toMatchObject({
      name: "ApiError",
      status: 200,
    });
  });
});
