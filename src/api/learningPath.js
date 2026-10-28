const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8001")
  .replace(/\/+$/, "");
const configuredTimeout = Number(import.meta.env.VITE_API_TIMEOUT_MS || 15000);
const API_TIMEOUT_MS = Number.isFinite(configuredTimeout) && configuredTimeout > 0
  ? configuredTimeout
  : 15000;

export async function generateLearningPath({ topic, level, goal, knownConcepts = [] }) {
  let response;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), API_TIMEOUT_MS);
  try {
    response = await fetch(`${API_BASE_URL}/api/learning-path`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        topic,
        level,
        goal: goal || null,
        known_concepts: knownConcepts,
      }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload?.detail;
      const message = Array.isArray(detail)
        ? detail.map((item) => item.msg).filter(Boolean).join(" ")
        : detail;
      throw new ApiResponseError(message || `Learning path request failed (${response.status}).`);
    }
    if (!payload || !Array.isArray(payload.learning_modules)) {
      throw new ApiResponseError("The LearnMap API returned an invalid learning path response.");
    }
    return payload;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error("Learning path generation timed out. Please try again.", { cause: error });
    }
    if (error instanceof ApiResponseError) {
      throw error;
    }
    throw new Error(
      `Couldn't reach the LearnMap API at ${API_BASE_URL}. Start the FastAPI backend and try again.`,
      { cause: error },
    );
  } finally {
    window.clearTimeout(timeoutId);
  }
}

class ApiResponseError extends Error {}
