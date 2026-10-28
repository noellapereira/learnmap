const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8001")
  .replace(/\/+$/, "");
const API_TIMEOUT_MS = 30000;

async function postJson(path, body) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), API_TIMEOUT_MS);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify(body),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload?.detail;
      const message = Array.isArray(detail)
        ? detail.map((item) => item.msg).filter(Boolean).join(" ")
        : detail;
      throw new AdaptiveQuizApiError(message || `Adaptive Quiz request failed (${response.status}).`);
    }
    if (!payload || typeof payload !== "object") {
      throw new AdaptiveQuizApiError("The Adaptive Quiz API returned an invalid response.");
    }
    return payload;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error("Quiz generation timed out. Please try again.", { cause: error });
    }
    if (error instanceof AdaptiveQuizApiError) {
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

export function generateAdaptiveQuiz({ topic, mastery, seenQuestionIds }) {
  return postJson("/api/adaptive-quiz", {
    topic,
    question_count: 8,
    mastery,
    seen_question_ids: seenQuestionIds,
  }).then((payload) => {
    if (typeof payload.quiz_id !== "string" || !Array.isArray(payload.questions)) {
      throw new AdaptiveQuizApiError("The Adaptive Quiz API returned an invalid quiz.");
    }
    return payload;
  });
}

export function submitAdaptiveQuiz({ quizId, answers }) {
  return postJson("/api/adaptive-quiz/submit", {
    quiz_id: quizId,
    answers,
  }).then((payload) => {
    if (!Array.isArray(payload.concept_scores) || !Array.isArray(payload.answers)) {
      throw new AdaptiveQuizApiError("The Adaptive Quiz API returned an invalid review.");
    }
    return payload;
  });
}

class AdaptiveQuizApiError extends Error {}
