const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8001")
  .replace(/\/+$/, "");
const configuredTimeout = Number(import.meta.env.VITE_API_TIMEOUT_MS || 60000);
const API_TIMEOUT_MS = Number.isFinite(configuredTimeout) && configuredTimeout > 0
  ? configuredTimeout
  : 60000;

export async function generateSmartNotes({ text, file }) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), API_TIMEOUT_MS);
  try {
    const request = file
      ? (() => {
        const body = new FormData();
        body.append("file", file);
        return {
          url: `${API_BASE_URL}/api/smart-notes/upload`,
          options: { method: "POST", body },
        };
      })()
      : {
        url: `${API_BASE_URL}/api/smart-notes`,
        options: {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        },
      };
    const response = await fetch(request.url, {
      ...request.options,
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload?.detail;
      const message = Array.isArray(detail)
        ? detail.map((item) => item.msg).filter(Boolean).join(" ")
        : detail;
      throw new SmartNotesApiError(message || `Smart Notes request failed (${response.status}).`);
    }
    if (!payload || !Array.isArray(payload.key_points) || typeof payload.organized_notes !== "string") {
      throw new SmartNotesApiError("The Smart Notes API returned an invalid response.");
    }
    return payload;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error("Smart Notes generation timed out. Try a shorter document.", { cause: error });
    }
    if (error instanceof SmartNotesApiError) {
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

class SmartNotesApiError extends Error {}
