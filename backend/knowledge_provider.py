"""Optional Wikipedia REST-backed discovery for topics outside the local catalog."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

import httpx

WIKIPEDIA_SUMMARY_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"


@dataclass(frozen=True)
class DiscoveredConcept:
    page_id: str | int
    title: str
    description: str
    source_url: str
    topic_search_rank: int = 0
    goal_search_rank: int | None = None
    source_name: str = "Wikipedia"


class KnowledgeProvider(Protocol):
    def discover(self, topic: str, goal: str | None = None) -> list[DiscoveredConcept]: ...


class KnowledgeProviderError(RuntimeError):
    """Raised when an external knowledge provider cannot return usable data."""


class WikipediaKnowledgeProvider:
    def __init__(
        self,
        timeout_seconds: float = 3.0,
        summary_api_url: str = WIKIPEDIA_SUMMARY_API_URL,
    ):
        if timeout_seconds <= 0:
            raise ValueError("Knowledge provider timeout must be positive.")
        self.timeout_seconds = timeout_seconds
        self.summary_api_url = summary_api_url.rstrip("/")

    def _get_json(self, url: str, params: dict[str, str | int] | None = None) -> dict:
        try:
            response = httpx.get(
                url,
                params=params,
                timeout=self.timeout_seconds,
                follow_redirects=True,
                headers={"User-Agent": "LearnMap/1.0 (educational learning path discovery)"},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise KnowledgeProviderError("Wikipedia knowledge discovery is unavailable.") from error

        if not isinstance(payload, dict) or "error" in payload:
            raise KnowledgeProviderError("Wikipedia returned an invalid knowledge response.")
        return payload

    def discover(self, topic: str, goal: str | None = None) -> list[DiscoveredConcept]:
        title_key = re.sub(r"\s+", "_", topic.strip())
        if not title_key:
            return []
        page = self._get_json(
            f"{self.summary_api_url}/{quote(title_key, safe='')}"
        )
        title = re.sub(r"\s+", " ", str(page.get("title", ""))).strip()
        description = re.sub(r"\s+", " ", str(page.get("extract", ""))).strip()
        source_url = str(
            page.get("content_urls", {}).get("desktop", {}).get("page", "")
        ).strip()
        try:
            page_id: str | int = int(page["pageid"])
        except (KeyError, TypeError, ValueError) as error:
            raise KnowledgeProviderError("Wikipedia returned an invalid article identifier.") from error
        if not title or len(description) < 80 or not source_url.startswith("https://en.wikipedia.org/"):
            return []
        return [DiscoveredConcept(
            page_id=page_id,
            title=title,
            description=description,
            source_url=source_url,
            topic_search_rank=1,
            source_name="Wikipedia",
        )]
