"""Optional scholarly-topic discovery using the public OpenAlex works API."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
OPENALEX_API_URL = "https://api.openalex.org/works"
OPENALEX_RESULTS_LIMIT = 40


class OpenAlexProviderError(RuntimeError):
    """Raised when OpenAlex cannot provide a usable scholarly search result."""


@dataclass(frozen=True)
class OpenAlexConcept:
    id: str
    title: str
    subfield: str
    field: str
    score: float
    supporting_work: str
    supporting_doi: str | None


@dataclass(frozen=True)
class OpenAlexSearchResult:
    concepts: list[OpenAlexConcept]
    score: float


class OpenAlexKnowledgeProvider:
    def __init__(
        self,
        timeout_seconds: float = 4.0,
        api_url: str = OPENALEX_API_URL,
        email: str | None = None,
    ):
        if timeout_seconds <= 0:
            raise ValueError("OpenAlex provider timeout must be positive.")
        self.timeout_seconds = timeout_seconds
        self.api_url = api_url
        self.email = email

    def discover(self, topic: str, goal: str | None = None) -> OpenAlexSearchResult | None:
        query = " ".join(part for part in (topic, goal) if part).strip()
        if not query:
            return None
        params = {
            "search": query,
            "per-page": OPENALEX_RESULTS_LIMIT,
            "select": (
                "id,doi,display_name,primary_topic,topics,publication_year,"
                "relevance_score"
            ),
        }
        headers = {"User-Agent": "LearnMap/1.0 (educational topic retrieval)"}
        if self.email:
            headers["User-Agent"] += f" mailto:{self.email}"
        try:
            response = httpx.get(
                self.api_url,
                params=params,
                timeout=self.timeout_seconds,
                follow_redirects=True,
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise OpenAlexProviderError("OpenAlex scholarly search is unavailable.") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise OpenAlexProviderError("OpenAlex returned a malformed works response.")

        candidates: dict[str, dict[str, Any]] = {}
        for work in payload["results"]:
            if not isinstance(work, dict):
                continue
            work_title = self._clean(work.get("display_name"))
            doi = work.get("doi")
            work_relevance = self._number(work.get("relevance_score"), 0.0)
            primary_topic = work.get("primary_topic")
            topics = [primary_topic] if isinstance(primary_topic, dict) else []
            if not topics and isinstance(work.get("topics"), list):
                topics = [item for item in work["topics"][:3] if isinstance(item, dict)]
            for research_topic in topics:
                topic_id = self._clean(research_topic.get("id"))
                title = self._clean(research_topic.get("display_name"))
                if not topic_id or not title:
                    continue
                subfield = self._nested_name(research_topic.get("subfield"))
                field = self._nested_name(research_topic.get("field"))
                record = candidates.setdefault(topic_id, {
                    "title": title,
                    "subfield": subfield,
                    "field": field,
                    "score": 0.0,
                    "supporting_work": work_title,
                    "supporting_doi": doi if isinstance(doi, str) else None,
                    "work_count": 0,
                    "document": "",
                })
                record["score"] = max(
                    record["score"],
                    self._number(research_topic.get("score"), 0.0),
                    work_relevance,
                )
                record["work_count"] += 1
                if not record["supporting_work"] and work_title:
                    record["supporting_work"] = work_title
                    record["supporting_doi"] = doi if isinstance(doi, str) else None
                record["document"] = (
                    f"{record['title']} {record['subfield']} {record['field']}"
                )

        if not candidates:
            return None
        candidate_items = list(candidates.items())
        documents = [record["document"] for _, record in candidate_items]
        try:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                sublinear_tf=True,
                max_features=20_000,
            )
            matrix = vectorizer.fit_transform([*documents, query])
        except ValueError:
            return None
        similarities = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        ranked = sorted(
            zip(candidate_items, similarities),
            key=lambda item: (
                float(item[1]),
                item[0][1]["score"],
                item[0][1]["work_count"],
            ),
            reverse=True,
        )
        best_score = float(ranked[0][1]) if ranked else 0.0
        if best_score < 0.08:
            return None
        selected = [
            item for item in ranked
            if float(item[1]) >= max(0.25, best_score * 0.6)
        ][:8]
        concepts = [
            OpenAlexConcept(
                id=topic_id,
                title=record["title"],
                subfield=record["subfield"],
                field=record["field"],
                score=float(similarity),
                supporting_work=record["supporting_work"],
                supporting_doi=record["supporting_doi"],
            )
            for ((topic_id, record), similarity) in selected
        ]
        return OpenAlexSearchResult(concepts=concepts, score=best_score)

    @staticmethod
    def _clean(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _nested_name(value: Any) -> str:
        if not isinstance(value, dict):
            return ""
        name = value.get("display_name")
        return re.sub(r"\s+", " ", name).strip() if isinstance(name, str) else ""

    @staticmethod
    def _number(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
