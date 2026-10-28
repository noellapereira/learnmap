"""LearnMap learning-path API: curated concept matching and DAG sequencing."""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from math import ceil
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nltk.tokenize import wordpunct_tokenize
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_base import CONCEPTS, TOPICS
from adaptive_quiz import configure_knowledge_sources, router as adaptive_quiz_router
from knowledge_provider import (
    DiscoveredConcept,
    KnowledgeProvider,
    KnowledgeProviderError,
    WikipediaKnowledgeProvider,
    WIKIPEDIA_SUMMARY_API_URL,
)
from mooccubex import MoocCubeXCatalog, MoocCubeXDataError, MoocCubeXSearchResult
from openalex_provider import (
    OPENALEX_API_URL,
    OpenAlexConcept,
    OpenAlexKnowledgeProvider,
    OpenAlexProviderError,
    OpenAlexSearchResult,
)
from smart_notes import router as smart_notes_router

logger = logging.getLogger(__name__)
STOP_WORDS = {
    "a", "an", "and", "at", "by", "for", "from", "i", "in", "into", "learn",
    "me", "of", "on", "please", "the", "to", "want", "with", "you",
}
CONCEPT_ORDER = {concept_id: index for index, concept_id in enumerate(CONCEPTS)}
SYNONYM_NORMALIZATION = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "nlp": "natural language processing",
    "dbms": "database management systems",
    "os": "operating systems",
    "cyber security": "cybersecurity",
    "deep neural networks": "deep learning",
    "statistical methods": "statistics",
}

provider_mode = os.getenv("LEARNMAP_KNOWLEDGE_PROVIDER", "openalex").strip().casefold()
provider_timeout = float(os.getenv("LEARNMAP_KNOWLEDGE_TIMEOUT_SECONDS", "3"))
provider_summary_api_url = os.getenv(
    "LEARNMAP_WIKIPEDIA_SUMMARY_API_URL",
    WIKIPEDIA_SUMMARY_API_URL,
)
if provider_mode not in {"openalex", "wikipedia", "local"}:
    raise ValueError("LEARNMAP_KNOWLEDGE_PROVIDER must be 'openalex', 'wikipedia', or 'local'.")
openalex_provider: OpenAlexKnowledgeProvider | None = (
    OpenAlexKnowledgeProvider(
        timeout_seconds=provider_timeout,
        api_url=os.getenv("LEARNMAP_OPENALEX_API_URL", OPENALEX_API_URL),
        email=os.getenv("LEARNMAP_OPENALEX_EMAIL"),
    )
    if provider_mode == "openalex"
    else None
)
knowledge_provider: KnowledgeProvider | None = (
    WikipediaKnowledgeProvider(provider_timeout, provider_summary_api_url)
    if provider_mode in {"openalex", "wikipedia"}
    else None
)
mooccubex_catalog = MoocCubeXCatalog(os.getenv("MOOCCUBEX_DATA_DIR"))
configure_knowledge_sources(knowledge_provider, mooccubex_catalog)

app = FastAPI(title="LearnMap Learning Path API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(smart_notes_router)
app.include_router(adaptive_quiz_router)


class LearnerLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class LearningPathRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1, max_length=200)
    level: LearnerLevel = LearnerLevel.beginner
    goal: str | None = Field(default=None, max_length=500)
    known_concepts: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("topic")
    @classmethod
    def topic_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Topic cannot be empty or whitespace.")
        return value

    @field_validator("goal")
    @classmethod
    def clean_goal(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("known_concepts")
    @classmethod
    def clean_known_concepts(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            value = value.strip()
            if not value:
                raise ValueError("Known concepts must not contain blank entries.")
            cleaned.append(value)
        return cleaned


class LearningResource(BaseModel):
    resource_type: str
    title: str
    source_id: str
    source: str = "MOOCCubeX"
    url: str | None = None


class LearningModule(BaseModel):
    id: str
    title: str
    description: str
    learning_objectives: list[str]
    prerequisites: list[str]
    estimated_minutes: int
    difficulty: LearnerLevel
    suggested_exercise: str
    order: int
    source_url: str | None = None
    source_name: str | None = None
    resources: list[LearningResource] = Field(default_factory=list)


class LearningPathResponse(BaseModel):
    requested_topic: str
    matched_topic: str
    learner_level: LearnerLevel
    detected_concepts: list[str]
    prerequisite_concepts: list[str]
    learning_modules: list[LearningModule]
    estimated_total_minutes: int
    total_concept_count: int
    matching_method: str
    limitations: list[str]


def preprocess(text: str) -> tuple[str, list[str]]:
    """Normalize text and extract informative tokens with NLTK's tokenizer."""
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    for synonym, canonical in SYNONYM_NORMALIZATION.items():
        normalized = re.sub(rf"(?<!\w){re.escape(synonym)}(?!\w)", canonical, normalized)
    normalized = re.sub(r"[\s_-]+", " ", normalized)
    tokens = [
        token
        for token in wordpunct_tokenize(normalized)
        if re.fullmatch(r"[a-z0-9][a-z0-9+#.-]*", token) and token not in STOP_WORDS
    ]
    return normalized, tokens


def _topic_document(topic: dict[str, Any]) -> str:
    concept_text = " ".join(
        f"{CONCEPTS[concept_id]['title']} {CONCEPTS[concept_id]['description']} "
        f"{' '.join(CONCEPTS[concept_id]['aliases'])}"
        for concept_id in topic["concept_ids"]
    )
    return f"{topic['title']} {' '.join(topic['aliases'])} {topic['description']} {concept_text}"


def _match_topic(text: str) -> tuple[str | None, float]:
    normalized, _ = preprocess(text)
    exact_matches = [
        (topic_id, alias)
        for topic_id, topic in TOPICS.items()
        for alias in topic["aliases"]
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized)
    ]
    if exact_matches:
        topic_id, _ = max(exact_matches, key=lambda match: len(match[1]))
        return topic_id, 1.0

    documents = [_topic_document(topic) for topic in TOPICS.values()]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    matrix = vectorizer.fit_transform([*documents, normalized])
    similarities = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    best_index = int(similarities.argmax())
    confidence = float(similarities[best_index])
    ranked_scores = sorted(similarities, reverse=True)
    second_score = float(ranked_scores[1]) if len(ranked_scores) > 1 else 0.0
    if confidence < 0.36 or confidence - second_score < 0.07:
        return None, confidence
    return list(TOPICS)[best_index], confidence


def _discovered_concept_matches(
    value: str,
    concepts: list[DiscoveredConcept],
) -> set[str | int]:
    normalized_value, _ = preprocess(value)
    matches = set()
    for concept in concepts:
        normalized_title, _ = preprocess(concept.title)
        if normalized_value == normalized_title or _concept_phrase_matches(
            normalized_value, [normalized_title]
        ):
            matches.add(concept.page_id)
    return matches


def _extract_keyphrases(text: str, limit: int = 4) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.split()) >= 4
    ]
    if not sentences:
        return []
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        stop_words="english",
        max_features=1000,
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return []
    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(scores, terms), reverse=True)
    keyphrases: list[str] = []
    for _, term in ranked:
        if not any(term in phrase or phrase in term for phrase in keyphrases):
            keyphrases.append(term)
        if len(keyphrases) == limit:
            break
    return keyphrases


def _discovered_path(
    request: LearningPathRequest,
    concepts: list[DiscoveredConcept],
    confidence: float,
) -> LearningPathResponse:
    _, topic_terms = preprocess(request.topic)
    topic_query = " ".join(topic_terms)
    goal_terms = preprocess(request.goal)[1] if request.goal else []
    combined_query = " ".join([*topic_terms, *goal_terms])
    documents = [f"{concept.title} {concept.description}" for concept in concepts]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform([*documents, topic_query, combined_query])
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=(
                "The knowledge source returned pages without enough distinct text to build a reliable path. "
                "Please add context or try a more specific topic."
            ),
        ) from None

    topic_scores = cosine_similarity(matrix[-2], matrix[:-2]).flatten()
    combined_scores = cosine_similarity(matrix[-1], matrix[:-2]).flatten()
    ranked = sorted(
        zip(concepts, topic_scores, combined_scores),
        key=lambda item: (
            float(item[2]),
            float(item[1]),
            -item[0].topic_search_rank if item[0].topic_search_rank else 0,
        ),
        reverse=True,
    )
    relevant = [
        item
        for item in ranked
        if float(item[1]) >= 0.03
        or item[0].topic_search_rank > 0
        or (item[0].goal_search_rank is not None and float(item[2]) >= 0.12)
    ]
    if not relevant:
        raise HTTPException(
            status_code=422,
            detail=(
                "The retrieved page did not contain enough relevant information about this topic. "
                "Please refine your topic or add a subject area or learning goal."
            ),
        )

    known_ids: set[str | int] = set()
    discovered_concepts = [item[0] for item in relevant]
    for known_concept in request.known_concepts:
        known_ids.update(_discovered_concept_matches(known_concept, discovered_concepts))
    relevant = [item for item in relevant if item[0].page_id not in known_ids]
    if not relevant:
        raise HTTPException(
            status_code=422,
            detail=(
                "All discovered concepts were already listed as known. Add a more specific goal "
                "or remove a known concept to generate a path."
            ),
        )

    module_limit = {
        LearnerLevel.beginner: 5,
        LearnerLevel.intermediate: 7,
        LearnerLevel.advanced: 9,
    }[request.level]
    selected = relevant[:module_limit]
    modules: list[LearningModule] = []
    for index, (concept, _, _) in enumerate(selected, start=1):
        keywords = _extract_keyphrases(concept.description)
        objectives = ["Summarize the main ideas using the cited source"]
        if keywords:
            objectives.append(f"Explain the role of {keywords[0]} in {concept.title}")
        if len(keywords) > 1:
            objectives.append(f"Relate {keywords[1]} to the topic using an example")

        word_count = len(concept.description.split())
        reading_rate = {
            LearnerLevel.beginner: 140,
            LearnerLevel.intermediate: 200,
            LearnerLevel.advanced: 260,
        }[request.level]
        estimated_minutes = max(20, min(75, ceil(word_count / reading_rate) * 5 + 15))
        modules.append(LearningModule(
            id=f"{concept.source_name.casefold()}-{concept.page_id}",
            title=concept.title,
            description=concept.description,
            learning_objectives=objectives,
            prerequisites=[],
            estimated_minutes=estimated_minutes,
            difficulty=request.level,
            suggested_exercise=(
                f"Create a short concept map for {concept.title} and {request.topic}; "
                "support each connection with a note from the linked source."
            ),
            order=index,
            source_url=concept.source_url,
            source_name=concept.source_name,
        ))

    limitations = [
        "The direct Wikipedia article summary is source material, not a verified course syllabus.",
        "No prerequisites are asserted for discovered concepts because source evidence did not establish prerequisite relationships.",
        "Objectives, exercises, and study-time estimates are generated heuristics, not educator-reviewed material.",
        f"Topic matching used curated-catalog similarity ({confidence:.2f}) and TF-IDF ranking of the direct Wikipedia article summary.",
    ]
    if known_ids:
        limitations.append(
            "Exact known concepts found in the retrieved pages were excluded; no prerequisite status was inferred."
        )

    return LearningPathResponse(
        requested_topic=request.topic,
        matched_topic=request.topic,
        learner_level=request.level,
        detected_concepts=[module.title for module in modules],
        prerequisite_concepts=[],
        learning_modules=modules,
        estimated_total_minutes=sum(module.estimated_minutes for module in modules),
        total_concept_count=len(modules),
        matching_method=(
            "NLTK wordpunct tokenization and synonym normalization; direct Wikipedia article-summary retrieval; "
            "TF-IDF keyphrase extraction and topic relevance ranking."
        ),
        limitations=limitations,
    )


def _openalex_path(
    request: LearningPathRequest,
    result: OpenAlexSearchResult,
) -> LearningPathResponse:
    known_names = [preprocess(value)[0] for value in request.known_concepts]

    def is_known(concept: OpenAlexConcept) -> bool:
        normalized_title = preprocess(concept.title)[0]
        return any(
            name == normalized_title
            or _concept_phrase_matches(name, [normalized_title])
            for name in known_names
        )

    concepts = [concept for concept in result.concepts if not is_known(concept)]
    module_limit = {
        LearnerLevel.beginner: 5,
        LearnerLevel.intermediate: 7,
        LearnerLevel.advanced: 9,
    }[request.level]
    concepts = concepts[:module_limit]
    if not concepts:
        raise HTTPException(
            status_code=422,
            detail=(
                "OpenAlex found relevant research classifications, but all matched concepts were listed "
                "as already known. Add a more specific goal or revise the known-concepts list."
            ),
        )

    modules: list[LearningModule] = []
    for index, concept in enumerate(concepts, start=1):
        area = " / ".join(value for value in (concept.subfield, concept.field) if value)
        description = (
            f"OpenAlex classifies relevant scholarly work under {concept.title}"
            + (f" ({area})" if area else "")
            + ". The linked OpenAlex record identifies this research topic; it is a subject classification, "
            "not a course syllabus."
        )
        if concept.supporting_work:
            description += f" A related indexed work is “{concept.supporting_work}”."
        objectives = [
            f"Explain what {concept.title} studies in relation to {request.topic}.",
            f"Identify a research question within {concept.title} using the cited topic record.",
        ]
        if request.goal:
            objectives.append(f"Connect {concept.title} to your goal: {request.goal}.")
        resources = (
            [LearningResource(
                resource_type="Scholarly work",
                title=concept.supporting_work,
                source_id=concept.supporting_doi,
                source="OpenAlex",
                url=concept.supporting_doi,
            )]
            if concept.supporting_work and concept.supporting_doi
            else []
        )
        modules.append(LearningModule(
            id=f"openalex-{concept.id.rsplit('/', maxsplit=1)[-1]}",
            title=concept.title,
            description=description,
            learning_objectives=objectives,
            prerequisites=[],
            estimated_minutes=25 if request.level == LearnerLevel.beginner else 35,
            difficulty=request.level,
            suggested_exercise=(
                f"Read the linked OpenAlex topic record and write a short explanation of "
                f"how {concept.title} relates to {request.topic}."
            ),
            order=index,
            source_url=concept.id,
            source_name="OpenAlex",
            resources=resources,
        ))

    limitations = [
        "OpenAlex primary-topic labels classify scholarly works; these results are a relevance-ranked topic map, not a vetted course sequence.",
        "OpenAlex does not provide verified prerequisite relationships here, so no prerequisite edges are asserted.",
        "Objectives, exercises, learner-level labels, and study-time estimates are generated heuristics.",
        "OpenAlex is a scholarly index, not an open course provider; linked papers may require access and their full text is not included.",
        "Topic order follows TF-IDF similarity to the requested topic and goal; it does not imply prerequisite order.",
        f"Matched OpenAlex research topics with TF-IDF/cosine confidence {result.score:.2f}.",
    ]
    excluded_known = [
        concept.title for concept in result.concepts if is_known(concept)
    ]
    if excluded_known:
        limitations.append(
            "Already-known concepts were excluded: " + ", ".join(excluded_known) + "."
        )
    return LearningPathResponse(
        requested_topic=request.topic,
        matched_topic=request.topic,
        learner_level=request.level,
        detected_concepts=[concept.title for concept in concepts],
        prerequisite_concepts=[],
        learning_modules=modules,
        estimated_total_minutes=sum(module.estimated_minutes for module in modules),
        total_concept_count=len(modules),
        matching_method=(
            "OpenAlex scholarly-work search and primary-topic taxonomy; NLTK input normalization; "
            "character n-gram TF-IDF cosine ranking of topic labels and subject fields."
        ),
        limitations=limitations,
    )


def _mooccubex_path(
    request: LearningPathRequest,
    result: MoocCubeXSearchResult,
) -> LearningPathResponse | None:
    candidates = result.concepts
    if not candidates:
        return None

    known_names = [preprocess(value)[0] for value in request.known_concepts]
    known_ids = {
        concept.id
        for concept in candidates
        if any(
            name == preprocess(concept.name)[0]
            or _concept_phrase_matches(name, [preprocess(concept.name)[0]])
            for name in known_names
        )
    }
    limit = {
        LearnerLevel.beginner: 5,
        LearnerLevel.intermediate: 7,
        LearnerLevel.advanced: 9,
    }[request.level]
    target_ids = [
        concept.id
        for concept in sorted(candidates, key=lambda item: item.similarity, reverse=True)
        if concept.id not in known_ids
    ][:limit]
    candidate_by_id = {concept.id: concept for concept in candidates}
    selected_ids: set[str] = set()
    pending = list(target_ids)
    while pending:
        concept_id = pending.pop()
        if concept_id in selected_ids or concept_id in known_ids:
            continue
        selected_ids.add(concept_id)
        concept = candidate_by_id.get(concept_id)
        if concept:
            pending.extend(concept.prerequisites)
    if not selected_ids:
        return None

    indegree = {
        concept_id: sum(
            prerequisite in selected_ids
            for prerequisite in candidate_by_id[concept_id].prerequisites
        )
        for concept_id in selected_ids
    }

    def order_key(concept_id: str) -> tuple[float, str]:
        concept = candidate_by_id[concept_id]
        return (-concept.similarity, concept.name.casefold())

    ready = sorted(
        (concept_id for concept_id, degree in indegree.items() if degree == 0),
        key=order_key,
    )
    ordered_ids: list[str] = []
    while ready:
        concept_id = ready.pop(0)
        ordered_ids.append(concept_id)
        for dependent in selected_ids:
            if concept_id in candidate_by_id[dependent].prerequisites:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
                    ready.sort(key=order_key)
    if len(ordered_ids) != len(selected_ids):
        raise MoocCubeXDataError("The MOOCCubeX ground-truth prerequisite graph contains a cycle.")

    modules: list[LearningModule] = []
    for index, concept_id in enumerate(ordered_ids, start=1):
        concept = candidate_by_id[concept_id]
        description = concept.description.strip() or (
            "MOOCCubeX identifies this concept but does not provide a description."
        )
        objectives = [
            f"Explain the central ideas in {concept.name} using the cited course material.",
            f"Connect {concept.name} to the other concepts in this learning sequence.",
        ]
        reading_minutes = min(45, max(10, ceil(len(description.split()) / 8)))
        resources = [
            LearningResource(
                resource_type=resource.resource_type,
                title=resource.title,
                source_id=resource.source_id,
            )
            for resource in concept.resources
        ]
        course_context = f" from {result.matched_course}" if result.matched_course else ""
        modules.append(LearningModule(
            id=f"mooccubex-{concept.id}",
            title=concept.name,
            description=description,
            learning_objectives=objectives,
            prerequisites=[
                candidate_by_id[prerequisite].name
                for prerequisite in concept.prerequisites
                if prerequisite in selected_ids or prerequisite in known_ids
            ],
            estimated_minutes=reading_minutes + (5 * len(resources)),
            difficulty=request.level,
            suggested_exercise=(
                f"Write a short explanation of {concept.name}{course_context}, then relate it to "
                "one prerequisite or neighboring concept in this path."
            ),
            order=index,
            resources=resources,
        ))

    known_prerequisites = {
        prerequisite
        for concept_id in ordered_ids
        for prerequisite in candidate_by_id[concept_id].prerequisites
        if prerequisite in known_ids
    }
    excluded_known = [
        concept.name for concept in candidates if concept.id in known_ids
    ]
    limitations = [
        "MOOCCubeX course and concept coverage is concentrated in its released fields; this path is not a universal curriculum.",
        "Prerequisites use only MOOCCubeX records labeled ground_truth=1. Course resources supply references but do not establish prerequisites or determine graph ordering.",
        "Difficulty selection, study-time estimates, objectives, and exercises are heuristics; the dataset does not certify these for an individual learner.",
        "MOOCCubeX resource titles and identifiers are shown when available; the loaded schema does not guarantee publicly accessible resource URLs.",
        "No prerequisite prediction model was used to create this path; only recorded ground-truth relationships were used.",
    ]
    if excluded_known:
        limitations.append(
            "Known concepts were excluded from the modules: " + ", ".join(excluded_known) + "."
        )
    return LearningPathResponse(
        requested_topic=request.topic,
        matched_topic=result.matched_course or request.topic,
        learner_level=request.level,
        detected_concepts=[concept.name for concept in candidates[:8]],
        prerequisite_concepts=sorted(
            candidate_by_id[concept_id].name for concept_id in known_prerequisites
        ),
        learning_modules=modules,
        estimated_total_minutes=sum(module.estimated_minutes for module in modules),
        total_concept_count=len(modules),
        matching_method=(
            "MOOCCubeX concept/course retrieval with character n-gram TF-IDF and cosine similarity; "
            "ground_truth prerequisite edges ordered by Kahn's algorithm, with relevance used only "
            "to break ties between currently eligible concepts."
        ),
        limitations=limitations,
    )


def _concept_phrase_matches(text: str, aliases: list[str]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(alias.casefold())}(?!\w)", text) for alias in aliases)


def _goal_concepts(goal: str, candidate_ids: list[str]) -> list[str]:
    _, terms = preprocess(goal)
    if not terms:
        return []
    candidates = [CONCEPTS[concept_id] for concept_id in candidate_ids]
    documents = [
        f"{concept['title']} {concept['description']} {' '.join(concept['aliases'])} "
        f"{' '.join(concept['objectives'])}"
        for concept in candidates
    ]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    matrix = vectorizer.fit_transform([*documents, " ".join(terms)])
    similarities = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    return [
        candidate_ids[index]
        for index, score in enumerate(similarities)
        if score >= 0.13
    ]


def _known_concept_ids(values: list[str]) -> tuple[set[str], list[str]]:
    known: set[str] = set()
    unknown: list[str] = []
    for value in values:
        normalized, _ = preprocess(value)
        matches = [
            concept_id
            for concept_id in CONCEPTS
            if normalized == concept_id
            or _concept_phrase_matches(
                normalized,
                [CONCEPTS[concept_id]["title"].casefold(), *CONCEPTS[concept_id]["aliases"]],
            )
        ]
        if matches:
            known.update(matches)
        else:
            unknown.append(value)
    return known, unknown


def _prerequisite_closure(target_ids: list[str], known_ids: set[str]) -> set[str]:
    selected: set[str] = set()
    pending = list(target_ids)
    while pending:
        concept_id = pending.pop()
        if concept_id in selected or concept_id in known_ids:
            continue
        if concept_id not in CONCEPTS:
            raise ValueError(f"Knowledge base references unknown concept '{concept_id}'.")
        selected.add(concept_id)
        pending.extend(CONCEPTS[concept_id]["prerequisites"])
    return selected


def _topological_order(selected: set[str]) -> list[str]:
    indegree = {
        concept_id: sum(
            prerequisite in selected
            for prerequisite in CONCEPTS[concept_id]["prerequisites"]
        )
        for concept_id in selected
    }
    ready = sorted(
        (concept_id for concept_id, degree in indegree.items() if degree == 0),
        key=CONCEPT_ORDER.__getitem__,
    )
    ordered: list[str] = []
    while ready:
        concept_id = ready.pop(0)
        ordered.append(concept_id)
        for dependent in sorted(selected):
            if concept_id in CONCEPTS[dependent]["prerequisites"]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
                    ready.sort(key=CONCEPT_ORDER.__getitem__)
    if len(ordered) != len(selected):
        raise ValueError("The concept knowledge base contains a prerequisite cycle.")
    return ordered


def _target_concepts(topic_ids: list[str], level: LearnerLevel) -> list[str]:
    if level == LearnerLevel.beginner:
        return list(topic_ids)
    elif level == LearnerLevel.intermediate:
        targets = [
            concept_id
            for concept_id in topic_ids
            if CONCEPTS[concept_id]["difficulty"] in {"intermediate", "advanced"}
        ]
    else:
        targets = [
            concept_id
            for concept_id in topic_ids
            if CONCEPTS[concept_id]["difficulty"] == "advanced"
        ]
    return targets or list(topic_ids)


def _search_module_resources(title: str) -> tuple[list[LearningResource], bool]:
    resources: list[LearningResource] = []
    failed = False
    if knowledge_provider is not None:
        try:
            pages = knowledge_provider.discover(title)
        except KnowledgeProviderError as error:
            logger.warning("Wikipedia resource search failed for %r.", title, exc_info=error)
            failed = True
        else:
            resources.extend(
                LearningResource(
                    resource_type="Article",
                    title=page.title,
                    source_id=str(page.page_id),
                    source=page.source_name,
                    url=page.source_url,
                )
                for page in pages
            )
    if openalex_provider is not None:
        try:
            result = openalex_provider.discover(title)
        except OpenAlexProviderError as error:
            logger.warning("OpenAlex resource search failed for %r.", title, exc_info=error)
            failed = True
        else:
            if result is not None:
                resources.extend(
                    LearningResource(
                        resource_type="Scholarly work",
                        title=concept.supporting_work,
                        source_id=concept.supporting_doi,
                        source="OpenAlex",
                        url=concept.supporting_doi,
                    )
                    for concept in result.concepts
                    if concept.supporting_work
                    and concept.supporting_doi
                    and concept.supporting_doi.startswith("https://")
                )
    unique: dict[tuple[str, str], LearningResource] = {}
    for resource in resources:
        if resource.url:
            unique[(resource.source, resource.url)] = resource
    return list(unique.values()), failed


def _attach_curated_resources(path: LearningPathResponse) -> LearningPathResponse:
    if knowledge_provider is None and openalex_provider is None:
        path.limitations.append(
            "External resource search is disabled; the generated prerequisite sequence is available without linked resources."
        )
        return path

    lookup_failures = False
    with ThreadPoolExecutor(max_workers=8) as executor:
        tasks = {
            executor.submit(_search_module_resources, module.title): module
            for module in path.learning_modules
        }
        for future in as_completed(tasks):
            module = tasks[future]
            resources, failed = future.result()
            lookup_failures = lookup_failures or failed
            existing_urls = {resource.url for resource in module.resources if resource.url}
            module.resources.extend(
                resource for resource in resources if resource.url not in existing_urls
            )

    if lookup_failures:
        path.limitations.append(
            "One or more external resource searches failed; the learning path and its prerequisite order are unchanged."
        )
    if any(module.resources for module in path.learning_modules):
        path.limitations.append(
            "Linked resources are source-backed article or scholarly-work search results; they are not educator-reviewed course recommendations."
        )
    unlinked_count = sum(not module.resources for module in path.learning_modules)
    if unlinked_count:
        path.limitations.append(
            f"No verified resource link was found for {unlinked_count} of {len(path.learning_modules)} concepts; missing links were not fabricated."
        )
    return path


def generate_learning_path(request: LearningPathRequest) -> LearningPathResponse:
    matched_topic_id, confidence = _match_topic(request.topic)
    if matched_topic_id is None:
        dataset_failure: str | None = None
        try:
            dataset_result = mooccubex_catalog.search(
                " ".join(preprocess(request.topic)[1]) or request.topic,
                " ".join(preprocess(request.goal)[1]) if request.goal else None,
            )
        except MoocCubeXDataError as error:
            logger.warning("MOOCCubeX retrieval failed; trying the external provider.", exc_info=error)
            dataset_result = None
            dataset_failure = "The configured MOOCCubeX files could not be read."
        if dataset_result is not None:
            try:
                dataset_path = _mooccubex_path(request, dataset_result)
            except MoocCubeXDataError as error:
                logger.warning("MOOCCubeX path ordering failed; trying the external provider.", exc_info=error)
                dataset_path = None
                dataset_failure = "The MOOCCubeX prerequisite graph contains a cycle."
            if dataset_path is not None:
                return dataset_path
        openalex_error: OpenAlexProviderError | None = None
        if openalex_provider is not None:
            try:
                openalex_result = openalex_provider.discover(
                    " ".join(preprocess(request.topic)[1]) or request.topic,
                    " ".join(preprocess(request.goal)[1]) if request.goal else None,
                )
            except OpenAlexProviderError as error:
                logger.warning("OpenAlex topic discovery failed; trying the next source.", exc_info=error)
                openalex_error = error
                openalex_result = None
            if openalex_result is not None and openalex_result.concepts:
                return _openalex_path(request, openalex_result)
        if knowledge_provider is None:
            if dataset_failure:
                _raise_unsupported_topic(
                    f"{dataset_failure} External topic discovery is disabled, so no path was generated."
                )
            if openalex_error:
                _raise_unsupported_topic(
                    "OpenAlex is unavailable, no local data matched this topic, and external discovery is disabled."
                )
            _raise_unsupported_topic(
                "External topic discovery is disabled. MOOCCubeX did not contain enough matching information. "
                "Configure a knowledge provider or add more context."
            )
        try:
            normalized_topic = request.topic.strip()
            discovered = knowledge_provider.discover(normalized_topic, request.goal)
        except KnowledgeProviderError as error:
            logger.warning("External topic discovery failed; local catalog was checked.", exc_info=error)
            if openalex_error:
                logger.warning("Both OpenAlex and Wikipedia discovery failed for the submitted topic.")
            raise HTTPException(
                status_code=503,
                detail=(
                    "No local topic data was available, and external knowledge retrieval failed. "
                    "Retry later or refine the topic with a subject area or learning goal."
                ),
            ) from error
        if discovered:
            return _discovered_path(request, discovered, confidence)
        _raise_unsupported_topic(
            "The local catalog and external knowledge source could not find enough reliable material. "
            "Please refine the topic or add a subject area and learning goal."
        )

    topic = TOPICS[matched_topic_id]
    topic_ids = topic["concept_ids"]
    _, submitted_terms = preprocess(request.topic)
    detected_ids = [
        concept_id
        for concept_id in topic_ids
        if _concept_phrase_matches(
            " ".join(submitted_terms),
            [CONCEPTS[concept_id]["title"].casefold(), *CONCEPTS[concept_id]["aliases"]],
        )
    ]

    known_ids, unknown_known = _known_concept_ids(request.known_concepts)
    targets = _target_concepts(topic_ids, request.level)
    limitations: list[str] = []
    if request.goal:
        goal_targets = _goal_concepts(request.goal, topic_ids)
        if goal_targets:
            targets = goal_targets
        else:
            limitations.append(
                "The goal did not match a specific catalog concept; the level-based path is shown."
            )
    if unknown_known:
        limitations.append(
            "These known concepts were not recognized in this topic's catalog and were not excluded: "
            + ", ".join(unknown_known)
            + "."
        )
    if confidence < 1:
        limitations.append(
            "Topic matching used character n-gram TF-IDF similarity against the documented catalog."
        )
    selected = _prerequisite_closure(targets, known_ids)
    ordered_ids = _topological_order(selected)
    position = {concept_id: index + 1 for index, concept_id in enumerate(ordered_ids)}
    modules = [
        LearningModule(
            id=concept_id,
            title=CONCEPTS[concept_id]["title"],
            description=CONCEPTS[concept_id]["description"],
            learning_objectives=CONCEPTS[concept_id]["objectives"],
            prerequisites=[
                CONCEPTS[prerequisite]["title"]
                for prerequisite in CONCEPTS[concept_id]["prerequisites"]
            ],
            estimated_minutes=CONCEPTS[concept_id]["minutes"],
            difficulty=CONCEPTS[concept_id]["difficulty"],
            suggested_exercise=CONCEPTS[concept_id]["exercise"],
            order=position[concept_id],
        )
        for concept_id in ordered_ids
    ]
    prerequisite_ids = {
        prerequisite
        for concept_id in ordered_ids
        for prerequisite in CONCEPTS[concept_id]["prerequisites"]
        if prerequisite not in selected
    }

    return LearningPathResponse(
        requested_topic=request.topic,
        matched_topic=topic["title"],
        learner_level=request.level,
        detected_concepts=[CONCEPTS[concept_id]["title"] for concept_id in detected_ids],
        prerequisite_concepts=sorted(CONCEPTS[concept_id]["title"] for concept_id in prerequisite_ids),
        learning_modules=modules,
        estimated_total_minutes=sum(module.estimated_minutes for module in modules),
        total_concept_count=len(modules),
        matching_method=(
            "NLTK wordpunct tokenization, stop-word filtering, and synonym normalization; character n-gram "
            "TF-IDF cosine matching; curated prerequisite DAG ordered by Kahn's algorithm."
        ),
        limitations=limitations,
    )


def _raise_unsupported_topic(message: str | None = None) -> None:
    supported = ", ".join(topic["title"] for topic in TOPICS.values())
    raise HTTPException(
        status_code=422,
        detail=message
        or (
            "We couldn't confidently match this topic or discover reliable source material. "
            f"Choose a supported local topic or add context: {supported}."
        ),
    )


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "mooccubex_configured": mooccubex_catalog.configured,
        "mooccubex_available": mooccubex_catalog.available,
        "prerequisite_model": mooccubex_catalog.training_status,
    }


@app.post("/api/learning-path", response_model=LearningPathResponse)
def create_learning_path(request: LearningPathRequest) -> LearningPathResponse:
    path = generate_learning_path(request)
    if any(module.id in CONCEPTS for module in path.learning_modules):
        return _attach_curated_resources(path)
    return path
