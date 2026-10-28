"""Source-grounded quiz generation and backend-side adaptive grading."""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from fastapi import APIRouter, HTTPException
from nltk.tokenize import wordpunct_tokenize
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_base import CONCEPTS, TOPICS
from knowledge_provider import KnowledgeProviderError
from mooccubex import MoocCubeXDataError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["adaptive-quiz"])
QUIZ_TTL_SECONDS = 3600
MAX_ACTIVE_QUIZZES = 500
MIN_QUIZ_QUESTIONS = 4
MAX_QUIZ_QUESTIONS = 15
_quiz_lock = threading.Lock()


class ConceptMastery(BaseModel):
    concept_id: str = Field(min_length=1, max_length=200)
    correct: int = Field(ge=0, le=1000)
    total: int = Field(ge=1, le=1000)

    @model_validator(mode="after")
    def correct_not_above_total(self) -> ConceptMastery:
        if self.correct > self.total:
            raise ValueError("Correct answers cannot exceed the total.")
        return self


class AdaptiveQuizRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1, max_length=200)
    question_count: int = Field(default=8, ge=MIN_QUIZ_QUESTIONS, le=MAX_QUIZ_QUESTIONS)
    mastery: list[ConceptMastery] = Field(default_factory=list, max_length=100)
    seen_question_ids: list[str] = Field(default_factory=list, max_length=1000)

    @field_validator("topic")
    @classmethod
    def topic_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Enter a topic to build a quiz.")
        return value


class QuizAnswer(BaseModel):
    question_id: str = Field(min_length=1, max_length=300)
    selected_option_id: str = Field(min_length=1, max_length=100)


class SubmitQuizRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quiz_id: str = Field(min_length=1, max_length=100)
    answers: list[QuizAnswer] = Field(min_length=MIN_QUIZ_QUESTIONS, max_length=MAX_QUIZ_QUESTIONS)


@dataclass(frozen=True)
class QuizQuestion:
    id: str
    concept_id: str
    concept: str
    prompt: str
    options: list[dict[str, str]]
    correct_option_id: str
    explanation: str
    source: str
    source_url: str | None = None


@dataclass(frozen=True)
class QuizSession:
    topic: str
    questions: list[QuizQuestion]
    expires_at: float
    source: str


_active_quizzes: dict[str, QuizSession] = {}
_knowledge_provider: Any = None
_mooccubex_catalog: Any = None


def configure_knowledge_sources(knowledge_provider: Any, mooccubex_catalog: Any) -> None:
    global _knowledge_provider, _mooccubex_catalog
    _knowledge_provider = knowledge_provider
    _mooccubex_catalog = mooccubex_catalog


def _normalize(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", re.sub(r"\s+", " ", value.casefold())).strip()


def _match_topic(topic: str) -> str | None:
    normalized = _normalize(topic)
    exact = [
        topic_id
        for topic_id, record in TOPICS.items()
        if any(
            re.search(rf"(?<!\w){re.escape(_normalize(alias))}(?!\w)", normalized)
            for alias in [record["title"], *record["aliases"]]
        )
    ]
    if exact:
        return max(exact, key=lambda topic_id: len(TOPICS[topic_id]["title"]))

    topic_ids = list(TOPICS)
    documents = [
        " ".join(
            [
                TOPICS[topic_id]["title"],
                TOPICS[topic_id]["description"],
                *TOPICS[topic_id]["aliases"],
                *(CONCEPTS[item]["title"] for item in TOPICS[topic_id]["concept_ids"]),
            ]
        )
        for topic_id in topic_ids
    ]
    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        matrix = vectorizer.fit_transform([*documents, normalized])
    except ValueError:
        return None
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    best_index = int(scores.argmax())
    return topic_ids[best_index] if float(scores[best_index]) >= 0.42 else None


def _prerequisite_closure(target_ids: list[str]) -> list[str]:
    selected: set[str] = set()
    pending = list(target_ids)
    while pending:
        concept_id = pending.pop()
        if concept_id in selected:
            continue
        selected.add(concept_id)
        pending.extend(CONCEPTS[concept_id]["prerequisites"])
    return [concept_id for concept_id in CONCEPTS if concept_id in selected]


def _options(correct_text: str, distractors: list[str], key: str) -> tuple[list[dict[str, str]], str]:
    correct_id = f"{key}-correct"
    values = [(correct_id, correct_text)]
    seen = {correct_text.casefold()}
    for index, value in enumerate(distractors):
        if value.casefold() in seen:
            continue
        values.append((f"{key}-option-{index}", value))
        seen.add(value.casefold())
        if len(values) == 4:
            break
    if len(values) < 4:
        return [], ""
    values.sort(key=lambda item: sha256(f"{key}:{item[1]}".encode("utf-8")).hexdigest())
    return [
        {"id": option_id, "text": text}
        for option_id, text in values
    ], correct_id


def _curated_questions(topic_id: str) -> list[QuizQuestion]:
    concept_ids = _prerequisite_closure(TOPICS[topic_id]["concept_ids"])
    records = [(concept_id, CONCEPTS[concept_id]) for concept_id in concept_ids]
    titles = [record["title"] for _, record in records]
    questions: list[QuizQuestion] = []
    for concept_id, record in records:
        alternatives = [title for title in titles if title != record["title"]]
        key = f"{concept_id}-meaning"
        options, correct_id = _options(record["title"], alternatives, key)
        if options:
            questions.append(QuizQuestion(
                id=key,
                concept_id=concept_id,
                concept=record["title"],
                prompt=f"Which concept is described by: {record['description']}",
                options=options,
                correct_option_id=correct_id,
                explanation=record["description"],
                source="LearnMap curated concept catalog",
            ))

    all_objectives = [
        (concept_id, record["title"], objective)
        for concept_id, record in records
        for objective in record["objectives"]
    ]
    for concept_id, record in records:
        own_objectives = record["objectives"]
        for objective_index, objective in enumerate(own_objectives):
            key = f"{concept_id}-objective-{objective_index + 1}"
            distractors = [
                candidate
                for other_id, _, candidate in all_objectives
                if other_id != concept_id or candidate != objective
            ]
            options, correct_id = _options(objective, distractors, key)
            if options:
                questions.append(QuizQuestion(
                    id=key,
                    concept_id=concept_id,
                    concept=record["title"],
                    prompt=f"Which learning outcome is listed for {record['title']}?",
                    options=options,
                    correct_option_id=correct_id,
                    explanation=f"{objective} This is an objective for {record['title']}.",
                    source="LearnMap curated concept catalog",
                ))
            reverse_key = f"{concept_id}-objective-concept-{objective_index + 1}"
            reverse_options, reverse_correct_id = _options(
                record["title"],
                [title for title in titles if title != record["title"]],
                reverse_key,
            )
            if reverse_options:
                questions.append(QuizQuestion(
                    id=reverse_key,
                    concept_id=concept_id,
                    concept=record["title"],
                    prompt=f"Which concept has this learning outcome? {objective}",
                    options=reverse_options,
                    correct_option_id=reverse_correct_id,
                    explanation=f"{objective} This is an objective for {record['title']}.",
                    source="LearnMap curated concept catalog",
                ))
    return questions


def _retrieved_questions(topic: str) -> list[QuizQuestion]:
    if _mooccubex_catalog is not None:
        try:
            result = _mooccubex_catalog.search(topic)
        except MoocCubeXDataError as error:
            logger.warning("MOOCCubeX quiz retrieval failed for %r.", topic, exc_info=error)
            result = None
        if result is not None and result.concepts:
            records = [
                {
                    "id": f"mooccubex-{concept.id}",
                    "title": concept.name,
                    "description": concept.description,
                    "source": "MOOCCubeX",
                    "source_url": None,
                }
                for concept in result.concepts
                if len(concept.description.split()) >= 8
            ]
            questions = _description_questions(records)
            if len(questions) >= MIN_QUIZ_QUESTIONS:
                return questions

    if _knowledge_provider is None:
        return []
    try:
        discovered = _knowledge_provider.discover(topic)
    except KnowledgeProviderError as error:
        logger.warning("External quiz knowledge retrieval failed for %r.", topic, exc_info=error)
        return []
    concepts = _extract_source_concepts(discovered)
    return _source_cloze_questions(concepts)


def _description_questions(records: list[dict[str, Any]]) -> list[QuizQuestion]:
    questions: list[QuizQuestion] = []
    titles = [record["title"] for record in records]
    for record in records:
        if len(record["description"].split()) < 8:
            continue
        key = f"{record['id']}-meaning"
        options, correct_id = _options(
            record["title"],
            [title for title in titles if title != record["title"]],
            key,
        )
        if not options:
            continue
        questions.append(QuizQuestion(
            id=key,
            concept_id=record["id"],
            concept=record["title"],
            prompt=f"Which concept is described by: {record['description']}",
            options=options,
            correct_option_id=correct_id,
            explanation=record["description"],
            source=record["source"],
        ))
    return questions


def _extract_source_concepts(pages: list[Any]) -> list[dict[str, Any]]:
    sentences: list[tuple[str, str | None]] = []
    for page in pages:
        description = re.sub(r"\s+", " ", page.description).strip()
        sentences.extend(
            (sentence.strip(), page.source_url)
            for sentence in re.split(r"(?<=[.!?])\s+", description)
            if len(wordpunct_tokenize(sentence)) >= 8
        )
    if len(sentences) < MIN_QUIZ_QUESTIONS:
        return []
    sentence_texts = [sentence for sentence, _ in sentences]
    try:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            stop_words="english",
            sublinear_tf=True,
            max_features=3000,
        )
        matrix = vectorizer.fit_transform(sentence_texts)
    except ValueError:
        return []
    terms = vectorizer.get_feature_names_out()
    scores = matrix.sum(axis=0).A1
    ranked_terms = [
        term
        for term, _ in sorted(zip(terms, scores), key=lambda item: float(item[1]), reverse=True)
        if len(term.split()) <= 3 and all(len(word) > 2 for word in term.split())
    ]
    concepts: list[dict[str, str]] = []
    used_sentences: set[int] = set()
    for term in ranked_terms:
        sentence_index = next(
            (
                index
                for index, (sentence, _) in enumerate(sentences)
                if index not in used_sentences
                and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", sentence, flags=re.IGNORECASE)
            ),
            None,
        )
        if sentence_index is None:
            continue
        used_sentences.add(sentence_index)
        concepts.append({
            "id": f"source-{sentence_index}",
            "title": term,
            "description": sentences[sentence_index][0],
            "source": "Wikipedia",
            "source_url": sentences[sentence_index][1],
        })
        if len(concepts) >= 8:
            break
    return concepts


def _source_cloze_questions(concepts: list[dict[str, Any]]) -> list[QuizQuestion]:
    terms = [concept["title"] for concept in concepts]
    questions: list[QuizQuestion] = []
    for concept in concepts:
        sentence = concept["description"]
        pattern = re.compile(rf"(?<!\w){re.escape(concept['title'])}(?!\w)", re.IGNORECASE)
        if not pattern.search(sentence):
            continue
        key = f"{concept['id']}-cloze"
        options, correct_id = _options(
            concept["title"],
            [term for term in terms if term.casefold() != concept["title"].casefold()],
            key,
        )
        if not options:
            continue
        questions.append(QuizQuestion(
            id=key,
            concept_id=concept["id"],
            concept=concept["title"],
            prompt=(
                "Which source term best completes this statement? "
                + pattern.sub("_____", sentence, count=1)
            ),
            options=options,
            correct_option_id=correct_id,
            explanation=sentence,
            source=concept["source"],
            source_url=concept.get("source_url"),
        ))
    return questions


def _select_questions(
    questions: list[QuizQuestion],
    count: int,
    mastery: list[ConceptMastery],
    seen_question_ids: list[str],
) -> list[QuizQuestion]:
    by_concept: dict[str, list[QuizQuestion]] = defaultdict(list)
    concept_names: dict[str, str] = {}
    for question in questions:
        by_concept[question.concept_id].append(question)
        concept_names[question.concept_id] = question.concept

    history = {
        item.concept_id: item.correct / item.total
        for item in mastery
    }
    known_ids = set(by_concept)
    weak_ids = {
        concept_id for concept_id, score in history.items()
        if concept_id in known_ids and score < 0.7
    }
    all_ids = list(by_concept)
    ordered_weak = sorted(weak_ids, key=lambda concept_id: (history[concept_id], concept_names[concept_id].casefold()))
    ordered_other = [concept_id for concept_id in all_ids if concept_id not in weak_ids]
    if not ordered_weak:
        ordered_weak = all_ids
        ordered_other = []

    selected: list[QuizQuestion] = []
    already_selected: set[str] = set()
    seen = set(seen_question_ids)

    def take_round_robin(concept_ids: list[str], quota: int) -> None:
        while quota > 0:
            added = False
            for concept_id in concept_ids:
                candidates = by_concept[concept_id]
                question = next(
                    (
                        item for item in candidates
                        if item.id not in already_selected and item.id not in seen
                    ),
                    None,
                )
                if question is None:
                    question = next(
                        (item for item in candidates if item.id not in already_selected),
                        None,
                    )
                if question is None:
                    continue
                selected.append(question)
                already_selected.add(question.id)
                quota -= 1
                added = True
                if quota == 0:
                    break
            if not added:
                return

    if weak_ids and ordered_other:
        weak_quota = min(count, max(1, round(count * 0.7)))
        take_round_robin(ordered_weak, weak_quota)
        take_round_robin(ordered_other, count - len(selected))
    else:
        take_round_robin(ordered_weak, count)
    if len(selected) < count:
        take_round_robin(all_ids, count - len(selected))
    return selected


def _store_quiz(quiz_id: str, session: QuizSession) -> None:
    now = time.monotonic()
    with _quiz_lock:
        expired = [
            key for key, value in _active_quizzes.items()
            if value.expires_at <= now
        ]
        for key in expired:
            del _active_quizzes[key]
        while len(_active_quizzes) >= MAX_ACTIVE_QUIZZES:
            del _active_quizzes[next(iter(_active_quizzes))]
        _active_quizzes[quiz_id] = session


@router.post("/adaptive-quiz")
def create_adaptive_quiz(request: AdaptiveQuizRequest) -> dict[str, Any]:
    topic_id = _match_topic(request.topic)
    if topic_id:
        candidates = _curated_questions(topic_id)
        source = "LearnMap curated concept catalog"
    else:
        candidates = _retrieved_questions(request.topic)
        source = candidates[0].source if candidates else "knowledge retrieval"

    if len(candidates) < MIN_QUIZ_QUESTIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                "There is not enough reliable topic material to build a fair multiple-choice quiz. "
                "Try a more specific topic or add learning material first."
            ),
        )
    selected = _select_questions(
        candidates,
        request.question_count,
        request.mastery,
        request.seen_question_ids,
    )
    if len(selected) < MIN_QUIZ_QUESTIONS:
        raise HTTPException(
            status_code=422,
            detail="This topic has too few distinct questions. Try a broader topic.",
        )

    quiz_id = str(uuid.uuid4())
    _store_quiz(
        quiz_id,
        QuizSession(
            topic=request.topic,
            questions=selected,
            expires_at=time.monotonic() + QUIZ_TTL_SECONDS,
            source=source,
        ),
    )
    return {
        "quiz_id": quiz_id,
        "topic": request.topic,
        "source": source,
        "sources": [
            {"name": name, "url": url}
            for name, url in dict.fromkeys(
                (question.source, question.source_url)
                for question in selected
                if question.source_url
            )
        ],
        "question_count": len(selected),
        "questions": [
            {
                "id": question.id,
                "concept_id": question.concept_id,
                "concept": question.concept,
                "prompt": question.prompt,
                "options": question.options,
                "source": question.source,
                "source_url": question.source_url,
            }
            for question in selected
        ],
        "limitations": [
            "Questions are generated from the cited concept descriptions and objectives; no trained question-generation model is used."
        ],
    }


@router.post("/adaptive-quiz/submit")
def submit_adaptive_quiz(request: SubmitQuizRequest) -> dict[str, Any]:
    with _quiz_lock:
        session = _active_quizzes.get(request.quiz_id)
    if session is None or session.expires_at <= time.monotonic():
        raise HTTPException(
            status_code=404,
            detail="This quiz session expired or was not found. Generate a new quiz and try again.",
        )

    answers = {answer.question_id: answer.selected_option_id for answer in request.answers}
    expected_ids = {question.id for question in session.questions}
    if len(answers) != len(request.answers) or set(answers) != expected_ids:
        raise HTTPException(
            status_code=422,
            detail="Submit exactly one answer for every question in this quiz.",
        )

    concept_counts: dict[str, dict[str, int | str]] = {}
    graded: list[dict[str, Any]] = []
    for question in session.questions:
        selected_option_id = answers[question.id]
        valid_option_ids = {option["id"] for option in question.options}
        if selected_option_id not in valid_option_ids:
            raise HTTPException(status_code=422, detail="A submitted answer is not a valid option.")
        correct = selected_option_id == question.correct_option_id
        counts = concept_counts.setdefault(
            question.concept_id,
            {"concept_id": question.concept_id, "concept": question.concept, "correct": 0, "total": 0},
        )
        counts["total"] = int(counts["total"]) + 1
        if correct:
            counts["correct"] = int(counts["correct"]) + 1
        graded.append({
            "question_id": question.id,
            "concept_id": question.concept_id,
            "concept": question.concept,
            "is_correct": correct,
            "selected_option_id": selected_option_id,
            "correct_option_id": question.correct_option_id,
            "correct_answer": next(
                option["text"] for option in question.options
                if option["id"] == question.correct_option_id
            ),
            "explanation": question.explanation,
            "source": question.source,
        })

    correct_count = sum(bool(item["is_correct"]) for item in graded)
    return {
        "quiz_id": request.quiz_id,
        "topic": session.topic,
        "source": session.source,
        "correct_count": correct_count,
        "total_count": len(graded),
        "score_percent": round(correct_count * 100 / len(graded)),
        "concept_scores": [
            {
                **counts,
                "score_percent": round(int(counts["correct"]) * 100 / int(counts["total"])),
            }
            for counts in concept_counts.values()
        ],
        "answers": graded,
    }
