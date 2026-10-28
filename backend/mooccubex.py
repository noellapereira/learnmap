"""Optional loader and TF-IDF retriever for a local MOOCCubeX data release."""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class MoocCubeXDataError(RuntimeError):
    """Raised when configured MOOCCubeX files cannot be read safely."""


@dataclass(frozen=True)
class MoocCubeXResource:
    resource_type: str
    title: str
    source_id: str


@dataclass(frozen=True)
class MoocCubeXConcept:
    id: str
    name: str
    description: str
    similarity: float
    prerequisites: list[str]
    resources: list[MoocCubeXResource]


@dataclass(frozen=True)
class MoocCubeXSearchResult:
    concepts: list[MoocCubeXConcept]
    matched_course: str
    matching_score: float


def _text(value: Any) -> str:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, list):
        return " ".join(part for part in (_text(item) for item in value) if part)
    if isinstance(value, dict):
        return " ".join(part for part in (_text(item) for item in value.values()) if part)
    return ""


def _records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            try:
                payload = json.load(file)
            except json.JSONDecodeError:
                file.seek(0)
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise MoocCubeXDataError(
                            f"Invalid JSON in {path.name} at line {line_number}."
                        ) from error
                    if isinstance(record, dict):
                        yield record
                return
    except OSError as error:
        raise MoocCubeXDataError(f"Unable to read MOOCCubeX file {path}.") from error

    if isinstance(payload, list):
        for record in payload:
            if isinstance(record, dict):
                yield record
    elif isinstance(payload, dict):
        records = payload.get("data")
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    yield record
        else:
            yield payload
    else:
        raise MoocCubeXDataError(f"Expected a JSON object or array in {path.name}.")


def _mapping(path: Path) -> Iterator[tuple[str, str]]:
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.reader(file, delimiter="\t"):
                if len(row) >= 2 and row[0].strip().casefold() not in {"concept_id", "concept"}:
                    yield row[0].strip(), row[1].strip()
    except OSError as error:
        raise MoocCubeXDataError(f"Unable to read MOOCCubeX relation file {path}.") from error


def _label(value: Any) -> int | None:
    try:
        label = int(value)
    except (TypeError, ValueError):
        return None
    return label if label in {0, 1} else None


class MoocCubeXCatalog:
    """Build a bounded concept index from the documented entities/relations files."""

    def __init__(self, data_dir: str | Path | None):
        self.data_dir = Path(data_dir).expanduser() if data_dir else None
        self._loaded = False
        self._load_error: MoocCubeXDataError | None = None
        self._concepts: dict[str, dict[str, Any]] = {}
        self._courses: dict[str, dict[str, Any]] = {}
        self._concept_courses: dict[str, set[str]] = {}
        self._concept_videos: dict[str, set[str]] = {}
        self._video_ccids: dict[str, set[str]] = {}
        self._prerequisites: dict[str, set[str]] = {}
        self._training_status = "not trained: dataset not loaded"
        self._enriched: dict[str, dict[str, Any]] = {}
        self._course_ids: list[str] = []
        self._course_documents: list[str] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._concept_matrix: Any = None
        self._course_matrix: Any = None

    @property
    def configured(self) -> bool:
        return self.data_dir is not None

    @property
    def available(self) -> bool:
        if not self.data_dir:
            return False
        return (
            (self.data_dir / "entities" / "concept.json").is_file()
            and (self.data_dir / "entities" / "course.json").is_file()
        )

    @property
    def training_status(self) -> str:
        return self._training_status

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            if self._load_error:
                raise self._load_error
            return bool(self._concepts)
        self._loaded = True
        if not self.available or self.data_dir is None:
            self._training_status = "not trained: MOOCCubeX files are not configured"
            return False
        try:
            self._load()
        except MoocCubeXDataError as error:
            self._load_error = error
            raise
        return bool(self._concepts)

    def _load(self) -> None:
        assert self.data_dir is not None
        entities = self.data_dir / "entities"
        relations = self.data_dir / "relations"
        for record in _records(entities / "concept.json"):
            concept_id = _text(record.get("id"))
            name = _text(record.get("name"))
            if concept_id and name:
                self._concepts[concept_id] = {
                    "name": name,
                    "context": _text(record.get("context")),
                }
        for record in _records(entities / "course.json"):
            course_id = _text(record.get("id"))
            name = _text(record.get("name"))
            if course_id and name:
                self._courses[course_id] = {
                    "name": name,
                    "about": _text(record.get("about")),
                    "field": _text(record.get("field")),
                    "resources": record.get("resource", []),
                }
        for concept_id, course_id in _mapping(relations / "concept-course.txt"):
            self._concept_courses.setdefault(concept_id, set()).add(course_id)
        for concept_id, video_id in _mapping(relations / "concept-video.txt"):
            self._concept_videos.setdefault(concept_id, set()).add(video_id)
        for video_id, ccid in _mapping(relations / "video_id-ccid.txt"):
            self._video_ccids.setdefault(video_id, set()).add(ccid)

        labeled_edges: list[tuple[str, str, int]] = []
        prerequisite_dir = self.data_dir / "prerequisites"
        for filename in ("cs.json", "math.json", "psy.json"):
            path = prerequisite_dir / filename
            if not path.is_file():
                continue
            for record in _records(path):
                prerequisite = _text(record.get("c1"))
                concept = _text(record.get("c2"))
                label = _label(record.get("ground_truth"))
                if prerequisite and concept and label is not None:
                    labeled_edges.append((prerequisite, concept, label))
                    if label == 1:
                        self._prerequisites.setdefault(concept, set()).add(prerequisite)
        self._training_status = self._train_and_evaluate(labeled_edges)
        self._prepare_index()

    def _train_and_evaluate(self, edges: list[tuple[str, str, int]]) -> str:
        counts = {label: sum(edge[2] == label for edge in edges) for label in (0, 1)}
        if min(counts.values()) < 5 or len(edges) < 20:
            return (
                "not trained: prerequisite labels are unavailable or too few "
                f"(negative={counts[0]}, positive={counts[1]})"
            )
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            from sklearn.model_selection import train_test_split

            names = {concept_id: value["name"] for concept_id, value in self._concepts.items()}
            texts = [
                f"{names.get(source, source.replace('_', ' '))} prerequisite relation "
                f"{names.get(target, target.replace('_', ' '))}"
                for source, target, _ in edges
            ]
            labels = [label for _, _, label in edges]
            train_text, test_text, train_labels, test_labels = train_test_split(
                texts,
                labels,
                test_size=0.2,
                random_state=42,
                stratify=labels,
            )
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
            train_matrix = vectorizer.fit_transform(train_text)
            test_matrix = vectorizer.transform(test_text)
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
            model.fit(train_matrix, train_labels)
            predictions = model.predict(test_matrix)
        except (ValueError, KeyError) as error:
            logger.warning("MOOCCubeX prerequisite model training was skipped: %s", error)
            return f"not trained: labeled data could not be split safely ({error})"

        metrics = {
            "accuracy": accuracy_score(test_labels, predictions),
            "precision": precision_score(test_labels, predictions, zero_division=0),
            "recall": recall_score(test_labels, predictions, zero_division=0),
            "f1": f1_score(test_labels, predictions, zero_division=0),
        }
        logger.info("MOOCCubeX prerequisite holdout metrics: %s", metrics)
        return (
            f"trained/evaluated on {len(edges)} labeled edges; "
            + ", ".join(f"{key}={value:.3f}" for key, value in metrics.items())
        )

    def _prepare_index(self) -> None:
        enriched: dict[str, dict[str, Any]] = {}
        course_documents: dict[str, str] = {}
        for course_id, course in self._courses.items():
            resources = course["resources"]
            if not isinstance(resources, list):
                resources = []
            resource_terms: list[str] = []
            for resource in resources:
                if not isinstance(resource, dict):
                    continue
                title = _text(resource.get("titles")) or _text(resource.get("chapter"))
                if title:
                    resource_terms.append(title)
            course_documents[course_id] = " ".join(
                [course["name"], course["about"], course["field"], *resource_terms]
            )

        for concept_id, concept in self._concepts.items():
            courses = [
                self._courses[course_id]
                for course_id in self._concept_courses.get(concept_id, set())
                if course_id in self._courses
            ]
            course_text = " ".join(
                part
                for course in courses
                for part in (course["name"], course["about"], course["field"])
            )
            course_names = [course["name"] for course in courses]
            resources: list[MoocCubeXResource] = []
            linked_video_ids = self._concept_videos.get(concept_id, set())
            linked_resource_ids = set(linked_video_ids)
            for video_id in linked_video_ids:
                linked_resource_ids.update(self._video_ccids.get(video_id, set()))
            for course_id in self._concept_courses.get(concept_id, set()):
                course = self._courses.get(course_id)
                if not course:
                    continue
                course_resources = course["resources"]
                if not isinstance(course_resources, list):
                    continue
                for resource in course_resources:
                    if not isinstance(resource, dict):
                        continue
                    resource_id = _text(resource.get("resource_id"))
                    if linked_resource_ids and resource_id not in linked_resource_ids:
                        continue
                    title = _text(resource.get("titles")) or _text(resource.get("chapter"))
                    if title:
                        resources.append(MoocCubeXResource(
                            resource_type=_text(resource.get("resource_type")) or "course material",
                            title=title,
                            source_id=resource_id or course_id,
                        ))
            enriched[concept_id] = {
                "name": concept["name"],
                "description": concept["context"] or course_text,
                "course_names": course_names,
                "resources": resources[:5],
                "document": " ".join(
                    [concept["name"], concept["context"], course_text]
                )[:5000],
            }

        if not enriched:
            return
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            sublinear_tf=True,
            max_features=150_000,
        )
        try:
            documents = [record["document"] for record in enriched.values()]
            course_ids = list(course_documents)
            course_texts = [course_documents[course_id] for course_id in course_ids]
            matrix = vectorizer.fit_transform([*documents, *course_texts])
        except ValueError as error:
            raise MoocCubeXDataError("Could not build the MOOCCubeX TF-IDF index.") from error
        self._enriched = enriched
        self._course_ids = course_ids
        self._course_documents = course_texts
        self._vectorizer = vectorizer
        self._concept_matrix = matrix[:len(documents)]
        self._course_matrix = matrix[len(documents):]

    def search(self, topic: str, goal: str | None = None) -> MoocCubeXSearchResult | None:
        if not self._ensure_loaded() or self._vectorizer is None or self._concept_matrix is None:
            return None

        queries = [topic, f"{topic} {goal or ''}"]
        try:
            query_matrix = self._vectorizer.transform(queries)
        except ValueError:
            return None
        topic_scores = cosine_similarity(query_matrix[0], self._concept_matrix).flatten()
        goal_scores = cosine_similarity(query_matrix[1], self._concept_matrix).flatten()
        ranked = sorted(
            zip(self._enriched.items(), topic_scores, goal_scores),
            key=lambda item: (float(item[1]) * 0.7 + float(item[2]) * 0.3, float(item[1])),
            reverse=True,
        )
        best_score = float(ranked[0][1]) if ranked else 0.0
        if best_score < 0.08:
            return None
        selected = [
            item
            for item in ranked
            if float(item[1]) >= max(0.08, best_score * 0.35)
        ][:12]
        selected_ids = {item[0][0] for item in selected}

        # Include only the ancestors explicitly labeled as prerequisites in the dataset.
        pending = list(selected_ids)
        while pending:
            concept_id = pending.pop()
            for prerequisite_id in self._prerequisites.get(concept_id, set()):
                if prerequisite_id in self._concepts and prerequisite_id not in selected_ids:
                    selected_ids.add(prerequisite_id)
                    pending.append(prerequisite_id)
        result_concepts = []
        for (concept_id, record), topic_score, combined_score in selected:
            if concept_id in selected_ids:
                result_concepts.append(MoocCubeXConcept(
                    id=concept_id,
                    name=record["name"],
                    description=record["description"],
                    similarity=float(topic_score) * 0.7 + float(combined_score) * 0.3,
                    prerequisites=sorted(
                        self._prerequisites.get(concept_id, set()) & selected_ids
                    ),
                    resources=record["resources"],
                ))
        for concept_id in selected_ids - {concept.id for concept in result_concepts}:
            record = self._enriched[concept_id]
            result_concepts.append(MoocCubeXConcept(
                id=concept_id,
                name=record["name"],
                description=record["description"],
                similarity=0.0,
                prerequisites=sorted(
                    self._prerequisites.get(concept_id, set()) & selected_ids
                ),
                resources=record["resources"],
            ))

        course_scores = (
            cosine_similarity(query_matrix[1], self._course_matrix).flatten()
            if self._course_matrix is not None and self._course_ids
            else []
        )
        matched_course = (
            self._courses[self._course_ids[int(course_scores.argmax())]]["name"]
            if len(course_scores)
            else topic
        )
        result_concepts.sort(
            key=lambda concept: (
                -concept.similarity,
                concept.name.casefold(),
            )
        )
        return MoocCubeXSearchResult(
            concepts=result_concepts,
            matched_course=matched_course or topic,
            matching_score=best_score,
        )
