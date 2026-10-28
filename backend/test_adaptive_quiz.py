from fastapi.testclient import TestClient

from knowledge_provider import DiscoveredConcept
import adaptive_quiz
from main import app


client = TestClient(app)


def _create_quiz(topic: str, **payload):
    response = client.post(
        "/api/adaptive-quiz",
        json={"topic": topic, **payload},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_varied_curated_topics_generate_topic_relevant_multiple_choice_questions(monkeypatch):
    adaptive_quiz.configure_knowledge_sources(None, None)
    examples = [
        ("Machine Learning", "ml."),
        ("Natural Language Processing", "nlp."),
        ("Cybersecurity", "cyber."),
        ("Cloud Computing", "cloud."),
    ]
    results = {
        topic: _create_quiz(topic)
        for topic, _ in examples
    }

    for topic, prefix in examples:
        result = results[topic]
        assert result["question_count"] == 8
        assert result["source"] == "LearnMap curated concept catalog"
        assert all(question["options"] and len(question["options"]) == 4 for question in result["questions"])
        assert all(question["concept_id"].startswith(prefix) or (
            prefix == "ml." and question["concept_id"].startswith("python.")
        ) for question in result["questions"])
        assert all("correct_answer" not in question for question in result["questions"])

    concept_sets = {
        topic: {question["concept"] for question in result["questions"]}
        for topic, result in results.items()
    }
    assert len({tuple(sorted(concepts)) for concepts in concept_sets.values()}) == len(examples)


def test_backend_grades_answers_and_reports_concept_level_performance(monkeypatch):
    adaptive_quiz.configure_knowledge_sources(None, None)
    quiz_result = _create_quiz("Cybersecurity")
    session = adaptive_quiz._active_quizzes[quiz_result["quiz_id"]]
    target_id = session.questions[0].id
    answers = [
        {
            "question_id": question.id,
            "selected_option_id": (
                next(option["id"] for option in question.options if option["id"] != question.correct_option_id)
                if question.id == target_id
                else question.correct_option_id
            ),
        }
        for question in session.questions
    ]

    response = client.post(
        "/api/adaptive-quiz/submit",
        json={"quiz_id": quiz_result["quiz_id"], "answers": answers},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["total_count"] == len(quiz_result["questions"])
    assert result["score_percent"] < 100
    assert any(score["score_percent"] < 100 for score in result["concept_scores"])
    assert any(not answer["is_correct"] and answer["explanation"] for answer in result["answers"])


def test_followup_quiz_prioritizes_weak_concepts_and_keeps_revision_items(monkeypatch):
    adaptive_quiz.configure_knowledge_sources(None, None)
    first = _create_quiz("Natural Language Processing")
    weak_id = "nlp.transformers"
    followup = _create_quiz(
        "Natural Language Processing",
        mastery=[
            {"concept_id": weak_id, "correct": 1, "total": 5},
            {"concept_id": "nlp.embeddings", "correct": 4, "total": 5},
        ],
        seen_question_ids=[question["id"] for question in first["questions"]],
    )
    concept_ids = [question["concept_id"] for question in followup["questions"]]
    weak_count = concept_ids.count(weak_id)

    assert weak_count > len(concept_ids) / 2
    assert any(concept_id != weak_id for concept_id in concept_ids)


def test_unlisted_topic_uses_retrieved_source_and_source_grounded_questions(monkeypatch):
    class RetrievedTopicProvider:
        def discover(self, topic, goal=None):
            assert topic == "Photosynthesis"
            sentences = [
                "Photosynthesis converts light energy into chemical energy in plants.",
                "Chlorophyll absorbs sunlight inside the chloroplast structures.",
                "Plants use carbon dioxide and water during this biological process.",
                "Glucose stores energy that supports plant growth and cellular activity.",
                "Oxygen is released into the surrounding environment as a product.",
                "Light intensity can affect the overall rate of photosynthesis.",
            ]
            return [
                DiscoveredConcept(
                    page_id=100,
                    title="Photosynthesis",
                    description=" ".join(sentences),
                    source_url="https://en.wikipedia.org/wiki/Photosynthesis",
                )
            ]

    adaptive_quiz.configure_knowledge_sources(RetrievedTopicProvider(), None)
    result = _create_quiz("Photosynthesis")
    assert result["source"] == "Wikipedia"
    assert result["questions"]
    assert result["sources"] == [{
        "name": "Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Photosynthesis",
    }]
    assert any("source term" in question["prompt"] for question in result["questions"])


def test_insufficient_source_and_invalid_submissions_return_clear_errors(monkeypatch):
    class EmptyProvider:
        def discover(self, topic, goal=None):
            return []

    adaptive_quiz.configure_knowledge_sources(EmptyProvider(), None)
    unsupported = client.post("/api/adaptive-quiz", json={"topic": "unfamiliar unsupported subject"})
    assert unsupported.status_code == 422
    assert "not enough reliable" in unsupported.json()["detail"].casefold()

    adaptive_quiz.configure_knowledge_sources(None, None)
    quiz_result = _create_quiz("Machine Learning")
    incomplete = client.post(
        "/api/adaptive-quiz/submit",
        json={
            "quiz_id": quiz_result["quiz_id"],
            "answers": [{"question_id": "unknown", "selected_option_id": "unknown"}] * 4,
        },
    )
    assert incomplete.status_code == 422
    invalid_topic = client.post("/api/adaptive-quiz", json={"topic": "  "})
    assert invalid_topic.status_code == 422
