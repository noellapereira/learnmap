from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from knowledge_base import CONCEPTS, TOPICS
import main as learning_path_main
from mooccubex import MoocCubeXCatalog
from knowledge_provider import DiscoveredConcept, KnowledgeProviderError, WikipediaKnowledgeProvider
from openalex_provider import OpenAlexKnowledgeProvider

app = learning_path_main.app
_topological_order = learning_path_main._topological_order

client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_network_openalex_for_tests(monkeypatch):
    monkeypatch.setattr(learning_path_main, "openalex_provider", None)
    monkeypatch.setattr(learning_path_main, "knowledge_provider", None)


def post_path(payload: dict) -> dict:
    response = client.post("/api/learning-path", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_python_path_has_python_concepts_and_ordered_prerequisites():
    result = post_path({"topic": "I want to learn Python programming from scratch"})
    assert result["matched_topic"] == "Python programming"
    assert result["learning_modules"][0]["id"] == "python.foundations"
    assert any("Python" in module["title"] for module in result["learning_modules"])


def test_machine_learning_path_is_distinct_and_relevant():
    result = post_path({"topic": "machine learning", "level": "intermediate"})
    ids = {module["id"] for module in result["learning_modules"]}
    assert result["matched_topic"] == "Machine learning"
    assert "ml.statistics" in ids
    assert "ml.supervised" in ids
    assert not any(module["id"].startswith("db.") for module in result["learning_modules"])


def test_database_management_system_path():
    result = post_path({"topic": "database management systems"})
    ids = [module["id"] for module in result["learning_modules"]]
    assert result["matched_topic"] == "Database management systems"
    assert ids.index("db.relational") < ids.index("db.sql") < ids.index("db.joins")


def test_data_structures_and_web_development_are_supported():
    dsa = post_path({"topic": "data structures and algorithms"})
    web = post_path({"topic": "web development"})
    assert dsa["matched_topic"] == "Data structures and algorithms"
    assert any(module["id"].startswith("dsa.") for module in dsa["learning_modules"])
    assert web["matched_topic"] == "Web development"
    assert any(module["id"].startswith("web.") for module in web["learning_modules"])


def test_beginner_gets_full_staged_roadmap_and_advanced_keeps_prerequisites():
    beginner = post_path({"topic": "Python", "level": "beginner"})
    advanced = post_path({"topic": "Python", "level": "advanced"})
    assert {module["difficulty"] for module in beginner["learning_modules"]} == {
        "beginner", "intermediate", "advanced",
    }
    positions = {module["title"]: index for index, module in enumerate(advanced["learning_modules"])}
    for module in advanced["learning_modules"]:
        for prerequisite in module["prerequisites"]:
            assert prerequisite not in positions or positions[prerequisite] < positions[module["title"]]
    assert advanced["learning_modules"][-1]["difficulty"] == "advanced"


def test_goal_narrows_sequence_to_matching_concepts_and_prerequisites():
    result = post_path({"topic": "machine learning", "goal": "evaluate model performance"})
    ids = [module["id"] for module in result["learning_modules"]]
    assert "ml.evaluation" in ids
    assert ids.index("ml.supervised") < ids.index("ml.evaluation")


def test_known_concepts_are_excluded_but_reported_as_prerequisites():
    result = post_path({
        "topic": "Python programming",
        "known_concepts": ["Python fundamentals"],
    })
    ids = [module["id"] for module in result["learning_modules"]]
    assert "python.foundations" not in ids
    assert "python.control-flow" in ids
    assert "Python fundamentals" in result["prerequisite_concepts"]


def test_blank_and_whitespace_topics_are_rejected():
    for topic in ("", "   \t"):
        response = client.post("/api/learning-path", json={"topic": topic})
        assert response.status_code == 422


def test_unsupported_topic_returns_refinement_guidance(monkeypatch):
    class EmptyProvider:
        def discover(self, topic: str, goal: str | None = None) -> list[DiscoveredConcept]:
            return []

    monkeypatch.setattr(learning_path_main, "knowledge_provider", EmptyProvider())
    response = client.post("/api/learning-path", json={"topic": "underwater basket weaving"})
    assert response.status_code == 422
    assert "refine the topic" in response.json()["detail"]


def test_previously_unsupported_topic_uses_discovered_source_pages(monkeypatch):
    class QuantumProvider:
        def discover(self, topic: str, goal: str | None = None) -> list[DiscoveredConcept]:
            assert "quantum mechanics" in topic.casefold()
            pages = [
                DiscoveredConcept(1, "Quantum mechanics", "Quantum mechanics describes physical phenomena at atomic scales. " * 8, "https://en.wikipedia.org/wiki/Quantum_mechanics"),
                DiscoveredConcept(2, "Wave function", "A wave function represents the quantum state and predicts measurement probabilities. " * 8, "https://en.wikipedia.org/wiki/Wave_function"),
                DiscoveredConcept(3, "Quantum measurement", "Quantum measurement describes the interaction of a quantum system and an observation. " * 8, "https://en.wikipedia.org/wiki/Measurement_in_quantum_mechanics"),
                DiscoveredConcept(4, "Quantum entanglement", "Quantum entanglement is a physical phenomenon involving correlations between systems. " * 8, "https://en.wikipedia.org/wiki/Quantum_entanglement"),
            ]
            return [replace(page, topic_search_rank=index) for index, page in enumerate(pages, start=1)]

    monkeypatch.setattr(learning_path_main, "knowledge_provider", QuantumProvider())
    result = post_path({"topic": "Quantum mechanics", "level": "beginner"})
    assert result["matched_topic"] == "Quantum mechanics"
    assert result["learning_modules"]
    assert all(module["source_url"].startswith("https://en.wikipedia.org/") for module in result["learning_modules"])
    assert all(not module["prerequisites"] for module in result["learning_modules"])
    assert any("no prerequisites" in limitation.lower() for limitation in result["limitations"])
    assert result["total_concept_count"] == len(result["learning_modules"])


def test_external_topic_is_adapted_to_level_goal_and_known_concepts(monkeypatch):
    pages = [
        DiscoveredConcept(11, "Artificial intelligence", "Artificial intelligence studies computational systems and intelligent agents. " * 8, "https://en.wikipedia.org/wiki/Artificial_intelligence"),
        DiscoveredConcept(12, "Machine learning", "Machine learning studies algorithms that learn patterns from data and experience. " * 8, "https://en.wikipedia.org/wiki/Machine_learning"),
        DiscoveredConcept(13, "Neural network", "Neural networks are computational models used to learn representations from data. " * 8, "https://en.wikipedia.org/wiki/Neural_network"),
        DiscoveredConcept(14, "Computer vision", "Computer vision studies methods for interpreting images and visual data. " * 8, "https://en.wikipedia.org/wiki/Computer_vision"),
        DiscoveredConcept(15, "Natural language processing", "Natural language processing studies computational methods for language. " * 8, "https://en.wikipedia.org/wiki/Natural_language_processing"),
        DiscoveredConcept(16, "Reinforcement learning", "Reinforcement learning studies agents learning through rewards and interaction. " * 8, "https://en.wikipedia.org/wiki/Reinforcement_learning"),
        DiscoveredConcept(17, "Expert system", "An expert system applies encoded knowledge and rules to a focused problem. " * 8, "https://en.wikipedia.org/wiki/Expert_system"),
    ]

    class AIProvider:
        def discover(self, topic: str, goal: str | None = None) -> list[DiscoveredConcept]:
            assert goal == "computer vision"
            return [
                replace(
                    page,
                    topic_search_rank=index if page.title != "Computer vision" else 0,
                    goal_search_rank=1 if page.title == "Computer vision" else None,
                )
                for index, page in enumerate(pages, start=1)
            ]

    monkeypatch.setattr(learning_path_main, "knowledge_provider", AIProvider())
    beginner = post_path({
        "topic": "artificial intelligence",
        "level": "beginner",
        "goal": "computer vision",
        "known_concepts": ["Artificial intelligence"],
    })
    advanced = post_path({"topic": "artificial intelligence", "level": "advanced", "goal": "computer vision"})
    assert len(beginner["learning_modules"]) <= 5
    assert len(advanced["learning_modules"]) > len(beginner["learning_modules"])
    assert all(module["difficulty"] == "beginner" for module in beginner["learning_modules"])
    assert all(module["difficulty"] == "advanced" for module in advanced["learning_modules"])
    assert "Artificial intelligence" not in [module["title"] for module in beginner["learning_modules"]]
    assert any("known concepts" in limitation for limitation in beginner["limitations"])
    assert beginner["learning_modules"][0]["title"] == "Computer vision"


def test_provider_failure_returns_clear_recoverable_error_and_does_not_fabricate(monkeypatch):
    class FailedProvider:
        def discover(self, topic: str, goal: str | None = None) -> list[DiscoveredConcept]:
            raise KnowledgeProviderError("network timeout")

    monkeypatch.setattr(learning_path_main, "knowledge_provider", FailedProvider())
    response = client.post("/api/learning-path", json={"topic": "quantum mechanics"})
    assert response.status_code == 503
    assert "external knowledge retrieval failed" in response.json()["detail"]


def test_local_only_mode_keeps_curated_paths_and_explains_catalog_limit(monkeypatch):
    monkeypatch.setattr(learning_path_main, "knowledge_provider", None)
    curated = post_path({"topic": "python programming"})
    unknown = client.post("/api/learning-path", json={"topic": "quantum mechanics"})
    assert curated["matched_topic"] == "Python programming"
    assert unknown.status_code == 422
    assert "External topic discovery is disabled" in unknown.json()["detail"]


def test_abbreviations_and_variants_normalize_to_canonical_terms():
    assert learning_path_main.preprocess("I want NLP, please")[1] == [
        "natural", "language", "processing"
    ]
    assert learning_path_main._match_topic("learn DBMS")[0] == "database-management"


def test_common_outside_catalog_subjects_are_sent_to_discovery_before_rejection(monkeypatch):
    class RecordingProvider:
        topics = []

        def discover(self, topic: str, goal: str | None = None) -> list[DiscoveredConcept]:
            self.topics.append(topic)
            return []

    provider = RecordingProvider()
    monkeypatch.setattr(learning_path_main, "knowledge_provider", provider)
    subjects = [
        "artificial intelligence", "deep learning", "computer vision", "operating systems",
        "computer networks", "mathematics", "statistics", "physics", "chemistry",
        "biology", "economics", "history",
    ]
    for subject in subjects:
        response = client.post("/api/learning-path", json={"topic": subject})
        assert response.status_code == 422
    assert len(provider.topics) == len(subjects)
    assert provider.topics[0] == "artificial intelligence"
    assert provider.topics[1] == "deep learning"
    assert provider.topics[2] == "computer vision"


def test_wikipedia_provider_queries_search_and_extracts_articles(monkeypatch):
    def fake_get(url, *, params, **kwargs):
        assert params is None
        assert url.endswith("/quantum_mechanics")
        return httpx.Response(200, json={
            "pageid": 101,
            "title": "Quantum mechanics",
            "extract": "Quantum mechanics describes physical phenomena at atomic scales. " * 3,
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Quantum_mechanics"}},
        }, request=httpx.Request("GET", url))

    monkeypatch.setattr("knowledge_provider.httpx.get", fake_get)
    provider = WikipediaKnowledgeProvider(timeout_seconds=0.25)
    concepts = provider.discover("quantum mechanics", "wave function")
    assert len(concepts) == 1
    assert concepts[0].topic_search_rank == 1
    assert concepts[0].goal_search_rank is None
    assert concepts[0].source_name == "Wikipedia"
    assert concepts[0].source_url == "https://en.wikipedia.org/wiki/Quantum_mechanics"


def test_wikipedia_provider_applies_timeout_and_surfaces_transport_errors(monkeypatch):
    assert WikipediaKnowledgeProvider(timeout_seconds=0.5).timeout_seconds == 0.5

    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr("knowledge_provider.httpx.get", timeout)
    provider = WikipediaKnowledgeProvider(timeout_seconds=0.01)
    try:
        provider.discover("unknown topic")
    except KnowledgeProviderError:
        pass
    else:
        raise AssertionError("Provider timeout should be reported.")


def test_openalex_provider_extracts_ranked_primary_topic_records(monkeypatch):
    works = [
        {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.0000/ai",
            "display_name": "Artificial intelligence methods",
            "relevance_score": 0.9,
            "primary_topic": {
                "id": "https://openalex.org/T100",
                "display_name": "Artificial Intelligence",
                "score": 0.98,
                "subfield": {"display_name": "Artificial Intelligence"},
                "field": {"display_name": "Computer Science"},
            },
        },
        {
            "id": "https://openalex.org/W2",
            "doi": "https://doi.org/10.0000/ml",
            "display_name": "Machine learning for intelligent agents",
            "relevance_score": 0.8,
            "primary_topic": {
                "id": "https://openalex.org/T101",
                "display_name": "Machine Learning",
                "score": 0.95,
                "subfield": {"display_name": "Artificial Intelligence"},
                "field": {"display_name": "Computer Science"},
            },
        },
    ]

    def fake_get(url, *, params, **kwargs):
        assert url == "https://api.openalex.org/works"
        assert params["search"] == "artificial intelligence computer vision"
        assert params["per-page"] == 40
        assert "primary_topic" in params["select"]
        return httpx.Response(200, json={"results": works}, request=httpx.Request("GET", url))

    monkeypatch.setattr("openalex_provider.httpx.get", fake_get)
    result = OpenAlexKnowledgeProvider().discover("artificial intelligence", "computer vision")
    assert result is not None
    assert [concept.title for concept in result.concepts] == [
        "Artificial Intelligence",
        "Machine Learning",
    ]
    assert result.concepts[0].field == "Computer Science"


def test_openalex_api_returns_relevant_source_linked_modules(monkeypatch):
    from openalex_provider import OpenAlexConcept, OpenAlexSearchResult

    class OpenAlexStub:
        def discover(self, topic, goal=None):
            assert topic == "artificial intelligence"
            return OpenAlexSearchResult(concepts=[
                OpenAlexConcept(
                    id="https://openalex.org/T100",
                    title="Artificial Intelligence",
                    subfield="Artificial Intelligence",
                    field="Computer Science",
                    score=0.9,
                    supporting_work="Artificial intelligence methods",
                    supporting_doi=None,
                ),
                OpenAlexConcept(
                    id="https://openalex.org/T101",
                    title="Machine Learning",
                    subfield="Artificial Intelligence",
                    field="Computer Science",
                    score=0.7,
                    supporting_work="Machine learning for intelligent agents",
                    supporting_doi=None,
                ),
            ], score=0.82)

    monkeypatch.setattr(learning_path_main, "openalex_provider", OpenAlexStub())
    result = post_path({"topic": "Artificial Intelligence"})
    assert result["matching_method"].startswith("OpenAlex scholarly-work search")
    assert [module["title"] for module in result["learning_modules"]] == [
        "Artificial Intelligence",
        "Machine Learning",
    ]
    assert all(module["source_name"] == "OpenAlex" for module in result["learning_modules"])
    assert all(module["source_url"].startswith("https://openalex.org/T") for module in result["learning_modules"])
    assert all(not module["prerequisites"] for module in result["learning_modules"])
    assert any("no prerequisite edges" in value for value in result["limitations"])
    assert result["estimated_total_minutes"] == sum(
        module["estimated_minutes"] for module in result["learning_modules"]
    )


def test_invalid_bodies_and_levels_are_rejected():
    for payload in ({}, {"topic": 42}, {"topic": "python", "level": "expert"}, {"topic": "python", "unexpected": True}):
        assert client.post("/api/learning-path", json=payload).status_code == 422


def test_totals_match_returned_modules():
    result = post_path({"topic": "database management systems"})
    modules = result["learning_modules"]
    assert result["total_concept_count"] == len(modules)
    assert result["estimated_total_minutes"] == sum(module["estimated_minutes"] for module in modules)


def test_all_supported_topic_graphs_are_acyclic_and_prerequisites_precede_dependents():
    full_order = _topological_order(set(CONCEPTS))
    full_positions = {concept_id: index for index, concept_id in enumerate(full_order)}
    for concept_id, concept in CONCEPTS.items():
        for prerequisite in concept["prerequisites"]:
            assert full_positions[prerequisite] < full_positions[concept_id]

    for topic in TOPICS.values():
        selected = set(topic["concept_ids"])
        ordered = _topological_order(selected)
        positions = {concept_id: index for index, concept_id in enumerate(ordered)}
        for concept_id in selected:
            for prerequisite in CONCEPTS[concept_id]["prerequisites"]:
                if prerequisite in selected:
                    assert positions[prerequisite] < positions[concept_id]


@pytest.mark.parametrize(
    ("topic", "prefix"),
    [
        ("Machine Learning", "ml."),
        ("NLP", "nlp."),
        ("Cybersecurity", "cyber."),
        ("Cloud Computing", "cloud."),
    ],
)
def test_requested_subjects_return_relevant_ordered_three_stage_paths(topic, prefix):
    result = post_path({"topic": topic})
    modules = result["learning_modules"]
    ids = [module["id"] for module in modules]
    positions = {module["title"]: index for index, module in enumerate(modules)}

    assert modules
    assert all(
        module_id.startswith(prefix)
        or (prefix == "ml." and module_id.startswith("python."))
        for module_id in ids
    )
    assert len(ids) == len(set(ids))
    assert {module["difficulty"] for module in modules} == {
        "beginner", "intermediate", "advanced",
    }
    for module in modules:
        for prerequisite in module["prerequisites"]:
            assert prerequisite in positions
            assert positions[prerequisite] < positions[module["title"]]
    assert result["total_concept_count"] == len(modules)
    assert result["estimated_total_minutes"] == sum(module["estimated_minutes"] for module in modules)


def test_curated_roadmap_attaches_real_source_links_when_resource_search_succeeds(monkeypatch):
    class WikipediaStub:
        def discover(self, topic, goal=None):
            return [
                DiscoveredConcept(
                    page_id=topic.casefold().replace(" ", "-"),
                    title=topic,
                    description=f"Reference material about {topic}. " * 8,
                    source_url=f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                )
            ]

    monkeypatch.setattr(learning_path_main, "knowledge_provider", WikipediaStub())
    result = post_path({"topic": "Cloud Computing"})
    assert all(module["resources"] for module in result["learning_modules"])
    assert all(
        resource["url"].startswith("https://en.wikipedia.org/wiki/")
        for module in result["learning_modules"]
        for resource in module["resources"]
    )


def test_resource_search_failure_does_not_discard_curated_roadmap(monkeypatch):
    class FailedProvider:
        def discover(self, topic, goal=None):
            raise KnowledgeProviderError("resource source unavailable")

    monkeypatch.setattr(learning_path_main, "knowledge_provider", FailedProvider())
    result = post_path({"topic": "Natural language processing"})
    assert result["learning_modules"]
    assert any("resource searches failed" in item.lower() for item in result["limitations"])
    assert any("missing links were not fabricated" in item.lower() for item in result["limitations"])


def _write_mooccubex_fixture(root: Path) -> None:
    entities = root / "entities"
    relations = root / "relations"
    prerequisites = root / "prerequisites"
    entities.mkdir(parents=True)
    relations.mkdir()
    prerequisites.mkdir()

    concepts = [
        {"id": "K_AI_cs", "name": "Artificial intelligence", "context": "Artificial intelligence studies intelligent agents, reasoning, and problem solving."},
        {"id": "K_ML_cs", "name": "Machine learning", "context": "Machine learning studies algorithms that learn patterns from data."},
        {"id": "K_Cyber_cs", "name": "Cybersecurity", "context": "Cybersecurity protects computer systems and information from digital attacks."},
        {"id": "K_Network_cs", "name": "Network security", "context": "Network security protects connected computer networks and their communications."},
        {"id": "K_Photosynthesis_bio", "name": "Photosynthesis", "context": "Photosynthesis converts light energy, water, and carbon dioxide into chemical energy."},
        {"id": "K_Chlorophyll_bio", "name": "Chlorophyll", "context": "Chlorophyll is a pigment that absorbs light energy during photosynthesis."},
        {"id": "K_Quantum_computing", "name": "Quantum computing", "context": "Quantum computing uses quantum states to process and represent information."},
        {"id": "K_Quantum_mechanics", "name": "Quantum mechanics", "context": "Quantum mechanics describes physical systems and states at small scales."},
    ]
    (entities / "concept.json").write_text(json.dumps(concepts), encoding="utf-8")

    courses = [
        {
            "id": "C_AI", "name": "Introduction to Artificial Intelligence",
            "about": "Foundations of intelligent systems and machine learning.",
            "field": "Computer science",
            "prerequisites": [],
            "resource": [
                {"resource_type": "video", "resource_id": "ccid-ml", "chapter": "Learning from data", "titles": ["Machine learning"]},
                {"resource_type": "video", "resource_id": "ccid-ai", "chapter": "AI foundations", "titles": ["Introduction to artificial intelligence"]},
            ],
        },
        {"id": "C_Cyber", "name": "Cybersecurity fundamentals", "about": "Security principles for systems and networks.", "field": "Computer science", "prerequisites": [], "resource": []},
        {"id": "C_Bio", "name": "Plant biology", "about": "Plant cells, pigments, and photosynthesis.", "field": "Biology", "prerequisites": [], "resource": []},
        {"id": "C_Quantum", "name": "Quantum computing", "about": "Quantum information and computation.", "field": "Physics", "prerequisites": [], "resource": []},
    ]
    (entities / "course.json").write_text(json.dumps(courses), encoding="utf-8")
    (relations / "concept-course.txt").write_text(
        "K_AI_cs\tC_AI\nK_ML_cs\tC_AI\nK_Cyber_cs\tC_Cyber\nK_Network_cs\tC_Cyber\n"
        "K_Photosynthesis_bio\tC_Bio\nK_Chlorophyll_bio\tC_Bio\n"
        "K_Quantum_computing\tC_Quantum\nK_Quantum_mechanics\tC_Quantum\n",
        encoding="utf-8",
    )
    (relations / "concept-video.txt").write_text(
        "K_AI_cs\tccid-ai\nK_ML_cs\tccid-ml\n",
        encoding="utf-8",
    )
    (prerequisites / "cs.json").write_text(
        json.dumps([
            {"c1": "K_AI_cs", "c2": "K_ML_cs", "ground_truth": 1, "text_predict": 0.9, "graph_predict": 0.9},
            {"c1": "K_Cyber_cs", "c2": "K_Network_cs", "ground_truth": 0, "text_predict": 0.99, "graph_predict": 0.99},
        ]),
        encoding="utf-8",
    )


def test_mooccubex_fixture_retrieves_requested_unfamiliar_topics_and_orders_evidence(tmp_path, monkeypatch):
    _write_mooccubex_fixture(tmp_path)
    monkeypatch.setattr(learning_path_main, "mooccubex_catalog", MoocCubeXCatalog(tmp_path))
    monkeypatch.setattr(learning_path_main, "knowledge_provider", None)

    results = {
        topic: post_path({"topic": topic})
        for topic in (
            "Artificial Intelligence",
            "Photosynthesis",
            "Quantum Computing",
        )
    }
    all_titles = {
        topic: [module["title"] for module in result["learning_modules"]]
        for topic, result in results.items()
    }
    assert "Artificial intelligence" in all_titles["Artificial Intelligence"]
    ai_path = results["Artificial Intelligence"]["learning_modules"]
    assert [module["title"] for module in ai_path].index("Artificial intelligence") < [
        module["title"] for module in ai_path
    ].index("Machine learning")
    assert ai_path[0]["resources"][0]["source"] == "MOOCCubeX"
    assert any("Photosynthesis" in title or "Chlorophyll" in title for title in all_titles["Photosynthesis"])
    assert any("Quantum" in title for title in all_titles["Quantum Computing"])
    assert len({tuple(titles) for titles in all_titles.values()}) == 3
    for result in results.values():
        modules = result["learning_modules"]
        assert result["total_concept_count"] == len(modules)
        assert result["estimated_total_minutes"] == sum(module["estimated_minutes"] for module in modules)


def test_mooccubex_personalization_and_ground_truth_prerequisites(tmp_path, monkeypatch):
    _write_mooccubex_fixture(tmp_path)
    monkeypatch.setattr(learning_path_main, "mooccubex_catalog", MoocCubeXCatalog(tmp_path))
    monkeypatch.setattr(learning_path_main, "knowledge_provider", None)
    result = post_path({
        "topic": "AI",
        "level": "advanced",
        "goal": "machine learning",
        "known_concepts": ["Artificial intelligence"],
    })
    titles = [module["title"] for module in result["learning_modules"]]
    assert "Artificial intelligence" not in titles
    assert "Machine learning" in titles
    assert result["learning_modules"][0]["difficulty"] == "advanced"
    assert result["learning_modules"][0]["prerequisites"] == ["Artificial intelligence"]
    assert result["prerequisite_concepts"] == ["Artificial intelligence"]
    assert any("ground_truth=1" in value for value in result["limitations"])
    assert "Network security" not in titles
    for level in ("beginner", "intermediate", "advanced"):
        level_result = post_path({"topic": "Artificial Intelligence", "level": level})
        assert all(module["difficulty"] == level for module in level_result["learning_modules"])


def test_mooccubex_cycle_does_not_produce_an_ordered_path(tmp_path, monkeypatch):
    _write_mooccubex_fixture(tmp_path)
    prerequisite_file = tmp_path / "prerequisites" / "cs.json"
    prerequisite_file.write_text(
        json.dumps([
            {"c1": "K_AI_cs", "c2": "K_ML_cs", "ground_truth": 1},
            {"c1": "K_ML_cs", "c2": "K_AI_cs", "ground_truth": 1},
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(learning_path_main, "mooccubex_catalog", MoocCubeXCatalog(tmp_path))
    monkeypatch.setattr(learning_path_main, "knowledge_provider", None)
    response = client.post("/api/learning-path", json={"topic": "Artificial Intelligence"})
    assert response.status_code == 422
    assert "prerequisite graph contains a cycle" in response.json()["detail"].lower()


def test_mooccubex_model_training_is_explicitly_evaluated_when_labeled_data_is_sufficient():
    catalog = MoocCubeXCatalog(None)
    catalog._concepts = {
        f"concept-{index}": {"name": f"concept {index}", "context": ""}
        for index in range(40)
    }
    edges = [
        (f"concept-{index}", f"concept-{index + 1}", int(index % 2 == 0))
        for index in range(30)
    ]
    status = catalog._train_and_evaluate(edges)
    assert status.startswith("trained/evaluated on 30 labeled edges")
    assert "f1=" in status
