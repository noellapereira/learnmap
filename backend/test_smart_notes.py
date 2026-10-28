from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _one_page_pdf(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, content in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(content)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


def test_pasted_text_returns_source_based_structured_notes():
    text = (
        "Photosynthesis is the process by which plants convert light energy into chemical energy. "
        "Chlorophyll absorbs light in the chloroplast. "
        "During photosynthesis, carbon dioxide and water are used to produce glucose and oxygen. "
        "Glucose stores energy that the plant can use for growth."
    )
    response = client.post("/api/smart-notes", json={"text": text})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["source_name"] == "Pasted text"
    assert "Photosynthesis is the process" in result["summary"]
    assert result["key_points"]
    assert any("chlorophyll" in item["term"].casefold() for item in result["keywords"])
    assert result["flashcards"]
    assert result["practice_questions"]
    assert "# Smart Notes" in result["organized_notes"]
    assert "extractive" in " ".join(result["limitations"]).casefold()


def test_different_input_changes_generated_summary_and_keywords():
    first = client.post("/api/smart-notes", json={
        "text": "Gravity is an attractive force between masses. Planets orbit stars because of gravity."
    }).json()
    second = client.post("/api/smart-notes", json={
        "text": "Mitosis is a type of cell division. It produces two genetically identical daughter cells."
    }).json()
    assert first["summary"] != second["summary"]
    first_terms = {item["term"] for item in first["keywords"]}
    second_terms = {item["term"] for item in second["keywords"]}
    assert "gravity" in first_terms
    assert "mitosis" in second_terms
    assert first_terms != second_terms


def test_multiword_keywords_are_prioritized_and_grounded_in_source():
    response = client.post("/api/smart-notes", json={
        "text": (
            "Plants use carbon dioxide and water during photosynthesis. "
            "Carbon dioxide enters the leaves through small openings. "
            "The process produces glucose and oxygen."
        )
    })
    assert response.status_code == 200, response.text
    terms = [item["term"].casefold() for item in response.json()["keywords"]]
    assert terms[0] == "carbon dioxide"
    assert "carbon" not in terms
    assert "dioxide" not in terms


def test_pdf_upload_extracts_text_and_generates_notes():
    pdf = _one_page_pdf(
        "Photosynthesis converts sunlight into stored chemical energy. Chlorophyll absorbs light. "
        "Plants use carbon dioxide and water to make glucose."
    )
    response = client.post(
        "/api/smart-notes/upload",
        files={"file": ("plant-biology.pdf", pdf, "application/pdf")},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["source_name"] == "plant-biology.pdf"
    assert "Photosynthesis" in result["summary"]
    assert any("chlorophyll" in point.casefold() for point in result["key_points"])


def test_empty_text_and_empty_text_pdf_return_helpful_errors():
    for text in ("", "   "):
        response = client.post("/api/smart-notes", json={"text": text})
        assert response.status_code == 422
    empty_pdf = _one_page_pdf("")
    response = client.post(
        "/api/smart-notes/upload",
        files={"file": ("empty.pdf", empty_pdf, "application/pdf")},
    )
    assert response.status_code == 422
    assert "readable" in response.json()["detail"].casefold()


def test_pdf_upload_rejects_bad_file_and_wrong_extension():
    bad_pdf = client.post(
        "/api/smart-notes/upload",
        files={"file": ("broken.pdf", b"not a pdf", "application/pdf")},
    )
    assert bad_pdf.status_code == 422
    wrong_extension = client.post(
        "/api/smart-notes/upload",
        files={"file": ("notes.txt", b"%PDF-1.4", "application/pdf")},
    )
    assert wrong_extension.status_code == 415


def test_text_input_validation_and_extra_fields():
    for payload in ({}, {"text": "   "}, {"text": 42}, {"text": "valid text input", "extra": True}):
        response = client.post("/api/smart-notes", json=payload)
        assert response.status_code == 422
