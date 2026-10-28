"""Source-grounded Smart Notes generation using extractive NLP baselines."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["smart-notes"])
MAX_TEXT_CHARACTERS = 150_000
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 200
STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "before",
    "being", "but", "can", "could", "did", "does", "during", "each", "for", "from",
    "have", "having", "into", "its", "more", "most", "not", "our", "over", "same",
    "some", "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "under", "very", "was", "were", "what", "when",
    "where", "which", "while", "with", "would", "your",
}


class SmartNotesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)

    @field_validator("text")
    @classmethod
    def text_must_contain_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Study material cannot be empty or whitespace.")
        return value


class KeywordDefinition(BaseModel):
    term: str
    definition: str
    basis: str


class Flashcard(BaseModel):
    front: str
    back: str


class SmartNotesResponse(BaseModel):
    source_name: str
    source_characters: int
    summary: str
    key_points: list[str]
    keywords: list[KeywordDefinition]
    flashcards: list[Flashcard]
    practice_questions: list[str]
    organized_notes: str
    method: str
    limitations: list[str]


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sentences(text: str) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+|(?<=:)\s+(?=[A-Z0-9])|\n+", text)
    return [
        sentence.strip(" \t-*#")
        for sentence in candidates
        if len(re.findall(r"\w+", sentence)) >= 3
    ]


def _tokenize(text: str) -> list[str]:
    return [
        word.casefold()
        for word in re.findall(r"[^\W_][\w'-]*", text, flags=re.UNICODE)
        if len(word) > 2 and word.casefold() not in STOP_WORDS
    ]


def _term_matches(term: str, sentence: str) -> bool:
    words = term.split()
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(word) for word in words) + r"(?!\w)"
    return re.search(pattern, sentence, flags=re.IGNORECASE) is not None


def _extract_keywords(sentences: Sequence[str]) -> list[tuple[str, float]]:
    if not sentences:
        return []
    selected: list[tuple[str, float]] = []
    for ngram_range, limit in (((2, 3), 4), ((1, 1), 6)):
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            token_pattern=r"(?u)\b[\w'-]{3,}\b",
            stop_words=sorted(STOP_WORDS | set(ENGLISH_STOP_WORDS)),
            sublinear_tf=True,
            max_features=4000,
        )
        try:
            matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            continue
        totals = matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        ranked = sorted(zip(terms, totals), key=lambda item: float(item[1]), reverse=True)
        added = 0
        for term, score in ranked:
            term = term.replace("'", "").strip()
            if not term or any(existing == term for existing, _ in selected):
                continue
            if not any(_term_matches(term, sentence) for sentence in sentences):
                continue
            words = term.split()
            if ngram_range[0] == 1 and any(
                len(existing.split()) > 1 and term in existing.split()
                for existing, _ in selected
            ):
                continue
            if not set(_tokenize(term)) >= set(words):
                continue
            selected.append((term, float(score)))
            added += 1
            if added == limit:
                break
    return selected[:10]


def _sentence_scores(sentences: Sequence[str]) -> list[float]:
    if not sentences:
        return []
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w'-]{3,}\b",
        stop_words=sorted(STOP_WORDS | set(ENGLISH_STOP_WORDS)),
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return [0.0] * len(sentences)
    weights = matrix.sum(axis=0).A1
    scores = matrix.dot(weights)
    return [float(score) / max(1, len(_tokenize(sentence))) for score, sentence in zip(scores, sentences)]


def _key_points(sentences: Sequence[str], scores: Sequence[float], limit: int = 7) -> list[str]:
    ranked = sorted(enumerate(sentences), key=lambda item: (scores[item[0]], -item[0]), reverse=True)
    selected: list[tuple[int, str]] = []
    selected_tokens: list[set[str]] = []
    for index, sentence in ranked:
        tokens = set(_tokenize(sentence))
        if not tokens:
            continue
        if any(len(tokens & prior) / max(1, len(tokens | prior)) > 0.72 for prior in selected_tokens):
            continue
        selected.append((index, sentence))
        selected_tokens.append(tokens)
        if len(selected) == limit:
            break
    selected.sort(key=lambda item: item[0])
    return [sentence for _, sentence in selected]


def _definition_evidence(term: str, sentences: Sequence[str]) -> tuple[str, str] | None:
    definition_markers = (
        r"\bis\b", r"\bare\b", r"\brefers to\b", r"\bmeans\b", r"\bdefined as\b",
        r"\bdescribes\b", r"\bconsists of\b", r"\bis known as\b",
    )
    matching = [sentence for sentence in sentences if _term_matches(term, sentence)]
    for sentence in matching:
        suffix = sentence[sentence.lower().find(term.lower()) + len(term):]
        if any(re.match(marker, suffix.strip(), flags=re.IGNORECASE) for marker in definition_markers):
            return sentence, "Definition-like wording found in supplied material."
    if matching:
        return matching[0], "Source context only; the material does not explicitly define this term."
    return None


def _questions(key_points: Sequence[str], keywords: Sequence[str], limit: int = 5) -> list[str]:
    questions: list[str] = []
    for point in key_points:
        matching_term = next((term for term in keywords if _term_matches(term, point)), None)
        if matching_term:
            question = f"What does the supplied material state about {matching_term}?"
        else:
            question = f"What is one main point from the supplied material, starting with: “{point[:90]}”?"
        if question not in questions:
            questions.append(question)
        if len(questions) == limit:
            break
    return questions


def generate_smart_notes(text: str, source_name: str = "Pasted text") -> SmartNotesResponse:
    cleaned = _clean_text(text)
    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="No readable study text was found. Paste text or use a text-based PDF.",
        )
    source_characters = len(cleaned)
    was_truncated = source_characters > MAX_TEXT_CHARACTERS
    if len(cleaned) > MAX_TEXT_CHARACTERS:
        cleaned = cleaned[:MAX_TEXT_CHARACTERS]
    sentences = _sentences(cleaned)
    if not sentences:
        raise HTTPException(
            status_code=422,
            detail="The supplied material did not contain enough readable sentences to create notes.",
        )
    scores = _sentence_scores(sentences)
    summary_limit = min(6, max(1, (len(sentences) + 2) // 3))
    ranked_indices = sorted(
        range(1, len(sentences)),
        key=lambda index: (scores[index], -index),
        reverse=True,
    )
    selected_indices = sorted({0, *ranked_indices[:max(0, summary_limit - 1)]})
    summary = " ".join(sentences[index] for index in selected_indices)
    key_points = _key_points(sentences, scores)
    keyword_scores = _extract_keywords(sentences)
    keyword_names = [term for term, _ in keyword_scores]
    keywords: list[KeywordDefinition] = []
    flashcards: list[Flashcard] = []
    for term in keyword_names:
        evidence = _definition_evidence(term, sentences)
        if not evidence:
            continue
        definition, basis = evidence
        keywords.append(KeywordDefinition(term=term, definition=definition, basis=basis))
        flashcards.append(Flashcard(front=f"What does “{term}” mean in this material?", back=definition))
        if len(keywords) == 8:
            break

    practice_questions = _questions(key_points, keyword_names)
    organized_lines = [
        "# Smart Notes",
        "",
        "## Summary",
        summary,
        "",
        "## Key Points",
        *[f"- {point}" for point in key_points],
        "",
        "## Keywords and Definitions",
    ]
    organized_lines.extend(
        f"- **{item.term}** — {item.definition} _({item.basis})_"
        for item in keywords
    )
    if not keywords:
        organized_lines.append("- No clear technical terms with source-supported context were extracted.")
    organized_lines.extend(["", "## Practice Questions", *[f"{i}. {question}" for i, question in enumerate(practice_questions, 1)]])
    limitations = [
        "Summary and key points are extractive: they reuse text from the supplied material and may need editing.",
        "No generative model or named-entity-recognition model is used; no model download is required.",
        "Keyword definitions and flashcards quote source context. Definition-like wording is flagged; otherwise context is explicitly not presented as a verified definition.",
        "Practice questions are template-based prompts grounded in extracted points, not model-generated questions.",
    ]
    if was_truncated:
        limitations.append(
            f"The source exceeded {MAX_TEXT_CHARACTERS:,} characters; notes were generated from the first {MAX_TEXT_CHARACTERS:,} characters."
        )
    return SmartNotesResponse(
        source_name=source_name,
        source_characters=source_characters,
        summary=summary,
        key_points=key_points,
        keywords=keywords,
        flashcards=flashcards,
        practice_questions=practice_questions,
        organized_notes="\n".join(organized_lines),
        method=(
            "Text normalization; rule-based sentence splitting; TF-IDF sentence salience and keyphrase extraction; "
            "extractive summary and source-context definition matching."
        ),
        limitations=limitations,
    )


def extract_pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                if reader.decrypt("") == 0:
                    raise HTTPException(status_code=422, detail="This PDF is password-protected and cannot be read.")
            except PdfReadError as error:
                raise HTTPException(status_code=422, detail="This PDF is password-protected and cannot be read.") from error
        if len(reader.pages) > MAX_PDF_PAGES:
            raise HTTPException(
                status_code=413,
                detail=f"PDFs must contain no more than {MAX_PDF_PAGES} pages.",
            )
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except HTTPException:
        raise
    except (PdfReadError, ValueError, OSError, KeyError) as error:
        logger.info("Smart Notes rejected an unreadable PDF: %s", error)
        raise HTTPException(
            status_code=422,
            detail="The PDF could not be read. Upload a valid, non-encrypted PDF with selectable text.",
        ) from error


@router.post("/smart-notes", response_model=SmartNotesResponse)
def create_smart_notes(request: SmartNotesRequest) -> SmartNotesResponse:
    return generate_smart_notes(request.text)


@router.post("/smart-notes/upload", response_model=SmartNotesResponse)
async def create_smart_notes_from_pdf(file: UploadFile = File(...)) -> SmartNotesResponse:
    filename = file.filename or ""
    if not filename.casefold().endswith(".pdf") or file.content_type not in {
        "application/pdf",
        "application/octet-stream",
    }:
        raise HTTPException(status_code=415, detail="Upload a PDF file to generate Smart Notes.")
    data = await file.read(MAX_PDF_BYTES + 1)
    await file.close()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF file size must not exceed {MAX_PDF_BYTES // (1024 * 1024)} MB.",
        )
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid PDF.")
    extracted = extract_pdf_text(data)
    result = generate_smart_notes(extracted, source_name=filename)
    return result
