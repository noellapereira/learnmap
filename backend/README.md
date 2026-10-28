# LearnMap learning-path discovery

The learning-path API retains its curated concept catalog and can optionally
retrieve concepts, courses, course materials, and labeled prerequisite edges
from a local MOOCCubeX release. Topics without a local match can then use
OpenAlex's public scholarly-topic index, with a direct Wikipedia article
summary as a final fallback. Neither dataset files nor network access are
required for the existing curated paths.

The curated catalog includes Python programming, machine learning, data
structures and algorithms, database management systems, web development,
natural language processing, cybersecurity, and cloud computing. Curated
concepts are divided into beginner, intermediate, and advanced stages and use
explicit, cycle-checked prerequisite edges. Beginner requests receive the
complete staged roadmap; intermediate and advanced requests target their
corresponding stages while retaining graph prerequisites.

## Discovery behavior

1. Normalize input, common abbreviations, and punctuation; try exact curated
   topic aliases, then conservative character n-gram TF-IDF matching.
2. Return an existing curated path when the local match is confident. Its
   prerequisite edges are the explicit relationships in `knowledge_base.py`.
   The frontend groups modules by their catalog difficulty while retaining
   topological order within each stage.
3. If no curated topic matches, search the configured MOOCCubeX release. The
   loader reads `entities/course.json`, `entities/concept.json`,
   `relations/concept-course.txt`, optional `relations/concept-video.txt` and
   `relations/video_id-ccid.txt`, and any available
   `prerequisites/{cs,math,psy}.json` files. It accepts JSON arrays, single
   JSON records, or JSON Lines. Character n-gram TF-IDF and cosine similarity
   rank concepts and course material; resource records are attached as
   references and do not determine the learning sequence.
4. Only prerequisite pairs with `ground_truth=1` become graph edges.
   `text_predict` and `graph_predict` scores are not treated as verified
relationships. The graph is topologically ordered, with relevance used
only to break ties between eligible concepts; cycles invalidate the dataset
path rather than being silently reordered.
5. If enough labeled prerequisite records are present, the backend trains and
   evaluates a TF-IDF + logistic-regression edge classifier using a stratified
   holdout split. Its metrics are logged and surfaced by `/api/health`. Its
   predictions do not create path prerequisites; paths use only ground-truth
   edges. No model is trained when no real dataset is configured or labels are
   insufficient.
6. If MOOCCubeX cannot provide a sufficiently relevant path, the configured
   OpenAlex provider searches scholarly works and retrieves their
   `primary_topic` classification, field labels, and work titles. TF-IDF and
   cosine similarity rank the actual OpenAlex research-area labels against the
   topic and goal. The returned modules link to OpenAlex topic records. They
   are a research-topic map, not an educator-authored syllabus; OpenAlex
   records do not supply verified prerequisites here, so no prerequisite edges
   are asserted.
7. If OpenAlex is unavailable or has no sufficiently relevant topic labels,
   the Wikipedia fallback requests the exact article summary for the submitted
   topic. This fallback may yield one module; it does not invent related
   concepts or prerequisites. Wikipedia summaries are attributed with their
   article link and CC BY-SA 4.0 link.
8. When local and external retrieval do not provide reliable matching
   information, the API reports the limitation instead of generating an
   unrelated path.
9. For curated paths, search each selected concept against the configured
   Wikipedia and OpenAlex providers for source-backed article or scholarly-work
   links. Resource lookup runs separately from graph construction and ordering;
   failed lookups add a limitation but do not discard or reorder the roadmap.
   In local-only mode, paths remain available without external resource links.

The dataset is not bundled or downloaded automatically. MOOCCubeX is large,
its concept/prerequisite coverage is limited to released fields, and this
project has not been tested against a full downloaded release. The MOOCCubeX
repository is published under GPL-3.0; check the terms accompanying the exact
dataset release before using or redistributing its data. See the
[MOOCCubeX repository](https://github.com/THU-KEG/MOOCCubeX) for dataset files
and attribution details.

Requests to the Wikipedia provider contain the submitted topic and optional
goal. Do not send sensitive information in either field.

## Configuration

Copy `.env.example` to `.env` as a reference. Uvicorn does not automatically
load that file; set the variables in the terminal that starts the backend.

- `LEARNMAP_KNOWLEDGE_PROVIDER=openalex` enables OpenAlex with Wikipedia
  summary fallback (default).
- `LEARNMAP_KNOWLEDGE_PROVIDER=wikipedia` uses only the direct Wikipedia
  summary fallback.
- `LEARNMAP_KNOWLEDGE_PROVIDER=local` disables network discovery.
- `LEARNMAP_KNOWLEDGE_TIMEOUT_SECONDS=3` bounds each provider request.
- `LEARNMAP_WIKIPEDIA_SUMMARY_API_URL=https://en.wikipedia.org/api/rest_v1/page/summary`
  selects the article-summary endpoint base.
- `LEARNMAP_OPENALEX_API_URL=https://api.openalex.org/works` selects the
  OpenAlex works API.
- `LEARNMAP_OPENALEX_EMAIL` optionally adds the operator's contact address to
  the OpenAlex User-Agent for API etiquette. Do not put a private learner email
  here.
- `MOOCCUBEX_DATA_DIR` is optional and points to the extracted MOOCCubeX
  dataset root. It must contain `entities/` and may contain `relations/` and
  `prerequisites/`; for example:
  `C:\datasets\MOOCCubeX`.
- `/api/health` reports whether the dataset files are available and whether
  enough labeled prerequisite data was available to train/evaluate the optional
  model.
- Wikipedia-derived descriptions are attributed with a source link and are
  available under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).
- `VITE_API_BASE_URL=http://127.0.0.1:8000` configures the frontend API base.
- `VITE_API_TIMEOUT_MS=15000` configures the frontend request timeout.

## Windows Command Prompt

From the project root, create and install the backend environment:

```cmd
"C:\Users\Bernadette\AppData\Local\Programs\Python\Python312\python.exe" -m venv backend\.venv
backend\.venv\Scripts\activate.bat
python -m pip install -r backend\requirements.txt
set LEARNMAP_KNOWLEDGE_PROVIDER=openalex
set LEARNMAP_KNOWLEDGE_TIMEOUT_SECONDS=3
set "MOOCCUBEX_DATA_DIR=C:\datasets\MOOCCubeX"
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Leave `MOOCCUBEX_DATA_DIR` unset (or empty) to run without the dataset.

In a second terminal, start the existing Vite frontend:

```cmd
cd /d C:\Users\Bernadette\nlp
set VITE_API_BASE_URL=http://127.0.0.1:8000
set VITE_API_TIMEOUT_MS=15000
npm run dev
```

Run backend tests from the project root:

```cmd
cd /d C:\Users\Bernadette\nlp\backend
python -m pytest -q
```

## Smart Notes

The Smart Notes API is implemented separately from the learning-path API:

- `POST /api/smart-notes` accepts JSON such as `{"text":"study material..."}`.
- `POST /api/smart-notes/upload` accepts a multipart PDF file.
- PDF files are limited to 10 MB and 200 pages; extracted text is limited to
  150,000 characters. Scanned/image-only PDFs require OCR, which is not included.

The feature uses whitespace cleanup, rule-based sentence splitting, TF-IDF
sentence/keyphrase ranking, extractive summarization, and definition-like
phrase detection. It does not use a trained summarization, NER, or question
generation model. Definitions are source context with a label indicating
whether definition-like wording appeared in the source; practice questions are
template-generated prompts about source sentences. Generated organized notes
are editable in the UI.

No model download is needed. The PDF endpoint uses `pypdf`, and FastAPI file
uploads use `python-multipart`; both are listed in `requirements.txt`.

## Adaptive Quiz Generator

The independent quiz API offers `POST /api/adaptive-quiz` to generate
multiple-choice questions from the curated topic concept catalog, configured
MOOCCubeX records, or retrieved Wikipedia text, and
`POST /api/adaptive-quiz/submit` to grade a quiz session on the backend. The UI
reports scores by concept and sends accumulated results on the next generation
request; concepts below 70% receive most available follow-up questions while
other concepts remain in the revision mix.

Curated questions are assembled from concept descriptions and learning
objectives. For unfamiliar subjects, TF-IDF keyphrase extraction creates
source-text cloze questions only when enough usable facts and distractors are
available. Topic matching uses TF-IDF cosine similarity. This baseline does
not use a trained question-generation, classification, or semantic-answer
model; submitted multiple-choice answers are checked against the stored
correct option. External question-source links are shown when available, and
the feature returns an informative error rather than inventing a quiz when
reliable source material is insufficient. Quiz sessions are held in memory
for one hour and are lost when the backend restarts.
