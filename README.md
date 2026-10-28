# LearnMap Website — Project Overview

## 1. Project purpose / problem statement

LearnMap is a personalized learning website designed to help learners organize study material and practice concepts. Instead of treating learning as one large topic, it provides tools to:

- Break a subject into a structured learning path.
- Turn supplied text or a PDF into revision notes.
- Test understanding with multiple-choice quizzes.
- Use quiz results to emphasize concepts the learner may need to review.

The main purpose is to make study activities more focused: **plan what to learn, review material, practice, and use performance to guide further practice**.

---

## 2. What is currently in the website

The website currently has three main features:

1. **Learning Path Generator**
2. **Smart Notes Generator**
3. **Adaptive Quiz Generator**

The separate **Resources** section has been removed. Source links may still appear inside individual learning-path or quiz results when the backend has them.

The interface uses a light dashboard design, with a left navigation area, cards and panels, purple accents, and responsive layouts.

---

## 3. Learning Path Generator

### What the learner can do

The learner can enter a topic and choose a current level:

- Beginner
- Intermediate
- Advanced

They can also provide an optional learning goal and list concepts they already know.

### How it works

1. The React frontend sends the submitted topic and preferences to FastAPI at `POST /api/learning-path`.
2. The backend normalizes the input and tries to match it to the local topic catalog. It supports aliases and uses character n-gram TF-IDF similarity when an exact alias is not found.
3. For a curated topic, the backend selects target concepts based on the learner’s level and goal.
4. It follows explicit prerequisite relationships to include necessary foundations.
5. The backend topologically sorts the selected concepts so prerequisites appear before concepts that depend on them.
6. The UI displays the returned modules in Beginner, Intermediate, and Advanced sections. A learner can mark modules complete, and the progress indicator updates accordingly.

### Curated topics

The current local catalog includes:

- Python programming
- Machine Learning
- Data Structures and Algorithms
- Database Management Systems
- Web Development
- Natural Language Processing
- Cybersecurity
- Cloud Computing

The concepts, their difficulty, and their prerequisite relationships are stored in `knowledge_base.py`. Matching, personalization, graph ordering, and response generation are handled in `main.py`.

### What each roadmap module may show

- Concept name and description
- Learning objectives
- Prerequisites
- Estimated study time
- Suggested exercise
- Resource links, if available
- Completion control

### Topic matching and fallback sources

For topics that do not match the curated catalog, the backend can try:

1. **MOOCCubeX**, if a local dataset has been configured.
2. **OpenAlex**, a scholarly works and topic index.
3. **Wikipedia**, for an article summary fallback.

These sources have different levels of educational detail. For example, a scholarly classification or a Wikipedia summary is not the same thing as a reviewed syllabus. When the source does not establish prerequisites, the backend should not claim it has verified them. MOOCCubeX is optional and is not included in the repository by default. Configuration and limitations are documented in `README.md`.

### NLP and graph concepts used in this feature

- **Normalization:** Cleans and standardizes the topic text so wording differences are easier to handle.
- **Aliases and synonym handling:** Maps abbreviations or alternate names, such as “NLP,” to a known subject name.
- **TF-IDF:** Weights terms according to how useful they are in distinguishing one text from another. Here, character n-gram TF-IDF supports topic matching and text retrieval.
- **Cosine similarity:** Compares vector representations of text to estimate how similar a submitted topic is to a catalog entry or retrieved concept.
- **Prerequisite graph:** Represents concepts as nodes and “must be learned first” relationships as directed edges.
- **Topological sorting:** Produces an ordering of graph nodes where prerequisite concepts appear before dependent concepts. The graph is checked for cycles rather than silently presenting a cyclic order.

**Important distinction:** TF-IDF helps find related text; it does not create or verify a curriculum. The curated prerequisite edges are explicit catalog data. Where external sources do not provide evidence for prerequisites, those relationships are not presented as verified.

---

## 4. Smart Notes Generator

### What the learner can do

The learner can either:

- Paste study material, or
- Upload a text-based PDF.

Smart Notes returns:

- **Smart Summary**
- **Key Points**
- **Keywords and Definitions**
- **Flashcards**
- **Practice Questions**
- **Organized Notes**, which can be edited in the UI

Text requests are sent to `POST /api/smart-notes`; PDF uploads use `POST /api/smart-notes/upload`. The implementation is in `smart_notes.py`, and the frontend request handling is in `smartNotes.js`.

### NLP techniques used

- **Text cleaning:** Normalizes spaces and removes unwanted characters before processing.
- **Sentence splitting:** Separates the supplied material into sentence-like units for ranking and selection.
- **TF-IDF keyphrase ranking:** Finds terms and phrases that are prominent in the input text.
- **TF-IDF sentence ranking:** Assigns importance scores to sentences based on their terms.
- **Extractive summarization:** Selects important sentences from the source material and uses them in the summary. It does not write a new abstractive summary.
- **Source-context matching:** Associates a keyword with text in the supplied material rather than claiming an outside definition was verified.
- **Rule/template-based questions and flashcards:** Creates prompts from extracted content using patterns.

The generated notes are intended to remain grounded in the supplied source. The feature does not use a trained summarization, named-entity-recognition, or question-generation model. Scanned PDFs require OCR, which is not included.

---

## 5. Adaptive Quiz Generator

### What the learner can do

The learner enters a topic, answers multiple-choice questions, submits the quiz, reviews concept-level results, and requests another quiz.

### How adaptation works

1. The frontend requests a quiz from `POST /api/adaptive-quiz`.
2. The backend tries to match a curated topic. For a curated topic, it builds questions from the catalog’s concept descriptions and learning objectives.
3. When a topic is not in the curated catalog, the backend can attempt to use configured MOOCCubeX data or Wikipedia text. If it cannot find enough usable material to create a fair quiz, it returns an explanatory error instead of inventing questions.
4. Correct answers are held on the backend, not included in the quiz questions sent to the browser.
5. The frontend submits the learner’s choices to `POST /api/adaptive-quiz/submit`. The backend grades them and returns an overall score, concept scores, correct answers, and explanations.
6. The frontend sends accumulated concept scores with the next quiz request. Concepts scoring below 70% receive most of the available questions; other concepts remain in the revision mix. The quiz generator prefers unseen question IDs when it can.

Quiz selection, question generation, and grading are implemented in `adaptive_quiz.py`. The frontend API calls are in `adaptiveQuiz.js`.

### NLP concepts used

- **Topic matching with TF-IDF and cosine similarity:** Helps match a submitted topic to known subjects.
- **Keyphrase extraction:** For retrieved text, TF-IDF ranks phrases that can be used to form source-text cloze questions.
- **Question construction from information:** Curated questions are assembled from concept descriptions and objectives; retrieved-source questions use extracted text. This is a rule-based baseline, not a trained question-generation model.
- **Answer matching:** For multiple-choice questions, the backend compares the selected option ID with the stored correct option ID. This is deterministic option grading, not semantic analysis of a written answer.
- **Weak-area tracking:** Per-concept correct and total counts are used to estimate mastery and allocate more questions to weaker concepts. This is a simple performance-based adaptation rule, not a learned student model.

Quiz sessions are held in backend memory for up to one hour. Quiz history and mastery are not stored in a database, so they do not persist across a page reload or backend restart.

---

## 6. NLP concepts explained in the context of LearnMap

| Concept | Meaning | How LearnMap uses it |
|---|---|---|
| **Normalization** | Standardizing text, such as case and whitespace, so equivalent wording is easier to compare. | Helps process topic input and text before matching or analysis. |
| **Tokenization** | Splitting text into units such as words or punctuation. | NLTK tokenization is used in the backend’s text processing. |
| **Stop-word filtering** | Excluding common words that often contribute little to a text comparison, such as “the” or “and.” | Used with vectorization and ranking to emphasize more informative words. |
| **TF-IDF** | A statistical weighting scheme that gives more weight to terms that are informative within a collection of documents. | Used for topic/concept matching, sentence ranking, and keyword extraction. |
| **Cosine similarity** | A measure of how close two text vectors point in the same direction. | Helps rank candidate topics and retrieved concepts by textual similarity. |
| **Keyphrase extraction** | Selecting terms or phrases that appear important in a text. | Powers keyword output in Smart Notes and helps form retrieved-text quiz questions. |
| **Extractive summarization** | Selecting important sentences from a source rather than generating new sentences. | Smart Notes uses this approach to keep the summary tied to the supplied material. |
| **Information retrieval** | Finding and ranking potentially relevant items from a collection or source. | Used in Learning Path fallback discovery and per-concept resource lookup. |
| **Knowledge graph / prerequisite graph** | A graph that represents concepts and their dependency relationships. | Stores the curated learning dependencies used to build a sensible order. |
| **Topological sort** | A graph algorithm that orders items while respecting directed dependencies. | Ensures a prerequisite is listed before the concept that depends on it. |
| **Classification** | Assigning an item to a category based on its properties. | The current Quiz feature does not train a question classifier. Quiz grading is direct option comparison. |
| **Semantic similarity** | Similarity in meaning, beyond exact word overlap, often computed using embeddings. | The project’s main matching baseline uses TF-IDF, not a Sentence Transformers embedding model. |
| **Named Entity Recognition (NER)** | Detecting entities such as people, organizations, or locations in text. | Not currently part of the Smart Notes processing pipeline. |
| **Abstractive summarization** | Generating new sentences that paraphrase a source. | Not currently used; Smart Notes is extractive. |
| **Question generation** | Creating assessment questions from content. | Implemented with rules and templates from catalog/source text; no trained generation model is used. |

---

## 7. Technology stack

### Frontend

- **React 18:** Builds the interactive interface and manages feature state.
- **Vite:** Runs the development server and creates production builds.
- **Tailwind CSS:** Included for styling support; the current dashboard also uses CSS in `index.css`.
- **Lucide React:** Supplies interface icons.
- **Recharts:** Listed as a dependency, but I did not find it used by the current `src` components.

### Backend

- **Python:** Backend language.
- **FastAPI:** HTTP API and endpoint routing.
- **Pydantic:** Validates incoming request data and response models.
- **NLTK:** Tokenization utilities.
- **scikit-learn:** TF-IDF vectorization, cosine similarity support, and related machine-learning utilities.
- **pypdf:** Extracts text from text-based PDF files.
- **python-multipart:** Enables PDF upload handling through FastAPI.

### Optional data and external sources

- **MOOCCubeX:** Optional local educational dataset; it must be separately downloaded/configured.
- **OpenAlex:** Optional online scholarly-topic discovery and resource lookup.
- **Wikipedia:** Optional article-summary fallback and resource lookup.

### Technologies not currently integrated

- **MongoDB:** Not currently used for user accounts, learning progress, quiz mastery, or persistence.
- **spaCy:** Not currently used by the features described above.
- **Sentence Transformers or another embedding model:** Not currently used for semantic embeddings.
- **Trained summarization or question-generation model:** Not currently used.
- **OCR:** Not included for scanned PDFs.

The API base URL is configurable through `VITE_API_BASE_URL`; the current frontend default is `http://localhost:8001`. The relevant endpoint and setup details are documented in `README.md`.

---

## 8. Overall workflow

A learner can use the site in this sequence:

1. **Choose a subject and build a Learning Path.** The curated path uses topic matching and explicit prerequisites to order concepts.
2. **Study content and make notes.** Paste text or upload a text-based PDF to generate summaries, key points, keywords, flashcards, and editable notes.
3. **Practice with an Adaptive Quiz.** Answer questions, see concept-level results, and request a follow-up quiz weighted toward weaker concepts.
4. **Use the returned links where available.** Resource links appear within relevant modules or quiz results; there is no longer a separate Resources section.

## 9. Testing status and limitations

The latest recorded backend test run passed **44 tests**, and the frontend production build succeeded. The tests cover the backend Learning Path, Smart Notes, and Adaptive Quiz behaviors, including multiple subject areas and adaptive quiz selection.

LearnMap currently uses **NLP/statistical baselines, curated knowledge, external retrieval, and rules** rather than a universal trained AI model. The curated topic list is finite; external discovery depends on network availability or a configured dataset; and progress/mastery is not persisted to MongoDB.

## 10. How to run Code

Open **two Command Prompt windows** from the project folder.

**Terminal 1 — FastAPI backend**
```bat
cd /d "C:\Users\<username>\<project-folder>"
backend\.venv\Scripts\activate.bat
set LEARNMAP_KNOWLEDGE_PROVIDER=openalex
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

**Terminal 2 — Vite frontend**
```bat
cd /d "C:\Users\<username>\<project-folder>"
set VITE_API_BASE_URL=http://127.0.0.1:8001
npm run dev
```

Then open the **Local** URL Vite prints—usually `http://localhost:5173/`—in your browser.

If the backend dependencies have not been installed in the virtual environment yet, run this once before starting the backend:
```bat
cd /d "C:\Users\<username>\<project-folder>"
backend\.venv\Scripts\activate.bat
python -m pip install -r backend\requirements.txt
```
