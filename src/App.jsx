import { useState } from "react";
import {
  ArrowDown,
  ArrowRight,
  BookOpen,
  Brain,
  Check,
  FileText,
  Flame,
  GraduationCap,
  Layers3,
  Map,
  Sparkles,
  Upload,
  WandSparkles,
} from "lucide-react";
import { generateLearningPath } from "./api/learningPath.js";
import { generateSmartNotes } from "./api/smartNotes.js";
import { generateAdaptiveQuiz, submitAdaptiveQuiz } from "./api/adaptiveQuiz.js";

const sampleTopics = [
  { id: "sample-preprocessing", title: "Text preprocessing", estimated_minutes: 15 },
  { id: "sample-tokenization", title: "Tokenization & stemming", estimated_minutes: 20 },
  { id: "sample-tfidf", title: "TF-IDF & keywords", estimated_minutes: 25 },
  { id: "sample-classification", title: "Text classification", estimated_minutes: 35 },
];

const features = [
  { id: "path", label: "Learning path", icon: Map, color: "lavender" },
  { id: "notes", label: "Smart notes", icon: FileText, color: "mint" },
  { id: "quiz", label: "Adaptive quiz", icon: Brain, color: "blue" },
];

function Logo() {
  return (
    <a className="brand" href="#top" aria-label="LearnMap home">
      <span className="brand-mark"><Map size={19} strokeWidth={2.2} /></span>
      <span>learn<span className="brand-accent">map</span></span>
    </a>
  );
}

function Sidebar({ active, onNavigate }) {
  return (
    <aside className="sidebar">
      <Logo />
      <div className="workspace-label">WORKSPACE</div>
      <nav className="side-nav" aria-label="Main navigation">
        <button className={`nav-item ${active === "top" ? "nav-active" : ""}`} onClick={() => onNavigate("top")}>
          <Layers3 size={18} /> Overview
        </button>
        {features.map(({ id, label, icon: Icon }) => (
          <button key={id} className={`nav-item ${active === id ? "nav-active" : ""}`} onClick={() => onNavigate(id)}>
            <Icon size={18} /> {label}
          </button>
        ))}
      </nav>
    </aside>
  );
}

function SectionHeading({ eyebrow, title, action, href }) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h2>{title}</h2>
      </div>
      {action && <a className="text-link" href={href}>{action}<ArrowRight size={15} /></a>}
    </div>
  );
}

function App() {
  const [active, setActive] = useState("top");
  const [topic, setTopic] = useState("");
  const [learnerLevel, setLearnerLevel] = useState("beginner");
  const [learnerGoal, setLearnerGoal] = useState("");
  const [knownConceptsInput, setKnownConceptsInput] = useState("");
  const [learningPath, setLearningPath] = useState(null);
  const [completedModuleIds, setCompletedModuleIds] = useState(() => new Set());
  const [isGeneratingPath, setIsGeneratingPath] = useState(false);
  const [hasRequestedPath, setHasRequestedPath] = useState(false);
  const [pathError, setPathError] = useState("");
  const [notesText, setNotesText] = useState("");
  const [notesFile, setNotesFile] = useState(null);
  const [notesResult, setNotesResult] = useState(null);
  const [isGeneratingNotes, setIsGeneratingNotes] = useState(false);
  const [notesError, setNotesError] = useState("");
  const [quizTopic, setQuizTopic] = useState("");
  const [adaptiveQuiz, setAdaptiveQuiz] = useState(null);
  const [quizAnswers, setQuizAnswers] = useState({});
  const [quizResult, setQuizResult] = useState(null);
  const [quizMastery, setQuizMastery] = useState([]);
  const [seenQuizQuestionIds, setSeenQuizQuestionIds] = useState([]);
  const [isBuildingQuiz, setIsBuildingQuiz] = useState(false);
  const [isSubmittingQuiz, setIsSubmittingQuiz] = useState(false);
  const [quizError, setQuizError] = useState("");

  const navigate = (id) => {
    setActive(id);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const generatePath = async (event) => {
    event.preventDefault();
    const newTopic = topic.trim();
    if (!newTopic) {
      setPathError("Enter a topic before building your learning path.");
      return;
    }

    setHasRequestedPath(true);
    setIsGeneratingPath(true);
    setPathError("");
    setLearningPath(null);
    setCompletedModuleIds(new Set());
    try {
      const result = await generateLearningPath({
        topic: newTopic,
        level: learnerLevel,
        goal: learnerGoal.trim() || null,
        knownConcepts: knownConceptsInput.split(",").map((concept) => concept.trim()).filter(Boolean),
      });
      setLearningPath(result);
      setActive("path");
      document.getElementById("path")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      setPathError(error instanceof Error ? error.message : "The learning path could not be generated.");
    } finally {
      setIsGeneratingPath(false);
    }
  };

  const createNotes = async (event) => {
    event.preventDefault();
    if (!notesFile && !notesText.trim()) {
      setNotesError("Paste study material or choose a PDF before generating notes.");
      return;
    }
    setNotesError("");
    setNotesResult(null);
    setIsGeneratingNotes(true);
    try {
      const result = await generateSmartNotes({
        text: notesFile ? undefined : notesText.trim(),
        file: notesFile || undefined,
      });
      setNotesResult(result);
    } catch (error) {
      setNotesError(error instanceof Error ? error.message : "Smart Notes could not be generated.");
    } finally {
      setIsGeneratingNotes(false);
    }
  };

  const buildAdaptiveQuiz = async (event) => {
    event.preventDefault();
    const newTopic = quizTopic.trim();
    if (!newTopic) {
      setQuizError("Enter a topic before building your quiz.");
      return;
    }

    setQuizError("");
    setIsBuildingQuiz(true);
    try {
      const result = await generateAdaptiveQuiz({
        topic: newTopic,
        mastery: quizMastery,
        seenQuestionIds: seenQuizQuestionIds,
      });
      setAdaptiveQuiz(result);
      setQuizAnswers({});
      setQuizResult(null);
      setSeenQuizQuestionIds((previous) => [
        ...new Set([...previous, ...result.questions.map((question) => question.id)]),
      ]);
    } catch (error) {
      setQuizError(error instanceof Error ? error.message : "The quiz could not be generated.");
    } finally {
      setIsBuildingQuiz(false);
    }
  };

  const reviewAdaptiveQuiz = async (event) => {
    event.preventDefault();
    if (!adaptiveQuiz) return;
    const answers = adaptiveQuiz.questions.map((question) => ({
      question_id: question.id,
      selected_option_id: quizAnswers[question.id],
    }));
    setQuizError("");
    setIsSubmittingQuiz(true);
    try {
      const result = await submitAdaptiveQuiz({ quizId: adaptiveQuiz.quiz_id, answers });
      setQuizResult(result);
      setQuizMastery((previous) => {
        const totals = new globalThis.Map(previous.map((item) => [item.concept_id, { ...item }]));
        for (const score of result.concept_scores) {
          const current = totals.get(score.concept_id) ?? {
            concept_id: score.concept_id,
            correct: 0,
            total: 0,
          };
          current.correct += score.correct;
          current.total += score.total;
          totals.set(score.concept_id, current);
        }
        return [...totals.values()];
      });
    } catch (error) {
      setQuizError(error instanceof Error ? error.message : "Your quiz could not be reviewed.");
    } finally {
      setIsSubmittingQuiz(false);
    }
  };

  const changeQuizTopic = (value) => {
    setQuizTopic(value);
    setAdaptiveQuiz(null);
    setQuizAnswers({});
    setQuizResult(null);
    setQuizMastery([]);
    setSeenQuizQuestionIds([]);
    setQuizError("");
  };

  const isSamplePath = !learningPath && !hasRequestedPath && !isGeneratingPath;
  const displayedModules = learningPath?.learning_modules ?? (isSamplePath ? sampleTopics : []);
  const completedCount = learningPath
    ? learningPath.learning_modules.filter((module) => completedModuleIds.has(module.id)).length
    : 0;
  const moduleCount = learningPath?.total_concept_count ?? (isSamplePath ? sampleTopics.length : 0);
  const totalMinutes = learningPath?.estimated_total_minutes
    ?? (isSamplePath ? sampleTopics.reduce((total, module) => total + module.estimated_minutes, 0) : 0);
  const progressPercent = moduleCount ? Math.round((completedCount / moduleCount) * 100) : 0;
  const currentModuleIndex = displayedModules.findIndex((module) => !completedModuleIds.has(module.id));
  const roadmapGroups = learningPath
    ? ["beginner", "intermediate", "advanced"]
      .map((level) => ({
        level,
        modules: displayedModules.filter((module) => module.difficulty === level),
      }))
      .filter((group) => group.modules.length > 0)
    : isSamplePath
      ? [{ level: "beginner", modules: displayedModules }]
      : [];

  const toggleModuleComplete = (moduleId) => {
    setCompletedModuleIds((previous) => {
      const next = new Set(previous);
      if (next.has(moduleId)) next.delete(moduleId);
      else next.add(moduleId);
      return next;
    });
  };

  return (
    <div className="app-shell" id="top">
      <Sidebar active={active} onNavigate={navigate} />
      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><span className="crumb-slash">/</span><strong>Overview</strong></div>
        </header>

        <div className="content-wrap">
          <section className="welcome-row">
            <div>
              <p className="welcome-subtitle">A little progress each day adds up to big things.</p>
            </div>
          </section>

          <section className="hero-card">
            <div className="hero-content">
              <div className="hero-kicker"><Sparkles size={14} /> YOUR NEXT LEARNING ADVENTURE</div>
              <h2>Big goals, meet<br /><span>a clear path.</span></h2>
              <p>Tell us what you want to learn. We'll map out the steps to get you there.</p>
              <form className="path-form" onSubmit={generatePath}>
                <label className="sr-only" htmlFor="learning-topic">What do you want to learn?</label>
                <input
                  id="learning-topic"
                  value={topic}
                  onChange={(event) => {
                    setTopic(event.target.value);
                    setPathError("");
                  }}
                  placeholder="e.g. I want to learn data science"
                  aria-invalid={Boolean(pathError && !topic.trim())}
                />
                <button type="submit" disabled={isGeneratingPath}>
                  <WandSparkles size={16} /> {isGeneratingPath ? "Building path…" : "Build my path"}
                </button>
              </form>
              <div style={{ display: "flex", gap: 8, maxWidth: 446, marginTop: 8 }}>
                <label className="sr-only" htmlFor="learner-level">Your current level</label>
                <select
                  id="learner-level"
                  value={learnerLevel}
                  onChange={(event) => setLearnerLevel(event.target.value)}
                  style={{ minWidth: 112, padding: "7px 8px", color: "#77778e", fontSize: 9, border: "1px solid #e6e3f7", borderRadius: 7, background: "white" }}
                >
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                </select>
                <label className="sr-only" htmlFor="learning-goal">Optional learning goal</label>
                <input
                  id="learning-goal"
                  value={learnerGoal}
                  onChange={(event) => setLearnerGoal(event.target.value)}
                  placeholder="Optional goal, e.g. evaluate model performance"
                  style={{ minWidth: 0, flex: 1, padding: "7px 9px", color: "#77778e", fontSize: 9, outline: "none", border: "1px solid #e6e3f7", borderRadius: 7, background: "white" }}
                />
              </div>
              <label className="sr-only" htmlFor="known-concepts">Concepts you already understand, separated by commas</label>
              <input
                id="known-concepts"
                value={knownConceptsInput}
                onChange={(event) => setKnownConceptsInput(event.target.value)}
                placeholder="Already know? Add concepts separated by commas"
                style={{ display: "block", width: "100%", maxWidth: 446, marginTop: 6, padding: "7px 9px", color: "#77778e", fontSize: 9, outline: "none", border: "1px solid #e6e3f7", borderRadius: 7, background: "white" }}
              />
              {pathError && (
                <p role="alert" style={{ maxWidth: 446, margin: "8px 0 0", color: "#b94c5b", fontSize: 10, lineHeight: 1.45 }}>
                  {pathError}
                </p>
              )}
              <div className="hero-hint"><span className="hint-spark">✳</span> Personalized to your pace, goals, and what you already know</div>
            </div>
            <div className="hero-art" aria-hidden="true">
              <div className="orbit orbit-one" /><div className="orbit orbit-two" />
              <div className="art-sun"><Sparkles size={34} strokeWidth={1.2} /></div>
              <div className="art-node node-top"><span>✦</span></div>
              <div className="art-node node-left"><Check size={16} /></div>
              <div className="art-node node-right"><BookOpen size={18} /></div>
              <div className="art-node node-bottom"><span>✳</span></div>
              <div className="art-path path-one" /><div className="art-path path-two" />
              <span className="art-label label-one">learn</span><span className="art-label label-two">grow</span>
            </div>
          </section>

          <section className="feature-strip" aria-label="LearnMap features">
            <span className="feature-strip-label">YOUR LEARNING, CONNECTED</span>
            {features.map(({ id, label, icon: Icon, color }) => (
              <button key={id} className="feature-chip" onClick={() => navigate(id)}>
                <span className={`chip-icon ${color}`}><Icon size={14} /></span>{label}
              </button>
            ))}
          </section>

          <section className="stats-row" aria-label="Learning progress">
            <div className="stat-card">
              <div className="stat-icon stat-violet"><Map size={18} /></div>
              <div><span className="stat-label">ACTIVE LEARNING PATH</span><strong>1 <small>path</small></strong></div>
              <span className="stat-trend">On track</span>
            </div>
            <div className="stat-card">
              <div className="stat-icon stat-green"><Check size={18} /></div>
              <div><span className="stat-label">CONCEPTS MASTERED</span><strong>12 <small>of 24</small></strong></div>
              <span className="stat-trend trend-green">+3 this week</span>
            </div>
            <div className="stat-card">
              <div className="stat-icon stat-orange"><Flame size={18} /></div>
              <div><span className="stat-label">LEARNING STREAK</span><strong>4 <small>days</small></strong></div>
              <span className="stat-trend trend-orange">Personal best 7</span>
            </div>
          </section>

          <div className="dashboard-grid">
            <section className="panel path-panel" id="path">
              <SectionHeading
                eyebrow={learningPath ? `YOUR ROADMAP · ${learningPath.learner_level.toUpperCase()}` : isSamplePath ? "SAMPLE ROADMAP · EXAMPLE ONLY" : "YOUR ROADMAP"}
                title={learningPath?.matched_topic ?? (isSamplePath ? "Natural Language Processing" : isGeneratingPath ? "Building your path…" : "No path generated")}
                action={learningPath ? "View full path" : undefined}
                href="#path"
              />
              {learningPath && learningPath.requested_topic !== learningPath.matched_topic && (
                <p className="panel-description" style={{ marginTop: 6 }}>Matched from: “{learningPath.requested_topic}”</p>
              )}
              <div className="path-meta">
                <span><GraduationCap size={15} /> {learningPath ? `${learningPath.learner_level} path` : isSamplePath ? "Beginner sample" : `${learnerLevel} request`}</span>
                <span><BookOpen size={15} /> {moduleCount} concepts</span>
                <span><Flame size={15} /> {totalMinutes} min total</span>
                <span className="path-progress-text">{completedCount}/{moduleCount} complete · {progressPercent}%</span>
              </div>
              <div className="progress-track" aria-label={`${progressPercent}% complete`}>
                <span style={{ width: `${progressPercent}%` }} />
              </div>
              {learningPath && (
                <>
                  <div className="process-caption"><span>NLP NORMALIZATION</span><i>→</i><span>TF-IDF RETRIEVAL</span><i>→</i><span>EVIDENCE-BASED ORDER</span></div>
                  {learningPath.detected_concepts.length > 0 && (
                    <p className="panel-description" style={{ margin: "8px 0" }}>
                      Matched concepts: {learningPath.detected_concepts.join(", ")}
                    </p>
                  )}
                  {learningPath.prerequisite_concepts.length > 0 && (
                    <p className="panel-description" style={{ margin: "8px 0" }}>
                      Already-known prerequisites: {learningPath.prerequisite_concepts.join(", ")}
                    </p>
                  )}
                  {learningPath.limitations.map((limitation) => (
                    <p key={limitation} className="panel-description" style={{ margin: "8px 0", color: "#9b7952" }}>
                      {limitation}
                    </p>
                  ))}
                </>
              )}
              {isGeneratingPath ? (
                <p role="status" className="panel-description" style={{ padding: "18px 0" }}>Matching your topic and ordering its prerequisites…</p>
              ) : pathError && !learningPath ? (
                <p className="panel-description" style={{ padding: "18px 0" }}>No new path was generated. Correct the topic or try again after resolving the error above.</p>
              ) : (
                <div className="roadmap-list">
                  {roadmapGroups.map((group) => (
                    <section key={group.level} aria-label={`${group.level} learning stage`}>
                      <h3 style={{ margin: "12px 0 5px", color: "#7771dc", fontSize: 10, fontWeight: 700, letterSpacing: ".08em", textTransform: "uppercase" }}>
                        {group.level}
                      </h3>
                      {group.modules.map((module, groupIndex) => {
                        const index = displayedModules.findIndex((item) => item.id === module.id);
                        const isCompleted = completedModuleIds.has(module.id);
                        const isCurrent = !isCompleted && index === currentModuleIndex;
                        const stepState = isCompleted ? "done" : isCurrent ? "current" : "upcoming";
                        return (
                          <div className={`roadmap-step step-${stepState}`} key={module.id} style={{ alignItems: "flex-start", padding: "9px 0" }}>
                            <div className="step-marker" style={{ marginTop: 3 }}>
                              {isCompleted ? <Check size={15} /> : <span>{String(module.order ?? index + 1).padStart(2, "0")}</span>}
                            </div>
                            <div className="step-copy" style={{ minWidth: 0, flex: 1, gap: 5 }}>
                              <strong>{module.title}</strong>
                              <span>
                                {isCompleted ? "COMPLETED" : isCurrent ? "UP NEXT" : "UPCOMING"}
                                {learningPath && ` · ${module.difficulty.toUpperCase()} · ${module.estimated_minutes} MIN`}
                                {!learningPath && ` · ${module.estimated_minutes} MIN`}
                              </span>
                              {learningPath && (
                                <>
                                  <p className="panel-description" style={{ margin: "2px 0", lineHeight: 1.5 }}>{module.description}</p>
                                  <ul style={{ margin: "2px 0 2px 14px", padding: 0, color: "#858599", fontSize: 9, lineHeight: 1.55 }}>
                                    {module.learning_objectives.map((objective) => <li key={objective}>{objective}</li>)}
                                  </ul>
                                  {module.resources?.length > 0 && (
                                    <ul aria-label="Related learning resources" style={{ margin: "2px 0 2px 14px", padding: 0, color: "#7771dc", fontSize: 9, lineHeight: 1.55 }}>
                                      {module.resources.map((resource) => (
                                        <li key={`${resource.source_id}-${resource.title}`}>
                                          {resource.resource_type}:{" "}
                                          {resource.url ? (
                                            <a href={resource.url} target="_blank" rel="noreferrer" style={{ color: "#7771dc" }}>
                                              {resource.title}
                                            </a>
                                          ) : resource.title}
                                          {" · "}{resource.source}
                                        </li>
                                      ))}
                                    </ul>
                                  )}
                                  {module.prerequisites.length > 0 && (
                                    <span style={{ color: "#8a85d5", fontSize: 8 }}>PREREQUISITES · {module.prerequisites.join(", ")}</span>
                                  )}
                                  {module.source_url && (
                                    <span style={{ fontSize: 9 }}>
                                      <a href={module.source_url} target="_blank" rel="noreferrer" style={{ color: "#7771dc" }}>
                                        {module.source_name || "Knowledge source"}
                                      </a>
                                      {module.source_name === "Wikipedia" && (
                                        <>
                                          {" · "}
                                          <a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noreferrer" style={{ color: "#7771dc" }}>
                                            CC BY-SA 4.0
                                          </a>
                                        </>
                                      )}
                                    </span>
                                  )}
                                  <details style={{ color: "#7771dc", fontSize: 9 }}>
                                    <summary style={{ cursor: "pointer" }}>Suggested exercise</summary>
                                    <p className="panel-description" style={{ margin: "5px 0" }}>{module.suggested_exercise}</p>
                                  </details>
                                </>
                              )}
                            </div>
                            {learningPath && (
                              <button
                                type="button"
                                aria-pressed={isCompleted}
                                aria-label={`${isCompleted ? "Mark incomplete" : "Mark complete"}: ${module.title}`}
                                onClick={() => toggleModuleComplete(module.id)}
                                style={{ flex: "0 0 auto", marginTop: 4, padding: 0, color: isCompleted ? "#50a98c" : "#7771dc", fontSize: 8, whiteSpace: "nowrap", border: 0, background: "transparent" }}
                              >
                                {isCompleted ? "Completed" : "Mark done"}
                              </button>
                            )}
                            {groupIndex < group.modules.length - 1 && <span className="step-connector" />}
                          </div>
                        );
                      })}
                    </section>
                  ))}
                </div>
              )}
            </section>

          </div>

          <div className="dashboard-grid lower-grid">
            <section className="panel notes-panel" id="notes">
              <SectionHeading eyebrow="STUDY SMARTER" title="Smart notes" action="Open notes" href="#notes" />
              <p className="panel-description">Paste study material or upload a text-based PDF to create source-grounded revision notes.</p>
              <form className="notes-input-wrap" onSubmit={createNotes}>
                <textarea
                  value={notesText}
                  onChange={(event) => {
                    setNotesText(event.target.value);
                    setNotesFile(null);
                    setNotesError("");
                  }}
                  placeholder="Paste a transcript, textbook extract, or your study material..."
                  aria-label="Study material to turn into notes"
                  disabled={isGeneratingNotes}
                />
                <div className="notes-actions">
                  <label className="upload-button">
                    <Upload size={15} /> {notesFile ? notesFile.name : "Upload PDF"}
                    <input
                      type="file"
                      accept="application/pdf,.pdf"
                      aria-label="Upload a PDF document"
                      disabled={isGeneratingNotes}
                      onChange={(event) => {
                        const file = event.target.files?.[0] ?? null;
                        setNotesFile(file);
                        if (file) setNotesText("");
                        setNotesError("");
                      }}
                    />
                  </label>
                  <button className="small-primary" type="submit" disabled={isGeneratingNotes}>
                    <Sparkles size={14} /> {isGeneratingNotes ? "Generating…" : "Generate notes"}
                  </button>
                </div>
              </form>
              {notesError && <p className="notes-error" role="alert">{notesError}</p>}
              {isGeneratingNotes && <p className="panel-description" role="status">Analyzing your material and preparing revision notes…</p>}
              {notesResult && (
                <div className="notes-result">
                  <div className="result-badge"><Check size={14} /> NOTES READY · {notesResult.source_name}</div>
                  <p className="notes-method">{notesResult.method} · {notesResult.source_characters.toLocaleString()} source characters</p>
                  <section className="smart-notes-section">
                    <h3>Smart Summary</h3>
                    <p>{notesResult.summary}</p>
                  </section>
                  <section className="smart-notes-section">
                    <h3>Key Points</h3>
                    <ul>{notesResult.key_points.map((point, index) => <li key={`${index}-${point}`}>{point}</li>)}</ul>
                  </section>
                  <section className="smart-notes-section">
                    <h3>Keywords and Definitions</h3>
                    {notesResult.keywords.length ? (
                      <ul className="notes-definitions">
                        {notesResult.keywords.map((keyword) => (
                          <li key={keyword.term}>
                            <strong>{keyword.term}</strong> — {keyword.definition}
                            <small>{keyword.basis}</small>
                          </li>
                        ))}
                      </ul>
                    ) : <p>No source-supported keyword definitions were identified.</p>}
                  </section>
                  <section className="smart-notes-section">
                    <h3>Flashcards</h3>
                    {notesResult.flashcards.length ? notesResult.flashcards.map((card, index) => (
                      <details className="notes-flashcard" key={`${index}-${card.front}`}>
                        <summary>{card.front}</summary>
                        <p>{card.back}</p>
                      </details>
                    )) : <p>No flashcards could be formed from the supplied material.</p>}
                  </section>
                  <section className="smart-notes-section">
                    <h3>Practice Questions</h3>
                    <ol>{notesResult.practice_questions.map((question, index) => <li key={`${index}-${question}`}>{question}</li>)}</ol>
                  </section>
                  <section className="smart-notes-section">
                    <h3>Organized Notes · editable</h3>
                    <textarea
                      className="organized-notes-editor"
                      aria-label="Edit your organized notes"
                      value={notesResult.organized_notes}
                      onChange={(event) => setNotesResult((current) => ({
                        ...current,
                        organized_notes: event.target.value,
                      }))}
                    />
                  </section>
                  <section className="smart-notes-section notes-limitations">
                    <h3>Method and limitations</h3>
                    <ul>{notesResult.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>
                  </section>
                </div>
              )}
            </section>
            <section className="panel adaptive-quiz-panel" id="quiz">
              <SectionHeading eyebrow="PRACTICE WITH PURPOSE" title="Adaptive Quiz Generator" />
              <p className="panel-description">
                Answer topic-based questions, review your results, then get a follow-up quiz weighted toward concepts that need work.
              </p>
              <form className="quiz-topic-form" onSubmit={buildAdaptiveQuiz}>
                <label className="sr-only" htmlFor="adaptive-quiz-topic">Quiz topic</label>
                <input
                  id="adaptive-quiz-topic"
                  value={quizTopic}
                  onChange={(event) => changeQuizTopic(event.target.value)}
                  placeholder="Enter a topic, e.g. machine learning"
                  disabled={isBuildingQuiz || isSubmittingQuiz}
                />
                <button className="small-primary" type="submit" disabled={isBuildingQuiz || isSubmittingQuiz}>
                  <Brain size={14} /> {isBuildingQuiz ? "Building quiz…" : quizResult ? "Build adaptive quiz" : "Build my quiz"}
                </button>
              </form>
              {quizError && <p className="notes-error" role="alert">{quizError}</p>}
              {isBuildingQuiz && <p className="panel-description" role="status">Finding reliable concepts and preparing questions…</p>}
              {adaptiveQuiz && (
                <div className="quiz-result">
                  <div className="result-badge">
                    <Check size={14} /> {adaptiveQuiz.topic} · {adaptiveQuiz.question_count} QUESTIONS · {adaptiveQuiz.source}
                  </div>
                  {adaptiveQuiz.sources?.map((source) => (
                    <a className="quiz-source-link" key={source.url} href={source.url} target="_blank" rel="noreferrer">
                      View source: {source.name}
                    </a>
                  ))}
                  <p className="notes-method">
                    {quizResult
                      ? `Quiz score: ${quizResult.correct_count}/${quizResult.total_count} (${quizResult.score_percent}%).`
                      : "Choose one answer per question. Submit to see your concept scores and explanations."}
                  </p>
                  {quizMastery.length > 0 && (
                    <div className="quiz-mastery" aria-label="Concept mastery history">
                      <strong>Current concept mastery</strong>
                      <div>
                        {quizMastery.map((item) => (
                          <span key={item.concept_id}>
                            {item.concept_id}: {Math.round(item.correct * 100 / item.total)}%
                          </span>
                        ))}
                      </div>
                      <p>Follow-up quizzes prioritize concepts below 70%, while keeping revision questions from other concepts.</p>
                    </div>
                  )}
                  <form className="quiz-questions" onSubmit={reviewAdaptiveQuiz}>
                    {adaptiveQuiz.questions.map((question, index) => (
                      <fieldset className="quiz-question" key={question.id} disabled={Boolean(quizResult) || isSubmittingQuiz}>
                        <legend>
                          <span>QUESTION {index + 1} · {question.concept}</span>
                          {question.prompt}
                        </legend>
                        <div className="quiz-options">
                          {question.options.map((option) => (
                            <label key={option.id} className={quizAnswers[question.id] === option.id ? "quiz-option quiz-option-selected" : "quiz-option"}>
                              <input
                                type="radio"
                                name={`answer-${question.id}`}
                                value={option.id}
                                checked={quizAnswers[question.id] === option.id}
                                onChange={() => setQuizAnswers((previous) => ({
                                  ...previous,
                                  [question.id]: option.id,
                                }))}
                              />
                              <span>{option.text}</span>
                            </label>
                          ))}
                        </div>
                      </fieldset>
                    ))}
                    {!quizResult && (
                      <button
                        className="small-primary quiz-submit"
                        type="submit"
                        disabled={isSubmittingQuiz || Object.keys(quizAnswers).length !== adaptiveQuiz.questions.length}
                      >
                        {isSubmittingQuiz ? "Reviewing answers…" : "Submit answers"}
                      </button>
                    )}
                  </form>
                  {quizResult && (
                    <div className="quiz-review">
                      <section className="smart-notes-section">
                        <h3>Concept results</h3>
                        <ul>
                          {quizResult.concept_scores.map((score) => (
                            <li key={score.concept_id}>
                              {score.concept}: {score.correct}/{score.total} correct · {score.score_percent}%
                            </li>
                          ))}
                        </ul>
                      </section>
                      {quizResult.answers.some((answer) => !answer.is_correct) && (
                        <section className="smart-notes-section">
                          <h3>Review missed questions</h3>
                          <ul>
                            {quizResult.answers.filter((answer) => !answer.is_correct).map((answer) => (
                              <li key={answer.question_id}>
                                <strong>{answer.concept}</strong> · Correct answer: {answer.correct_answer}. {answer.explanation}
                              </li>
                            ))}
                          </ul>
                        </section>
                      )}
                      <button className="small-primary quiz-submit" type="button" onClick={buildAdaptiveQuiz} disabled={isBuildingQuiz}>
                        {isBuildingQuiz ? "Building quiz…" : "Build next adaptive quiz"}
                      </button>
                    </div>
                  )}
                  <p className="notes-method">{adaptiveQuiz.limitations.join(" ")}</p>
                </div>
              )}
            </section>
          </div>

          <footer className="page-footer"><Logo /><span>Learn at your pace. Get there with purpose.</span><button onClick={() => navigate("top")}>Back to top <ArrowDown size={14} /></button></footer>
        </div>
      </main>
    </div>
  );
}

export default App;
