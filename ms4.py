import os
import re
import json
import uuid
import enum
import hashlib
import logging
import traceback
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles  # <-- for serving frontend
from pydantic import BaseModel, ConfigDict

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, DateTime, Enum as SAEnum,
    ForeignKey, create_engine, desc, func as sa_func
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.sql import func

import chromadb
from chromadb.utils import embedding_functions
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bug-diagnosis")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./bug_analyzer.db")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Severity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SourceType(str, enum.Enum):
    PASTE = "paste"
    UPLOAD = "upload"

class BugSubmission(Base):
    __tablename__ = "bug_submissions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    bug_number = Column(Integer, nullable=False)
    bug_id = Column(String(20), unique=True, index=True, nullable=False)
    title = Column(String(255), nullable=False)
    severity = Column(SAEnum(Severity), nullable=False, default=Severity.MEDIUM)
    source_type = Column(SAEnum(SourceType), nullable=False)
    original_filename = Column(String(255), nullable=True)
    file_type = Column(String(50), nullable=True)
    content = Column(Text, nullable=False)
    content_length = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    kb_defect_id = Column(String(20), nullable=True)
    analysis = relationship("AgentAnalysis", back_populates="bug", uselist=False, cascade="all, delete-orphan")

class AgentAnalysis(Base):
    __tablename__ = "agent_analyses"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    bug_id = Column(String, ForeignKey("bug_submissions.id"), unique=True, nullable=False, index=True)
    triage_severity = Column(String(20), nullable=False)
    triage_priority = Column(String(10), nullable=False)
    triage_type = Column(String(30), nullable=False)
    triage_component = Column(String(100), nullable=False)
    triage_confidence = Column(Float, nullable=False)
    triage_reasoning = Column(Text, nullable=False)
    log_exception_type = Column(String(150), nullable=True)
    log_message = Column(Text, nullable=True)
    log_language = Column(String(30), nullable=True)
    log_failure_file = Column(String(255), nullable=True)
    log_failure_line = Column(String(20), nullable=True)
    log_affected_paths = Column(Text, nullable=True)
    log_error_signature = Column(String(64), nullable=True)
    log_confidence = Column(Float, nullable=False)
    log_reasoning = Column(Text, nullable=False)
    root_cause_hypothesis = Column(Text, nullable=True)
    root_cause_confidence = Column(Float, nullable=True)
    root_cause_evidence = Column(Text, nullable=True)
    duplicate_matches = Column(Text, nullable=True)
    remediation_summary = Column(Text, nullable=True)
    remediation_steps = Column(Text, nullable=True)
    remediation_confidence = Column(Float, nullable=True)
    remediation_sources = Column(Text, nullable=True)
    combined_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    bug = relationship("BugSubmission", back_populates="analysis")

class HistoricalDefect(Base):
    __tablename__ = "historical_defects"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    defect_id = Column(String(20), unique=True, nullable=False)
    title = Column(String(255), nullable=False)
    symptom_text = Column(Text, nullable=False)
    root_cause = Column(Text, nullable=False)
    resolution_summary = Column(Text, nullable=False)
    component = Column(String(100), nullable=False)
    tags = Column(String(255), nullable=True)

class BugSubmissionResponse(BaseModel):
    id: str
    bug_id: str
    title: str
    severity: str
    source_type: str
    original_filename: Optional[str]
    file_type: Optional[str]
    content_length: Optional[str]
    created_at: datetime
    is_resolved: bool = False
    resolution_notes: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)

class BugSubmissionCreatedResponse(BaseModel):
    id: str
    bug_id: str
    message: str
    analysis: Optional[Dict[str, Any]] = None

SEED_HISTORICAL_DEFECTS = [
    {"defect_id": "HDB-001", "title": "Checkout fails with CardError on Stripe charge",
     "symptom_text": "Traceback... stripe.error.CardError",
     "root_cause": "Stripe API key pointed to test-mode key.",
     "resolution_summary": "Corrected STRIPE_SECRET_KEY env var.",
     "component": "Payments", "tags": "stripe,charge,payment"},
    {"defect_id": "HDB-002", "title": "Login endpoint throws NullPointerException",
     "symptom_text": "java.lang.NullPointerException at SessionService.validate",
     "root_cause": "Session cache expired without null-check.",
     "resolution_summary": "Added null check and redirect.",
     "component": "Authentication", "tags": "auth,login,nullpointer"},
    {"defect_id": "HDB-003", "title": "Database connection pool exhausted",
     "symptom_text": "connection refused timeout database postgres",
     "root_cause": "Pool size default (5) too small for load.",
     "resolution_summary": "Increased pool_size and added metrics.",
     "component": "Database", "tags": "database,postgres,pool"},
    {"defect_id": "HDB-004", "title": "Uncaught TypeError on checkout button",
     "symptom_text": "TypeError: Cannot read properties of undefined",
     "root_cause": "Cart read before async fetch resolved.",
     "resolution_summary": "Added loading guard and default state.",
     "component": "Frontend/UI", "tags": "typeerror,ui,cart"},
    {"defect_id": "HDB-005", "title": "File upload 500 for large PDFs",
     "symptom_text": "500 internal server error upload file pdf",
     "root_cause": "No request body size limit configured.",
     "resolution_summary": "Set explicit max upload size.",
     "component": "File Upload", "tags": "upload,pdf,api"},
]

PLAYBOOKS = {
    "Authentication": ["Add explicit null/expiry checks around session/token lookups.",
                       "Add a regression test covering the expired-session path.",
                       "Ensure failed auth attempts degrade gracefully."],
    "Payments": ["Verify payment provider API keys/environment config.",
                 "Add a startup/config validation check.",
                 "Add idempotency handling for retried payments."],
    "Database": ["Review connection pool sizing against current concurrent load.",
                 "Add row-level locking or optimistic concurrency control.",
                 "Add query performance monitoring."],
    "API": ["Check for N+1 query patterns in the affected endpoint.",
            "Add explicit request size/timeout limits.",
            "Add an integration test reproducing the reported failure."],
    "Frontend/UI": ["Add a loading/guard state for async data.",
                    "Add a default/fallback value for any state accessed before load.",
                    "Add a visual regression test covering this component."],
    "File Upload": ["Set explicit max upload size limits and stream large files to disk.",
                    "Return a clear 413/validation error.",
                    "Add a test with a large/edge-case file."],
    "Networking": ["Set explicit client-side timeouts with exponential backoff retry.",
                   "Add a circuit breaker around the failing dependency.",
                   "Add monitoring/alerting on timeout rate."],
    "Infrastructure": ["Confirm the last deploy config against the previous known-good release.",
                       "Add a pre-deploy config validation step to CI.",
                       "Add health-check-based automatic rollback."],
    "Unclassified": ["Reproduce the issue locally to confirm scope before fixing.",
                     "Add logging around the suspected failure point.",
                     "Once root cause is confirmed, add a regression test."],
}

# ------------------------------------------------------------
#  Initialize ChromaDB and SentenceTransformer
# ------------------------------------------------------------
try:
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=SENTENCE_TRANSFORMER_MODEL
    )
    logger.info(f"SentenceTransformer model '{SENTENCE_TRANSFORMER_MODEL}' loaded.")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}")
    raise RuntimeError("SentenceTransformer could not be loaded. Please install 'sentence-transformers'.") from e

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    defect_collection = chroma_client.get_or_create_collection(
        name="historical_defects", embedding_function=sentence_transformer_ef
    )
    submission_collection = chroma_client.get_or_create_collection(
        name="bug_submissions", embedding_function=sentence_transformer_ef
    )
    logger.info(f"ChromaDB collections created/opened at {CHROMA_PATH}")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    raise RuntimeError("ChromaDB initialization failed. Check path permissions.") from e

ollama_client = ollama.Client(host=OLLAMA_HOST)

def call_ollama_json(system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
    try:
        response = ollama_client.chat(
            model=OLLAMA_CHAT_MODEL,
            format="json",
            options={"temperature": 0.1},
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}]
        )
        content = response["message"]["content"]
        return json.loads(content)
    except Exception as exc:
        logger.warning("Ollama call failed, falling back to deterministic logic: %s", exc)
        return None

# ------------------------------------------------------------
#  All analysis functions (unchanged)
# ------------------------------------------------------------
def keyword_triage(content: str) -> Dict[str, Any]:
    text = content.lower()
    scores = {s: 0 for s in ["critical","high","medium","low"]}
    matched = {s: [] for s in ["critical","high","medium","low"]}
    sev_keywords = {
        "critical": ["data loss", "corrupt", "security", "exploit", "vulnerability", "breach", "outage", "down",
                     "crash", "deadlock", "unresponsive", "segmentation fault", "fatal", "kernel panic"],
        "high": ["exception", "error", "fail", "failed", "failure", "broken", "blocked", "null pointer",
                 "nullpointerexception", "stack trace", "traceback", "cannot login", "regression", "timeout",
                 "connection refused"],
        "medium": ["warning", "warn", "intermittent", "slow", "performance", "deprecated", "inconsistent",
                   "unexpected behavior", "edge case"],
        "low": ["typo", "cosmetic", "misaligned", "ui glitch", "minor", "spacing", "tooltip", "enhancement"]
    }
    for sev, words in sev_keywords.items():
        for w in words:
            if w in text:
                scores[sev] += 1
                matched[sev].append(w)
    total = sum(scores.values())
    reasoning = []
    if total > 0:
        best = max(scores, key=lambda k: (scores[k], {"critical":4,"high":3,"medium":2,"low":1}[k]))
        reasoning.append(f"matched {len(matched[best])} {best}-severity keyword(s): {', '.join(matched[best][:5])}")
        confidence = min(0.5 + (total/10)*0.4, 0.95)
    else:
        best = "medium"
        reasoning.append("no strong severity keywords found; defaulted to Medium")
        confidence = 0.5
    comp_keywords = {
        "Authentication": ["login","auth","token","session","password","oauth","jwt"],
        "Payments": ["payment","checkout","stripe","billing","invoice","charge","refund"],
        "Database": ["database","sql","query","postgres","sqlite","orm","connection pool","deadlock"],
        "API": ["api","endpoint","request","response","rest","http"],
        "Frontend/UI": ["ui","button","css","layout","render","dom","browser","misaligned","tooltip"],
        "File Upload": ["upload","file","attachment","pdf","docx","multipart"],
        "Networking": ["timeout","connection refused","network","socket","dns"],
        "Infrastructure": ["deploy","docker","kubernetes","server","outage","down","crash"]
    }
    comp_scores = {c: sum(1 for kw in kws if kw in text) for c, kws in comp_keywords.items()}
    component = max(comp_scores, key=comp_scores.get) if max(comp_scores.values()) > 0 else "Unclassified"
    if component != "Unclassified":
        reasoning.append(f"component inferred as '{component}'")
    priority = {"critical":"P0","high":"P1","medium":"P2","low":"P3"}[best]
    bug_type = "Enhancement" if "enhancement" in text or "feature request" in text else "Bug"
    return {
        "severity": best, "priority": priority, "type": bug_type,
        "component": component, "confidence": round(confidence, 2),
        "reasoning": "; ".join(reasoning)
    }

def run_triage(content: str) -> Dict[str, Any]:
    baseline = keyword_triage(content)
    llm_result = call_ollama_json(
        system_prompt=(
            "You are a bug triage engine. Classify the bug report into JSON with keys: "
            "severity (low|medium|high|critical), priority (P0|P1|P2|P3), type (Bug|Enhancement), "
            "component (Authentication|Payments|Database|API|Frontend/UI|File Upload|Networking|Infrastructure|Unclassified), "
            "confidence (0-1 float), reasoning (short string). Respond with JSON only."
        ),
        user_prompt=f"Bug report:\n{content}\n\nKeyword baseline (for reference): {json.dumps(baseline)}",
    )
    if llm_result and all(k in llm_result for k in ("severity","priority","component","confidence")):
        llm_result.setdefault("type", baseline["type"])
        llm_result.setdefault("reasoning", baseline["reasoning"])
        llm_result["reasoning"] = f"[ollama:{OLLAMA_CHAT_MODEL}] " + str(llm_result["reasoning"])
        llm_result["confidence"] = round(float(llm_result["confidence"]), 2)
        return llm_result
    baseline["reasoning"] = "[keyword-fallback] " + baseline["reasoning"]
    return baseline

def run_log_analysis(content: str) -> Dict[str, Any]:
    reasoning = []
    exception_type = message = language = failure_file = failure_line = None
    confidence = 0.4

    if "Traceback (most recent call last)" in content:
        language = "Python"
        reasoning.append("detected Python traceback")
        matches = re.findall(r'File "([^"]+)", line (\d+)', content)
        if matches:
            failure_file, failure_line = matches[-1]
        exc = re.search(r"^([A-Za-z_.]+(?:Error|Exception|Warning))\s*:\s*(.*)$", content, re.MULTILINE)
        if exc:
            exception_type, message = exc.group(1), exc.group(2).strip()
    elif "Exception in thread" in content or ".java:" in content:
        language = "Java"
        reasoning.append("detected Java stack trace")
        m = re.search(r"\(([\w.$]+\.java):(\d+)\)", content)
        if m:
            failure_file, failure_line = m.group(1), m.group(2)
        exc = re.search(r"([\w.]+(?:Exception|Error))(?::\s*(.*))?", content)
        if exc:
            exception_type = exc.group(1)
            message = (exc.group(2) or "").strip() or None
    elif "Uncaught" in content or "TypeError" in content or "ReferenceError" in content:
        language = "JavaScript"
        reasoning.append("detected JS error")
        exc = re.search(r"(?:Uncaught\s+)?([A-Za-z]+(?:Error|Exception))\s*:\s*(.*)", content)
        if exc:
            exception_type, message = exc.group(1), exc.group(2).strip().split("\n")[0]
        m = re.search(r"([\w./\\-]+\.(?:js|ts|jsx|tsx)):(\d+):(\d+)", content)
        if m:
            failure_file, failure_line = m.group(1), m.group(2)
    else:
        m = re.search(r"([\w./\\-]+\.(?:py|js|ts|java|go|rb|cpp|c|cs|php|kt|rs)):(\d+)", content)
        if m:
            failure_file, failure_line = m.group(1), m.group(2)
            reasoning.append("failure point via file:line pattern")
        reasoning.append("no recognized stack trace; treated as unstructured")

    paths = re.findall(r"([\w./\\-]+\.(?:py|js|ts|java|go|rb|cpp|c|cs|php|kt|rs))", content)
    affected_paths = list(dict.fromkeys(paths))[:10]
    if failure_file and failure_file not in affected_paths:
        affected_paths.insert(0, failure_file)

    sig = f"{exception_type or 'unknown'}::{ (message or '')[:200]}"
    error_signature = hashlib.sha256(sig.encode()).hexdigest()[:16]

    if language:
        confidence += 0.25
    if exception_type:
        confidence += 0.2
    if failure_file:
        confidence += 0.1
    confidence = round(min(confidence, 0.97), 2)

    return {
        "exception_type": exception_type, "message": message, "language": language,
        "failure_file": failure_file, "failure_line": failure_line,
        "affected_paths": affected_paths, "error_signature": error_signature,
        "confidence": confidence, "reasoning": "; ".join(reasoning),
    }

def sklearn_fallback_search(query: str, corpus: List[Dict[str, str]], k: int) -> List[Dict[str, Any]]:
    if not corpus:
        return []
    texts = [c["text"] for c in corpus]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()
    ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    return [{"index": i, "score": round(float(sims[i]), 4)} for i in ranked if sims[i] > 0.01]

def chroma_query(collection, query_text: str, n_results: int) -> Optional[Dict[str, Any]]:
    try:
        return collection.query(query_texts=[query_text], n_results=n_results)
    except Exception as exc:
        logger.warning("ChromaDB query failed, using scikit-learn fallback: %s", exc)
        return None

def run_root_cause(content: str, triage: Dict, log_result: Dict, historical: List[HistoricalDefect]) -> Dict:
    query = f"{content} {triage.get('component', '')} {log_result.get('exception_type') or ''}"
    evidence = []

    chroma_result = chroma_query(defect_collection, query, 3)
    if chroma_result and chroma_result.get("ids") and chroma_result["ids"][0]:
        by_defect_id = {d.defect_id: d for d in historical}
        for i, doc_id in enumerate(chroma_result["ids"][0]):
            defect_id = doc_id.replace("kb:", "")
            d = by_defect_id.get(defect_id)
            if not d:
                continue
            distance = chroma_result["distances"][0][i]
            similarity = max(0.0, 1.0 - distance)
            evidence.append({"defect_id": d.defect_id, "title": d.title, "similarity": round(similarity, 4),
                              "root_cause": d.root_cause, "component": d.component})
    else:
        corpus = [{"text": f"{d.title} {d.symptom_text} {d.tags or ''}"} for d in historical]
        for m in sklearn_fallback_search(query, corpus, 3):
            d = historical[m["index"]]
            evidence.append({"defect_id": d.defect_id, "title": d.title, "similarity": m["score"],
                              "root_cause": d.root_cause, "component": d.component})

    if evidence and evidence[0]["similarity"] >= 0.2:
        base_confidence = min(0.55 + evidence[0]["similarity"] * 0.5, 0.95)
    elif log_result.get("exception_type"):
        base_confidence = 0.35
    else:
        base_confidence = 0.2

    llm_result = call_ollama_json(
        system_prompt=(
            "You are a root-cause analysis engine for software bugs. Given a bug report, its "
            "log analysis, and retrieved similar historical defects (evidence), propose the most "
            "likely root cause. Respond as JSON with keys: hypothesis (string), confidence (0-1 float)."
        ),
        user_prompt=json.dumps({
            "bug_report": content[:2000], "triage": triage, "log_analysis": log_result, "evidence": evidence,
        }),
    )
    if llm_result and "hypothesis" in llm_result:
        hypothesis = f"[ollama:{OLLAMA_CHAT_MODEL}] " + str(llm_result["hypothesis"])
        confidence = round(float(llm_result.get("confidence", base_confidence)), 2)
    else:
        if evidence and evidence[0]["similarity"] >= 0.2:
            top = evidence[0]
            hypothesis = f"[fallback] Likely same root cause as {top['defect_id']} ({top['title']}): {top['root_cause']}"
        elif log_result.get("exception_type"):
            hypothesis = f"[fallback] No close historical match. Based on '{log_result['exception_type']}' and '{triage.get('component')}', likely a fault in that code path."
        else:
            hypothesis = "[fallback] No close historical match or structured exception; manual investigation recommended."
        confidence = round(base_confidence, 2)

    return {"hypothesis": hypothesis, "confidence": confidence, "evidence": evidence,
            "reasoning": f"retrieved {len(evidence)} candidate(s) via ChromaDB/Sentence-Transformers RAG"}

def run_duplicate_detection(content: str, triage: Dict, log_result: Dict, historical: List[HistoricalDefect]) -> Dict:
    query = f"{content} {triage.get('component', '')} {log_result.get('exception_type') or ''}"
    matches: List[Dict[str, Any]] = []

    for collection, source in ((defect_collection, "historical_kb"), (submission_collection, "prior_submission")):
        result = chroma_query(collection, query, 3)
        if not result or not result.get("ids") or not result["ids"][0]:
            continue
        for i, doc_id in enumerate(result["ids"][0]):
            distance = result["distances"][0][i]
            similarity = max(0.0, 1.0 - distance)
            if similarity < 0.05:
                continue
            meta = result["metadatas"][0][i] or {}
            matches.append({
                "source": source, "id": meta.get("ref_id", doc_id), "title": meta.get("title", ""),
                "similarity": round(similarity, 4), "similarity_pct": round(similarity * 100, 1),
                "resolution_summary": meta.get("resolution_summary") or "No resolution recorded",
                "component": meta.get("component"),
            })

    if not matches:
        corpus = [{"text": f"{d.title} {d.symptom_text} {d.tags or ''}"} for d in historical]
        for m in sklearn_fallback_search(query, corpus, 3):
            d = historical[m["index"]]
            matches.append({"source": "historical_kb", "id": d.defect_id, "title": d.title,
                             "similarity": m["score"], "similarity_pct": round(m["score"] * 100, 1),
                             "resolution_summary": d.resolution_summary, "component": d.component})

    matches.sort(key=lambda m: m["similarity"], reverse=True)
    return {"matches": matches[:3], "reasoning": "searched ChromaDB (historical_defects + bug_submissions)"}

def run_remediation(triage: Dict, log_result: Dict, root_cause: Dict, duplicates: Dict) -> Dict:
    component = triage.get("component", "Unclassified")
    playbook = PLAYBOOKS.get(component, PLAYBOOKS["Unclassified"])
    top_dup = duplicates["matches"][0] if duplicates.get("matches") else None

    llm_result = call_ollama_json(
        system_prompt=(
            "You are a remediation-recommendation engine. Given the triage, root cause, and top "
            "duplicate match, produce actionable remediation. Respond as JSON with keys: "
            "summary (string), steps (array of strings)."
        ),
        user_prompt=json.dumps({
            "triage": triage, "root_cause": root_cause, "top_duplicate": top_dup,
            "component_playbook": playbook,
        }),
    )

    sources = []
    if top_dup:
        sources.append(f"{top_dup['source']}:{top_dup['id']}")
    if root_cause.get("evidence"):
        sources.append(f"root_cause_evidence:{root_cause['evidence'][0]['defect_id']}")
    sources.append(f"best_practice_playbook:{component}")

    if llm_result and "summary" in llm_result and "steps" in llm_result:
        steps = list(dict.fromkeys(list(llm_result["steps"]) + playbook))
        summary = f"[ollama:{OLLAMA_CHAT_MODEL}] " + str(llm_result["summary"])
    else:
        steps_seed = []
        if top_dup and top_dup["similarity"] >= 0.15:
            steps_seed.append(f"Apply fix from {top_dup['source']} {top_dup['id']}: {top_dup['resolution_summary']}")
        if root_cause.get("confidence", 0) >= 0.4:
            steps_seed.append(f"Address root cause: {root_cause['hypothesis']}")
        elif not top_dup:
            steps_seed.append("Root cause confidence low — reproduce and add targeted logging.")
        steps = list(dict.fromkeys(steps_seed + playbook))
        if top_dup:
            summary = f"[fallback] Recommended fix grounded in matching resolved issue ({top_dup['id']}, {top_dup['similarity_pct']}%), supplemented with {component} best practices."
        elif root_cause.get("confidence", 0) >= 0.4:
            summary = f"[fallback] Recommended fix targets hypothesized root cause, supplemented with {component} best practices."
        else:
            summary = f"[fallback] No strong historical match — leading with investigation steps, followed by {component} best practices."

    dup_conf = top_dup["similarity"] if top_dup else 0
    rc_conf = root_cause.get("confidence", 0)
    confidence = round(min(0.9, max(dup_conf, rc_conf) * 0.9 + 0.1), 2)

    return {"summary": summary, "steps": steps, "confidence": confidence, "sources": sources}

def run_orchestration(content: str, db: Session) -> Dict[str, Any]:
    triage = run_triage(content)
    log_result = run_log_analysis(content)
    historical = db.query(HistoricalDefect).all()
    root_cause = run_root_cause(content, triage, log_result, historical)
    duplicates = run_duplicate_detection(content, triage, log_result, historical)
    remediation = run_remediation(triage, log_result, root_cause, duplicates)
    return {
        "triage": triage, "log_analysis": log_result, "root_cause": root_cause,
        "duplicates": duplicates, "remediation": remediation,
        "analyzed_at": datetime.utcnow().isoformat(),
    }

def index_submission_in_chroma(bug: "BugSubmission", triage: Dict, root_cause_hypothesis: str):
    try:
        submission_collection.add(
            ids=[bug.bug_id],
            documents=[f"{bug.title} {bug.content}"],
            metadatas=[{"ref_id": bug.bug_id, "title": bug.title,
                        "resolution_summary": root_cause_hypothesis or "", "component": triage.get("component", "")}],
        )
    except Exception as exc:
        logger.warning("Failed to index submission %s in ChromaDB: %s", bug.bug_id, exc)

def seed_historical_defects(db: Session):
    if db.query(HistoricalDefect).count() == 0:
        for d in SEED_HISTORICAL_DEFECTS:
            db.add(HistoricalDefect(id=str(uuid.uuid4()), **d))
        db.commit()
        logger.info("Database seeded with %d historical defects.", len(SEED_HISTORICAL_DEFECTS))

    try:
        existing_ids = set(defect_collection.get()["ids"])
    except Exception:
        existing_ids = set()
    to_add = [d for d in SEED_HISTORICAL_DEFECTS if f"kb:{d['defect_id']}" not in existing_ids]
    if to_add:
        try:
            defect_collection.add(
                ids=[f"kb:{d['defect_id']}" for d in to_add],
                documents=[f"{d['title']} {d['symptom_text']} {d['tags']}" for d in to_add],
                metadatas=[{"ref_id": d["defect_id"], "title": d["title"],
                            "resolution_summary": d["resolution_summary"], "component": d["component"]} for d in to_add],
            )
            logger.info("ChromaDB seeded with %d historical defect embeddings.", len(to_add))
        except Exception as exc:
            logger.warning("Failed to seed ChromaDB: %s", exc)

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema recreated.")
    
    db = SessionLocal()
    try:
        seed_historical_defects(db)
    finally:
        db.close()
    yield
    logger.info("Shutting down.")

# ------------------------------------------------------------
#  FastAPI App
# ------------------------------------------------------------
app = FastAPI(title="Intelligent Bug Diagnosis API", version="3.0.0", lifespan=lifespan)

# CORS (still needed for API calls, even if same-origin, but harmless)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                    allow_methods=["*"], allow_headers=["*"])

# ------------------------------------------------------------
#  API Routes (defined BEFORE static mount)
# ------------------------------------------------------------

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "traceback": traceback.format_exc()
        }
    )

def _next_bug_number(db: Session) -> int:
    max_num = db.query(sa_func.max(BugSubmission.bug_number)).scalar()
    return (max_num or 0) + 1

def _format_bug_id(num: int) -> str:
    return f"BUG-{num:05d}"

def _persist_analysis(db: Session, bug: BugSubmission, combined: Dict) -> AgentAnalysis:
    t, l, r, d, rem = (combined["triage"], combined["log_analysis"], combined["root_cause"],
                        combined["duplicates"], combined["remediation"])
    record = AgentAnalysis(
        bug_id=bug.id,
        triage_severity=t["severity"], triage_priority=t["priority"], triage_type=t["type"],
        triage_component=t["component"], triage_confidence=t["confidence"], triage_reasoning=t["reasoning"],
        log_exception_type=l["exception_type"], log_message=l["message"], log_language=l["language"],
        log_failure_file=l["failure_file"], log_failure_line=l["failure_line"],
        log_affected_paths=json.dumps(l["affected_paths"]), log_error_signature=l["error_signature"],
        log_confidence=l["confidence"], log_reasoning=l["reasoning"],
        root_cause_hypothesis=r["hypothesis"], root_cause_confidence=r["confidence"],
        root_cause_evidence=json.dumps(r["evidence"]),
        duplicate_matches=json.dumps(d["matches"]),
        remediation_summary=rem["summary"], remediation_steps=json.dumps(rem["steps"]),
        remediation_confidence=rem["confidence"], remediation_sources=json.dumps(rem["sources"]),
        combined_json=json.dumps({"bug_id": bug.bug_id, **combined}),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def _serialize_analysis(record: AgentAnalysis, bug_id: str) -> Dict:
    return {
        "bug_id": bug_id,
        "triage": {"severity": record.triage_severity, "priority": record.triage_priority,
                   "type": record.triage_type, "component": record.triage_component,
                   "confidence": record.triage_confidence, "reasoning": record.triage_reasoning},
        "log_analysis": {"exception_type": record.log_exception_type, "message": record.log_message,
                          "language": record.log_language, "failure_file": record.log_failure_file,
                          "failure_line": record.log_failure_line,
                          "affected_paths": json.loads(record.log_affected_paths or "[]"),
                          "error_signature": record.log_error_signature,
                          "confidence": record.log_confidence, "reasoning": record.log_reasoning},
        "root_cause": {"hypothesis": record.root_cause_hypothesis, "confidence": record.root_cause_confidence,
                        "evidence": json.loads(record.root_cause_evidence or "[]")},
        "duplicates": {"matches": json.loads(record.duplicate_matches or "[]")},
        "remediation": {"summary": record.remediation_summary, "steps": json.loads(record.remediation_steps or "[]"),
                         "confidence": record.remediation_confidence,
                         "sources": json.loads(record.remediation_sources or "[]")},
        "analyzed_at": record.created_at,
    }

@app.post("/api/bugs/paste", response_model=BugSubmissionCreatedResponse)
def submit_pasted_bug(title: str = Form(...), content: str = Form(...), db: Session = Depends(get_db)):
    if not content.strip():
        raise HTTPException(400, "Content cannot be empty")
    combined = run_orchestration(content, db)
    bug_number = _next_bug_number(db)
    bug = BugSubmission(
        bug_number=bug_number, bug_id=_format_bug_id(bug_number), title=title.strip(),
        severity=Severity(combined["triage"]["severity"]), source_type=SourceType.PASTE,
        content=content, content_length=str(len(content)),
    )
    db.add(bug)
    db.commit()
    db.refresh(bug)
    record = _persist_analysis(db, bug, combined)
    index_submission_in_chroma(bug, combined["triage"], combined["root_cause"]["hypothesis"])
    return {"id": bug.id, "bug_id": bug.bug_id, "message": "Bug analyzed successfully.",
            "analysis": _serialize_analysis(record, bug.bug_id)}

@app.post("/api/bugs/upload", response_model=BugSubmissionCreatedResponse)
async def submit_uploaded_bug(title: str = Form(...), files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    if not files:
        raise HTTPException(400, "At least one file required")
    file = files[0]
    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except Exception:
        content = f"[Binary file: {file.filename}]"
    combined = run_orchestration(content, db)
    bug_number = _next_bug_number(db)
    bug = BugSubmission(
        bug_number=bug_number, bug_id=_format_bug_id(bug_number), title=title.strip(),
        severity=Severity(combined["triage"]["severity"]), source_type=SourceType.UPLOAD,
        original_filename=file.filename, file_type=file.filename.split(".")[-1],
        content=content, content_length=str(len(content)),
    )
    db.add(bug)
    db.commit()
    db.refresh(bug)
    record = _persist_analysis(db, bug, combined)
    index_submission_in_chroma(bug, combined["triage"], combined["root_cause"]["hypothesis"])
    return {"id": bug.id, "bug_id": bug.bug_id, "message": "File uploaded and analyzed.",
            "analysis": _serialize_analysis(record, bug.bug_id)}

@app.get("/api/bugs", response_model=List[BugSubmissionResponse])
def list_bugs(db: Session = Depends(get_db)):
    return db.query(BugSubmission).order_by(desc(BugSubmission.created_at)).all()

@app.get("/api/bugs/{bug_id}/analysis")
def get_analysis(bug_id: str, db: Session = Depends(get_db)):
    bug = db.query(BugSubmission).filter(
        (BugSubmission.id == bug_id) | (BugSubmission.bug_id == bug_id)
    ).first()
    if not bug:
        raise HTTPException(404, "Bug not found")
    if not bug.analysis:
        raise HTTPException(404, "No analysis found")
    return _serialize_analysis(bug.analysis, bug.bug_id)

@app.delete("/api/bugs/{bug_id}")
def delete_bug(bug_id: str, db: Session = Depends(get_db)):
    bug = db.query(BugSubmission).filter(
        (BugSubmission.id == bug_id) | (BugSubmission.bug_id == bug_id)
    ).first()
    if not bug:
        raise HTTPException(404, "Bug not found")
    db.delete(bug)
    db.commit()
    try:
        submission_collection.delete(ids=[bug.bug_id])
    except Exception:
        pass
    return {"message": "Deleted"}

@app.put("/api/bugs/{bug_id}/resolve")
def resolve_bug(bug_id: str, notes: str = Form(...), db: Session = Depends(get_db)):
    bug = db.query(BugSubmission).filter(
        (BugSubmission.id == bug_id) | (BugSubmission.bug_id == bug_id)
    ).first()
    if not bug:
        raise HTTPException(404, "Bug not found")
    bug.is_resolved = True
    bug.resolved_at = datetime.utcnow()
    bug.resolution_notes = notes
    bug.kb_defect_id = f"KB-{bug.bug_number:05d}"

    existing = db.query(HistoricalDefect).filter_by(defect_id=bug.kb_defect_id).first()
    if not existing and bug.analysis:
        new_defect = HistoricalDefect(
            id=str(uuid.uuid4()), defect_id=bug.kb_defect_id, title=bug.title,
            symptom_text=bug.content[:500], root_cause=bug.analysis.root_cause_hypothesis or notes,
            resolution_summary=notes, component=bug.analysis.triage_component or "Unclassified", tags="",
        )
        db.add(new_defect)
        db.commit()
        try:
            defect_collection.add(
                ids=[f"kb:{new_defect.defect_id}"],
                documents=[f"{new_defect.title} {new_defect.symptom_text}"],
                metadatas=[{"ref_id": new_defect.defect_id, "title": new_defect.title,
                            "resolution_summary": new_defect.resolution_summary,
                            "component": new_defect.component}],
            )
        except Exception as exc:
            logger.warning("Failed to index resolved defect in ChromaDB: %s", exc)
    else:
        db.commit()
    return {"message": "Bug resolved and added to knowledge base"}

# ------------------------------------------------------------
#  Static files (FRONTEND) – mounted LAST so API routes take priority
# ------------------------------------------------------------
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info("Starting Bug Diagnosis API on http://localhost:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)