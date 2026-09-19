# HealthAgentX — Collaborative Multi-Agent Healthcare Assistant

A privacy-preserving multi-agent AI system that analyzes patient symptoms,
medical history, lab reports, and medications entirely on local
infrastructure — no patient data ever leaves the machine. Built with
**LangGraph** for agent orchestration, **RAGWire** (this project's local
retrieval-augmented-generation subsystem) for evidence grounding, and a
**local LLM** (served via [Ollama](https://ollama.com)) for reasoning.

## Architecture

```
Coordinator ─▶ Symptom Analysis ─▶ Medical Report ─▶ Medical RAG (RAGWire)
                                                             │
                          ┌──────────────────────────────────┘
                          ▼
                     Medication ─▶ Risk Assessment ─▶ Generator ─▶ Critic/Verifier
                                                                       │
                                                              ┌────────┴────────┐
                                                         passed?           failed?
                                                              │                  │
                                                          Finalize      Loop Controller
                                                                              │
                                                              (loop back to Medical RAG
                                                               with follow-up query,
                                                               up to MAX_VERIFICATION_LOOPS)
```

**Loop Engineering**: `Analyze → Retrieve → Generate → Critique → Verify →
Re-analyze/Retrieve → Finalize`. The Critic/Verifier Agent checks the draft
assessment for contradictions, unsupported claims, and hallucinations
(evidence-grounding is checked with a hard heuristic gate that runs
regardless of LLM availability, plus an optional LLM self-critique pass).
If verification fails, the Loop Controller routes back to the Medical RAG
Agent with a targeted follow-up query and the pipeline re-generates and
re-critiques — up to `MAX_VERIFICATION_LOOPS` times — before finalizing a
best-effort result with outstanding issues recorded transparently.

### Agents (`backend/app/agents/`)

| Agent | Responsibility |
|---|---|
| Coordinator | Plans the workflow, seeds the initial retrieval query |
| Symptom Analysis | Extracts clinical findings & candidate conditions from symptoms/history |
| Medical Report | Parses lab values from uploaded documents, flags abnormal results |
| Medical RAG (RAGWire) | Retrieves supporting medical knowledge; re-runs on each loop |
| Medication | Checks reported medications for known interactions |
| Risk Assessment | Synthesizes red flags / urgency from all upstream findings |
| Generator | Synthesizes the final structured, evidence-cited assessment |
| Critic/Verifier | Grounding + contradiction/hallucination checks |
| Loop Controller | Decides retry vs. finalize (Loop Engineering) |

### RAGWire (`backend/app/ragwire/`)

The local RAG subsystem: `document_loader.py` (PyMuPDF + pytesseract OCR +
pdfplumber fallback) → `embeddings.py` (local sentence-transformers, with an
offline hashing-vectorizer fallback) → `vector_store.py` (persistent
ChromaDB) → `reranker.py` (local cross-encoder, with a lexical-overlap
fallback) → `retriever.py` (the `RagWire` retrieve-and-rerank facade).

Every "local model unavailable" path (no internet to fetch a
sentence-transformers/cross-encoder model on first run, or no local Ollama
server) automatically falls back to a dependency-light heuristic so the full
pipeline still runs end-to-end offline — this is a resilience feature, not
a design shortcut: install/cache the real local models for full retrieval
and generation quality.

## Tech Stack

| Component | Choice |
|---|---|
| LLM | Local, via Ollama (`OLLAMA_MODEL`, default `llama3.2`) |
| RAG | RAGWire (this repo's `backend/app/ragwire/`) |
| Agent framework | LangGraph |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector store | ChromaDB (persistent, local) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Document processing | PyMuPDF + pdfplumber |
| OCR | pytesseract (Tesseract) |
| Backend | FastAPI |
| Database | SQLite (via SQLAlchemy) |
| Frontend | Streamlit |

## Setup

### 1. Install dependencies
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. (Optional but recommended) Install a local LLM
```bash
# https://ollama.com
ollama pull llama3.2
ollama serve
```
Without a running Ollama server, every agent automatically falls back to
transparent, deterministic heuristics so the app still works for demos.

### 3. Configure environment
```bash
cp .env.example .env
# edit .env if you want a different model / paths
```

### 4. Ingest the local medical knowledge base into RAGWire
```bash
python scripts/ingest_knowledge_base.py --reset
```

### 5. Run the backend
```bash
cd backend
uvicorn app.main:app --reload
```
API docs at http://localhost:8000/docs.

### 6. Run the frontend
```bash
BACKEND_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
```

### Quick end-to-end demo (no server needed)
```bash
python scripts/run_demo.py
```
Prints the final structured assessment, critique, and the full agent
execution / Loop Engineering trace for a sample case.

## API

- `GET /api/health` — service + RAGWire index status
- `POST /api/knowledge/ingest?reset=false` — (re)build the RAGWire index
- `POST /api/cases/analyze` — multipart form: `symptoms`, `history`,
  `medications` (comma-separated), `files` (PDF/image lab reports) →
  returns the structured assessment, critique, evidence, and trace
- `GET /api/cases/{case_id}` — fetch a previously stored case

## Output format

Each analysis returns:

**Patient Summary → Clinical Findings → Possible Conditions → Report
Insights → Medication Considerations → Risk Indicators → Supporting
Evidence → Suggested Next Steps**

...alongside a full **agent execution & verification trace** showing every
agent's contribution and how Loop Engineering corrected or improved the
output across iterations.

## Tests

```bash
cd backend
pytest tests/ -v
```

## Disclaimer

HealthAgentX is a clinical decision-support prototype for hackathon/research
purposes. It never issues definitive diagnoses and is not a substitute for
professional medical judgment.
