# HealthAgentX — Collaborative Multi-Agent Healthcare Assistant

A privacy-preserving multi-agent AI system that analyzes patient symptoms,
medical history, lab reports, and medications entirely on local
infrastructure — no patient data ever leaves the machine. Built with
**LangGraph** for agent orchestration, **[RAGWire](https://github.com/laxmimerit/ragwire)**
for the full retrieval-augmented-generation stack, and a **local LLM**
(served via [Ollama](https://ollama.com)) for reasoning.

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

### RAGWire (`backend/app/ragwire/`, configured by `ragwire_config.yaml`)

The RAG stack is the [`ragwire`](https://pypi.org/project/ragwire/) library
itself (`pip install ragwire`), not something hand-rolled here — it already
provides document loading/chunking, local embeddings, a local Qdrant vector
store, and local cross-encoder reranking end to end, all driven by
`ragwire_config.yaml`. `backend/app/ragwire/` is a thin integration layer:

- `retriever.py` builds the shared `ragwire.RAGWire` pipeline (once) from
  that config and exposes `RagWire().retrieve(query)`, translating RAGWire's
  LangChain `Document` results into the plain dicts the rest of the app uses.
- `ingest.py` calls the pipeline's own `ingest_directory()` to (re)build the
  index from `data/knowledge_base/`.

The vector store runs as **embedded local Qdrant** (`vectorstore.url` is a
plain disk path, no Docker/server needed) — set it to an `http(s)://` URL in
`.env` (`QDRANT_URL`) to point at a real Qdrant server instead, which is
required if the API server and the ingestion CLI need to run at the same
time (local Qdrant storage allows exactly one reader/writer process).

Uploaded **patient** documents (lab reports) are handled separately, in
`backend/app/documents/loader.py` (PyMuPDF + pytesseract OCR + pdfplumber
fallback) — deliberately *not* run through RAGWire's ingestion, since a
patient's lab report is transient, case-specific PHI, not shared medical
knowledge that should join the persistent, cross-case knowledge base.

## Tech Stack

| Component | Choice |
|---|---|
| LLM | Local, via Ollama (`OLLAMA_MODEL`, default `llama3.2`) |
| RAG | [RAGWire](https://github.com/laxmimerit/ragwire) (`pip install ragwire`) |
| Agent framework | LangGraph |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via RAGWire (local) |
| Vector store | Qdrant, embedded/local by default, via RAGWire |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` via RAGWire (local) |
| Document processing (patient uploads) | PyMuPDF + pdfplumber |
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

### 2. Install a local LLM
```bash
# https://ollama.com
ollama pull llama3.2
ollama serve
```
The reasoning agents (Symptom Analysis, Medication, Risk Assessment,
Generator, Critic) each fall back to transparent, deterministic heuristics
if Ollama isn't reachable, so the multi-agent pipeline still runs without it
for demos. RAGWire itself is less forgiving: it needs Ollama for
per-document metadata extraction at ingest time (a document that fails
extraction is still ingested and stays searchable, just without metadata
filters), and its embedding/reranker models are downloaded from HuggingFace
on first use — first run needs internet access once, then everything is
fully local and cached.

### 3. Configure environment
```bash
cp .env.example .env
# edit .env if you want a different model / paths
```

### 4. Ingest the local medical knowledge base into RAGWire
```bash
python scripts/ingest_knowledge_base.py --reset
```
This downloads the embedding/reranker models on first run (cached after
that) and builds the local Qdrant index from `data/knowledge_base/`.

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
Tests that exercise retrieval build the real RAGWire pipeline (downloading
its embedding/reranker models on first run) and skip with a clear reason if
that's not possible in the current environment (no internet, no cached
models).

## Disclaimer

HealthAgentX is a clinical decision-support prototype for hackathon/research
purposes. It never issues definitive diagnoses and is not a substitute for
professional medical judgment.
