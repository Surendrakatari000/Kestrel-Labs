# Kestrel Labs Multi-Agent Research Assistant

An enterprise-grade, evidence-grounded research assistant built with **LangGraph**, **Groq**, and **ChromaDB** that navigates Kestrel Labs' internal documentation. It retrieves verified evidence, adjudicates conflicting policies by publication date, detects unsupported queries with honest refusals, fact-checks every factual claim through a critic verifier, and visibly displays its full reasoning workflow with clickable citations.

Built for **Take-Home Assignment 3: Multi-Agent Research Assistant**.

---

## 1. System Architecture

```
                      ┌──────────────────────┐
                      │   User Query / Turn  │
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │     Router Agent     │ ◄── Resolves pronouns from history &
                      └──────────┬───────────┘     classifies into 5 query types
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │   Retriever Agent    │ ◄── Dynamic-k vector search over
                      └──────────┬───────────┘     local ChromaDB embeddings
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │  Synthesizer Agent   │ ◄── Evidence-grounded draft with
                      └──────────┬───────────┘     dual citations for conflicting dates
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │    Verifier Agent    │ ◄── Fact-checks claims; surgically revises;
                      └──────────┬───────────┘     preserves [chunk_id: title] citations
                                 │
                    [Claim verdict supported?]
                                 │
                   YES ──────────┴────────── NO (retry_count == 0)
                    │                                 │
                    │                                 ▼
                    │                     ┌──────────────────────┐
                    │                     │    Retriever Retry   │ (k+2 chunks)
                    │                     └──────────┬───────────┘
                    │                                │
                    │                                └──► Synthesizer
                    ▼
       ┌──────────────────────────────┐
       │ Verified Answer + Citations  │
       └──────────────────────────────┘
```

### The Multi-Agent Team

| Agent | Responsibility | Hand-off & Tool Capabilities |
| :--- | :--- | :--- |
| **Router Agent** | • Multi-turn pronoun resolution<br>• Query intent classification<br>• Query decomposition for multi-hop | Ingests `messages` state history; outputs a clean standalone query and assigns query type (`single_hop`, `multi_hop`, `conflicting`, `unsupported`, `follow_up`). |
| **Retriever Agent** | • Targeted semantic retrieval<br>• Metadata-rich document ingestion | Queries local ChromaDB with **dynamic $k$** (3 to 6 chunks depending on query complexity). Uses local sentence-transformers embeddings. |
| **Synthesizer Agent** | • Evidence-grounded drafting<br>• Conflict resolution by date precedence | Ingests chunks with `published` dates. Strictly cites sources as `[chunk_id: Document Title]`. For conflicting policies, it cites *both* older and newer documents. |
| **Verifier Agent (Critic)** | • Claim-by-claim audit<br>• Surgical factual revision<br>• Reapplication of citation tags | Verifies claims against context: `supported`, `partially_supported`, `conflicting_evidence`, or `insufficient_evidence`. If claims are unbacked, triggers a targeted re-retrieval loop (max 1 retry). |

---

## 2. Tech Stack & Requirements Compliance

- **Orchestration**: **LangGraph** (`StateGraph` with conditional routing and dynamic retry loop).
- **LLM / Generation**: **Groq API** (`qwen/qwen3.8-27b` with automatic fallback to `openai/gpt-oss-20b`). Output capped, with Tenacity exponential backoff on HTTP 429.
- **Embedding Model**: **Local model** `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace (runs 100% locally; zero hosted embedding API dependency per Section 4).
- **Vector Store**: **ChromaDB** (`chromadb.PersistentClient`, auto-indexed from root `corpus.jsonl`).
- **Observability**: **LangSmith** (`LANGCHAIN_TRACING_V2=true`) with dataset logging and custom feedback metrics.
- **Frontend**: **Streamlit** multi-turn chat UI with live agent step tracking (`st.status`), response streaming, and expandable source viewers.

---

## 3. Quick Start (Single-Command Execution)

### Step 1: Clone and Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Surendrakatari000/Kestrel-Labs.git
cd Kestrel-Labs

# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

Create your `.env` file from `.env.example`:

```bash
# Windows PowerShell:
Copy-Item .env.example .env

# Linux / macOS:
cp .env.example .env
```

Edit `.env` with your API keys:
```ini
GROQ_API_KEY=gsk_...
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=kestrel-research-assistant
GROQ_MODEL=qwen/qwen3.8-27b
```

### Step 3: Run the Full Evaluation Suite (Single Command)

Per Section 4 of the assignment guidelines, the complete evaluation suite executes with a single command:

```bash
python run_evals.py
```

> **Automatic Ingestion:** If `chroma_db/` does not already exist, the script automatically parses and indexes `corpus.jsonl` into ChromaDB before executing evaluations.

### Step 4: Launch the Interactive Web Frontend

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 4. Evaluation Suite & Verified Benchmark Metrics

The evaluation suite (`run_evals.py`) runs 15 original questions probing all hard cases specified in the assignment prompt. Every question is logged as a LangSmith dataset example and evaluated against **6 defensible metrics**.

### Verified Evaluation Summary

| Metric | Initial Baseline | Final Improved System | Improvement |
| :--- | :---: | :---: | :---: |
| **Total Questions / Errors** | 15 / 0 errors | 15 / 0 errors | **100% Reliable** |
| **Avg Citation Precision** | 36.7% | **82.2%** | **+124%** |
| **Avg Citation Recall** | 33.3% | **64.4%** | **+93%** |
| **Avg Faithfulness** | 100.0% | **100.0%** | **100% Sustained** |
| **Avg Answer Relevance** | 61.3% | **98.7%** | **+61%** |
| **Avg Answer Correctness** | 54.0% | **93.3%** | **+73%** |
| **Avg Latency (per query)** | 42.1s | **19.5s** | **54% Faster** |
| **Total Wall-Clock Time** | 893.2s (~14.9 min) | **633.6s** (~10.5 min) | **29% Time Saved** |

### Breakdown by Question Type

- **Conflicting Questions ($n=3$):** **100% Citation Precision**, **88.9% Citation Recall**, **100% Answer Correctness**, **100% Faithfulness**. Synthesizer cites both older and newer documents with date-based explanations.
- **Single-Hop Questions ($n=4$):** **100% Citation Precision**, **100% Answer Correctness**, **100% Faithfulness**.
- **Multi-Hop Questions ($n=4$):** **100% Answer Correctness**, **100% Faithfulness**. Correct cross-document reasoning linking post-mortems, specs, and release notes.
- **Unsupported Questions ($n=2$):** **100% Precision**, **100% Recall**, **100% Correctness**. Clear, polite refusals with 0 hallucinated citations.
- **Follow-Up Questions ($n=2$):** **100% Multi-Turn Flow**. Router pronoun resolution rewrites vague queries (`"What about on the Growth plan?"` $\to$ `"How many Beacons can a Growth project have?"`).

---

## 5. What Was Improved (Evaluation-Driven Changes)

Detailed before-and-after analysis is documented in [`results/improvement.md`](file:///c:/Users/suren/Desktop/projects/internship/MINT/ccoding%20assignment%20-1/results/improvement.md):

1. **Citation Stripping in Verifier Fixed:** When the Verifier edited text for conflicting claims, it stripped out citation brackets. We implemented deterministic citation preservation and reapplication in `src/agents/verifier.py`, raising Conflicting Citation Precision from **0% to 100%**.
2. **Pronoun Resolution in Router:** Implemented `RouteDecision` structured output in `src/agents/router.py` to rewrite multi-turn follow-ups, resolving pronoun ambiguity before retrieval.
3. **Dual Citation for Conflicting Policies:** Prompted the Synthesizer to compare publication dates and cite both superseded and authoritative documents.
4. **Citation Normalization:** Created `src/utils/citations.py` to deterministically map raw IDs to `[chunk_id: Document Title]`.
5. **Rate-Limit Resilience:** Integrated Tenacity exponential backoff and multi-model quota fallback in `src/agents/utils.py`.

---

## 6. Observability & LangSmith Project Access

Every single execution from both `run_evals.py` and the Streamlit UI is traced to LangSmith under project `kestrel-research-assistant`.

- 🔗 **Public LangSmith Traces & Dataset Link:**  
  [https://smith.langchain.com/public/193ee189-07b8-4629-9319-eebd5b780402/d](https://smith.langchain.com/public/193ee189-07b8-4629-9319-eebd5b780402/d)  
  *(Anyone can click this link to directly inspect the evaluation dataset, all 15 question runs, multi-agent trace steps, and attached feedback metrics without needing to log in).*

- **Reviewer Access Email:** As required by Section 3.4 of the assignment, project access has also been provisioned for:
  ```
  radialpulse@nxtwave.co.in
  ```

- **LangSmith Traces Include:**
  - Full router classification and standalone query rewriting.
  - ChromaDB vector retrieval similarity scores and ranked chunk IDs.
  - Synthesizer drafting step with prompt metadata.
  - Verifier claim-by-claim verdicts and final surgical edits.
  - Attached feedback metrics (Retrieval Recall, Citation Precision/Recall, Faithfulness, Relevance, Correctness).

---

## 7. Submission Directory Structure

```
├── corpus.jsonl               # Read-only corpus (154 chunks, SHA verified)
├── .env.example               # Environment variables template
├── requirements.txt           # All Python dependencies
├── README.md                  # System overview, setup, and run instructions
├── DESIGN.md                  # Architecture diagram, agent roles, handoffs, and tradeoffs
├── REFLECTION.md              # Engineering reflection, costs, latency, and future roadmap
├── app.py                     # Streamlit frontend with live agent tracing & streaming
├── run_evals.py               # Evaluation runner, LangSmith logger, and metric calculator
├── src/
│   ├── __init__.py
│   ├── state.py               # LangGraph AgentState typed schema
│   ├── vectorstore.py         # ChromaDB client & sentence-transformers embeddings
│   ├── graph.py               # LangGraph workflow, conditional edges, retry cycle
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── router.py          # Pronoun resolution & query intent classifier
│   │   ├── retriever.py       # Dynamic-k vector retriever tool
│   │   ├── synthesizer.py     # Grounded drafting with date precedence
│   │   ├── verifier.py        # Fact-checking critic & surgical reviser
│   │   └── utils.py           # LLM client with exponential backoff & failover
│   └── utils/
│       ├── __init__.py
│       └── citations.py       # Deterministic citation normalizer ([chunk_id: title])
├── results/
│   ├── eval_questions.jsonl   # 15 original benchmark questions across 5 categories
│   ├── eval_results.jsonl     # Per-question outputs, scores, and LangSmith URLs
│   ├── metrics_summary.json   # Aggregate metrics, token usage, and category breakdown
│   └── improvement.md         # Narrative improvement report with before/after data
├── tests/
│   └── test_citations.py      # Unit tests for citation normalization and extraction
└── chroma_db/                 # Persistent local vector store (auto-generated)
```
