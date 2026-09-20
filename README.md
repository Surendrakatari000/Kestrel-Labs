# Kestrel Labs Research Assistant

A multi-agent RAG (Retrieval-Augmented Generation) assistant that answers questions about Kestrel Labs' internal documentation with grounded citations and fact-checking.

## Architecture

```
User ─► Router ─► Retriever ─► Synthesizer ─► Verifier ─► Answer
                       ▲                          │
                       └──── retry (max 1) ◄──────┘
```

**Four agents**, orchestrated with **LangGraph**:

| Agent | Role |
|---|---|
| **Router** | Resolves pronouns, classifies query type, decomposes multi-hop queries |
| **Retriever** | Searches ChromaDB with dynamic k (3–6 chunks) based on query type |
| **Synthesizer** | Drafts grounded answers with `[chunk_id: title]` citations |
| **Verifier** | Fact-checks every claim, issues verdicts, rewrites if needed |

## Tech Stack

- **LLM**: Groq (`llama-3.3-70b-versatile`) with tenacity retry on 429
- **Embeddings**: Local `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: ChromaDB (persistent, local)
- **Orchestration**: LangGraph (shared state, conditional edges, retry loop)
- **UI**: Streamlit
- **Observability**: LangSmith

## Quick Start

### 1. Setup Environment
```bash
# Clone and enter the project
git clone https://github.com/Surendrakatari000/Kestrel-Labs.git
cd Kestrel-Labs

# Create & activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your actual keys:
#   GROQ_API_KEY=gsk_...
#   LANGCHAIN_API_KEY=lsv2_...
```

### 2. Single-Command End-to-End Run
Per the assignment specification, after setting `.env`, the entire system runs with a single command:
```bash
python run_evals.py
```
> **Note:** You do **not** need to manually ingest the corpus first. `run_evals.py` and `app.py` automatically detect if `chroma_db/` exists, and will auto-ingest `corpus.jsonl` on the first run.

### 3. Launch Interactive Chat UI
```bash
streamlit run app.py
```

## Evaluation

The evaluation suite (`run_evals.py`) tests 15 questions across 5 categories:

| Category | Count | Tests |
|---|---|---|
| **single_hop** | 4 | Direct factual lookup |
| **multi_hop** | 4 | Combining info from multiple documents |
| **conflicting** | 3 | Metadata-based version precedence |
| **unsupported** | 2 | Honest "I don't know" responses |
| **follow_up** | 2 | Multi-turn pronoun resolution |

Results are written to:
- `results/eval_results.jsonl` — per-question outputs (matching exact required schema)
- `results/metrics_summary.json` — aggregate scores, latency, models, and type breakdowns
- LangSmith traces — full agent execution logs and traces

## Observability & LangSmith Sharing

All runs from both the evaluation suite and the Streamlit UI are traced to LangSmith (`LANGCHAIN_PROJECT=kestrel-research-assistant`). Reviewers can inspect the full agent workflow: from router classification, to semantic retrieval, to synthesis, to fact-checking verification.
- **Reviewer access:** Per the assignment instructions, project access should be shared with `radialpulse@nxtwave.co.in`.

## File Structure

```
├── corpus.jsonl          # Read-only knowledge base (154 chunks)
├── .env.example          # Environment template
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── DESIGN.md             # Architecture diagram, agent roles & handoffs
├── REFLECTION.md         # Engineering reflection, trade-offs & latency
├── app.py                # Streamlit frontend
├── run_evals.py          # Evaluation runner
├── src/
│   ├── __init__.py
│   ├── state.py          # LangGraph AgentState
│   ├── vectorstore.py    # ChromaDB + local embeddings
│   ├── agents.py         # Router, Retriever, Synthesizer, Verifier
│   └── graph.py          # LangGraph workflow + retry loop
├── results/
│   ├── eval_questions.jsonl
│   ├── eval_results.jsonl
│   ├── metrics_summary.json
│   └── improvement.md
└── chroma_db/            # Auto-generated vector store
```
