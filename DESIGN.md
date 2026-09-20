# Architecture & Design Document: Kestrel Research Assistant

## 1. System Overview & Objectives
The **Kestrel Research Assistant** is an agentic Retrieval-Augmented Generation (RAG) system engineered to answer questions regarding Kestrel Labs' internal product analytics platform. It retrieves evidence from 154 documents across 7 categories, enforces date precedence during document conflicts, resolves pronouns in follow-up conversations, and verifies every claim before delivering an answer.

---

## 2. One-Page Architecture Flowchart

```text
               User Prompt & Conversation History
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │             ROUTER AGENT             │
            │  • Pronoun resolution & rewriting    │
            │  • Query classification & sub-queries│
            └──────────────────┬───────────────────┘
                               │
             ┌─────────────────┴─────────────────┐
             │ [needs_retrieval == False]        │ [needs_retrieval == True]
             ▼                                   ▼
    ┌──────────────────┐               ┌──────────────────┐
    │ UNSUPPORTED NODE │               │ RETRIEVER AGENT  │
    │ Direct greeting  │               │ Dynamic k (3-6)  │
    │ or out-of-domain │               │ ChromaDB search  │
    └────────┬─────────┘               └────────┬─────────┘
             │                                  │
             │                                  ▼
             │                         ┌──────────────────┐
             │                         │SYNTHESIZER AGENT │
             │                         │Grounded drafting │
             │                         │Conflict handling │
             │                         └────────┬─────────┘
             │                                  │
             │                                  ▼
             │                         ┌──────────────────┐
             │                         │  VERIFIER AGENT  │
             │                         │Claim-by-claim    │
             │                         │audits & revision │
             │                         └────────┬─────────┘
             │                                  │
             │                    ┌─────────────┴─────────────┐
             │                    ▼                           ▼
             │             [all supported /            [unsupported claims &
             │              conflicts resolved]         retry_count < 1]
             │                    │                           │
             │                    │                           ▼ (Loop back)
             │                    │                    ┌──────────────────┐
             │                    │                    │ RETRIEVER RETRY  │
             │                    │                    └──────────────────┘
             │                    │
             ▼                    ▼
     [Final Verified Answer with [chunk_id: title] Citations]
             │
             ├──► Streamlit UI (real-time progress + claim audit expanders)
             └──► LangSmith Trace (end-to-end telemetry)
```

---

## 3. The 4 Agents & Hand-Off Protocol

| Agent | Responsibility | Tool & I/O |
|---|---|---|
| **Router** | Analyzes user messages and full conversational history. Resolves ambiguous pronouns (*"What about on Growth?"* $\rightarrow$ *"How many Beacons can a Growth project have?"*), classifies query type (`single_hop`, `multi_hop`, `conflicting`, `unsupported`, `follow_up`, `greeting`), and decomposes multi-hop queries into sub-queries. | **Output:** Pydantic `RouteDecision` |
| **Retriever** | Performs semantic retrieval over local ChromaDB using local SentenceTransformer embeddings (`all-MiniLM-L6-v2`). Executes dynamic $k$ retrieval ($k=3$ for single-hop, $k=6$ with sub-query deduplication for multi-hop, $k=5$ for conflicting). | **Tool:** `search_corpus(query, k)` |
| **Synthesizer** | Generates an initial draft answer grounded exclusively in the retrieved chunks. Enforces metadata precedence: when documents conflict, the newer `published` date overrides the older one. Appends citations formatted strictly as `[chunk_id: title]`. | **LLM:** Groq (`openai/gpt-oss-120b` / `llama-3.3-70b-versatile`) |
| **Verifier (Critic)** | Inspects each factual claim against retrieved context and issues structured verdicts: `supported`, `partially_supported`, `conflicting_evidence`, or `insufficient_evidence`. Generates a surgical `revised_answer` qualifying unproven statements. | **Output:** Pydantic `VerifierOutput` |

---

## 4. Orchestration Choice: Why LangGraph?

We chose **LangGraph** over simple linear chains or autonomous loops for the following technical reasons:
1. **Explicit Shared State (`AgentState`)**: Agents communicate through a typed schema tracking `messages`, `current_query`, `retrieved_chunks`, `draft_answer`, `verifier_verdicts`, and `retry_count`.
2. **Conditional Branching & Bypasses**: Fast-paths bypass the vector store for conversational greetings, reducing latency to ~1.5s.
3. **Controlled Retry Loop**: If the Verifier identifies missing evidence, LangGraph dynamically routes back to the Retriever (bounded to max 1 retry), preventing hallucinated answers without risking infinite loops.
4. **First-Class LangSmith Observability**: LangGraph natively exposes node-level inputs, outputs, latencies, and metadata to LangSmith.
