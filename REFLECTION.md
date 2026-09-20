# Engineering Reflection: Kestrel Labs Research Assistant

## 1. Trade-offs Made

### Multi-Agent Pipeline vs. Single-Shot RAG
- **Trade-off:** Introducing 4 dedicated agents (Router $\rightarrow$ Retriever $\rightarrow$ Synthesizer $\rightarrow$ Verifier) added latency compared to a naive single-prompt RAG approach.
- **Why it was worth it:** The naive approach failed on multi-turn pronoun queries (*"What about Growth?"* scored 0% citation recall without the Router) and hallucinated on out-of-corpus topics. The Verifier agent provided strict factual grounding and prevented hallucinations by revising claims that lacked evidence.

### Single-Pass Surgical Verification vs. Full Generator-Critic Loop
- **Trade-off:** When the Verifier identifies `insufficient_evidence`, it directly produces an honest `revised_answer` with caveats rather than looping back to the Synthesizer for another rewrite.
- **Why it was worth it:** Looping back to the Synthesizer for rewriting and then re-verifying added 15–20 seconds of latency and risked introducing fresh hallucinations. The Verifier's surgical revision fixed claims in a single pass at zero additional latency cost.

---

## 2. Latency & Cost Observations

1. **Free-Tier Zero Inference Cost:**
   Using Groq's high-speed inference for LLM calls and local `sentence-transformers/all-MiniLM-L6-v2` embeddings resulted in **$0.00 total API cost**.
2. **Cold-Start vs. Warm Latency:**
   - **Initial Cold-Start:** The initial load of SentenceTransformer weights from local disk took ~16 seconds.
   - **Optimization:** By caching the embedding function and ChromaDB client as process-level singletons and pre-warming them via `@st.cache_resource` on Streamlit startup, subsequent retrievals dropped to **12–17 milliseconds**.
3. **Conversational vs. RAG Latency:**
   - Greetings and out-of-domain queries take **~1.3s – 1.8s** due to the router bypass.
   - Grounded queries take **~7s – 15s**, which is well within acceptable limits for a 4-agent stateful pipeline.

---

## 3. What Worked Well

- **Router Pronoun Resolution:** Rewriting conversational follow-ups into self-contained search queries increased follow-up citation recall from 0% to 50–100%.
- **Date Precedence Rules:** Explicitly instructing the Synthesizer to compare `published` metadata successfully resolved documented conflicts (e.g., modern 30-day retention overriding older 90-day retention; $180 on-call stipend overriding older $150).
- **LangSmith Tracing:** Having full trace visualization allowed us to identify exact latency bottlenecks and confirm agent hand-offs. The public traces and dataset evaluation can be inspected at: [https://smith.langchain.com/public/193ee189-07b8-4629-9319-eebd5b780402/d](https://smith.langchain.com/public/193ee189-07b8-4629-9319-eebd5b780402/d).

---

## 4. What Would Be Done Next

1. **Hybrid Retrieval (BM25 + Semantic):**
   Adding BM25 sparse keyword search alongside ChromaDB dense vectors would boost recall for exact error codes and release tags (e.g. `INC-2025-07`, `session.timeout.ms`).
2. **Parallel Sub-Query Execution:**
   For multi-hop queries decomposed into multiple sub-queries, running the ChromaDB vector searches concurrently via `asyncio` would shave 1–2 seconds off complex retrievals.
3. **Automated LLM-as-a-Judge Evaluation in LangSmith:**
   Deploying automated LangSmith evaluators to continuously score faithfulness, answer relevance, and citation precision on production traces.
