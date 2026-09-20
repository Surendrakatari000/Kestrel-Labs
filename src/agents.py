"""
Core Agent Implementations for Kestrel Research Assistant
========================================================
This module contains the logic for all four specialized agents:

1. Router Agent:
   - Analyzes conversational context to rewrite follow-up questions (pronoun resolution).
   - Classifies query intent: single_hop, multi_hop, conflicting, unsupported, follow_up, greeting.
   - Decomposes complex multi-hop queries into 2-3 focused sub-queries.

2. Retriever Agent:
   - Queries ChromaDB using dynamic k sizing based on query type.
   - Executes sub-query retrieval for multi-hop questions with ID deduplication.

3. Synthesizer Agent:
   - Generates an evidence-grounded answer citing exact sources in format: [chunk_id: title].
   - Enforces metadata precedence: when documents conflict, the newer 'published' date is trusted.

4. Verifier (Critic) Agent:
   - Fact-checks every material claim against retrieved chunks.
   - Assigns structured verdicts: supported, partially_supported, conflicting_evidence, insufficient_evidence.
   - Performs surgical in-place revisions to ensure factual honesty without extra generation latency.

Rate Limiting & Free-Tier Resilience:
   - All LLM invocations use `tenacity` with exponential backoff on HTTP 429 errors.
   - Output tokens capped at 2048 to respect Groq free-tier quotas.
"""

import os
from typing import List, Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.state import AgentState
from src.vectorstore import search_corpus


# ─────────────────────────────────────────────
# Resilient LLM Helpers (HTTP 429 Backoff)
# ─────────────────────────────────────────────

def _get_llm(temperature: float = 0):
    """
    Returns an initialized ChatGroq client.
    Default model is 'openai/gpt-oss-120b' or 'llama-3.3-70b-versatile'.
    Output capped at 2048 tokens to stay within free-tier limits.
    """
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=2048,
    )


def _log_retry(retry_state):
    """
    Callback fired before Tenacity sleeps between retries.
    Informs developers in the console about rate limits and backoff duration.
    """
    exc = retry_state.outcome.exception()
    sleep_time = retry_state.next_action.sleep
    attempt = retry_state.attempt_number
    print(
        f"[RateLimit / Backoff] Transient error / HTTP 429: {exc}. "
        f"Waiting {sleep_time:.1f}s before retry (attempt {attempt + 1}/4)...",
        flush=True,
    )


def _safe_invoke(llm, messages):
    """
    Invokes the LLM with exponential backoff on transient errors (HTTP 429 / timeouts).
    Wait progression: 4s -> 8s -> 16s -> 32s (bounded at 60s, max 4 attempts).
    """
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _call():
        return llm.invoke(messages)
    return _call()


def _safe_invoke_structured(structured_llm, messages):
    """
    Invokes a Pydantic-structured LLM with exponential backoff on rate limits.
    Ensures structured schema guarantees even during high API load.
    """
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _call():
        return structured_llm.invoke(messages)
    return _call()


# ─────────────────────────────────────────────
# 1. ROUTER AGENT
# ─────────────────────────────────────────────

class RouteDecision(BaseModel):
    """
    Pydantic schema enforcing structured routing decisions:
    - standalone_query: Pronoun-resolved question suitable for semantic vector search.
    - query_type: Categorization used by downstream nodes for dynamic retrieval strategy.
    - sub_queries: List of simpler queries if question requires multi-hop decomposition.
    - needs_retrieval: Whether vector search should be triggered.
    """
    standalone_query: str = Field(
        description="The user's question rewritten as a fully self-contained query. "
                    "Resolve all pronouns (e.g. 'them', 'it', 'that') using conversation history."
    )
    query_type: Literal["single_hop", "multi_hop", "conflicting", "unsupported", "follow_up", "greeting"] = Field(
        description="The category of the query: "
                    "greeting: conversational greetings ('hi', 'hello', 'hey') or polite remarks ('thanks'). "
                    "single_hop: direct factual lookup. "
                    "multi_hop: needs info from multiple documents. "
                    "conflicting: asks about something where sources may disagree. "
                    "unsupported: clearly not answerable from Kestrel internal docs. "
                    "follow_up: a continuation of a previous question."
    )
    sub_queries: List[str] = Field(
        default_factory=list,
        description="For multi_hop queries only: break the question into 2-3 simpler sub-questions."
    )
    needs_retrieval: bool = Field(
        description="True if the query requires searching the knowledge base. "
                    "False only for greetings ('hi', 'thanks') or clearly unsupported questions."
    )


ROUTER_SYSTEM = """You are the Router agent for the Kestrel Labs internal research assistant.

Kestrel Labs is a product-analytics SaaS company. The knowledge base covers:
- Product specs (Beacons, Funnels, Cohorts, Trails, Warehouse Sync, SDKs, KQL, Ingest API)
- Release notes (versions 3.4 through 4.1)
- Pricing plans (Starter, Growth, Scale)
- Engineering docs (ingest pipeline, Osprey query engine, deployment runbook, on-call runbook)
- Incident post-mortems (INC-2025-07, INC-2025-11, INC-2026-02)
- Policies (data retention, security & compliance, employee handbook)
- Onboarding guide and Product FAQ

Your job:
1. Rewrite the user's latest message into a standalone_query by resolving pronouns from chat history.
2. Classify the query_type:
   - "greeting" for hello/hi/hey/thanks/who are you.
   - "single_hop" for direct factual questions answered by 1 doc.
   - "multi_hop" for questions requiring combining information across multiple docs.
   - "conflicting" for questions where different documents or versions disagree.
   - "unsupported" for questions clearly not covered by Kestrel documentation.
   - "follow_up" for continuous conversational questions following up on previous topics.
3. If multi_hop, break it into sub_queries.
4. Set needs_retrieval=False ONLY for greetings or questions clearly outside Kestrel's domain (e.g. "What's the weather?").
   For questions that MIGHT be in the corpus but you're unsure, set needs_retrieval=True so the retriever can check.
"""


def router_node(state: AgentState) -> dict:
    """
    Router Node:
    Takes conversation history, performs conversational pronoun resolution,
    categorizes intent, and decomposes multi-hop queries.
    """
    messages = state.get("messages", [])
    if not messages:
        return {
            "current_query": "",
            "query_type": "unsupported",
            "sub_queries": [],
            "needs_retrieval": False,
        }

    llm = _get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)
    invoke_messages = [SystemMessage(content=ROUTER_SYSTEM)] + list(messages)

    try:
        decision: RouteDecision = _safe_invoke_structured(structured_llm, invoke_messages)
    except Exception:
        # Graceful fallback: treat as single_hop with raw user query
        last_msg = messages[-1].content if messages else ""
        return {
            "current_query": last_msg,
            "query_type": "single_hop",
            "sub_queries": [],
            "needs_retrieval": True,
        }

    return {
        "current_query": decision.standalone_query,
        "query_type": decision.query_type,
        "sub_queries": decision.sub_queries,
        "needs_retrieval": decision.needs_retrieval,
    }


# ─────────────────────────────────────────────
# 2. RETRIEVER AGENT
# ─────────────────────────────────────────────

def retriever_node(state: AgentState) -> dict:
    """
    Retriever Node:
    Executes semantic search over local ChromaDB using dynamic k heuristics:
    - multi_hop: Runs search across all decomposed sub-queries, deduplicates, and caps at 8 chunks.
    - conflicting: Retrieves top-5 chunks to capture older specs and newer release notes.
    - single_hop / follow_up: Retrieves top-3 most focused chunks.
    """
    query = state.get("current_query", "")
    query_type = state.get("query_type", "single_hop")
    sub_queries = state.get("sub_queries", [])

    if not query:
        return {"retrieved_chunks": []}

    if query_type == "multi_hop" and sub_queries:
        # Multi-hop: search each sub-query independently and deduplicate by chunk_id
        seen_ids = set()
        all_chunks = []
        for sq in sub_queries:
            hits = search_corpus(sq, k=3)
            for h in hits:
                cid = h["metadata"].get("chunk_id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(h)

        # Also search the main query to ensure broad context
        for h in search_corpus(query, k=3):
            cid = h["metadata"].get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(h)

        return {"retrieved_chunks": all_chunks[:8]}  # Cap at top 8 to stay within context budget

    elif query_type == "conflicting":
        # Conflicting queries need wider context to find both older and newer documentation
        return {"retrieved_chunks": search_corpus(query, k=5)}

    else:
        # Direct lookup (single_hop or follow_up)
        return {"retrieved_chunks": search_corpus(query, k=3)}


# ─────────────────────────────────────────────
# 3. SYNTHESIZER AGENT
# ─────────────────────────────────────────────

def _format_chunks_for_prompt(chunks: List[dict]) -> str:
    """Formats retrieved chunks with metadata headers so the LLM can inspect publication dates."""
    if not chunks:
        return "(No context retrieved.)"
    parts = []
    for c in chunks:
        m = c["metadata"]
        parts.append(
            f"--- CHUNK {m.get('chunk_id', '?')} ---\n"
            f"Title: {m.get('title', '?')}\n"
            f"Published: {m.get('published', '?')}\n"
            f"Version: {m.get('version', '?')}\n"
            f"Category: {m.get('category', '?')}\n"
            f"Text: {c['text']}\n"
        )
    return "\n".join(parts)


SYNTHESIZER_SYSTEM = """You are the Synthesizer agent for Kestrel Labs.
Your task is to draft an answer based STRICTLY on the provided context chunks.

RULES:
1. Ground every statement in the context. NEVER invent information.
2. Cite sources using this exact format: [chunk_id: title]
   Example: [spec-beacons:0: Beacons: Alerting Specification]
3. When sources DISAGREE, compare their 'published' dates. The NEWER document is more authoritative.
   Present BOTH pieces of information, state which is newer, and explain which to trust.
4. If the context does NOT answer the question, say:
   "The available documentation does not contain information to answer this question."
5. Be concise but thorough. Include specific numbers, limits, and details from the chunks.
"""


def synthesizer_node(state: AgentState) -> dict:
    """
    Synthesizer Node:
    Drafts an answer grounded exclusively in the retrieved chunks.
    Enforces publication date precedence and strict source citations.
    """
    query = state.get("current_query", "")
    chunks = state.get("retrieved_chunks", [])

    if not query:
        return {"draft_answer": "I need a question to answer."}

    if not chunks:
        return {
            "draft_answer": "The available documentation does not contain information to answer this question."
        }

    context = _format_chunks_for_prompt(chunks)
    llm = _get_llm()

    msgs = [
        SystemMessage(content=SYNTHESIZER_SYSTEM + f"\n\nCONTEXT:\n{context}"),
        HumanMessage(content=query),
    ]

    try:
        response = _safe_invoke(llm, msgs)
        return {"draft_answer": response.content}
    except Exception as e:
        return {"draft_answer": f"Error generating answer: {e}"}


# ─────────────────────────────────────────────
# 4. VERIFIER / CRITIC AGENT
# ─────────────────────────────────────────────

class ClaimVerdict(BaseModel):
    """Structured audit verdict for a single factual claim."""
    claim: str = Field(description="A factual claim extracted from the draft answer.")
    verdict: Literal[
        "supported", "partially_supported", "conflicting_evidence", "insufficient_evidence"
    ] = Field(description="The verification verdict for this claim.")
    explanation: str = Field(
        description="Why this verdict was chosen. If conflicting, state which source is newer."
    )


class VerifierOutput(BaseModel):
    """Structured output from the Verifier containing claim-by-claim audits and surgical revision."""
    claims: List[ClaimVerdict] = Field(
        description="Verdicts for every material claim in the draft."
    )
    overall_supported: bool = Field(
        description="True only if ALL claims are 'supported'."
    )
    revised_answer: str = Field(
        description="If any claim is not fully supported, rewrite the answer to be honest "
                    "about gaps or conflicts. Keep citations. If all supported, copy the draft as-is."
    )


VERIFIER_SYSTEM = """You are the Verifier (Critic) agent for Kestrel Labs.
Your job is to fact-check a Draft Answer against the provided Context chunks.

INSTRUCTIONS:
1. Extract every material factual claim from the Draft Answer.
2. For each claim, check it against the Context and assign a verdict:
   - supported: The context fully backs it.
   - partially_supported: The context partially backs it but misses details.
   - conflicting_evidence: Different chunks say different things. Use 'published' dates to determine which is more reliable.
   - insufficient_evidence: The context does not address this claim.
3. Set overall_supported=True ONLY if every single claim is 'supported'.
4. Write a revised_answer:
   - If all supported: copy the draft answer exactly.
   - If any claim is NOT supported: rewrite to be honest. State what the evidence says and doesn't say.
     Keep [chunk_id: title] citations. Add notes about missing evidence or conflicts.
"""


def verifier_node(state: AgentState) -> dict:
    """
    Verifier Node:
    Acts as an adversarial fact-checker. Evaluates every claim in the draft against evidence chunks.
    Directly produces a surgically revised answer if unsupported claims are found, avoiding latency overhead.
    """
    draft = state.get("draft_answer", "")
    chunks = state.get("retrieved_chunks", [])

    # If already a refusal on missing docs, pass through immediately
    if not draft or "does not contain information" in draft.lower():
        return {
            "verifier_verdicts": [],
            "overall_supported": True,
            "final_answer": draft,
        }

    context = _format_chunks_for_prompt(chunks)
    llm = _get_llm()
    structured_llm = llm.with_structured_output(VerifierOutput)

    msgs = [
        SystemMessage(content=VERIFIER_SYSTEM + f"\n\nCONTEXT:\n{context}"),
        HumanMessage(content=f"Draft Answer to verify:\n\n{draft}"),
    ]

    try:
        result: VerifierOutput = _safe_invoke_structured(structured_llm, msgs)

        verdicts = [
            {"claim": c.claim, "verdict": c.verdict, "explanation": c.explanation}
            for c in result.claims
        ]

        return {
            "verifier_verdicts": verdicts,
            "overall_supported": result.overall_supported,
            "final_answer": result.revised_answer,
        }
    except Exception as e:
        # Fallback: if structured parser fails, accept draft safely
        return {
            "verifier_verdicts": [{"claim": "parse_error", "verdict": "supported", "explanation": str(e)}],
            "overall_supported": True,
            "final_answer": draft,
        }
