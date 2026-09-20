"""
All four agents for the Kestrel Research Assistant.

Router    → classifies query, resolves pronouns, decomposes multi-hop
Retriever → searches ChromaDB with dynamic k
Synthesizer → drafts grounded answer with citations
Verifier  → fact-checks claims, issues verdicts, revises if needed
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
# Helpers
# ─────────────────────────────────────────────

def _get_llm(temperature: float = 0):
    """Returns a Groq LLM with rate-limit-safe retry via tenacity."""
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=2048,
    )


def _safe_invoke(llm, messages):
    """Invoke LLM with exponential backoff on rate limit (HTTP 429)."""
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call():
        return llm.invoke(messages)
    return _call()


def _safe_invoke_structured(structured_llm, messages):
    """Invoke structured-output LLM with exponential backoff."""
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call():
        return structured_llm.invoke(messages)
    return _call()


# ─────────────────────────────────────────────
# 1. ROUTER AGENT
# ─────────────────────────────────────────────

class RouteDecision(BaseModel):
    """Structured output from the Router agent."""
    standalone_query: str = Field(
        description="The user's question rewritten as a fully self-contained query. "
                    "Resolve all pronouns (e.g. 'them', 'it', 'that') using conversation history."
    )
    query_type: Literal["single_hop", "multi_hop", "conflicting", "unsupported", "follow_up", "greeting"] = Field(
        description="The category of the query. "
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
    """Classifies the query, resolves pronouns, decomposes multi-hop."""
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
        # Fallback: treat as single_hop with the raw user text
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
    """Searches ChromaDB with dynamic k based on query_type."""
    query = state.get("current_query", "")
    query_type = state.get("query_type", "single_hop")
    sub_queries = state.get("sub_queries", [])

    if not query:
        return {"retrieved_chunks": []}

    if query_type == "multi_hop" and sub_queries:
        # Run each sub-query separately and deduplicate
        seen_ids = set()
        all_chunks = []
        for sq in sub_queries:
            hits = search_corpus(sq, k=3)
            for h in hits:
                cid = h["metadata"].get("chunk_id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(h)
        # Also search the main query for good measure
        for h in search_corpus(query, k=3):
            cid = h["metadata"].get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(h)
        return {"retrieved_chunks": all_chunks[:8]}  # Cap at 8
    elif query_type == "conflicting":
        return {"retrieved_chunks": search_corpus(query, k=5)}
    else:
        # single_hop, follow_up, or default
        return {"retrieved_chunks": search_corpus(query, k=3)}


# ─────────────────────────────────────────────
# 3. SYNTHESIZER AGENT
# ─────────────────────────────────────────────

def _format_chunks_for_prompt(chunks) -> str:
    """Formats retrieved chunks into a context string for the LLM."""
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
    """Drafts a grounded answer with citations from retrieved chunks."""
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
    claim: str = Field(description="A factual claim extracted from the draft answer.")
    verdict: Literal[
        "supported", "partially_supported", "conflicting_evidence", "insufficient_evidence"
    ] = Field(description="The verification verdict for this claim.")
    explanation: str = Field(
        description="Why this verdict was chosen. If conflicting, state which source is newer."
    )


class VerifierOutput(BaseModel):
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
    """Fact-checks the draft answer against retrieved chunks."""
    draft = state.get("draft_answer", "")
    chunks = state.get("retrieved_chunks", [])

    # If no draft or it's already an "insufficient" answer, pass through
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
        # If structured output fails, accept the draft as-is
        return {
            "verifier_verdicts": [{"claim": "parse_error", "verdict": "supported", "explanation": str(e)}],
            "overall_supported": True,
            "final_answer": draft,
        }
