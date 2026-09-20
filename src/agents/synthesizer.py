"""Synthesizer Agent: evidence-grounded drafting with date precedence."""

from typing import List
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState
from src.agents.utils import _get_llm, _safe_invoke


def _format_chunks_for_prompt(chunks: List[dict]) -> str:
    """Formats retrieved chunks with metadata headers for date precedence."""
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
Draft an answer based STRICTLY on the provided context chunks.

RULES:
1. Ground every statement in context; never invent facts.
2. Cite sources using exact format: [chunk_id: title]
   Example: "All Beacons are evaluated every 5 minutes [spec-beacons:1: Beacons: Alerting Specification]."
   Always include BOTH the chunk_id and title inside the square brackets.
3. If sources disagree, compare 'published' dates. Trust the newer document and mention both.
4. If context does not answer the query, reply:
   "The available documentation does not contain information to answer this question."
5. Be concise and include specific numbers, limits, and details.
"""


def synthesizer_node(state: AgentState) -> dict:
    """Drafts an evidence-backed answer citing chunks and resolving conflicts by date."""
    from src.utils.citations import normalize_citations

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
        normalized_draft = normalize_citations(response.content, chunks)
        return {"draft_answer": normalized_draft}
    except Exception as e:
        return {"draft_answer": f"Error generating answer: {e}"}

