import os
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage

from src.state import AgentState
from src.agents.utils import _get_llm, _safe_invoke_structured

ROUTER_MODEL = os.getenv("GROQ_ROUTER_MODEL", "qwen/qwen3.8-27b")



class RouteDecision(BaseModel):
    """Schema for routing and query rewriting decisions."""
    standalone_query: str = Field(
        description="User question rewritten with pronouns resolved from conversation history."
    )
    query_type: Literal[
        "single_hop", "multi_hop", "conflicting", "unsupported", "follow_up", "greeting"
    ] = Field(
        description="Category: greeting, single_hop, multi_hop, conflicting, unsupported, or follow_up."
    )
    sub_queries: List[str] = Field(
        default_factory=list,
        description="For multi_hop queries: 2-3 simpler decomposed sub-questions."
    )
    needs_retrieval: bool = Field(
        description="False for greetings or unsupported queries; True otherwise."
    )


ROUTER_SYSTEM = """You are the Router agent for Kestrel Labs internal research assistant.
The knowledge base covers product specs, release notes, pricing, engineering docs, post-mortems, and policies.

Your job:
1. Rewrite user's query into standalone_query by resolving pronouns or missing context from conversation history.
2. Classify query_type into:
   - "greeting": general chitchat or hello (needs_retrieval=False).
   - "unsupported": questions unrelated to Kestrel or explicitly outside the corpus (needs_retrieval=False).
   - "multi_hop": questions asking for multiple distinct facts, root causes AND release fixes, or cross-document comparisons (e.g., plans vs releases). For multi_hop, you MUST provide 2-3 focused sub_queries.
   - "conflicting": questions touching policies, retention windows, timeouts, or stipends that may have changed across versions or documents.
   - "follow_up": questions that depend on previous chat turns.
   - "single_hop": standard direct factual questions.
3. If multi_hop, decompose into 2-3 specific sub_queries that search for the separate facts.
4. Set needs_retrieval=False ONLY for greetings or clearly unsupported questions.
"""


def router_node(state: AgentState) -> dict:
    """Classifies query intent and rewrites pronouns into a standalone query."""
    messages = state.get("messages", [])
    if not messages:
        return {
            "current_query": "",
            "query_type": "unsupported",
            "sub_queries": [],
            "needs_retrieval": False,
        }

    llm = _get_llm(model=ROUTER_MODEL, max_tokens=512)
    structured_llm = llm.with_structured_output(RouteDecision)

    invoke_messages = [SystemMessage(content=ROUTER_SYSTEM)] + list(messages)

    try:
        decision = _safe_invoke_structured(structured_llm, invoke_messages)
        if isinstance(decision, dict):
            standalone = decision.get("standalone_query")
            q_type = decision.get("query_type", "single_hop")
            sub_q = decision.get("sub_queries", [])
            retrieval = decision.get("needs_retrieval", True)
        elif hasattr(decision, "standalone_query"):
            standalone = decision.standalone_query
            q_type = decision.query_type
            sub_q = decision.sub_queries
            retrieval = decision.needs_retrieval
        else:
            raise ValueError(f"Unexpected decision output format: {type(decision)}")

        last_msg = messages[-1].content if messages else ""
        return {
            "current_query": standalone if standalone else last_msg,
            "query_type": q_type,
            "sub_queries": sub_q,
            "needs_retrieval": retrieval,
        }
    except Exception as e:
        # Fallback: treat last user message as a direct single_hop query
        last_msg = messages[-1].content if messages else ""
        return {
            "current_query": last_msg,
            "query_type": "single_hop",
            "sub_queries": [],
            "needs_retrieval": True,
        }
