"""Router Agent: query classification and pronoun resolution."""

from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage

from src.state import AgentState
from src.agents.utils import _get_llm, _safe_invoke_structured


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
1. Rewrite user's query into standalone_query by resolving pronouns from conversation history.
2. Classify query_type into: greeting, single_hop, multi_hop, conflicting, unsupported, or follow_up.
3. If multi_hop, decompose into 2-3 sub_queries.
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

    llm = _get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)
    invoke_messages = [SystemMessage(content=ROUTER_SYSTEM)] + list(messages)

    try:
        decision: RouteDecision = _safe_invoke_structured(structured_llm, invoke_messages)
    except Exception:
        # Fallback: treat last user message as a direct single_hop query
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
