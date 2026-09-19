"""
LangGraph workflow that wires the 4 agents together.

Flow:
  START → router
           ├─ unsupported / no retrieval → END
           └─ needs retrieval → retriever → synthesizer → verifier
                                                            ├─ all supported → END
                                                            └─ retry_count < 1 → retriever (loop)
                                                               else → END (best effort)
"""

from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.agents import router_node, retriever_node, synthesizer_node, verifier_node


# ─────────────────────────────────────────────
# Conditional edge functions
# ─────────────────────────────────────────────

def after_router(state: AgentState) -> str:
    """Decide where to go after the Router."""
    if not state.get("needs_retrieval", True):
        return "end_unsupported"
    return "retriever"


def after_verifier(state: AgentState) -> str:
    """Decide whether to accept, retry, or bail."""
    if state.get("overall_supported", True):
        return "accept"

    retry_count = state.get("retry_count", 0)
    if retry_count < 1:
        return "retry"

    # Max retries exhausted — accept best effort
    return "accept"


# ─────────────────────────────────────────────
# Helper nodes
# ─────────────────────────────────────────────

def unsupported_node(state: AgentState) -> dict:
    """Generates a polite response for unsupported/conversational queries."""
    query_type = state.get("query_type", "")
    query = state.get("current_query", "")

    if query_type == "unsupported":
        return {
            "final_answer": (
                "I'm a research assistant for Kestrel Labs internal documentation. "
                "The available documents don't contain information about that topic. "
                "I can help with questions about Kestrel's product specs, pricing, "
                "engineering architecture, incident reports, and company policies."
            )
        }
    # Generic conversational (greetings, thanks)
    return {
        "final_answer": (
            "Hello! I'm the Kestrel Labs research assistant. "
            "Ask me anything about Kestrel's products, pricing, engineering, or policies."
        )
    }


def increment_retry(state: AgentState) -> dict:
    """Bumps the retry counter before looping back to the retriever."""
    return {"retry_count": state.get("retry_count", 0) + 1}


# ─────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────

def build_graph():
    """Compiles the LangGraph with conditional routing and a retry loop."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("unsupported", unsupported_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("increment_retry", increment_retry)

    # START → Router
    workflow.add_edge(START, "router")

    # Router → Retriever or Unsupported
    workflow.add_conditional_edges(
        "router",
        after_router,
        {
            "retriever": "retriever",
            "end_unsupported": "unsupported",
        },
    )

    # Unsupported → END
    workflow.add_edge("unsupported", END)

    # Retriever → Synthesizer → Verifier
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", "verifier")

    # Verifier → Accept (END) or Retry (loop back to retriever)
    workflow.add_conditional_edges(
        "verifier",
        after_verifier,
        {
            "accept": END,
            "retry": "increment_retry",
        },
    )

    # Retry increment → Retriever (loop)
    workflow.add_edge("increment_retry", "retriever")

    return workflow.compile()
