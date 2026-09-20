"""
LangGraph Multi-Agent Orchestration Workflow
============================================
This module constructs and compiles the stateful graph connecting all four agents.

Architecture Flow:
  START ──► Router
              │
              ├─► [needs_retrieval == False] ──► Unsupported Node (Fast-path ~1.5s) ──► END
              │
              └─► [needs_retrieval == True] ──► Retriever ──► Synthesizer ──► Verifier
                                                                                │
                                                                                ├─► [All claims supported] ──► END
                                                                                │
                                                                                └─► [Unsupported claims & retry < 1]
                                                                                          │ (Loop back)
                                                                                          ▼
                                                                                   increment_retry ──► Retriever

Key Interview Talking Points:
1. State Machine vs Linear Chain:
   - LangGraph treats the RAG pipeline as a directed cyclic graph with state.
   - Allows dynamic branching (bypassing retrieval for greetings) and cyclic self-correction (retrying retrieval).
2. Bounded Cyclic Self-Correction:
   - If the Verifier discovers that a draft claim lacks support, it triggers a retry loop back to the Retriever.
   - Retries are strictly capped at 1 (`retry_count < 1`) to prevent infinite looping and excessive token burn.
3. Latency Optimization:
   - Conversational greetings or thanks take the "fast path" directly to END (~1.5s latency).
   - Conflicting questions skip vector re-queries because the conflict is resolved via date precedence in the Synthesizer.
"""

from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.agents import router_node, retriever_node, synthesizer_node, verifier_node


# ─────────────────────────────────────────────
# 1. Conditional Edge Functions (Routing Logic)
# ─────────────────────────────────────────────

def after_router(state: AgentState) -> str:
    """
    Decides the next node after the Router agent.
    - If needs_retrieval is False (e.g., greetings, general questions), routes to fast unsupported_node.
    - If needs_retrieval is True, routes to the semantic Retriever agent.
    """
    if not state.get("needs_retrieval", True):
        return "end_unsupported"
    return "retriever"


def after_verifier(state: AgentState) -> str:
    """
    Evaluates the Verifier's factual claim audit to determine whether to finalize or retry.
    - If overall_supported is True: accepts answer and routes to END.
    - If query is 'conflicting' or 'unsupported': accepts without retry (date precedence already resolved it).
    - If claims failed verification and retry_count < 1: loops back to increment_retry -> retriever.
    - Otherwise: accepts best-effort qualified answer to avoid infinite loops.
    """
    if state.get("overall_supported", True):
        return "accept"

    # Conflicting or unsupported queries are already resolved via publication metadata
    query_type = state.get("query_type", "")
    if query_type in ("conflicting", "unsupported"):
        return "accept"

    # Bounded retry: allow at most 1 re-retrieval
    retry_count = state.get("retry_count", 0)
    if retry_count < 1:
        return "retry"

    # Retries exhausted — deliver qualified, honest answer
    return "accept"


# ─────────────────────────────────────────────
# 2. Fast-Path / Helper Nodes
# ─────────────────────────────────────────────

GREETING_PATTERNS = {
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "howdy", "sup", "what's up", "hiya", "yo"
}
THANKS_PATTERNS = {
    "thanks", "thank you", "thx", "appreciate it", "thank you very much"
}
IDENTITY_PATTERNS = {
    "who are you", "what are you", "what can you do", "help"
}


def unsupported_node(state: AgentState) -> dict:
    """
    Generates an immediate, polite response for conversational messages or out-of-domain queries
    without performing expensive vector store lookups.
    """
    query_type = state.get("query_type", "")
    query = (state.get("current_query", "") or "").strip().lower()
    clean_query = "".join(c for c in query if c.isalnum() or c.isspace()).strip()

    # Conversational: Thank you
    if clean_query in THANKS_PATTERNS or any(clean_query.startswith(t) for t in THANKS_PATTERNS):
        return {
            "final_answer": "You're very welcome! Feel free to ask if you have any other questions about Kestrel Labs."
        }

    # Conversational: Identity / Capabilities
    if clean_query in IDENTITY_PATTERNS or any(clean_query.startswith(i) for i in IDENTITY_PATTERNS):
        return {
            "final_answer": (
                "I am a research assistant dedicated to Kestrel Labs internal documentation. "
                "I retrieve evidence from company docs, cite exact sources, and verify claims. "
                "Ask me anything about our product features, engineering runbooks, release notes, or policies!"
            )
        }

    # Conversational: Greetings
    if query_type == "greeting" or clean_query in GREETING_PATTERNS or any(clean_query.startswith(g + " ") for g in GREETING_PATTERNS):
        return {
            "final_answer": (
                "Hello! I'm the Kestrel Labs research assistant. How can I help you today? "
                "You can ask me anything about Kestrel's product specifications (Beacons, Funnels, Trails, Warehouse Sync), "
                "pricing plans, engineering architecture, incident post-mortems, or company policies."
            )
        }

    # Out-of-corpus / Unsupported queries
    return {
        "final_answer": (
            "I'm a research assistant for Kestrel Labs internal documentation. "
            "The available documents don't contain information about that topic. "
            "I can help with questions about Kestrel's product specs, pricing, "
            "engineering architecture, incident reports, and company policies."
        )
    }


def increment_retry(state: AgentState) -> dict:
    """Bumps the retry counter before looping back to the retriever."""
    return {"retry_count": state.get("retry_count", 0) + 1}


# ─────────────────────────────────────────────
# 3. Graph Assembly & Compilation
# ─────────────────────────────────────────────

def build_graph():
    """
    Assembles the LangGraph StateGraph, adds nodes, wires conditional edges,
    and returns the compiled executable runnable.
    """
    workflow = StateGraph(AgentState)

    # Register the agent nodes
    workflow.add_node("router", router_node)
    workflow.add_node("unsupported", unsupported_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("increment_retry", increment_retry)

    # Entry point: START -> Router
    workflow.add_edge(START, "router")

    # Conditional Branch: Router -> Retriever (if research needed) OR Unsupported (fast-path)
    workflow.add_conditional_edges(
        "router",
        after_router,
        {
            "retriever": "retriever",
            "end_unsupported": "unsupported",
        },
    )

    # Fast-path terminates at END
    workflow.add_edge("unsupported", END)

    # Sequential Core Pipeline: Retriever -> Synthesizer -> Verifier
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", "verifier")

    # Conditional Branch: Verifier -> Accept (END) OR Retry Loop (increment_retry -> retriever)
    workflow.add_conditional_edges(
        "verifier",
        after_verifier,
        {
            "accept": END,
            "retry": "increment_retry",
        },
    )

    # Connect retry back to Retriever
    workflow.add_edge("increment_retry", "retriever")

    return workflow.compile()
