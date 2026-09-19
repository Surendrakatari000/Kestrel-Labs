"""LangGraph shared state definition for the Kestrel Research Assistant."""

from typing import Annotated, Any, Dict, List, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state passed between all agents in the LangGraph.
    Each key is updated by the agent node that owns it.
    """
    # ── Conversation ──
    # Full message history (HumanMessage, AIMessage). Uses operator.add
    # so each node can append without overwriting.
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # ── Router outputs ──
    current_query: str          # Pronoun-resolved standalone query
    query_type: str             # single_hop | multi_hop | conflicting | unsupported | follow_up
    sub_queries: List[str]      # Decomposed sub-queries for multi_hop
    needs_retrieval: bool       # False for greetings or unsupported

    # ── Retriever outputs ──
    retrieved_chunks: List[Dict[str, Any]]  # Each has 'text' and 'metadata'

    # ── Synthesizer outputs ──
    draft_answer: str

    # ── Verifier outputs ──
    verifier_verdicts: List[Dict[str, Any]]  # Each has claim, verdict, explanation
    overall_supported: bool

    # ── Final ──
    final_answer: str
    retry_count: int            # Tracks verification retries (max 1)
