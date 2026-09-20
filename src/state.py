"""
LangGraph Shared State Schema: AgentState
=========================================
This module defines the central state schema passed between all four agents in
the multi-agent workflow.

Design Highlights for Interviews:
1. TypedDict Architecture:
   - Provides static type safety across graph nodes without runtime overhead.
   - Every node returns a subset of keys to update the state.
2. Append-Only Messages (`operator.add`):
   - Using Annotated[Sequence[BaseMessage], operator.add] allows nodes to append
     new conversation turns without overwriting previous history.
3. Explicit Hand-off Contracts:
   - Router writes: current_query, query_type, sub_queries, needs_retrieval.
   - Retriever writes: retrieved_chunks.
   - Synthesizer writes: draft_answer.
   - Verifier writes: verifier_verdicts, overall_supported, final_answer.
"""

from typing import Annotated, Any, Dict, List, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state container passed between all agents in the LangGraph workflow.
    Each key represents an explicit data contract owned and updated by a node.
    """
    # ── 1. Conversation History ──
    # Annotated with operator.add so each node appends messages rather than replacing the list.
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # ── 2. Router Outputs ──
    current_query: str          # Rewritten standalone query with all pronouns resolved
    query_type: str             # Category: single_hop | multi_hop | conflicting | unsupported | follow_up | greeting
    sub_queries: List[str]      # Decomposed search queries (used when query_type == "multi_hop")
    needs_retrieval: bool       # Flag indicating if vector store retrieval is required

    # ── 3. Retriever Outputs ──
    retrieved_chunks: List[Dict[str, Any]]  # List of relevant chunks with text & metadata (published, doc_id, chunk_id)

    # ── 4. Synthesizer Outputs ──
    draft_answer: str           # Grounded response with strict [chunk_id: title] citations

    # ── 5. Verifier (Critic) Outputs ──
    verifier_verdicts: List[Dict[str, Any]]  # Detailed claim audits (supported | partially_supported | conflicting | insufficient)
    overall_supported: bool     # True only when all factual claims are verified by retrieved evidence

    # ── 6. Final Outputs & Control Flow ──
    final_answer: str           # The verified, audited answer presented to the user
    retry_count: int            # Tracks bounded verification retries (max 1 retry to avoid infinite loops)
