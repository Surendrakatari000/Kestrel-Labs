"""Retriever Agent: dynamic k retrieval and sub-query deduplication."""

from src.state import AgentState
from src.vectorstore import search_corpus


def retriever_node(state: AgentState) -> dict:
    """Executes semantic search with dynamic k based on query classification."""
    query = state.get("current_query", "")
    query_type = state.get("query_type", "single_hop")
    sub_queries = state.get("sub_queries", [])

    if not query:
        return {"retrieved_chunks": []}

    # Multi-hop: retrieve for each sub-query and deduplicate by chunk_id
    if query_type == "multi_hop" and sub_queries:
        seen_ids = set()
        all_chunks = []

        for sq in sub_queries:
            for hit in search_corpus(sq, k=3):
                cid = hit["metadata"].get("chunk_id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(hit)

        # Also search main query for overall context
        for hit in search_corpus(query, k=3):
            cid = hit["metadata"].get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(hit)

        return {"retrieved_chunks": all_chunks[:8]}

    # Conflicting: retrieve top 5 to capture both older specs and newer release notes
    elif query_type == "conflicting":
        return {"retrieved_chunks": search_corpus(query, k=5)}

    # Single-hop or follow-up: retrieve top 3 focused chunks
    else:
        return {"retrieved_chunks": search_corpus(query, k=3)}
