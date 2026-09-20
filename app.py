"""Streamlit UI for the Kestrel Labs Research Assistant."""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.graph import build_graph
from src.vectorstore import get_collection

# Load .env (GROQ_API_KEY, LANGCHAIN_*)
load_dotenv()


@st.cache_resource
def _warmup_resources():
    """Pre-load local embeddings & vector store into memory on startup."""
    col = get_collection()
    return col.count()

_warmup_resources()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Kestrel Labs Research Assistant",
    page_icon="🐦",
    layout="wide",
)

st.title("🐦 Kestrel Labs Research Assistant")
st.caption(
    "Grounded answers from Kestrel's internal documentation. "
    "Every claim is cited and fact-checked."
)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────
# Chat history
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ─────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────
user_input = st.chat_input("Ask about Kestrel (e.g. How do Beacons work?)")

if user_input:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_area = st.empty()
        final_answer = ""
        verdicts = []
        router_info = {}

        state_input = {
            "messages": list(st.session_state.messages),
            "retry_count": 0,
        }

        try:
            with st.spinner("Thinking..."):
                for step_output in st.session_state.graph.stream(state_input):
                    for node_name, node_state in step_output.items():
                        if node_name == "router":
                            router_info = {
                                "query": node_state.get("current_query", ""),
                                "type": node_state.get("query_type", ""),
                                "sub_queries": node_state.get("sub_queries", []),
                            }
                            status_area.info(f"🔀 **Router** → Type: `{router_info['type']}` | Query: _{router_info['query']}_")

                        elif node_name == "retriever":
                            n_chunks = len(node_state.get("retrieved_chunks", []))
                            status_area.info(f"🔍 **Retriever** → Found {n_chunks} chunks")

                        elif node_name == "synthesizer":
                            status_area.info("✍️ **Synthesizer** → Drafting answer...")

                        elif node_name == "verifier":
                            verdicts = node_state.get("verifier_verdicts", [])
                            overall = node_state.get("overall_supported", True)
                            status_area.info(
                                f"✅ **Verifier** → {'All supported' if overall else 'Issues found — may retry'}"
                            )
                            final_answer = node_state.get("final_answer", "")

                        elif node_name == "increment_retry":
                            status_area.warning("🔄 **Retrying** — Verifier requested a re-search...")

                        elif node_name == "unsupported":
                            final_answer = node_state.get("final_answer", "")
                            status_area.info("⚠️ **Unsupported** — Not in the knowledge base")

            # Clear the status and show the answer
            status_area.empty()

            if final_answer:
                st.markdown("### Answer")
                st.markdown(final_answer)

                # Debug: Router info
                with st.expander("🔀 Router Agent — Query Analysis"):
                    st.json(router_info)

                # Verifier verdicts
                if verdicts:
                    with st.expander("🧐 Verifier Agent — Claim Verdicts"):
                        for v in verdicts:
                            verdict_emoji = {
                                "supported": "✅",
                                "partially_supported": "⚠️",
                                "conflicting_evidence": "⚡",
                                "insufficient_evidence": "❌",
                            }.get(v["verdict"], "❓")
                            st.markdown(f"{verdict_emoji} **{v['verdict'].upper()}**: {v['claim']}")
                            st.caption(v["explanation"])
                            st.divider()

                # Save to history
                st.session_state.messages.append(AIMessage(content=final_answer))

        except Exception as e:
            status_area.empty()
            st.error(f"Error: {e}")
            st.info("Check your `.env` file: is `GROQ_API_KEY` set correctly?")
