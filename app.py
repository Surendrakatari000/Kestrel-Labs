"""Streamlit UI for the Kestrel Labs Research Assistant."""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# ─────────────────────────────────────────────
# 1. Page Config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Kestrel Labs Research Assistant",
    page_icon="🐦",
    layout="centered",
)

# Load environment
load_dotenv()

from src.graph import build_graph
from src.vectorstore import get_collection, EMBEDDING_MODEL_NAME


@st.cache_resource(show_spinner="Initializing Kestrel knowledge base & local embeddings...")
def _warmup_resources():
    """Pre-load local embeddings & vector store into memory on startup."""
    col = get_collection()
    return col.count()

chunk_count = _warmup_resources()


# ─────────────────────────────────────────────
# 2. Main Header & Clear Conversation Button
# ─────────────────────────────────────────────
header_col1, header_col2 = st.columns([0.78, 0.22], vertical_alignment="bottom")
with header_col1:
    st.title("🐦 Kestrel Labs Research Assistant")
    st.caption(
        f"Grounded Multi-Agent RAG over Kestrel Labs internal documentation • {chunk_count} chunks indexed across 25 docs"
    )

with header_col2:
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────────
# 4. Welcome & Starter Suggestions (when empty)
# ─────────────────────────────────────────────
prompt_to_submit = None
welcome_placeholder = st.empty()

if not st.session_state.messages:
    with welcome_placeholder.container():
        st.markdown("""
        ### 👋 Welcome! How can I help you today?
        I can retrieve verified evidence from company docs, resolve conflicting policies using publication dates, and cross-reference multiple specs.
        """)

        st.markdown("**Try asking one of these common questions:**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔔 How do Beacons work and how often are they evaluated?", use_container_width=True):
                prompt_to_submit = "How do Beacons work and how often are they evaluated?"
            if st.button("⚡ What caused incident INC-2025-11 and how was it fixed?", use_container_width=True):
                prompt_to_submit = "What caused the Warehouse Sync duplicate rows incident (INC-2025-11) and how was deduplication fixed?"

        with col2:
            if st.button("💰 What is the overage cost and allowance on Growth?", use_container_width=True):
                prompt_to_submit = "How much does overage cost on the Growth plan, and what is the monthly event allowance?"
            if st.button("🛡️ How long are raw events kept in cold storage?", use_container_width=True):
                prompt_to_submit = "How long are raw events kept in cold storage after the plan retention window ends?"

        st.divider()


# ─────────────────────────────────────────────
# 5. Render Conversation History
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)


# ─────────────────────────────────────────────
# 6. Chat Input & Processing
# ─────────────────────────────────────────────
chat_input = st.chat_input("Ask about Kestrel (e.g. Is Trails available on the Starter plan?)")

user_input = prompt_to_submit or chat_input

if user_input:
    # Immediately clear the welcome banner and sample questions from the screen
    welcome_placeholder.empty()

    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_area = st.empty()
        status_area.markdown("🧠 *Understanding question...*")
        final_answer = ""

        state_input = {
            "messages": list(st.session_state.messages),
            "retry_count": 0,
        }

        try:
            for step_output in st.session_state.graph.stream(state_input):
                for node_name, node_state in step_output.items():
                    if node_name == "router":
                        status_area.markdown("🔍 *Searching company docs...*")

                    elif node_name == "retriever":
                        status_area.markdown("✍️ *Synthesizing answer...*")

                    elif node_name == "synthesizer":
                        status_area.markdown("🛡️ *Fact-checking claims...*")

                    elif node_name == "verifier":
                        final_answer = node_state.get("final_answer", "")

                    elif node_name == "increment_retry":
                        status_area.markdown("🔍 *Searching company docs...*")

                    elif node_name == "unsupported":
                        final_answer = node_state.get("final_answer", "")

            # Clear status and show answer
            status_area.empty()

            if final_answer:
                st.markdown(final_answer)
                st.session_state.messages.append(AIMessage(content=final_answer))

        except Exception as e:
            status_area.empty()
            st.error(f"Error: {e}")
            st.info("Check your `.env` file: is `GROQ_API_KEY` configured?")
