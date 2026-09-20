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

import time
from src.graph import build_graph
from src.vectorstore import get_collection, get_embedding_function, EMBEDDING_MODEL_NAME

# ─────────────────────────────────────────────
# 2. Knowledge Base & Model Warmup with Progress Bar
# ─────────────────────────────────────────────
if "chunk_count" not in st.session_state:
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        prog_bar = st.progress(0, text="Loading company documents & assistant... 0%")
        
        # Step 1: Connect to document storage
        prog_bar.progress(20, text="Loading company knowledge base... 20%")
        col = get_collection()
        chunk_count = col.count()
        
        # Step 2: Load search engine
        prog_bar.progress(50, text="Preparing smart search engine... 50%")
        _ = get_embedding_function()
        
        # Step 3: Warm up index
        prog_bar.progress(75, text="Indexing product guides and policies... 75%")
        _ = col.query(query_texts=["warmup"], n_results=1)
        
        # Step 4: Ready assistant
        prog_bar.progress(90, text="Starting research assistant... 90%")
        if "graph" not in st.session_state:
            st.session_state.graph = build_graph()
            
        # Step 5: Ready
        prog_bar.progress(100, text="Ready! 100%")
        time.sleep(0.3)
    
    loading_placeholder.empty()
    st.session_state.chunk_count = chunk_count
else:
    chunk_count = st.session_state.chunk_count


# ─────────────────────────────────────────────
# 2. Main Header & Clear Conversation Button
# ─────────────────────────────────────────────
header_col1, header_col2 = st.columns([0.78, 0.22], vertical_alignment="bottom")
with header_col1:
    st.title("🐦 Kestrel Labs Research Assistant")
    st.caption(
        "Your AI assistant for Kestrel Labs — product specs, pricing, engineering guides, and company policies."
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
if not st.session_state.messages:
    welcome_container = st.container()
    with welcome_container:
        st.markdown("""
        ### 👋 Welcome! How can I help you today?
        Ask any question about Kestrel Labs products, pricing plans, engineering runbooks, or company policies.
        I search official company documents to give you verified, up-to-date answers with source citations.
        """)

        st.markdown("**Try asking one of these common questions:**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔔 How do Beacons work and how often are they evaluated?", use_container_width=True):
                st.session_state.messages.append(HumanMessage(content="How do Beacons work and how often are they evaluated?"))
                st.rerun()
            if st.button("⚡ What caused incident INC-2025-11 and how was it fixed?", use_container_width=True):
                st.session_state.messages.append(HumanMessage(content="What caused the Warehouse Sync duplicate rows incident (INC-2025-11) and how was deduplication fixed?"))
                st.rerun()

        with col2:
            if st.button("💰 What is the overage fee and allowance on Growth?", use_container_width=True):
                st.session_state.messages.append(HumanMessage(content="How much does overage cost on the Growth plan, and what is the monthly event allowance?"))
                st.rerun()
            if st.button("🛡️ How long are raw events kept in cold storage?", use_container_width=True):
                st.session_state.messages.append(HumanMessage(content="How long are raw events kept in cold storage after the plan retention window ends?"))
                st.rerun()

        st.divider()


# ─────────────────────────────────────────────
# 5. Render Conversation & Active Turn
# ─────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    is_generating = bool(st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage))

    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if is_generating:
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

                status_area.empty()

                if final_answer:
                    st.session_state.messages.append(AIMessage(content=final_answer))
                    st.rerun()

            except Exception as e:
                status_area.empty()
                err_msg = f"Error: {e}"
                st.error(err_msg)
                st.info("Check your `.env` file: is `GROQ_API_KEY` configured?")
                st.session_state.messages.append(AIMessage(content=err_msg))
                st.rerun()


# ─────────────────────────────────────────────
# 6. Chat Input & Processing
# ─────────────────────────────────────────────
chat_input = st.chat_input("Ask about Kestrel (e.g. Is Trails available on the Starter plan?)")

# Auto-focus chat input bar
st.html(
    """
    <script>
    function focusInput() {
        try {
            const doc = window.parent ? window.parent.document : document;
            const textarea = doc.querySelector('textarea[data-testid="stChatInputTextArea"]');
            if (textarea) {
                textarea.focus();
            }
        } catch (e) {}
    }
    setTimeout(focusInput, 150);
    setTimeout(focusInput, 500);
    </script>
    """
)

if chat_input:
    st.session_state.messages.append(HumanMessage(content=chat_input))
    st.rerun()
