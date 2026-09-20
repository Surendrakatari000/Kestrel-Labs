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
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    },
)

# Hide Streamlit default UI chrome (deploy button, hamburger menu, header, footer)
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden !important; display: none !important;}
    header {visibility: hidden !important; height: 0px !important;}
    [data-testid="stHeader"] {visibility: hidden !important; height: 0px !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    .stDeployButton {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important; display: none !important;}
    #manage-app-button {display: none !important;}
    ul[data-testid="main-menu-list"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load environment
load_dotenv()

import time
from src.graph import build_graph
from src.vectorstore import get_collection, get_embedding_function, EMBEDDING_MODEL_NAME
from src.utils.citations import extract_cited_sources

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
header_col1, header_col2 = st.columns([0.72, 0.28], vertical_alignment="bottom")
with header_col1:
    st.title("🐦 Kestrel Labs Research Assistant")
    st.caption(
        f"Your AI assistant for Kestrel Labs • **{chunk_count} docs indexed** • **4 active agents**"
    )

with header_col2:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────────
# 3. Welcome & Starter Suggestions (when empty)
# ─────────────────────────────────────────────
def ask_question(q: str):
    """Callback fired on button click before the script runs."""
    st.session_state.messages.append(HumanMessage(content=q))


if not st.session_state.messages:
    st.markdown("""
    ### 👋 Welcome! How can I help you today?
    Ask any question about Kestrel Labs products, pricing plans, engineering runbooks, or company policies.
    I search official company documents to give you verified, up-to-date answers with source citations.
    """)

    st.markdown("**Try asking one of these common questions:**")

    col1, col2 = st.columns(2)
    with col1:
        st.button(
            "🔔 Beacons & Evaluation Frequency",
            use_container_width=True,
            help="What is a Beacon and how often is it evaluated?",
            on_click=ask_question,
            args=("How do Beacons work and how often are they evaluated?",),
        )
        st.button(
            "⚡ Incident INC-2025-11 Deduplication",
            use_container_width=True,
            help="What caused the Warehouse Sync duplicate rows incident (INC-2025-11) and how was deduplication fixed?",
            on_click=ask_question,
            args=("What caused the Warehouse Sync duplicate rows incident (INC-2025-11) and how was deduplication fixed?",),
        )

    with col2:
        st.button(
            "💰 Growth Plan Overage & Limits",
            use_container_width=True,
            help="How much does overage cost on the Growth plan, and what is the monthly event allowance?",
            on_click=ask_question,
            args=("How much does overage cost on the Growth plan, and what is the monthly event allowance?",),
        )
        st.button(
            "🛡️ Cold Storage Retention Policy",
            use_container_width=True,
            help="How long are raw events kept in cold storage after the plan retention window ends?",
            on_click=ask_question,
            args=("How long are raw events kept in cold storage after the plan retention window ends?",),
        )

    st.divider()


# ─────────────────────────────────────────────
# 4. Render Conversation & Active Turn
# ─────────────────────────────────────────────
def render_assistant_content(content: str, steps: list = None, stream: bool = False):
    """Renders the assistant response, agent workflow trace, and citations."""
    if steps:
        with st.expander("🤖 Multi-Agent Workflow Trace", expanded=False):
            for step in steps:
                st.markdown(step)

    if stream:
        def _token_stream():
            lines = content.split("\n")
            for i, line in enumerate(lines):
                words = line.split(" ")
                for j, word in enumerate(words):
                    yield word + (" " if j < len(words) - 1 else "")
                    time.sleep(0.01)
                if i < len(lines) - 1:
                    yield "\n"

        st.write_stream(_token_stream)
    else:
        st.markdown(content)

    sources = extract_cited_sources(content)
    if sources:
        with st.expander(f"📚 Citations & Evidence ({len(sources)} sources)", expanded=False):
            for s in sources:
                cat = f" • *{s['category'].title()}*" if s.get('category') else ""
                ver = f" (v{s['version']})" if s.get('version') else ""
                st.markdown(f"- **`{s['chunk_id']}`**: {s['title']}{ver}{cat}")


# Render all conversation history up to the latest turn
for msg in st.session_state.messages[:-1] if (st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage)) else st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        if role == "assistant":
            steps = msg.additional_kwargs.get("steps", [])
            render_assistant_content(msg.content, steps=steps, stream=False)

        else:
            st.markdown(msg.content)

# If the latest message is a user message needing an answer, run the assistant!
if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
    latest_user_msg = st.session_state.messages[-1]
    with st.chat_message("user"):
        st.markdown(latest_user_msg.content)

    with st.chat_message("assistant"):
        final_answer = ""
        recorded_steps = []

        state_input = {
            "messages": list(st.session_state.messages),
            "retry_count": 0,
        }

        with st.status("🤖 Multi-Agent Workflow in progress...", expanded=True) as status_box:
            try:
                for step_output in st.session_state.graph.stream(state_input):
                    for node_name, node_state in step_output.items():
                        if node_name == "router":
                            qtype = node_state.get("query_type", "single_hop")
                            q = node_state.get("current_query", "")
                            step_text = f"🧠 **Router Agent:** Classified as `{qtype}`" + (f" → *\"{q}\"*" if q else "")
                            st.write(step_text)
                            recorded_steps.append(step_text)

                        elif node_name == "retriever":
                            chunks = node_state.get("retrieved_chunks", [])
                            step_text = f"🔍 **Retriever Agent:** Retrieved **{len(chunks)} evidence chunks** from vector store"
                            st.write(step_text)
                            recorded_steps.append(step_text)

                        elif node_name == "synthesizer":
                            step_text = "✍️ **Synthesizer Agent:** Drafted answer with grounded citations"
                            st.write(step_text)
                            recorded_steps.append(step_text)

                        elif node_name == "verifier":
                            final_answer = node_state.get("final_answer", "")
                            verdicts = node_state.get("verifier_verdicts", [])
                            all_sup = node_state.get("overall_supported", True)
                            icon = "✅" if all_sup else "⚠️"
                            note = "All claims verified against evidence" if all_sup else "Audited claims & qualified unbacked statements"
                            step_text = f"🛡️ **Verifier Agent:** Audited {len(verdicts)} factual claims ({icon} {note})"
                            st.write(step_text)
                            recorded_steps.append(step_text)

                        elif node_name == "unsupported":
                            final_answer = node_state.get("final_answer", "")
                            step_text = "⚡ **Unsupported Node:** Conversational fast-path response"
                            st.write(step_text)
                            recorded_steps.append(step_text)

                        elif node_name == "increment_retry":
                            step_text = "🔄 **Retry Loop:** Re-retrieving context to verify unbacked claims..."
                            st.write(step_text)
                            recorded_steps.append(step_text)

                status_box.update(label="✅ Multi-Agent Workflow Complete", state="complete", expanded=False)

            except Exception as e:
                status_box.update(label="❌ Workflow Error", state="error", expanded=True)
                err_msg = f"Error: {e}"
                st.error(err_msg)
                st.info("Check your `.env` file: is `GROQ_API_KEY` configured?")
                final_answer = err_msg

        if final_answer:
            render_assistant_content(final_answer, steps=None, stream=True)
            st.session_state.messages.append(
                AIMessage(content=final_answer, additional_kwargs={"steps": recorded_steps})
            )



# ─────────────────────────────────────────────
# 5. Chat Input & Processing
# ─────────────────────────────────────────────
chat_input = st.chat_input("Ask about Kestrel (e.g. Is Trails available on the Starter plan?)")

if chat_input:
    st.session_state.messages.append(HumanMessage(content=chat_input))
    st.rerun()

