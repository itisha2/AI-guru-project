"""
Chat page — converse with the AI Guru.
Maintains per-session memory via LangGraph MemorySaver + thread_id.
Shows a full Answer Provenance Trail after each response.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from backend.config import EMBEDDING_MODEL, LLM_MODEL
from backend.data_loader import DATASET_REGISTRY
from backend.rag_graph import create_rag_graph, retrieve_docs, stream_guru
from backend.vector_store import collection_exists

st.set_page_config(page_title="Chat — AI Guru", page_icon="💬", layout="wide")

# ─── Session state ───────────────────────────────────────────────────────────

if "graph" not in st.session_state:
    st.session_state.graph = create_rag_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_provenance" not in st.session_state:
    st.session_state.show_provenance = True

# ─── Header ──────────────────────────────────────────────────────────────────

st.title("💬 Chat with the Guru")

col1, col2 = st.columns([3, 1])
with col2:
    st.session_state.show_provenance = st.toggle(
        "Show provenance", value=st.session_state.show_provenance
    )
    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.graph = create_rag_graph()
        st.rerun()

with col1:
    st.caption(f"Session ID: `{st.session_state.thread_id[:8]}…`")

if not collection_exists():
    st.error(
        "Knowledge base not found. Run `python scripts/ingest_data.py` first."
    )
    st.stop()


# ─── Source badge helpers ─────────────────────────────────────────────────────

_SOURCE_COLORS = {
    "gita_yaml":        ("🟩", "#2d6a4f"),
    "alpaca_qa":        ("🟦", "#1d3557"),
    "pranesh_json":     ("🟧", "#7f4f24"),
    "jdhruv14_qa":      ("🟪", "#4a0e8f"),
    "utkarsh_gita":     ("🟥", "#9b2226"),
    "modotte_infinity":  ("🟨", "#6d6a00"),
    "jdhruv14_dataset":  ("🟫", "#5c3d2e"),
}

def _source_badge(source: str) -> str:
    emoji, _ = _SOURCE_COLORS.get(source, ("⬜", "#333"))
    label = DATASET_REGISTRY.get(source, {}).get("label", source)
    return f"{emoji} `{label}`"

def _score_badge(score: float) -> str:
    if score > 0.7:
        return "🟢"
    if score > 0.5:
        return "🟡"
    return "🔴"


# ─── Provenance Trail renderer ───────────────────────────────────────────────

def _render_provenance(query: str, docs: list, turn_key: str) -> None:
    """Render a collapsible step-by-step provenance trail."""
    with st.expander(
        f"🔍 Answer Provenance Trail — {len(docs)} passages retrieved",
        expanded=False,
    ):
        # Step 1 — Query encoding
        st.markdown("##### Step 1 · Query Encoding")
        st.markdown(
            f"""
| Field | Value |
|-------|-------|
| Query | *{query}* |
| Embedding model | `{EMBEDDING_MODEL}` |
| Vector store | ChromaDB (cosine similarity) |
"""
        )

        st.divider()

        # Step 2 — Retrieved passages (sorted highest → lowest similarity)
        sorted_docs = sorted(docs, key=lambda d: float(d.get("score", 0)), reverse=True)
        st.markdown(f"##### Step 2 · Vector Retrieval (top-{len(sorted_docs)} passages, highest similarity first)")

        if not sorted_docs:
            st.warning("No passages were retrieved — knowledge base may be empty.")
        else:
            for rank, doc in enumerate(sorted_docs, 1):
                meta = doc["metadata"]
                score = float(doc.get("score", 0))
                chapter = meta.get("chapter", "?")
                verse = meta.get("verse", "?")
                source = meta.get("source", "unknown")
                sources_str = meta.get("sources", source)

                sbadge = _score_badge(score)

                # primary label: Chapter N · Verse V
                if chapter and chapter != 0:
                    loc = f"Chapter {chapter} · Verse {verse}"
                else:
                    loc = "Q&A entry"

                # contributing source badges
                source_badges = " ".join(
                    _source_badge(s.strip())
                    for s in sources_str.split(",") if s.strip()
                ) if sources_str else _source_badge(source)

                col_a, col_b = st.columns([1, 3])
                with col_a:
                    st.metric(
                        label=f"#{rank} Similarity",
                        value=f"{score:.3f}",
                        delta=f"{sbadge} {'High' if score > 0.7 else 'Medium' if score > 0.5 else 'Low'}",
                        delta_color="off",
                    )
                    doc_id = doc.get("id") or ""
                    if doc_id:
                        st.markdown(
                            f'<a href="/ChromaDB_Browser?doc_id={doc_id}" target="_blank" '
                            f'style="font-size:0.75rem;">🔗 View in Browser</a>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f"`{doc_id[:20]}…`")
                with col_b:
                    st.markdown(
                        f"**{loc}**  \n"
                        f"**Sources:** {source_badges}  \n"
                        f"**Preview:** {doc['content'][:300]}{'…' if len(doc['content']) > 300 else ''}"
                    )
                if rank < len(sorted_docs):
                    st.divider()

        st.divider()

        # Step 3 — Context assembly
        st.markdown("##### Step 3 · Context Assembly")
        st.markdown(
            f"""
The **{len(docs)} retrieved passages** were assembled into a grounding context block
and prepended to the LLM's system prompt. The Guru was instructed to use the
wisdom of these passages naturally — without citing chapter/verse numbers or quoting Sanskrit.
"""
        )

        with st.expander("View context block sent to LLM", expanded=False):
            if docs:
                context_parts = []
                for d in sorted_docs:
                    meta = d["metadata"]
                    ch = meta.get("chapter", "?")
                    vs = meta.get("verse", "?")
                    tag = (
                        f"[Gita Ch.{ch} V.{vs}]"
                        if ch and ch != 0
                        else "[Gita Teaching]"
                    )
                    context_parts.append(f"{tag}\n{d['content'][:400]}")
                st.code(
                    "\n\n---\n\n".join(context_parts),
                    language="text",
                )
            else:
                st.info("No context — Guru responded from base knowledge only.")

        st.divider()

        # Step 4 — Generation
        st.markdown("##### Step 4 · Generation")
        st.markdown(
            f"""
| Field | Value |
|-------|-------|
| LLM | `{LLM_MODEL}` (via Ollama) |
| Temperature | `0.7` |
| Conversation memory | LangGraph `MemorySaver` (per thread) |
| Session thread | `{st.session_state.thread_id[:8]}…` |
"""
        )


# ─── Render conversation history ──────────────────────────────────────────────

for i, turn in enumerate(st.session_state.chat_history):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if (
            turn["role"] == "assistant"
            and st.session_state.show_provenance
            and turn.get("docs") is not None
        ):
            _render_provenance(turn.get("query", ""), turn["docs"], f"hist_{i}")


# ─── Example prompts ─────────────────────────────────────────────────────────

EXAMPLE_PROMPTS = [
    "I'm overwhelmed by expectations from my family. What should I do?",
    "How do I stop being afraid of failure?",
    "What does it mean to live with purpose?",
    "I feel like my work doesn't matter. How do I find meaning?",
    "How do I deal with a person who constantly betrays my trust?",
]

with st.expander("💡 Example questions to get started"):
    for p in EXAMPLE_PROMPTS:
        if st.button(p, key=p):
            st.session_state["prefill"] = p
            st.rerun()

# ─── Input ────────────────────────────────────────────────────────────────────

prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input(
    "Ask the Guru anything about life, purpose, or inner struggle …",
    key="chat_input",
)
if prefill:
    user_input = prefill

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input, "docs": [], "query": user_input}
    )

    with st.chat_message("assistant"):
        with st.spinner("The digital agent is thinking..."):
            retrieved, query = retrieve_docs(user_input)

        # stream tokens as they arrive
        answer = st.write_stream(
            stream_guru(
                st.session_state.graph,
                user_input,
                st.session_state.thread_id,
                retrieved,
            )
        )

        if st.session_state.show_provenance:
            _render_provenance(query, retrieved, "latest")

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "docs": retrieved,
            "query": query,
        }
    )
    st.rerun()
