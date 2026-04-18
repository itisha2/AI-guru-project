"""
Visualize page — three interactive views:
  1. RAG Pipeline Diagram   — Plotly network graph of the data flow
  2. Vector Space Explorer  — UMAP 2-D projection of all document embeddings
  3. Retrieval Inspector    — Enter a query, see retrieved chunks + similarity scores
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.vector_store import collection_exists, get_raw_embeddings, similarity_search_with_scores

st.set_page_config(page_title="Visualize — AI Guru", page_icon="🔬", layout="wide")

st.title("🔬 RAG Visualisation")

if not collection_exists():
    st.error("Knowledge base not found. Run `python scripts/ingest_data.py` first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔄 RAG Pipeline", "🌐 Vector Space", "🔎 Retrieval Inspector"])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — RAG Pipeline Diagram
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.subheader("RAG Pipeline — Data Flow")
    st.caption(
        "How a user question travels through the system to become a Guru response."
    )

    _NODE = (
        "display:inline-block;padding:10px 14px;border-radius:8px;"
        "text-align:center;font-size:13px;font-weight:600;color:#fff;"
        "min-width:100px;line-height:1.4;"
    )
    _ARROW = "font-size:22px;color:#666;align-self:center;padding:0 4px;"
    _DARROW = "font-size:18px;color:#666;align-self:center;padding:0 4px;"

    st.markdown(
        f"""
<div style="display:flex;align-items:center;flex-wrap:nowrap;
            padding:28px 20px;background:#161b22;border-radius:12px;
            overflow-x:auto;gap:0px;">

  <div style="{_NODE}background:#4A90D9;">User<br>Question</div>
  <div style="{_ARROW}">&#8594;</div>
  <div style="{_NODE}background:#7B68EE;">Embedding<br><small style='font-weight:400;font-size:11px'>nomic-embed-text-v1.5</small></div>
  <div style="{_ARROW}">&#8594;</div>
  <div style="{_NODE}background:#E86B4F;">ChromaDB<br><small style='font-weight:400;font-size:11px'>13,307 docs</small></div>
  <div style="{_ARROW}">&#8594;</div>
  <div style="{_NODE}background:#50C878;">Top-5<br>Retrieval</div>
  <div style="{_ARROW}">&#8594;</div>

  <div style="display:flex;flex-direction:column;align-items:center;gap:6px;">
    <div style="{_NODE}background:#888;font-size:11px;min-width:90px;padding:7px 10px;">
      LangGraph<br>MemorySaver
    </div>
    <div style="color:#666;font-size:14px;">&#8597;</div>
    <div style="{_NODE}background:#FFB347;">LLaMA 3.1 8B<br><small style='font-weight:400;font-size:11px'>via Groq</small></div>
  </div>

  <div style="{_ARROW}">&#8594;</div>
  <div style="{_NODE}background:#4A90D9;">Guru<br>Response</div>

</div>
""",
        unsafe_allow_html=True,
    )

    # LangGraph mermaid
    st.subheader("LangGraph State Machine")
    st.markdown(
        """
```mermaid
graph LR
    START([__start__]) --> retrieve
    retrieve["🔍 retrieve<br/>(ChromaDB · top-5 · cosine similarity)"]
    retrieve --> generate
    generate["🤖 generate<br/>(LLaMA 3.1 8B via Groq + system prompt)"]
    generate --> END([__end__])

    memory["💾 MemorySaver<br/>(per-thread checkpointer)"]
    generate -.->|checkpoint| memory
    memory -.->|restore history| generate
```
"""
    )

    st.subheader("State Schema")
    st.code(
        """\
class GururState(TypedDict):
    messages      : Annotated[list, add_messages]  # full conversation history
    retrieved_docs: List[Dict[str, Any]]            # top-K Gita passages
    query         : str                             # last user question
""",
        language="python",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Vector Space (UMAP)
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Vector Space Explorer — UMAP 2-D Projection")
    st.caption(
        "Each dot = one document chunk. Colour = chapter (grey = Q&A pairs). "
        "Hover for the text preview. Clusters show semantic similarity."
    )

    @st.cache_data(show_spinner="Computing UMAP projection …")
    def compute_umap(max_docs: int = 2000):
        raw = get_raw_embeddings()
        embeddings = np.array(raw["embeddings"])
        documents = raw["documents"]
        metadatas = raw["metadatas"]

        # subsample for speed
        if len(embeddings) > max_docs:
            idx = np.random.choice(len(embeddings), max_docs, replace=False)
            embeddings = embeddings[idx]
            documents = [documents[i] for i in idx]
            metadatas = [metadatas[i] for i in idx]

        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(embeddings)

        return pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
                "chapter": [str(m.get("chapter", "Q&A")) for m in metadatas],
                "source": [m.get("source", "?") for m in metadatas],
                "text": [d[:120] + "…" if len(d) > 120 else d for d in documents],
                "verse": [str(m.get("verse", "")) for m in metadatas],
            }
        )

    max_docs = st.slider(
        "Max documents to project (higher = slower but more complete)",
        min_value=200, max_value=3000, value=1000, step=200,
    )

    if st.button("Compute / Refresh Projection", type="primary"):
        st.cache_data.clear()

    df = compute_umap(max_docs)

    color_map = {
        str(c): px.colors.qualitative.Plotly[i % 10]
        for i, c in enumerate(sorted(df["chapter"].unique()))
    }
    color_map["Q&A"] = "#888888"
    color_map["0"] = "#888888"

    fig2 = px.scatter(
        df,
        x="x", y="y",
        color="chapter",
        hover_data={"x": False, "y": False, "text": True, "verse": True, "source": True},
        color_discrete_map=color_map,
        labels={"chapter": "Chapter"},
        title=f"Document Embeddings — UMAP ({len(df)} docs)",
    )
    fig2.update_traces(marker=dict(size=5, opacity=0.7))
    fig2.update_layout(height=600, legend_title_text="Chapter")
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "**How to read this:** Documents that are semantically similar are plotted close together. "
        "Chapter clusters emerge naturally from the embeddings — no chapter labels were used during projection."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Retrieval Inspector
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Retrieval Inspector")
    st.caption(
        "Enter any question to see exactly which Gita passages are retrieved and their similarity scores."
    )

    query = st.text_input(
        "Query",
        placeholder="e.g. How do I stop being attached to results?",
        key="inspect_query",
    )
    k_inspect = st.slider("Top-K", 1, 15, 5, key="inspect_k")

    if query:
        with st.spinner("Retrieving …"):
            results = similarity_search_with_scores(query, k=k_inspect)

        # bar chart of similarity scores
        labels = []
        scores = []
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            ch = meta.get("chapter", "?")
            v = meta.get("verse", "?")
            labels.append(f"#{i} Ch{ch}/V{v}" if ch != 0 else f"#{i} Q&A")
            scores.append(float(score))

        score_df = pd.DataFrame({"Document": labels, "Similarity": scores})
        fig3 = px.bar(
            score_df, x="Document", y="Similarity",
            color="Similarity",
            color_continuous_scale="RdYlGn",
            range_y=[0, 1],
            title=f"Similarity scores for: \"{query}\"",
        )
        fig3.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

        st.divider()

        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            chapter = meta.get("chapter", "?")
            verse = meta.get("verse", "?")
            source = meta.get("source", "unknown")
            badge = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"

            label = (
                f"Ch. {chapter}, V. {verse}" if chapter != 0 else f"Q&A #{i}"
            )
            with st.expander(f"{badge} #{i} — {label}  |  score: {score:.4f}  |  {source}"):
                st.markdown(doc.page_content)
                if meta.get("sanskrit"):
                    st.caption(f"Sanskrit: {meta['sanskrit']}")
