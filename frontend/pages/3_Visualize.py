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

    nodes = [
        (0.05, 0.5, "User\nQuestion", "#4A90D9", 40),
        (0.22, 0.5, "Embedding\n(nomic-embed-text)", "#7B68EE", 40),
        (0.40, 0.5, "ChromaDB\nVector Store", "#E86B4F", 50),
        (0.58, 0.5, "Top-K\nRetrieval", "#50C878", 40),
        (0.75, 0.5, "LLM\n(Mistral)", "#FFB347", 50),
        (0.92, 0.5, "Guru\nResponse", "#4A90D9", 40),
        # LangGraph memory loop
        (0.75, 0.15, "LangGraph\nMemorySaver", "#C0C0C0", 35),
    ]

    edges = [
        (0, 1, "encode query"),
        (1, 2, "similarity search"),
        (2, 3, "top-5 docs"),
        (3, 4, "augmented prompt"),
        (4, 5, "generate"),
        (4, 6, "checkpoint"),
        (6, 4, "history"),
    ]

    fig = go.Figure()

    # draw edges as Scatter lines + arrowhead annotations using data coords
    # (Plotly 6.x dropped axref='paper'; we use axref='x', ayref='y' instead)
    for src_i, dst_i, label in edges:
        x0, y0 = nodes[src_i][0], nodes[src_i][1]
        x1, y1 = nodes[dst_i][0], nodes[dst_i][1]

        # line segment
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color="#888", width=2),
            showlegend=False,
            hoverinfo="skip",
        ))
        # arrowhead at destination node
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3, arrowsize=1.5, arrowwidth=2,
            arrowcolor="#888",
            text="",
        )
        # edge label
        fig.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 0.06,
            xref="x", yref="y",
            text=label, showarrow=False,
            font=dict(size=10, color="#aaa"),
        )

    # draw nodes as filled circles (shapes) + text labels
    for x, y, label, color, size in nodes:
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=x - 0.07, y0=y - 0.22,
            x1=x + 0.07, y1=y + 0.22,
            fillcolor=color, opacity=0.9,
            line=dict(color="white", width=2),
        )
        fig.add_annotation(
            x=x, y=y, xref="x", yref="y",
            text=label.replace("\n", "<br>"),
            showarrow=False,
            font=dict(size=11, color="white"),
            align="center",
        )

    fig.update_layout(
        height=380,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.1, 1.0]),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # LangGraph mermaid
    st.subheader("LangGraph State Machine")
    st.markdown(
        """
```mermaid
graph LR
    START([__start__]) --> retrieve
    retrieve["🔍 retrieve<br/>(ChromaDB similarity search)"]
    retrieve --> generate
    generate["🤖 generate<br/>(Mistral LLM + system prompt)"]
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
