"""
Knowledge Base page — browse and search all seven indexed Gita datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from backend.data_loader import DATASET_REGISTRY
from backend.vector_store import collection_exists, collection_stats, similarity_search_with_scores

st.set_page_config(page_title="Knowledge Base — AI Guru", page_icon="📚", layout="wide")

st.title("📚 Knowledge Base")

if not collection_exists():
    st.error("Knowledge base not found. Run `python scripts/ingest_data.py` first.")
    st.stop()

# ─── Stats overview ──────────────────────────────────────────────────────────

stats = collection_stats()

col1, col2, col3 = st.columns(3)
col1.metric("Total Documents", stats["total"])
col2.metric("Datasets Indexed", len(stats["sources"]))
col3.metric("Chapters Covered", len([c for c in stats["chapters"] if c != 0]))

st.divider()

# ─── Dataset cards ───────────────────────────────────────────────────────────

st.subheader("Indexed Datasets")
st.caption(
    "All seven Gita datasets are merged into a single ChromaDB collection. "
    "Each document retains its `source` tag so the Guru can attribute answers."
)

_SOURCE_ICONS = {
    "gita_yaml":         "🟩",
    "alpaca_qa":         "🟦",
    "pranesh_json":      "🟧",
    "jdhruv14_qa":       "🟪",
    "utkarsh_gita":      "🟥",
    "modotte_infinity":  "🟨",
    "jdhruv14_dataset":  "🟫",
}

_TYPE_ICONS = {
    "Verse translations":    "📖",
    "Q&A instruction pairs": "❓",
    "Multi-field verse JSON":"🗂️",
    "Q&A pairs":             "💬",
    "Structured verse dataset": "🏗️",
    "Extended verse content":"🌐",
}

for src_key, info in DATASET_REGISTRY.items():
    count = stats["sources"].get(src_key, 0)
    icon = _SOURCE_ICONS.get(src_key, "⬜")
    type_icon = _TYPE_ICONS.get(info["type"], "📄")

    with st.container(border=True):
        header_col, count_col = st.columns([4, 1])
        with header_col:
            st.markdown(
                f"#### {icon} {info['label']}  \n"
                f"{type_icon} *{info['type']}*"
            )
        with count_col:
            st.metric("Documents", count if count else "—")

        st.markdown(info["description"])

        with st.expander("Schema & indexing details"):
            st.markdown(
                f"**Fields available:** `{info['fields']}`  \n"
                f"**How indexed:** {info['index_key']}  \n"
                f"**Source tag in metadata:** `source = \"{src_key}\"`  \n"
                f"**URL:** {info['url']}"
            )

st.divider()

# ─── Chapter distribution ────────────────────────────────────────────────────

chapter_data = {
    str(k): v for k, v in sorted(stats["chapters"].items()) if k != 0
}
if chapter_data:
    st.subheader("Documents per Chapter")
    st.caption("Counts across all verse-based datasets (Q&A entries not shown here).")
    ch_df = pd.DataFrame(
        {"Chapter": list(chapter_data.keys()), "Documents": list(chapter_data.values())}
    )
    st.bar_chart(ch_df.set_index("Chapter"))

st.divider()

# ─── Source breakdown table ──────────────────────────────────────────────────

st.subheader("Source Distribution")
src_rows = []
for k, v in sorted(stats["sources"].items(), key=lambda x: -x[1]):
    info = DATASET_REGISTRY.get(k, {})
    src_rows.append({
        "Source key": k,
        "Dataset": info.get("label", k),
        "Type": info.get("type", "—"),
        "Documents": v,
        "% of total": f"{v / stats['total'] * 100:.1f}%" if stats["total"] else "—",
    })

st.dataframe(
    pd.DataFrame(src_rows),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ─── Semantic search ─────────────────────────────────────────────────────────

st.subheader("🔍 Semantic Search")
st.caption(
    "Search across all seven datasets simultaneously using natural language. "
    "Returns the most semantically similar passages ranked by cosine similarity."
)

search_col, filter_col = st.columns([3, 1])
with search_col:
    query = st.text_input(
        "Search query",
        placeholder="e.g. detachment from outcomes, fear of death, purpose of life …",
    )
with filter_col:
    k = st.slider("Results", min_value=1, max_value=20, value=5)

if query:
    with st.spinner("Searching …"):
        results = similarity_search_with_scores(query, k=k)

    st.markdown(f"**{len(results)} results** for *\"{query}\"*")
    st.divider()

    for rank, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        chapter = meta.get("chapter", "?")
        verse = meta.get("verse", "?")
        source = meta.get("source", "unknown")
        source_label = meta.get("source_label") or DATASET_REGISTRY.get(source, {}).get("label", source)

        sbadge = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"
        icon = _SOURCE_ICONS.get(source, "⬜")
        loc = f"Ch. {chapter}, V. {verse}" if chapter and chapter != 0 else "Q&A entry"

        with st.expander(
            f"{sbadge} #{rank} — {loc}  |  similarity: {score:.4f}  |  {icon} {source_label}"
        ):
            st.markdown(doc.page_content)

            detail_cols = st.columns(3)
            if meta.get("sanskrit"):
                detail_cols[0].caption(f"**Sanskrit:** {meta['sanskrit']}")
            if meta.get("transliteration"):
                detail_cols[1].caption(f"**Transliteration:** {meta['transliteration']}")
            if meta.get("question"):
                detail_cols[2].caption(f"**Original question:** {meta['question']}")
