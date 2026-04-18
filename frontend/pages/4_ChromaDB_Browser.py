"""
ChromaDB Browser — MongoDB-Compass-style document explorer.

Documents are indexed by (chapter, verse). Each entry merges content
from all contributing sources for that verse.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from backend.vector_store import browse_collection, collection_exists, collection_stats

st.set_page_config(page_title="ChromaDB Browser — AI Guru", page_icon="🗄️", layout="wide")

st.title("🗄️ ChromaDB Browser")
st.caption(
    "Every document in the vector store — indexed by chapter & verse. "
    "Hover a row to expand full content and metadata."
)

if not collection_exists():
    st.error("Knowledge base not found. Run `python scripts/ingest_data.py` first.")
    st.stop()

# ─── Index structure overview ────────────────────────────────────────────────

stats = collection_stats()
verse_count = stats["total"] - stats["chapters"].get(0, 0)
qa_count = stats["chapters"].get(0, 0)
chapter_count = len([c for c in stats["chapters"] if c != 0])

st.markdown("### Index Structure")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Documents", f"{stats['total']:,}")
col2.metric("Verse Documents", f"{verse_count:,}", help="One merged doc per unique chapter+verse")
col3.metric("Q&A Documents", f"{qa_count:,}", help="Standalone Q&A pairs without a verse coordinate")
col4.metric("Chapters Covered", chapter_count)

st.markdown(
    """
**How the index is organised:**

| Index key | Type | Count | Description |
|-----------|------|-------|-------------|
| `(chapter, verse)` | Merged verse | **{verse}** | One document per unique Gita verse (Ch. 1–18). Content from all contributing datasets is merged into a single rich document. Embedding captures combined meaning. |
| `chapter = 0` | Q&A / unlocated | **{qa}** | Standalone question-answer pairs that are not linked to a specific verse. Indexed by semantic content only. |

**Embedding model:** `nomic-embed-text` — 768-dimensional vectors via Ollama
**Similarity metric:** Cosine similarity
**Collection name:** `gita_knowledge`
**Storage:** ChromaDB persistent store at `data/chroma_db/`
""".format(verse=f"{verse_count:,}", qa=f"{qa_count:,}")
)

# per-chapter document count
chapter_data = {str(k): v for k, v in sorted(stats["chapters"].items()) if k != 0}
if chapter_data:
    import plotly.express as px
    ch_df = pd.DataFrame({"Chapter": list(chapter_data.keys()), "Documents": list(chapter_data.values())})
    fig_ch = px.bar(
        ch_df, x="Chapter", y="Documents",
        color="Documents", color_continuous_scale="Blues",
        title="Indexed documents per chapter",
        height=250,
    )
    fig_ch.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_ch, use_container_width=True)

st.divider()

# ─── Sidebar: collection stats + filters ─────────────────────────────────────

with st.sidebar:
    st.header("Collection Info")
    st.metric("Total Documents", stats["total"])
    st.metric("Verse Documents", verse_count)
    st.metric("Q&A Documents", qa_count)
    st.metric("Chapters Covered", chapter_count)

    st.divider()
    st.header("Filters")

    chapter_nums = sorted([c for c in stats["chapters"].keys() if c != 0])
    chapter_options = ["(all chapters)"] + [f"Chapter {c}" for c in chapter_nums] + ["Q&A only"]
    selected_chapter = st.selectbox("Chapter", chapter_options)

    verse_input = st.text_input("Verse number", placeholder="e.g. 47")

    keyword = st.text_input("Keyword in text", placeholder="e.g. detachment, duty …")

    page_size = st.select_slider("Per page", options=[10, 25, 50, 100], value=25)

    st.divider()
    st.header("Jump to Document ID")
    doc_id_input = st.text_input("ChromaDB ID", placeholder="Paste a document ID …")


# ─── Build `where` filter ────────────────────────────────────────────────────

where: dict | None = None

if selected_chapter == "Q&A only":
    where = {"chapter": {"$eq": 0}}
elif selected_chapter != "(all chapters)":
    ch_val = int(selected_chapter.split()[1])
    where = {"chapter": {"$eq": ch_val}}


# ─── Direct ID lookup ────────────────────────────────────────────────────────

if doc_id_input.strip():
    import chromadb
    from backend.config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    col = client.get_collection(CHROMA_COLLECTION_NAME)
    try:
        res = col.get(ids=[doc_id_input.strip()], include=["documents", "metadatas"])
        if res["ids"]:
            st.subheader(f"Document: `{res['ids'][0]}`")
            st.markdown(res["documents"][0])
            st.json(res["metadatas"][0])
        else:
            st.warning("No document found with that ID.")
    except Exception as e:
        st.error(f"Lookup failed: {e}")
    st.divider()


# ─── Paginated browser ───────────────────────────────────────────────────────

if "browser_page" not in st.session_state:
    st.session_state.browser_page = 0

filter_key = (selected_chapter, verse_input, keyword, page_size)
if st.session_state.get("_last_filter") != filter_key:
    st.session_state.browser_page = 0
    st.session_state["_last_filter"] = filter_key

offset = st.session_state.browser_page * page_size

# combine keyword + verse filters for client-side pass
combined_keyword = keyword.strip() or None
if verse_input.strip():
    combined_keyword = (combined_keyword + " " if combined_keyword else "") + f"Verse {verse_input.strip()}"

with st.spinner("Loading …"):
    result = browse_collection(
        offset=offset,
        limit=page_size,
        where=where,
        keyword=combined_keyword,
    )

total = result["total"]
total_pages = max(1, (total + page_size - 1) // page_size)
current_page = st.session_state.browser_page


# ─── Header + nav ────────────────────────────────────────────────────────────

header_col, nav_col = st.columns([3, 2])
with header_col:
    parts = []
    if selected_chapter != "(all chapters)":
        parts.append(selected_chapter)
    if verse_input.strip():
        parts.append(f"verse {verse_input.strip()}")
    if keyword:
        parts.append(f'"{keyword}"')
    subtitle = "  ·  ".join(parts) if parts else "all documents"
    st.markdown(f"**{total:,} documents** — {subtitle}")

with nav_col:
    btn_cols = st.columns([1, 2, 1])
    with btn_cols[0]:
        if st.button("← Prev", disabled=current_page == 0):
            st.session_state.browser_page -= 1
            st.rerun()
    with btn_cols[1]:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px'>Page {current_page+1} / {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with btn_cols[2]:
        if st.button("Next →", disabled=current_page >= total_pages - 1):
            st.session_state.browser_page += 1
            st.rerun()

st.divider()


# ─── Document cards ───────────────────────────────────────────────────────────

SOURCE_ICONS = {
    "gita_yaml":         "🟩",
    "alpaca_qa":         "🟦",
    "pranesh_json":      "🟧",
    "jdhruv14_qa":       "🟪",
    "utkarsh_gita":      "🟥",
    "modotte_infinity":  "🟨",
    "jdhruv14_dataset":  "🟫",
    "merged":            "🔵",
}

if not result["documents"]:
    st.info("No documents match the current filters.")
else:
    for doc_text, meta, doc_id in zip(result["documents"], result["metadatas"], result["ids"]):
        chapter = meta.get("chapter", 0)
        verse = meta.get("verse", "?")
        sources_str = meta.get("sources", meta.get("source", "unknown"))
        source_icons = " ".join(
            SOURCE_ICONS.get(s.strip(), "⬜") for s in sources_str.split(",") if s.strip()
        )

        if chapter and chapter != 0:
            loc = f"Ch. {chapter}  ·  V. {verse}"
        else:
            loc = "Q&A"

        n_sources = len([s for s in sources_str.split(",") if s.strip()])
        src_label = meta.get("source_label", sources_str)
        header = f"{source_icons}  **{loc}**  ·  {src_label}"

        with st.expander(header):
            content_col, meta_col = st.columns([3, 2])

            with content_col:
                st.markdown("**Content**")
                st.markdown(doc_text)

            with meta_col:
                st.markdown("**Metadata**")

                # show contributing sources as badges
                if sources_str and sources_str != "unknown":
                    badge_line = " ".join(
                        f"{SOURCE_ICONS.get(s.strip(), '⬜')} `{s.strip()}`"
                        for s in sources_str.split(",") if s.strip()
                    )
                    st.markdown(f"**Sources:** {badge_line}")

                meta_rows = [
                    {"Field": k, "Value": str(v)[:200]}
                    for k, v in meta.items()
                    if k not in ("source", "sources", "source_label")
                ]
                if meta_rows:
                    st.dataframe(pd.DataFrame(meta_rows), hide_index=True, use_container_width=True)
                st.caption(f"**ID:** `{doc_id}`")


# ─── Bottom nav ───────────────────────────────────────────────────────────────

st.divider()
bot_cols = st.columns([1, 3, 1])
with bot_cols[0]:
    if st.button("← Prev ", disabled=current_page == 0, key="prev_bot"):
        st.session_state.browser_page -= 1
        st.rerun()
with bot_cols[1]:
    showing_from = offset + 1
    showing_to = min(offset + page_size, total)
    st.markdown(
        f"<div style='text-align:center;padding-top:6px'>"
        f"Showing {showing_from:,}–{showing_to:,} of {total:,}</div>",
        unsafe_allow_html=True,
    )
with bot_cols[2]:
    if st.button("Next → ", disabled=current_page >= total_pages - 1, key="next_bot"):
        st.session_state.browser_page += 1
        st.rerun()
