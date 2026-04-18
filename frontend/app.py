"""
AI Guru — Main Streamlit Entry Point
Run: streamlit run frontend/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="AI Guru",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI Guru")
st.subheader("A handy philosopher in your pocket")

st.markdown(
    """
Welcome to **AI Guru** — a wisdom companion grounded in the **Bhagavad Gita**.

> *"The wisdom informs the answer, not the vocabulary."*

---

### What is AI Guru?

AI Guru is a **Retrieval-Augmented Generation (RAG)** system powered by:
- **Mistral** (via Ollama) as the language brain
- **ChromaDB** as the vector knowledge base
- **LangGraph** for orchestration and conversation memory
- **LangChain** for the RAG pipeline

It is designed for **philosophical interactions** — existential questions, life dilemmas,
inner conflicts — not for factual lookups.

---

### Navigate using the sidebar:

| Page | Description |
|------|-------------|
| 💬 **Chat** | Converse with the Guru |
| 📚 **Knowledge Base** | Browse and search indexed Gita content |
| 🔬 **Visualize** | Explore the vector space and RAG pipeline |

---

### Quick setup check
"""
)

from backend.vector_store import collection_exists, collection_stats

if collection_exists():
    stats = collection_stats()
    st.success(
        f"✅ Knowledge base is ready — **{stats['total']} documents** indexed "
        f"({', '.join(f'{v} from {k}' for k, v in stats['sources'].items())})"
    )
else:
    st.error(
        "❌ Knowledge base not found. "
        "Run the ingestion script first:\n\n"
        "```bash\npython scripts/ingest_data.py\n```"
    )
