"""
ChromaDB vector store management.
Provides create / load / retrieval helpers used by the RAG graph and frontend.
"""

from __future__ import annotations

from typing import List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from .config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def create_vector_store(documents: List[Document], batch_size: int = 100) -> Chroma:
    """Embed all documents and persist to ChromaDB."""
    print(f"Creating vector store with {len(documents)} documents …")
    embeddings = _get_embeddings()

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )

    # add in batches to avoid OOM on large corpora
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        vector_store.add_documents(batch)
        print(f"  Indexed {min(i + batch_size, len(documents))} / {len(documents)}")

    print("Vector store created and persisted.")
    return vector_store


def load_vector_store() -> Chroma:
    """Load an existing persisted ChromaDB collection."""
    embeddings = _get_embeddings()
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )


def get_retriever(k: int = 5):
    """Return a LangChain retriever over the persisted collection."""
    vs = load_vector_store()
    return vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def similarity_search_with_scores(query: str, k: int = 5):
    """Return (Document, score) pairs for a query."""
    vs = load_vector_store()
    return vs.similarity_search_with_score(query, k=k)


def collection_exists() -> bool:
    """Return True if the ChromaDB collection exists and has documents."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        col = client.get_collection(CHROMA_COLLECTION_NAME)
        return col.count() > 0
    except Exception:
        return False


def get_raw_embeddings():
    """
    Return raw embeddings + metadata from ChromaDB for visualisation.
    Returns dict with keys: embeddings, documents, metadatas, ids
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    col = client.get_collection(CHROMA_COLLECTION_NAME)
    return col.get(include=["embeddings", "documents", "metadatas"])


def collection_stats() -> dict:
    """Return basic stats about the collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    col = client.get_collection(CHROMA_COLLECTION_NAME)
    total = col.count()

    # chapter distribution
    result = col.get(include=["metadatas"])
    from collections import Counter
    chapters = Counter(
        m.get("chapter", 0) for m in result["metadatas"]
    )
    sources = Counter(
        m.get("source", "unknown") for m in result["metadatas"]
    )
    return {"total": total, "chapters": dict(chapters), "sources": dict(sources)}
