"""
ChromaDB vector store management.
Provides create / load / retrieval helpers used by the RAG graph and frontend.
"""

from __future__ import annotations

from typing import List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR, EMBEDDING_MODEL

_embeddings_cache: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_cache


_vs_cache: Optional[Chroma] = None


def get_vector_store() -> Chroma:
    """Return a cached Chroma instance (created once per process)."""
    global _vs_cache
    if _vs_cache is None:
        _vs_cache = load_vector_store()
    return _vs_cache


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
    return get_vector_store().similarity_search_with_score(query, k=k)


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


def browse_collection(
    offset: int = 0,
    limit: int = 50,
    where: Optional[dict] = None,
    keyword: Optional[str] = None,
) -> dict:
    """
    Paginated browse of the ChromaDB collection.

    Returns dict with:
      - documents: list of page_content strings
      - metadatas: list of metadata dicts
      - ids: list of document IDs
      - total: total documents matching the filter (before pagination)
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    col = client.get_collection(CHROMA_COLLECTION_NAME)

    get_kwargs: dict = {"include": ["documents", "metadatas"]}
    if where:
        get_kwargs["where"] = where

    result = col.get(**get_kwargs)
    docs = result["documents"]
    metas = result["metadatas"]
    ids = result["ids"]

    # keyword filter on page content (client-side)
    if keyword:
        kw = keyword.lower()
        filtered = [
            (d, m, i) for d, m, i in zip(docs, metas, ids)
            if kw in (d or "").lower()
        ]
        docs, metas, ids = zip(*filtered) if filtered else ([], [], [])
        docs, metas, ids = list(docs), list(metas), list(ids)

    total = len(docs)
    return {
        "documents": docs[offset : offset + limit],
        "metadatas": metas[offset : offset + limit],
        "ids": ids[offset : offset + limit],
        "total": total,
    }


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
