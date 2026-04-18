"""
LangGraph-based RAG pipeline with conversation memory.

Graph topology:
    START → retrieve → generate → END

Performance notes:
  - ChatOllama and Chroma are cached at module level (not recreated per call)
  - Context passed to LLM is capped at 400 chars per passage
  - A separate stream_guru() path enables token-by-token streaming in the UI
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, Generator, List, TypedDict

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import GROQ_API_KEY, LLM_MODEL, SYSTEM_PROMPT
from .vector_store import collection_exists, get_vector_store


# ─── Module-level LLM cache ─────────────────────────────────────────────────

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=LLM_MODEL,
            temperature=0.7,
            max_tokens=1024,
            api_key=GROQ_API_KEY,
        )
    return _llm


# ─── State ──────────────────────────────────────────────────────────────────

class GururState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_docs: List[Dict[str, Any]]
    query: str


# ─── Shared helpers ──────────────────────────────────────────────────────────

_CONTEXT_CHARS = 500  # max chars per passage sent to the LLM
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean_response(text: str) -> str:
    """Strip DeepSeek R1-style <think>...</think> reasoning blocks."""
    return _THINK_RE.sub("", text).strip()


def _fetch_docs(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top-k passages using the cached vector store."""
    results = get_vector_store().similarity_search_with_score(query, k=k)
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
            "id": getattr(doc, "id", None),
        }
        for doc, score in results
    ]


def _build_context_messages(docs: List[Dict[str, Any]]) -> list:
    """Build the system context block from retrieved passages."""
    if not docs:
        return []
    parts = []
    for i, d in enumerate(sorted(docs, key=lambda x: x["score"], reverse=True), 1):
        parts.append(f"[Insight {i}]\n{d['content'][:_CONTEXT_CHARS]}")
    context_block = "\n\n---\n\n".join(parts)
    return [
        SystemMessage(
            content=(
                "Use the following philosophical insights to inform your response. "
                "Do NOT quote them, reference them, or reveal their source. "
                "Absorb the wisdom and express it entirely in your own voice:\n\n"
                + context_block
            )
        )
    ]


# ─── Node helpers ───────────────────────────────────────────────────────────

def _retrieve(state: GururState) -> Dict[str, Any]:
    query = state["messages"][-1].content
    if not collection_exists():
        return {"retrieved_docs": [], "query": query}
    return {"retrieved_docs": _fetch_docs(query, k=5), "query": query}


def _generate(state: GururState) -> Dict[str, Any]:
    llm = _get_llm()
    docs = state.get("retrieved_docs", [])
    messages_to_send = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + _build_context_messages(docs)
        + list(state["messages"])
    )
    response = llm.invoke(messages_to_send)
    return {"messages": [response]}


# ─── Graph factory ──────────────────────────────────────────────────────────

def create_rag_graph():
    graph = StateGraph(GururState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate", _generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ─── Non-streaming wrapper (kept for compatibility) ──────────────────────────

def ask_guru(compiled_graph, question: str, thread_id: str) -> Dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    result = compiled_graph.invoke(
        {"messages": [HumanMessage(content=question)]}, config=config
    )
    answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            answer = _clean_response(msg.content)
            break
    return {
        "answer": answer,
        "retrieved_docs": result.get("retrieved_docs", []),
        "query": result.get("query", question),
    }


# ─── Streaming path ──────────────────────────────────────────────────────────

def retrieve_docs(question: str) -> tuple[List[Dict[str, Any]], str]:
    """
    Retrieve relevant passages without running the full graph.
    Returns (docs, query). Fast — uses cached vector store.
    """
    return _fetch_docs(question, k=5), question


def stream_guru(
    compiled_graph,
    question: str,
    thread_id: str,
    docs: List[Dict[str, Any]],
) -> Generator[str, None, None]:
    """
    Stream LLM tokens for a question given pre-fetched docs.

    After the full response is produced the AIMessage is committed to the
    LangGraph MemorySaver so conversation history is preserved.
    """
    llm = _get_llm()

    # Reconstruct message history from the graph's checkpointed state
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = compiled_graph.get_state(config)
    prior_messages: list = []
    if checkpoint and checkpoint.values:
        prior_messages = list(checkpoint.values.get("messages", []))

    messages_to_send = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + _build_context_messages(docs)
        + prior_messages
        + [HumanMessage(content=question)]
    )

    full_response = []
    buffer = ""
    in_think = False
    for chunk in llm.stream(messages_to_send):
        if not chunk.content:
            continue
        buffer += chunk.content
        full_response.append(chunk.content)

        # suppress <think>...</think> blocks from the streamed UI output
        if "<think>" in buffer:
            in_think = True
        if in_think:
            if "</think>" in buffer:
                in_think = False
                # yield only what comes after the closing tag
                after = buffer.split("</think>", 1)[1]
                buffer = after
                if after:
                    yield after
            # still inside think block — don't yield
        else:
            yield chunk.content
            buffer = ""

    # Persist the full turn into the graph's memory
    compiled_graph.invoke(
        {"messages": [HumanMessage(content=question), AIMessage(content="".join(full_response))]},
        config=config,
    )
