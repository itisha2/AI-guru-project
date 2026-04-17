"""
LangGraph-based RAG pipeline with conversation memory.

Graph topology:
    START → retrieve → generate → END

State carries:
  - messages      : full conversation history (HumanMessage / AIMessage)
  - retrieved_docs: last retrieval result (list of {content, metadata, score})
  - query         : most recent user question (for downstream display)
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import LLM_MODEL, OLLAMA_BASE_URL, SYSTEM_PROMPT
from .vector_store import collection_exists, similarity_search_with_scores


# ─── State ──────────────────────────────────────────────────────────────────

class GururState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_docs: List[Dict[str, Any]]
    query: str


# ─── Node helpers ───────────────────────────────────────────────────────────

def _retrieve(state: GururState) -> Dict[str, Any]:
    """Retrieve top-k relevant Gita passages for the latest user message."""
    query = state["messages"][-1].content

    if not collection_exists():
        return {"retrieved_docs": [], "query": query}

    results = similarity_search_with_scores(query, k=5)
    docs = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        }
        for doc, score in results
    ]
    return {"retrieved_docs": docs, "query": query}


def _generate(state: GururState) -> Dict[str, Any]:
    """Generate a response grounded in retrieved Gita passages."""
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
    )

    docs = state.get("retrieved_docs", [])

    # build context block from retrieved passages
    context_parts = []
    for d in docs:
        meta = d["metadata"]
        chapter = meta.get("chapter", "?")
        verse = meta.get("verse", "?")
        tag = (
            f"[Gita Ch.{chapter} V.{verse}]"
            if chapter != 0
            else "[Gita Teaching]"
        )
        context_parts.append(f"{tag}\n{d['content']}")

    messages_to_send: list = [SystemMessage(content=SYSTEM_PROMPT)]

    if context_parts:
        context_block = "\n\n---\n\n".join(context_parts)
        messages_to_send.append(
            SystemMessage(
                content=(
                    "Relevant passages from the Bhagavad Gita "
                    "(use their wisdom to inform your response, "
                    "but speak naturally — do not quote or cite them directly):\n\n"
                    + context_block
                )
            )
        )

    messages_to_send.extend(state["messages"])

    response = llm.invoke(messages_to_send)
    return {"messages": [response]}


# ─── Graph factory ──────────────────────────────────────────────────────────

def create_rag_graph():
    """
    Build and compile the LangGraph RAG pipeline.
    Each call creates a fresh graph with an in-memory checkpointer.
    The checkpointer enables per-thread conversation memory.
    """
    graph = StateGraph(GururState)

    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate", _generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ─── Convenience wrapper ────────────────────────────────────────────────────

def ask_guru(
    compiled_graph,
    question: str,
    thread_id: str,
) -> Dict[str, Any]:
    """
    Invoke the graph for a given thread, return the AI reply and retrieved docs.

    Returns:
        {
          "answer": str,
          "retrieved_docs": list[dict],
          "query": str,
        }
    """
    config = {"configurable": {"thread_id": thread_id}}
    state_input = {"messages": [HumanMessage(content=question)]}

    result = compiled_graph.invoke(state_input, config=config)

    # extract the last AIMessage
    answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            answer = msg.content
            break

    return {
        "answer": answer,
        "retrieved_docs": result.get("retrieved_docs", []),
        "query": result.get("query", question),
    }
