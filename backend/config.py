from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def _secret(key: str, default: str = "") -> str:
    """Read from env var, then st.secrets (Streamlit Cloud), then default."""
    val = os.getenv(key, "")
    if not val:
        try:
            import streamlit as st
            val = st.secrets.get(key, default)
        except Exception:
            val = default
    return val


HUGGINGFACEHUB_API_TOKEN = _secret("HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = _secret("GROQ_API_KEY")

# LLM via Groq (free, fast) — llama-3.1-8b-instant: Gemma 2 9B, strong reasoning, multilingual
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# nomic-embed-text v1.5 — same model as Ollama nomic-embed-text, 768-dim, strong retrieval
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-ai/nomic-embed-text-v1.5",
)

CHROMA_COLLECTION_NAME = "gita_knowledge"

SYSTEM_PROMPT = """You are a direct, practical advisor. Give a clear, solution-focused answer the person can act on immediately.

Rules:
- Lead with the solution or the single most important insight. No build-up.
- Be certain and decisive. Do not hedge or over-qualify.
- Sound like a trusted friend who has thought this through — not a therapist or philosopher.
- Never mention the Bhagavad Gita, chapters, verses, Sanskrit, or any religious source.
- No dramatic openings, filler phrases, or emotional preamble.
- Every sentence must earn its place. Cut anything vague.

Format: 2 short paragraphs maximum. End with one concrete action."""
