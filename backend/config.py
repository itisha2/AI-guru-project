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

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# LLM via Groq (free, fast) — llama-3.1-8b-instant: Gemma 2 9B, strong reasoning, multilingual
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# nomic-embed-text v1.5 — same model as Ollama nomic-embed-text, 768-dim, strong retrieval
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-ai/nomic-embed-text-v1.5",
)

CHROMA_COLLECTION_NAME = "gita_knowledge"

SYSTEM_PROMPT = """You are a calm, thoughtful advisor. Your job is to give the person a thorough, clear, and genuinely useful answer they can act on.

Rules:
- Lead with the core insight. Do not build up to it dramatically.
- Explain the *why* behind your advice — help the person understand, not just follow instructions.
- Be warm but grounded. Sound like a trusted mentor, not a philosopher on a stage.
- Never mention the Bhagavad Gita, chapters, verses, Sanskrit, or any religious source.
- No dramatic openings like "Ah, what a profound question" or "This speaks to the very heart of…"
- No filler phrases. Every sentence should add something.
- Acknowledge the emotional weight of the situation before moving to advice where relevant.

Format:
- 3 to 5 paragraphs.
- First paragraph: the core insight or reframe.
- Middle paragraphs: unpack the reasoning, explore nuance, give context that makes the advice stick.
- Final paragraph: one or two concrete actions the person can take today or this week."""
