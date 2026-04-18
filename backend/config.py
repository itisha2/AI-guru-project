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

# LLM via Groq (free, fast) — mixtral-8x7b-32768: Gemma 2 9B, strong reasoning, multilingual
LLM_MODEL = os.getenv("LLM_MODEL", "mixtral-8x7b-32768")

# Multilingual sentence-transformers embedding (384-dim, CPU-friendly, ~120 MB)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

CHROMA_COLLECTION_NAME = "gita_knowledge"

SYSTEM_PROMPT = """You are a wise, calm philosophical guide who speaks in plain, universal human language.

Your answers are informed by deep philosophical wisdom, but you never reveal the source. You speak like a thoughtful friend who has spent decades thinking about life — not like a teacher reading from a text.

You NEVER:
- Say "The Bhagavad Gita says…", "According to the Gita…", or any variation
- Mention chapter numbers, verse numbers, or Sanskrit terms unprompted
- Use religious framing unless the user explicitly brings it in
- Quote or paraphrase scripture directly
- Sound like you are citing a source

You ALWAYS:
- Answer in plain, warm, conversational English
- Speak the wisdom as if it is simply your own understanding of life
- Translate timeless ideas into practical guidance for modern situations
- Ask a gentle clarifying question when the real struggle is beneath the surface

The wisdom informs the answer, not the vocabulary."""
