---
title: AI Guru
emoji: 🧠
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
app_file: frontend/app.py
pinned: false
---

# AI Guru — Handy Philosopher in Your Pocket

> *"The wisdom informs the answer, not the vocabulary."*

AI Guru is a **Retrieval-Augmented Generation (RAG)** system that functions as a wise
philosophy teacher shaped by the **Bhagavad Gita**. It answers philosophical questions
about life, purpose, inner conflict, and existential struggle — speaking in plain modern
English, never reciting scripture.

---

## Table of Contents

1. [What It Is (and Is Not)](#what-it-is-and-is-not)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Datasets](#datasets)
5. [How Indexing Works](#how-indexing-works)
6. [Project Structure](#project-structure)
7. [Setup & Installation](#setup--installation)
8. [Application Pages](#application-pages)
9. [Explainability & Provenance](#explainability--provenance)
10. [Key Concepts](#key-concepts)
11. [Future Vision](#future-vision)

---

## What It Is (and Is Not)

| Designed for | Not designed for |
|---|---|
| Philosophical / subjective questions | Factual / scientific lookups |
| *"What does it mean to live with purpose?"* | *"Where does the sun rise?"* |
| PhD-level expertise in schools of philosophy | Science, engineering, math |

The Guru is a **digital twin of a philosophy teacher** — like an Ayurvedic doctor who
says "eat lighter, sleep earlier" instead of "tridosha imbalance." The Gita's wisdom
shapes the thinking; the response speaks plain English.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Question                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Embedding — nomic-embed-text (via Ollama)                           │
│  → 768-dimensional vector representation of the query               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ChromaDB Vector Store                                               │
│  → Cosine similarity search across all indexed documents             │
│  → Returns top-5 most semantically relevant Gita passages            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LangGraph RAG Pipeline  (START → retrieve → generate → END)         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  retrieve node  : similarity_search_with_scores(query, k=5) │   │
│  │  generate node  : Mistral + system prompt + context block   │   │
│  │  MemorySaver    : per-thread conversation checkpointing      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Guru Response                                                        │
│  + Answer Provenance Trail (step-by-step trace: which passages,       │
│    from which datasets, similarity scores, context sent to LLM)      │
└─────────────────────────────────────────────────────────────────────┘
```

### LangGraph State Machine

```
__start__  ──▶  retrieve  ──▶  generate  ──▶  __end__
                                   ↕
                              MemorySaver
                           (per-thread history)
```

**State schema:**

```python
class GururState(TypedDict):
    messages      : Annotated[list, add_messages]   # full conversation history
    retrieved_docs: List[Dict[str, Any]]             # top-K retrieved passages
    query         : str                              # latest user question
```

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | `llama-3.1-8b-instant` via Groq API | Free, fast cloud inference |
| Embeddings | `nomic-ai/nomic-embed-text-v1.5` via HuggingFace | 768-dim, runs locally on CPU |
| RAG orchestration | LangGraph | Typed state machine |
| Conversation memory | LangGraph MemorySaver | Per-thread checkpointing |
| LLM framework | LangChain + langchain-groq | Retrieval + prompt assembly |
| Vector store | ChromaDB (persistent) | Cosine similarity, disk-backed |
| Frontend | Streamlit (multi-page app) | 4 pages |
| Visualization | Plotly + UMAP | Interactive 2-D embedding projection |
| HuggingFace datasets | `datasets` library | Datasets 4–7 |
| Data helpers | PyYAML, requests, GitPython | Datasets 1–3 |

### Why Groq?

Groq's free API tier provides fast LLaMA 3.1 inference with no GPU required locally.
`llama-3.1-8b-instant` has strong instruction-following, making the system prompt
reliably shape the "practical advisor, not scripture reciter" persona.

### Why nomic-embed-text-v1.5?

Purpose-built for retrieval (8192 context window, 768 dims). Significantly
outperforms `all-MiniLM` on philosophical/semantic queries. Runs on CPU via
HuggingFace `sentence-transformers` — no Ollama required.

### Why ChromaDB?

- Persistent on-disk storage (no re-indexing on restart)
- Native Python client (no external service)
- Direct access to raw embeddings for UMAP visualisation
- Metadata filtering (filter by chapter, source, etc.)

### Why LangGraph?

Treats the RAG pipeline as a **typed state machine** — each node is an explicit,
testable function. `MemorySaver` makes multi-turn memory transparent. Easy to extend
with relevance grading, re-ranking, or self-critique nodes.

---

## Datasets

Seven open-source Gita datasets are ingested, merged, and indexed into a single
ChromaDB collection (**13,307 documents total**). Every document retains a `source`
metadata tag so the provenance trail can attribute each retrieved passage back to
its origin dataset.

---

### Dataset 1 — gita/gita (GitHub YAML)

| | |
|---|---|
| **Source tag** | `gita_yaml` |
| **URL** | https://github.com/gita/gita |
| **Type** | Verse-by-verse translations |
| **Format** | 18 YAML files (`b01.yaml` … `b18.yaml`) |

**What it contains:**
The official gita/gita open-source repository. Each YAML file is one chapter. Every
verse entry has three fields:
- `c` — Sanskrit text (Devanagari)
- `g` — Roman transliteration
- `m` — English meaning / translation

**How it's organised:**
Files are named `b01.yaml` through `b18.yaml` (one per chapter). Each file is a dict
keyed by verse number. The repo is Git-cloned to `data/raw/gita/`.

**How it's indexed:**
One `Document` per verse.
`page_content` = `"Chapter N, Verse V\nTransliteration: {g}\n\n{m}"`
Metadata: `source`, `chapter`, `verse`, `sanskrit`, `transliteration`

**Approximate size:** ~700 documents

---

### Dataset 2 — SatyaSanatan alpaca (HuggingFace)

| | |
|---|---|
| **Source tag** | `alpaca_qa` |
| **URL** | https://huggingface.co/datasets/SatyaSanatan/shrimad-bhagavad-gita-dataset-alpaca |
| **Type** | Q&A instruction pairs |
| **Format** | JSON array (alpaca format) — `instruction`, `input`, `output` |

**What it contains:**
Instruction-tuning style Q&A pairs covering philosophical themes across the Gita.
The "bridge" dataset — questions here are phrased like user questions, so they
improve retrieval recall for conversational queries.

**How it's organised:**
Flat JSON array downloaded to `data/raw/Shrimad-bhagvad-gita.json`.
Each entry: `instruction` (question) + optional `input` + `output` (Gita-grounded answer).

**How it's indexed:**
One document per Q&A pair.
`page_content` = `"{instruction}\n\nAnswer: {output}"`
Metadata: `source`, `chapter=0`, `verse="qa_N"`, `question`

**Approximate size:** ~500–1,000 documents

---

### Dataset 3 — praneshp1org/Bhagavad-Gita-JSON-data (GitHub)

| | |
|---|---|
| **Source tag** | `pranesh_json` |
| **URL** | https://github.com/praneshp1org/Bhagavad-Gita-JSON-data |
| **Type** | Multi-field verse JSON |
| **Format** | `verse.json` — flat JSON array, one object per verse |

**What it contains:**
All 700 Gita verses with richly structured per-verse fields: Sanskrit text,
word-by-word meanings, Roman transliteration, full translation, and commentary.
Complements Dataset 1 with deeper word-level breakdown and alternate translation phrasing.

**How it's organised:**
Downloaded to `data/raw/pranesh_verse.json`. Fields per verse:
`chapter_number`, `verse_number`, `text` (Sanskrit), `transliteration`,
`word_meanings`, `translation`, `commentary`

**How it's indexed:**
One document per verse.
`page_content` = `"{translation}\n\nWord meanings: {word_meanings}"`
Metadata: `source`, `chapter`, `verse`, `sanskrit`, `transliteration`

**Approximate size:** ~700 documents

---

### Dataset 4 — JDhruv14/Bhagavad-Gita-QA (HuggingFace)

| | |
|---|---|
| **Source tag** | `jdhruv14_qa` |
| **URL** | https://huggingface.co/datasets/JDhruv14/Bhagavad-Gita-QA |
| **Type** | Question-answer pairs |
| **Format** | HuggingFace dataset (downloaded via `datasets` library) |

**What it contains:**
Q&A pairs curated from the Bhagavad Gita. When a user asks a question, its embedding
will directly match Q&A entries whose questions are semantically similar — giving the
LLM both the question and the Gita-grounded answer as context.

**How it's organised:**
Loaded via `datasets.load_dataset("JDhruv14/Bhagavad-Gita-QA", split="train")`.
Cached to `data/raw/jdhruv14_qa.json` after first download.
Columns: `question`, `answer` (flexible column detection).

**How it's indexed:**
One document per Q&A pair.
`page_content` = `"{question}\n\nAnswer: {answer}"`
Metadata: `source`, `chapter=0`, `verse="qa_N"`, `question`

---

### Dataset 5 — utkarshpophli/bhagwat_gita (HuggingFace)

| | |
|---|---|
| **Source tag** | `utkarsh_gita` |
| **URL** | https://huggingface.co/datasets/utkarshpophli/bhagwat_gita |
| **Type** | Structured verse dataset |
| **Format** | HuggingFace dataset (downloaded via `datasets` library) |

**What it contains:**
Verse-level dataset with chapter/verse metadata, Sanskrit shlok, transliteration,
and English meaning. Provides an additional translation layer with different phrasing
from Dataset 1, increasing retrieval recall for paraphrased queries.

**How it's organised:**
Loaded via `datasets.load_dataset("utkarshpophli/bhagwat_gita", split="train")`.
Cached to `data/raw/utkarsh_gita.json` after first download.
Columns: `chapter_number`, `verse_number`, `shlok` (Sanskrit), `transliteration`, `meaning`

**How it's indexed:**
One document per verse.
`page_content` = `"Chapter N, Verse V\n\n{meaning}"`
Metadata: `source`, `chapter`, `verse`, `sanskrit`, `transliteration`

---

### Dataset 6 — Modotte/Bhagwat-Gita-Infinity (HuggingFace)

| | |
|---|---|
| **Source tag** | `modotte_infinity` |
| **URL** | https://huggingface.co/datasets/Modotte/Bhagwat-Gita-Infinity |
| **Type** | Extended verse content |
| **Format** | HuggingFace dataset (downloaded via `datasets` library) |

**What it contains:**
Extended and paraphrased Gita verse content with elaborated meanings. Increases
semantic coverage and retrieval recall for nuanced philosophical queries that may not
match the exact phrasing of a traditional translation.

**How it's organised:**
Loaded via `datasets.load_dataset("Modotte/Bhagwat-Gita-Infinity", split="train")`.
Cached to `data/raw/modotte_infinity.json` after first download.
Content columns detected flexibly from available text fields.

**How it's indexed:**
One document per entry.
`page_content` = `"Chapter N, Verse V\n\n{content}"` (when chapter known)
Metadata: `source`, `chapter`, `verse`, `source_label`

---

### Dataset 7 — JDhruv14/Bhagavad-Gita\_Dataset (HuggingFace)

| | |
|---|---|
| **Source tag** | `jdhruv14_dataset` |
| **URL** | https://huggingface.co/datasets/JDhruv14/Bhagavad-Gita_Dataset |
| **Type** | Structured verse dataset |
| **Format** | HuggingFace dataset (downloaded via `datasets` library) |

**What it contains:**
Verse-level dataset with chapter/verse coordinates, Sanskrit text, and English
translation. Adds an additional translation layer and increases retrieval recall
for paraphrased queries not captured by earlier datasets.

**How it's organised:**
Loaded via `datasets.load_dataset("JDhruv14/Bhagavad-Gita_Dataset", split="train")`.
Cached to `data/raw/jdhruv14_dataset.json` after first download.
Columns: `chapter_no`, `verse_no`, `translation` (and optional Sanskrit fields).

**How it's indexed:**
One document per verse.
`page_content` = `"Chapter N, Verse V\n\n{translation}"`
Metadata: `source`, `chapter`, `verse`

---

## How Indexing Works

### Pipeline

```
Raw data (YAML / JSON / HuggingFace)
           │
           ▼
   data_loader.py  —  seven parsers
   ├── _parse_gita_yaml_dir()       Dataset 1  →  verse-level  (chapter, verse)
   ├── _parse_alpaca_json()         Dataset 2  →  Q&A          (chapter = 0)
   ├── _parse_pranesh_json()        Dataset 3  →  verse-level  (chapter, verse)
   ├── _parse_jdhruv14_qa()         Dataset 4  →  verse-linked (chapter_no, verse_no)
   ├── _parse_utkarsh_gita()        Dataset 5  →  verse-level  (regex from text)
   ├── _parse_modotte_infinity()    Dataset 6  →  verse-level  (when available)
   └── _parse_jdhruv14_dataset()    Dataset 7  →  verse-level  (chapter, verse)
           │
           ▼
   _merge_by_verse()
   ├── Groups all docs by (chapter, verse)
   ├── Merges content from all sources into one Document per unique verse
   └── Q&A docs (chapter=0) kept as-is
           │
           ▼
   List[LangChain Document]
   (page_content + metadata{chapter, verse, sources, sanskrit, transliteration})
           │
           ▼
   HuggingFaceEmbeddings(nomic-embed-text-v1.5)  →  768-dim vector per document
           │
           ▼
   ChromaDB  →  persisted at data/chroma_db/
                collection: "gita_knowledge"
```

### Index organisation

| Index key | Document type | Count | Description |
|-----------|--------------|-------|-------------|
| `(chapter=N, verse=V)` | Merged verse | ~701 | One document per unique Gita verse (Ch. 1–18). Content from all contributing datasets merged into a single rich document. |
| `(chapter=N, verse=V)` | Individual verse | ~6,303 | Individual verse docs kept alongside merged versions to improve retrieval diversity. |
| `(chapter=0)` | Q&A / unlocated | ~6,303 | Standalone Q&A pairs not linked to a specific verse. Indexed by semantic content only. |

**Total indexed:** 13,307 documents

**Embedding:** Each document's `page_content` is embedded as a single 768-dimensional
vector using `nomic-ai/nomic-embed-text-v1.5` via HuggingFace `sentence-transformers`.
Similarity search uses **cosine distance**.

**Merge strategy:** When multiple datasets cover the same verse (e.g. Ch. 2, V. 47),
their translations and commentary are concatenated under labelled sections
(`[source label]`) into one document. This means a single retrieval hit contains
multiple perspectives on the verse.

**Why merge by verse?**
- Eliminates duplicate retrieval (without merging, the same verse from 4 sources
  would occupy 4 of the top-5 results)
- Richer context per retrieved document
- Cleaner provenance — the Guru's answer is grounded in a specific verse, not a
  fragment of one source's translation

### Caches

| Cache | Path | Purpose |
|---|---|---|
| Processed documents | `data/processed/documents.json` | Skip re-parsing on restart |
| HuggingFace datasets | `data/raw/*.json` | Skip re-downloading from HF Hub |
| ChromaDB index | `data/chroma_db/` | Persistent embedding store (survives restart) |

---

## Project Structure

```
ai-guru/
│
├── backend/
│   ├── __init__.py
│   ├── config.py               # Paths, env vars, system prompt, model names
│   ├── data_loader.py          # Seven dataset loaders + DATASET_REGISTRY dict
│   ├── rag_graph.py            # LangGraph RAG pipeline (retrieve → generate)
│   └── vector_store.py         # ChromaDB wrapper (create, load, search, browse, stats)
│
├── frontend/
│   ├── app.py                  # Streamlit home page — project overview
│   └── pages/
│       ├── 1_Chat.py           # Conversational interface + Answer Provenance Trail
│       ├── 2_Knowledge_Base.py # Dataset cards + semantic search browser
│       ├── 3_Visualize.py      # Pipeline diagram, UMAP vector space, retrieval inspector
│       └── 4_ChromaDB_Browser.py  # MongoDB-style document explorer for the vector store
│
├── scripts/
│   └── ingest_data.py          # One-time: download all 7 datasets + index ChromaDB
│
├── data/
│   ├── raw/                    # git-ignored — re-downloaded on ingest
│   ├── processed/              # git-ignored — intermediate parse cache
│   └── chroma_db/              # committed to git — 13,307 docs, ~97 MB
│
├── .streamlit/
│   └── config.toml             # Streamlit server config (headless, CORS off)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com) (LLM inference)
- A free [HuggingFace token](https://huggingface.co/settings/tokens) (dataset downloads)
- ~2 GB RAM, ~500 MB disk

No Ollama. No local GPU. Everything runs on CPU.

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/ai-guru.git
cd ai-guru
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=gsk_...
HUGGINGFACEHUB_API_TOKEN=hf_...
LLM_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
```

### 4. Ingest all seven datasets

```bash
python scripts/ingest_data.py
```

This will sequentially:
1. Clone `github.com/gita/gita` → `data/raw/gita/`
2. Download SatyaSanatan alpaca JSON from HuggingFace
3. Download praneshp1org `verse.json` from GitHub
4. Download `JDhruv14/Bhagavad-Gita-QA` via `datasets` library
5. Download `utkarshpophli/bhagwat_gita` via `datasets` library
6. Download `Modotte/Bhagwat-Gita-Infinity` via `datasets` library
7. Download `JDhruv14/Bhagavad-Gita_Dataset` via `datasets` library
8. Embed all 13,307 documents with `nomic-embed-text-v1.5` (batched, CPU)
9. Persist the ChromaDB collection to `data/chroma_db/`

First run: ~15–30 minutes (embedding 13k docs on CPU).
Re-index from scratch: `python scripts/ingest_data.py --force`

### 5. Start the app

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

---

## Application Pages

### 💬 Chat (`pages/1_Chat.py`)

The main conversational interface. Type any philosophical question and receive a
response grounded in Gita wisdom — spoken in plain English.

**Answer Provenance Trail** (toggle on/off):
After every response, expand the provenance panel to see the full RAG trace:

- **Step 1 — Query Encoding:** embedding model, query text
- **Step 2 — Vector Retrieval:** ranked list of 5 retrieved passages with similarity
  scores, chapter/verse location, dataset source, and a direct link to the raw
  document in the ChromaDB Browser
- **Step 3 — Context Assembly:** the exact context block that was sent to the LLM,
  expandable for inspection
- **Step 4 — Generation:** LLM model, temperature, session thread ID

### 📚 Knowledge Base (`pages/2_Knowledge_Base.py`)

Browse and understand the indexed content:
- **Dataset cards:** description, schema, indexing approach, URL, and document count
  for each of the 7 datasets
- **Chapter distribution chart:** documents per Gita chapter across all verse datasets
- **Source distribution table:** document counts and percentages per dataset
- **Semantic search:** natural-language search across all 7 datasets simultaneously,
  results ranked by cosine similarity with source attribution

### 🔬 Visualize (`pages/3_Visualize.py`)

Three interactive views:
- **RAG Pipeline tab:** HTML pipeline diagram (User → Embedding → ChromaDB → LLM → Response) + LangGraph state machine diagram
- **Vector Space tab:** UMAP 2-D projection of all document embeddings, coloured
  by chapter. Hover to see text previews. Semantic clusters emerge naturally.
- **Retrieval Inspector tab:** Enter any query, see retrieved documents ranked
  by similarity score rendered as a colour-coded bar chart

### 🗄️ ChromaDB Browser (`pages/4_ChromaDB_Browser.py`)

MongoDB Compass-style document explorer for the vector store:
- **Index overview:** total documents, verse docs, Q&A docs, chapters covered
- **Paginated document cards:** browse all 13,307 documents with metadata
- **Filters:** by chapter, verse number, keyword, source
- **Direct ID lookup:** paste a ChromaDB document ID to jump straight to it
  (linked from the Answer Provenance Trail in Chat)

---

## Explainability & Provenance

Every Guru answer is fully traceable. The **Answer Provenance Trail** in the Chat page
exposes the complete RAG pipeline trace for each response:

```
User query: "How do I stop being afraid of failure?"
    │
    ├─ Step 1: Embedding
    │    model: nomic-embed-text-v1.5 → 768-dim vector
    │
    ├─ Step 2: Retrieval (ChromaDB cosine similarity, k=5)
    │    #1  🟢 0.823  Ch.2 V.47   [🟩 gita_yaml]      🔗 View in Browser
    │    #2  🟢 0.771  Ch.18 V.66  [🟥 utkarsh_gita]   🔗 View in Browser
    │    #3  🟡 0.654  Q&A entry   [🟦 alpaca_qa]       🔗 View in Browser
    │    #4  🟡 0.621  Ch.3 V.19   [🟧 pranesh_json]    🔗 View in Browser
    │    #5  🔴 0.498  Ch.6 V.5    [🟪 jdhruv14_qa]     🔗 View in Browser
    │
    ├─ Step 3: Context assembly
    │    5 passages (500 chars each) prepended to system prompt
    │
    └─ Step 4: Generation
         model: llama-3.1-8b-instant (Groq)  temperature: 0.7
         session thread: abc12345…
```

Similarity score legend: 🟢 > 0.7 (high) · 🟡 > 0.5 (medium) · 🔴 ≤ 0.5 (low)

Dataset colour legend: 🟩 gita_yaml · 🟦 alpaca_qa · 🟧 pranesh_json · 🟪 jdhruv14_qa · 🟥 utkarsh_gita · 🟨 modotte_infinity · 🟫 jdhruv14_dataset

---

## Key Concepts

### Retrieval-Augmented Generation (RAG)

RAG anchors the LLM's responses in actual source material. Without RAG, the model
answers from general training knowledge — vague and inconsistent with the Gita's
specific teachings. With RAG, the 5 most relevant passages are injected as context,
grounding every response in a specific verse or teaching.

### Semantic Search vs. Keyword Search

Keyword search: finds documents containing the *exact words* in the query.
Semantic search: finds documents with *similar meaning*, even with completely
different vocabulary.

Example:
- Query: *"I'm terrified of dying"*
- Keyword match: passages containing "dying" or "terrified"
- Semantic match: passages about the soul's immortality and impermanence —
  conceptually relevant without word overlap

### Multi-Dataset Retrieval

Having 7 datasets with different phrasings, formats, and source translations means:
- The same verse concept appears in multiple phrasings → higher chance of matching any user query
- Q&A datasets (alpaca, jdhruv14) match conversational questions directly
- Verse datasets (yaml, pranesh, utkarsh) match topical/thematic queries
- Extended content (modotte) catches nuanced paraphrased queries

### LangGraph State Machine

Each node in the graph is a pure function that reads/writes typed state keys.
`MemorySaver` checkpoints the full state after every turn, keyed by `thread_id`.
New messages are appended to existing history — giving the LLM full multi-turn
context without any hidden state.

---

## Deployment

### Streamlit Community Cloud (recommended — free, no card)

1. Push this repo to GitHub (the `data/chroma_db/` folder must be committed — it's ~97 MB and already included)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Main file path** to `frontend/app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_..."
   HUGGINGFACEHUB_API_TOKEN = "hf_..."
   EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
   ```
5. Deploy — the pre-built index is cloned with the repo, so no ingestion runs on startup

### HuggingFace Spaces (free, no card)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) — SDK: **Streamlit**
2. Push the repo to the Space's git remote (`https://huggingface.co/spaces/USERNAME/ai-guru`)
3. Add `GROQ_API_KEY` and `HUGGINGFACEHUB_API_TOKEN` under **Settings → Repository secrets**
4. HuggingFace handles Git LFS automatically for large files

### Railway ($5/mo free credit, persistent disk)

1. Connect GitHub repo at [railway.app](https://railway.app)
2. Set start command: `python scripts/ingest_data.py && streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0`
3. Add env vars: `GROQ_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`
4. Add a volume mounted at `/app/data` for persistent ChromaDB storage

---

## Future Vision

The concept notes describe a broader product roadmap:

- [ ] **Voice-first interface:** Whisper STT → LLM → Piper TTS (all offline)
- [ ] **On-device deployment:** Quantized 4-bit GGUF via llama.cpp on Raspberry Pi or Qualcomm RB3 Gen 2
- [ ] **Learns from user:** Daily conversation logging + LoRA fine-tuning loop
- [ ] **Additional philosophy schools:** Stoicism, Buddhism (currently Gita only)
- [ ] **Hardware device:** Palm-sized, microphone + speaker + minimal touch surface
- [ ] **Multi-lingual:** Hindi, Sanskrit transliteration support
- [ ] **User preference memory:** Remembers which philosophical angle resonates per user
