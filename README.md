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
| LLM | Mistral 7B via Ollama | Fully local / offline |
| Embeddings | nomic-embed-text via Ollama | 768-dim, optimised for RAG |
| RAG orchestration | LangGraph | Typed state machine |
| Conversation memory | LangGraph MemorySaver | Per-thread checkpointing |
| LLM framework | LangChain + langchain-ollama | Retrieval + prompt assembly |
| Vector store | ChromaDB (persistent) | Cosine similarity, disk-backed |
| Frontend | Streamlit (multi-page app) | 3 pages |
| Visualization | Plotly + UMAP | Interactive 2-D embedding projection |
| HuggingFace datasets | `datasets` library | Datasets 4, 5, 6 |
| Data helpers | PyYAML, requests, GitPython | Datasets 1, 2, 3 |

### Why Mistral?

Mistral 7B balances reasoning quality and local inference speed. Its strong
instruction-following makes the system prompt reliably shape the "wise teacher,
not scripture reciter" persona.

### Why nomic-embed-text?

Purpose-built for retrieval (8192 context window, 768 dims). Significantly
outperforms `all-MiniLM` on philosophical/semantic queries and runs efficiently
alongside the main LLM via Ollama.

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

Six open-source Gita datasets are ingested, merged, and indexed into a single
ChromaDB collection. Every document retains a `source` metadata tag so the
Guru can attribute each retrieved passage back to its origin dataset.

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

## How Indexing Works

```
Raw data (YAML / JSON / HuggingFace)
           │
           ▼
   data_loader.py
   ├── _parse_gita_yaml_dir()      Dataset 1
   ├── _parse_alpaca_json()        Dataset 2
   ├── _parse_pranesh_json()       Dataset 3
   ├── _parse_jdhruv14_qa()        Dataset 4
   ├── _parse_utkarsh_gita()       Dataset 5
   └── _parse_modotte_infinity()   Dataset 6
           │
           ▼
   List[LangChain Document]
   (page_content + metadata{source, chapter, verse, …})
           │
           ▼
   vector_store.create_vector_store()
   ├── OllamaEmbeddings(nomic-embed-text)  ← 768-dim per document
   └── Chroma.add_documents(batch_size=100)
           │
           ▼
   ChromaDB  →  persisted at data/chroma_db/
                collection: "gita_knowledge"
```

**Processed document cache:**
After the first ingestion, all parsed documents are serialised to
`data/processed/documents.json`. Subsequent runs load from this cache
(skipping re-download and re-parsing) unless `--force` is passed.

**HuggingFace dataset cache:**
Each HuggingFace dataset is cached as a JSON file in `data/raw/` after the
first download, so the `datasets` library is only invoked once.

---

## Project Structure

```
ai-guru/
│
├── backend/
│   ├── __init__.py
│   ├── config.py               # Paths, env vars, system prompt, OLLAMA URLs
│   ├── data_loader.py          # Six dataset loaders + DATASET_REGISTRY dict
│   ├── rag_graph.py            # LangGraph RAG pipeline (retrieve → generate)
│   └── vector_store.py         # ChromaDB wrapper (create, load, search, stats)
│
├── frontend/
│   ├── app.py                  # Streamlit home page — setup check + project overview
│   └── pages/
│       ├── 1_Chat.py           # Conversational interface + Answer Provenance Trail
│       ├── 2_Knowledge_Base.py # Dataset cards + semantic search browser
│       └── 3_Visualize.py      # Pipeline diagram, UMAP vector space, retrieval inspector
│
├── scripts/
│   └── ingest_data.py          # One-time: download all 6 datasets + index ChromaDB
│
├── data/                       # Created at runtime (git-ignored)
│   ├── raw/
│   │   ├── gita/               # Cloned gita/gita GitHub repo
│   │   ├── Shrimad-bhagvad-gita.json  # Dataset 2 download
│   │   ├── pranesh_verse.json  # Dataset 3 download
│   │   ├── jdhruv14_qa.json    # Dataset 4 HF cache
│   │   ├── utkarsh_gita.json   # Dataset 5 HF cache
│   │   └── modotte_infinity.json  # Dataset 6 HF cache
│   ├── processed/
│   │   └── documents.json      # Merged, parsed document cache
│   └── chroma_db/              # ChromaDB persisted vector store
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- ~8 GB RAM (Mistral 7B + nomic-embed-text)
- ~6 GB free disk space

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull Ollama models

```bash
ollama pull mistral             # ~4 GB — the LLM brain
ollama pull nomic-embed-text    # ~270 MB — the embedding model
```

### 4. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env if you use a non-default Ollama URL or different models
```

`.env` variables:
```
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text
```

### 5. Ingest all six datasets

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
7. Embed all documents with `nomic-embed-text` (batched, 100 at a time)
8. Persist the ChromaDB collection to `data/chroma_db/`

First run: ~10–30 minutes depending on network and hardware.
Re-index from scratch: `python scripts/ingest_data.py --force`

### 6. Start the app

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
- **Step 2 — Vector Retrieval:** ranked list of retrieved passages with similarity
  scores, chapter/verse location, and which of the 6 datasets they came from
- **Step 3 — Context Assembly:** the exact context block that was sent to the LLM,
  expandable for inspection
- **Step 4 — Generation:** LLM model, temperature, session thread ID

### 📚 Knowledge Base (`pages/2_Knowledge_Base.py`)

Browse and understand the indexed content:
- **Dataset cards:** description, schema, indexing approach, URL, and document count
  for each of the 6 datasets
- **Chapter distribution chart:** documents per Gita chapter across all verse datasets
- **Source distribution table:** document counts and percentages per dataset
- **Semantic search:** natural-language search across all 6 datasets simultaneously,
  results ranked by cosine similarity with source attribution

### 🔬 Visualize (`pages/3_Visualize.py`)

Three interactive views:
- **RAG Pipeline tab:** Plotly network graph of the full data flow + LangGraph
  state machine Mermaid diagram + state schema
- **Vector Space tab:** UMAP 2-D projection of all document embeddings, coloured
  by chapter. Hover to see text previews. Semantic clusters emerge naturally.
- **Retrieval Inspector tab:** Enter any query, see retrieved documents ranked
  by similarity score rendered as a colour-coded bar chart

---

## Explainability & Provenance

Every Guru answer is fully traceable. The **Answer Provenance Trail** in the Chat page
exposes the complete RAG pipeline trace for each response:

```
User query: "How do I stop being afraid of failure?"
    │
    ├─ Step 1: Embedding
    │    model: nomic-embed-text → 768-dim vector
    │
    ├─ Step 2: Retrieval (ChromaDB cosine similarity, k=5)
    │    #1  🟢 0.823  Ch.2 V.47   [🟩 gita_yaml]
    │    #2  🟢 0.771  Ch.18 V.66  [🟥 utkarsh_gita]
    │    #3  🟡 0.654  Q&A entry   [🟦 alpaca_qa]
    │    #4  🟡 0.621  Ch.3 V.19   [🟧 pranesh_json]
    │    #5  🔴 0.498  Ch.6 V.5    [🟪 jdhruv14_qa]
    │
    ├─ Step 3: Context assembly
    │    5 passages formatted as [Gita Ch.X V.Y]\n{text}
    │    prepended to system prompt as grounding context
    │
    └─ Step 4: Generation
         model: mistral  temperature: 0.7
         session thread: abc12345…
```

Similarity score legend: 🟢 > 0.7 (high) · 🟡 > 0.5 (medium) · 🔴 ≤ 0.5 (low)

Dataset colour legend: 🟩 gita_yaml · 🟦 alpaca_qa · 🟧 pranesh_json · 🟪 jdhruv14_qa · 🟥 utkarsh_gita · 🟨 modotte_infinity

---

## Key Concepts

### Retrieval-Augmented Generation (RAG)

RAG anchors the LLM's responses in actual source material. Without RAG, Mistral
answers from general training knowledge — vague and inconsistent with the Gita's
specific teachings. With RAG, the 5 most relevant Gita passages are injected as
context, grounding every response.

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

Having 6 datasets with different phrasings, formats, and source translations means:
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

## Future Vision

The concept notes describe a broader product roadmap:

- [ ] **Voice-first interface:** Whisper STT → LLM → Piper TTS (all offline)
- [ ] **On-device deployment:** Quantized 4-bit GGUF via llama.cpp on Raspberry Pi or Qualcomm RB3 Gen 2
- [ ] **Learns from user:** Daily conversation logging + LoRA fine-tuning loop
- [ ] **Additional philosophy schools:** Stoicism, Buddhism (currently Gita only)
- [ ] **Hardware device:** Palm-sized, microphone + speaker + minimal touch surface
- [ ] **Multi-lingual:** Hindi, Sanskrit transliteration support
- [ ] **User preference memory:** Remembers which philosophical angle resonates per user
