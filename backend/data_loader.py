"""
Loads and processes six Gita datasets into LangChain Documents.

Dataset index
─────────────
  1. gita/gita GitHub repo (YAML)                       — b01…b18.yaml verse meanings
  2. SatyaSanatan alpaca JSON (HuggingFace)              — instruction/output Q&A pairs
  3. praneshp1org/Bhagavad-Gita-JSON-data (GitHub)      — multi-field verse JSON
  4. JDhruv14/Bhagavad-Gita-QA (HuggingFace)            — question-answer pairs
  5. utkarshpophli/bhagwat_gita (HuggingFace)           — structured verse teachings
  6. Modotte/Bhagwat-Gita-Infinity (HuggingFace)        — extended verse content

Each dataset is tagged with a `source` key in Document.metadata so the
frontend can show full provenance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import requests
import yaml
from langchain_core.documents import Document
from tqdm import tqdm

from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR


# ── Dataset URLs ─────────────────────────────────────────────────────────────

GITA_REPO_URL = "https://github.com/gita/gita.git"

ALPACA_JSON_URL = (
    "https://huggingface.co/datasets/SatyaSanatan/"
    "shrimad-bhagavad-gita-dataset-alpaca/resolve/main/Shrimad-bhagvad-gita.json"
)

PRANESH_JSON_URL = (
    "https://raw.githubusercontent.com/praneshp1org/"
    "Bhagavad-Gita-JSON-data/main/verse.json"
)


# ── Generic helpers ──────────────────────────────────────────────────────────

def _clone_or_pull(repo_url: str, dest: Path) -> None:
    import git

    if (dest / ".git").exists():
        print(f"  Repo already cloned at {dest}, pulling latest …")
        git.Repo(dest).remotes.origin.pull()
    else:
        print(f"  Cloning {repo_url} → {dest} …")
        git.Repo.clone_from(repo_url, dest)


def _download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    print(f"  Downloading {url} …")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  Saved to {dest}")


def _coerce_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val).strip()


def _pick(item: dict, *keys: str) -> str:
    """Return the first non-empty value from a dict for any of the given keys."""
    for k in keys:
        v = _coerce_str(item.get(k, ""))
        if v:
            return v
    return ""


# ── Dataset 1: gita/gita JSON ──────────────────────────────────────────────

def _parse_gita_yaml_dir(repo_root: Path) -> List[Document]:
    """
    Parse the cloned gita/gita repo.

    The repo stores data in data/verse.json + data/translation.json (JSON format,
    not YAML — despite the legacy function name kept for compatibility).

    verse.json       : Sanskrit text, transliteration, word_meanings (one per verse)
    translation.json : Multiple translations per verse across languages
    We use English translations; when multiple exist we concatenate them.
    """
    docs: List[Document] = []
    data_dir = repo_root / "data"

    verse_path = data_dir / "verse.json"
    trans_path = data_dir / "translation.json"

    if not verse_path.exists():
        print("  [WARN] data/verse.json not found in gita repo.")
        return docs

    try:
        verses = json.loads(verse_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [WARN] Could not parse verse.json: {e}")
        return docs

    # build lookup: verse_id → list of English translations
    eng_by_verse: dict = {}
    if trans_path.exists():
        try:
            translations = json.loads(trans_path.read_text(encoding="utf-8"))
            for t in translations:
                if t.get("lang") == "english" and t.get("description"):
                    vid = t["verse_id"]
                    eng_by_verse.setdefault(vid, []).append(
                        _coerce_str(t["description"])
                    )
        except Exception as e:
            print(f"  [WARN] Could not parse translation.json: {e}")

    for verse in tqdm(verses, desc="  Parsing gita/gita JSON"):
        chapter = int(verse.get("chapter_number") or verse.get("chapter_id") or 0)
        verse_num = verse.get("verse_number") or verse.get("verse_order") or "?"
        verse_id = verse.get("id")

        sanskrit = _coerce_str(verse.get("text", ""))
        transliteration = _coerce_str(verse.get("transliteration", ""))
        word_meanings = _coerce_str(verse.get("word_meanings", ""))

        # prefer English translations; fall back to word_meanings
        translations_list = eng_by_verse.get(verse_id, [])
        if translations_list:
            meaning = translations_list[0]  # use first English translation
        else:
            meaning = word_meanings

        if not meaning:
            continue

        page_content = (
            f"Chapter {chapter}, Verse {verse_num}\n"
            f"Transliteration: {transliteration}\n\n"
            f"{meaning}"
        )
        if word_meanings and word_meanings != meaning:
            page_content += f"\n\nWord meanings: {word_meanings[:300]}"

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": "gita_yaml",
                    "source_label": "gita/gita (GitHub JSON)",
                    "chapter": chapter,
                    "verse": str(verse_num),
                    "sanskrit": sanskrit[:200],
                    "transliteration": transliteration[:200],
                },
            )
        )

    print(f"  Loaded {len(docs)} verses from gita/gita JSON")
    return docs


# ── Dataset 2: SatyaSanatan alpaca Q&A JSON ───────────────────────────────

def _parse_alpaca_json(json_path: Path) -> List[Document]:
    """
    Parse the SatyaSanatan JSON file.

    Handles both alpaca format (instruction/input/output) and
    plain Q&A format (question/answer/text).
    """
    docs: List[Document] = []

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raw = raw.get("data", raw.get("train", []))

    for i, item in enumerate(tqdm(raw, desc="  Parsing alpaca Q&A")):
        # support alpaca format AND plain question/answer format
        question = _pick(item, "instruction", "question", "query", "input")
        answer = _pick(
            item, "output", "response", "answer",
            "text",   # some datasets use "text" for the body
        )

        if not answer or answer == question:
            # fall back: skip entries where answer == question (empty/invalid)
            continue

        page_content = f"{question}\n\nAnswer: {answer}" if question else answer

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": "alpaca_qa",
                    "source_label": "SatyaSanatan alpaca (HuggingFace)",
                    "chapter": 0,
                    "verse": f"qa_{i}",
                    "question": question[:300],
                },
            )
        )

    print(f"  Loaded {len(docs)} Q&A pairs from alpaca JSON")
    return docs


# ── Dataset 3: praneshp1org/Bhagavad-Gita-JSON-data ──────────────────────

def _parse_pranesh_json(json_path: Path) -> List[Document]:
    """
    Parse the praneshp1org verse.json — a flat array of verse objects.

    Expected fields (any subset may be present):
      chapter_number, verse_number, text, transliteration, word_meanings,
      translation, commentary, author_name
    """
    docs: List[Document] = []

    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [WARN] Could not parse pranesh JSON: {e}")
        return docs

    items = raw if isinstance(raw, list) else raw.get("verses", raw.get("data", []))

    for item in tqdm(items, desc="  Parsing pranesh verse.json"):
        if not isinstance(item, dict):
            continue

        chapter = int(item.get("chapter_number") or item.get("chapter") or 0)
        verse = str(item.get("verse_number") or item.get("verse") or "?")

        # prefer translation, fall back to commentary / text
        translation = _pick(
            item, "translation", "meaning", "commentary", "text"
        )
        if not translation:
            continue

        word_meanings = _coerce_str(item.get("word_meanings"))
        transliteration = _coerce_str(
            item.get("transliteration") or item.get("transliteration_of_verse", "")
        )
        sanskrit = _coerce_str(item.get("text") or item.get("verse_text", ""))

        page_content = f"Chapter {chapter}, Verse {verse}\n\n{translation}"
        if word_meanings:
            page_content += f"\n\nWord meanings: {word_meanings}"

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": "pranesh_json",
                    "source_label": "praneshp1org verse.json (GitHub)",
                    "chapter": chapter,
                    "verse": verse,
                    "sanskrit": sanskrit[:200],
                    "transliteration": transliteration[:200],
                },
            )
        )

    print(f"  Loaded {len(docs)} verses from pranesh JSON")
    return docs


# ── Dataset 4: JDhruv14/Bhagavad-Gita-QA (HuggingFace) ──────────────────

def _parse_jdhruv14_qa(cache_dir: Path) -> List[Document]:
    """
    Load JDhruv14/Bhagavad-Gita-QA from HuggingFace.

    Expected columns (flexible detection):
      question / context / answer / response / output
    """
    docs: List[Document] = []
    cache_file = cache_dir / "jdhruv14_qa.json"

    if cache_file.exists():
        print(f"  Loading cached JDhruv14 QA from {cache_file.name} …")
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        items = raw
    else:
        try:
            from datasets import load_dataset  # type: ignore
            print("  Downloading JDhruv14/Bhagavad-Gita-QA from HuggingFace …")
            ds = load_dataset("JDhruv14/Bhagavad-Gita-QA", split="train")
            items = [dict(row) for row in ds]
            cache_file.write_text(
                json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"  [WARN] JDhruv14/Bhagavad-Gita-QA loading failed: {e}")
            return docs

    for i, item in enumerate(tqdm(items, desc="  Parsing JDhruv14 QA")):
        question = _pick(item, "question", "instruction", "query", "input")
        answer = _pick(item, "answer", "output", "response", "context", "text")

        if not answer:
            continue

        page_content = f"{question}\n\nAnswer: {answer}" if question else answer

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": "jdhruv14_qa",
                    "source_label": "JDhruv14/Bhagavad-Gita-QA (HuggingFace)",
                    "chapter": 0,
                    "verse": f"qa_{i}",
                    "question": question[:300],
                },
            )
        )

    print(f"  Loaded {len(docs)} Q&A pairs from JDhruv14")
    return docs


# ── Dataset 5: utkarshpophli/bhagwat_gita (HuggingFace) ──────────────────

def _parse_utkarsh_gita(cache_dir: Path) -> List[Document]:
    """
    Load utkarshpophli/bhagwat_gita from HuggingFace.

    Expected columns: chapter, verse, text / meaning / translation / commentary
    """
    docs: List[Document] = []
    cache_file = cache_dir / "utkarsh_gita.json"

    if cache_file.exists():
        print(f"  Loading cached utkarshpophli gita from {cache_file.name} …")
        items = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        try:
            from datasets import load_dataset  # type: ignore
            print("  Downloading utkarshpophli/bhagwat_gita from HuggingFace …")
            ds = load_dataset("utkarshpophli/bhagwat_gita", split="train")
            items = [dict(row) for row in ds]
            cache_file.write_text(
                json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"  [WARN] utkarshpophli/bhagwat_gita loading failed: {e}")
            return docs

    for item in tqdm(items, desc="  Parsing utkarshpophli gita"):
        chapter = int(item.get("chapter_number") or item.get("chapter") or 0)
        verse = str(item.get("verse_number") or item.get("verse") or "?")

        content = _pick(
            item, "meaning", "translation", "commentary",
            "text", "verse_text", "description", "output"
        )
        if not content:
            continue

        page_content = f"Chapter {chapter}, Verse {verse}\n\n{content}"
        sanskrit = _coerce_str(
            item.get("text") or item.get("verse_text") or item.get("shlok", "")
        )
        transliteration = _coerce_str(
            item.get("transliteration") or item.get("transliteration_of_verse", "")
        )

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": "utkarsh_gita",
                    "source_label": "utkarshpophli/bhagwat_gita (HuggingFace)",
                    "chapter": chapter,
                    "verse": verse,
                    "sanskrit": sanskrit[:200],
                    "transliteration": transliteration[:200],
                },
            )
        )

    print(f"  Loaded {len(docs)} verses from utkarshpophli gita")
    return docs


# ── Dataset 6: Modotte/Bhagwat-Gita-Infinity (HuggingFace) ───────────────

def _parse_modotte_infinity(cache_dir: Path) -> List[Document]:
    """
    Load Modotte/Bhagwat-Gita-Infinity from HuggingFace.

    Flexible column detection for any text-rich columns.
    """
    docs: List[Document] = []
    cache_file = cache_dir / "modotte_infinity.json"

    if cache_file.exists():
        print(f"  Loading cached Modotte infinity from {cache_file.name} …")
        items = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        try:
            from datasets import load_dataset  # type: ignore
            print("  Downloading Modotte/Bhagwat-Gita-Infinity from HuggingFace …")
            ds = load_dataset("Modotte/Bhagwat-Gita-Infinity", split="train")
            items = [dict(row) for row in ds]
            cache_file.write_text(
                json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"  [WARN] Modotte/Bhagwat-Gita-Infinity loading failed: {e}")
            return docs

    for i, item in enumerate(tqdm(items, desc="  Parsing Modotte infinity")):
        chapter = int(item.get("chapter_number") or item.get("chapter") or 0)
        verse = str(item.get("verse_number") or item.get("verse") or f"mod_{i}")

        content = _pick(
            item, "meaning", "translation", "commentary", "text",
            "verse_text", "description", "content", "output", "response"
        )
        if not content:
            # last resort: join all string values
            parts = [str(v) for v in item.values() if isinstance(v, str) and len(v) > 30]
            content = " ".join(parts[:2])

        if not content:
            continue

        prefix = f"Chapter {chapter}, Verse {verse}\n\n" if chapter else ""
        page_content = prefix + content

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": "modotte_infinity",
                    "source_label": "Modotte/Bhagwat-Gita-Infinity (HuggingFace)",
                    "chapter": chapter,
                    "verse": verse,
                    "sanskrit": _coerce_str(item.get("text", ""))[:200],
                    "transliteration": _coerce_str(
                        item.get("transliteration", "")
                    )[:200],
                },
            )
        )

    print(f"  Loaded {len(docs)} entries from Modotte infinity")
    return docs


# ── Public API ────────────────────────────────────────────────────────────

# Human-readable registry used by the Knowledge Base page
DATASET_REGISTRY = {
    "gita_yaml": {
        "label": "gita/gita (GitHub YAML)",
        "url": "https://github.com/gita/gita",
        "type": "Verse translations",
        "description": (
            "Official gita/gita GitHub repo — 18 YAML chapter files (b01…b18). "
            "Each verse includes Sanskrit (Devanagari), Roman transliteration, "
            "and an English meaning. ~700 verses total."
        ),
        "fields": "chapter, verse, sanskrit (c), transliteration (g), meaning (m)",
        "index_key": "meaning text → embedded and indexed per verse",
    },
    "alpaca_qa": {
        "label": "SatyaSanatan alpaca (HuggingFace)",
        "url": "https://huggingface.co/datasets/SatyaSanatan/shrimad-bhagavad-gita-dataset-alpaca",
        "type": "Q&A instruction pairs",
        "description": (
            "Alpaca-format instruction/output Q&A dataset from SatyaSanatan. "
            "Covers philosophical questions with Gita-grounded answers. "
            "Format: instruction → output."
        ),
        "fields": "instruction, input (optional context), output",
        "index_key": "instruction + output text → embedded as a single document",
    },
    "pranesh_json": {
        "label": "praneshp1org verse.json (GitHub)",
        "url": "https://github.com/praneshp1org/Bhagavad-Gita-JSON-data",
        "type": "Multi-field verse JSON",
        "description": (
            "Flat JSON array of all 700 Gita verses with rich multi-field structure: "
            "Sanskrit text, word-by-word meanings, transliteration, and commentary. "
            "Complements gita_yaml with deeper per-word breakdown."
        ),
        "fields": "chapter_number, verse_number, text (Sanskrit), transliteration, word_meanings, translation, commentary",
        "index_key": "translation + word_meanings → embedded per verse",
    },
    "jdhruv14_qa": {
        "label": "JDhruv14/Bhagavad-Gita-QA (HuggingFace)",
        "url": "https://huggingface.co/datasets/JDhruv14/Bhagavad-Gita-QA",
        "type": "Q&A pairs",
        "description": (
            "Question-answer pairs specifically curated from the Bhagavad Gita. "
            "Useful for conversational retrieval — matches user questions "
            "to Gita-grounded answers."
        ),
        "fields": "question, answer",
        "index_key": "question + answer → embedded as a single document",
    },
    "utkarsh_gita": {
        "label": "utkarshpophli/bhagwat_gita (HuggingFace)",
        "url": "https://huggingface.co/datasets/utkarshpophli/bhagwat_gita",
        "type": "Structured verse dataset",
        "description": (
            "Structured verse-level dataset with chapter/verse metadata, "
            "Sanskrit shlok, transliteration, and meaning. Provides an "
            "additional translation layer alongside gita_yaml."
        ),
        "fields": "chapter_number, verse_number, shlok (Sanskrit), transliteration, meaning",
        "index_key": "meaning text → embedded per verse",
    },
    "modotte_infinity": {
        "label": "Modotte/Bhagwat-Gita-Infinity (HuggingFace)",
        "url": "https://huggingface.co/datasets/Modotte/Bhagwat-Gita-Infinity",
        "type": "Extended verse content",
        "description": (
            "Expanded/extended Gita verse content — includes paraphrased "
            "and elaborated verse meanings. Increases semantic coverage "
            "and retrieval recall for nuanced philosophical questions."
        ),
        "fields": "verse content with metadata",
        "index_key": "content text → embedded per entry",
    },
}


def load_all_documents(force: bool = False) -> List[Document]:
    """
    Download all six datasets, parse them, and return combined Documents.

    Uses a processed cache (data/processed/documents.json) to avoid
    re-downloading on subsequent runs. Pass force=True to rebuild.
    """
    processed_path = PROCESSED_DATA_DIR / "documents.json"

    if processed_path.exists() and not force:
        print("Loading cached processed documents …")
        raw = json.loads(processed_path.read_text(encoding="utf-8"))
        docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in raw
        ]
        print(f"  {len(docs)} documents loaded from cache")
        return docs

    all_docs: List[Document] = []

    # ── 1: gita/gita YAML ──────────────────────────────────────────────────
    print("\n[Dataset 1/6] gita/gita GitHub YAML repo …")
    gita_dest = RAW_DATA_DIR / "gita"
    try:
        _clone_or_pull(GITA_REPO_URL, gita_dest)
        all_docs.extend(_parse_gita_yaml_dir(gita_dest))
    except Exception as e:
        print(f"  [WARN] gita repo loading failed: {e}")

    # ── 2: SatyaSanatan alpaca JSON ────────────────────────────────────────
    print("\n[Dataset 2/6] SatyaSanatan alpaca JSON (HuggingFace) …")
    alpaca_dest = RAW_DATA_DIR / "Shrimad-bhagvad-gita.json"
    try:
        _download_file(ALPACA_JSON_URL, alpaca_dest)
        all_docs.extend(_parse_alpaca_json(alpaca_dest))
    except Exception as e:
        print(f"  [WARN] alpaca JSON loading failed: {e}")

    # ── 3: praneshp1org verse.json ─────────────────────────────────────────
    print("\n[Dataset 3/6] praneshp1org/Bhagavad-Gita-JSON-data (GitHub) …")
    pranesh_dest = RAW_DATA_DIR / "pranesh_verse.json"
    try:
        _download_file(PRANESH_JSON_URL, pranesh_dest)
        all_docs.extend(_parse_pranesh_json(pranesh_dest))
    except Exception as e:
        print(f"  [WARN] pranesh JSON loading failed: {e}")

    # ── 4: JDhruv14/Bhagavad-Gita-QA ──────────────────────────────────────
    print("\n[Dataset 4/6] JDhruv14/Bhagavad-Gita-QA (HuggingFace) …")
    try:
        all_docs.extend(_parse_jdhruv14_qa(RAW_DATA_DIR))
    except Exception as e:
        print(f"  [WARN] JDhruv14 QA loading failed: {e}")

    # ── 5: utkarshpophli/bhagwat_gita ─────────────────────────────────────
    print("\n[Dataset 5/6] utkarshpophli/bhagwat_gita (HuggingFace) …")
    try:
        all_docs.extend(_parse_utkarsh_gita(RAW_DATA_DIR))
    except Exception as e:
        print(f"  [WARN] utkarshpophli gita loading failed: {e}")

    # ── 6: Modotte/Bhagwat-Gita-Infinity ──────────────────────────────────
    print("\n[Dataset 6/6] Modotte/Bhagwat-Gita-Infinity (HuggingFace) …")
    try:
        all_docs.extend(_parse_modotte_infinity(RAW_DATA_DIR))
    except Exception as e:
        print(f"  [WARN] Modotte infinity loading failed: {e}")

    if not all_docs:
        raise RuntimeError(
            "No documents loaded from any dataset. "
            "Check your network connection and try again."
        )

    # ── Cache processed documents ──────────────────────────────────────────
    serialised = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in all_docs
    ]
    processed_path.write_text(
        json.dumps(serialised, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    source_counts: dict = {}
    for d in all_docs:
        src = d.metadata.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\n{'─'*50}")
    print(f"  Total documents ingested: {len(all_docs)}")
    for src, count in sorted(source_counts.items()):
        label = DATASET_REGISTRY.get(src, {}).get("label", src)
        print(f"    {label}: {count}")
    print(f"{'─'*50}")
    print(f"  Cached to {processed_path}")

    return all_docs
