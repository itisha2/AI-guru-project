"""
Loads and processes seven Gita datasets into LangChain Documents.

Dataset index
─────────────
  1. gita/gita GitHub repo (YAML)                       — b01…b18.yaml verse meanings
  2. SatyaSanatan alpaca JSON (HuggingFace)              — instruction/output Q&A pairs
  3. praneshp1org/Bhagavad-Gita-JSON-data (GitHub)      — multi-field verse JSON
  4. JDhruv14/Bhagavad-Gita-QA (HuggingFace)            — question-answer pairs
  5. utkarshpophli/bhagwat_gita (HuggingFace)           — structured verse teachings
  6. Modotte/Bhagwat-Gita-Infinity (HuggingFace)        — extended verse content
  7. JDhruv14/Bhagavad-Gita_Dataset (HuggingFace)       — verse-level structured dataset

Each dataset is tagged with a `source` key in Document.metadata so the
frontend can show full provenance.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
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

        # this dataset uses chapter_no / verse_no (not chapter_number)
        chapter = int(item.get("chapter_no") or item.get("chapter_number") or item.get("chapter") or 0)
        verse = str(item.get("verse_no") or item.get("verse_number") or item.get("verse") or f"qa_{i}")

        page_content = f"{question}\n\nAnswer: {answer}" if question else answer

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": "jdhruv14_qa",
                    "source_label": "JDhruv14/Bhagavad-Gita-QA (HuggingFace)",
                    "chapter": chapter,
                    "verse": verse,
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

    _VERSE_RE = re.compile(r'verse\s+(\d+)\.(\d+)', re.IGNORECASE)

    for i, item in enumerate(tqdm(items, desc="  Parsing utkarshpophli gita")):
        # dataset has only a 'text' field in instruction-tuning format:
        # "<s>[INST] ... verse X.Y ... [/INST] <answer>"
        raw_text = _coerce_str(item.get("text", ""))

        # extract chapter/verse from the instruction part
        m = _VERSE_RE.search(raw_text)
        if m:
            chapter = int(m.group(1))
            verse = str(int(m.group(2)))  # normalise "01" → "1"
        else:
            chapter = int(item.get("chapter_number") or item.get("chapter") or 0)
            verse = str(item.get("verse_number") or item.get("verse") or f"ut_{i}")

        # extract the answer portion (after [/INST])
        if "[/INST]" in raw_text:
            content = raw_text.split("[/INST]", 1)[1].strip()
        else:
            content = _pick(item, "meaning", "translation", "commentary",
                            "text", "verse_text", "description", "output")

        if not content:
            continue

        page_content = f"Chapter {chapter}, Verse {verse}\n\n{content}"
        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": "utkarsh_gita",
                    "source_label": "utkarshpophli/bhagwat_gita (HuggingFace)",
                    "chapter": chapter,
                    "verse": verse,
                    "sanskrit": "",
                    "transliteration": "",
                },
            )
        )

    print(f"  Loaded {len(docs)} verses from utkarshpophli gita")
    return docs


# ── Dataset 7: JDhruv14/Bhagavad-Gita_Dataset (HuggingFace) ─────────────

def _parse_jdhruv14_dataset(cache_dir: Path) -> List[Document]:
    """
    Load JDhruv14/Bhagavad-Gita_Dataset from HuggingFace.

    Flexible column detection — handles verse-level and Q&A style rows.
    """
    docs: List[Document] = []
    cache_file = cache_dir / "jdhruv14_dataset.json"

    if cache_file.exists():
        print(f"  Loading cached JDhruv14 Dataset from {cache_file.name} …")
        items = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        try:
            from datasets import load_dataset  # type: ignore
            print("  Downloading JDhruv14/Bhagavad-Gita_Dataset from HuggingFace …")
            ds = load_dataset("JDhruv14/Bhagavad-Gita_Dataset", split="train")
            items = [dict(row) for row in ds]
            cache_file.write_text(
                json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"  [WARN] JDhruv14/Bhagavad-Gita_Dataset loading failed: {e}")
            return docs

    for i, item in enumerate(tqdm(items, desc="  Parsing JDhruv14 Dataset")):
        chapter = int(item.get("chapter_number") or item.get("chapter") or 0)
        verse = str(item.get("verse_number") or item.get("verse") or f"ds_{i}")

        content = _pick(
            item, "meaning", "translation", "commentary", "answer",
            "output", "text", "verse_text", "description", "context"
        )
        if not content:
            parts = [str(v) for v in item.values() if isinstance(v, str) and len(v) > 30]
            content = " ".join(parts[:2])

        if not content:
            continue

        question = _pick(item, "question", "instruction", "query", "input")
        if question:
            page_content = f"{question}\n\nAnswer: {content}"
        elif chapter:
            page_content = f"Chapter {chapter}, Verse {verse}\n\n{content}"
        else:
            page_content = content

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": "jdhruv14_dataset",
                    "source_label": "JDhruv14/Bhagavad-Gita_Dataset (HuggingFace)",
                    "chapter": chapter,
                    "verse": verse,
                    "sanskrit": _coerce_str(item.get("text", "") or item.get("shlok", ""))[:200],
                    "transliteration": _coerce_str(item.get("transliteration", ""))[:200],
                    "question": question[:300] if question else "",
                },
            )
        )

    print(f"  Loaded {len(docs)} entries from JDhruv14 Dataset")
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


# ── Verse-level merge ─────────────────────────────────────────────────────

def _merge_by_verse(docs: List[Document]) -> List[Document]:
    """
    Combine all documents for the same (chapter, verse) into one rich document.

    Verse-located docs (chapter != 0) are grouped by (chapter, verse) and their
    content from each source is joined under labelled sections.
    Q&A / unlocated docs (chapter == 0) are kept as-is.
    """
    _VERSE_NUM_RE = re.compile(r'^\d+')

    verse_groups: dict = defaultdict(list)
    unlocated: List[Document] = []

    for doc in docs:
        chapter = doc.metadata.get("chapter", 0)
        verse = str(doc.metadata.get("verse", "")).strip()

        # skip docs with no real chapter
        if not chapter or chapter == 0:
            unlocated.append(doc)
            continue

        # normalise verse: "2.47" → "47", "047" → "47"
        if "." in verse:
            verse = verse.split(".")[-1]
        m = _VERSE_NUM_RE.match(verse)
        verse_key = str(int(m.group())) if m else verse

        verse_groups[(int(chapter), verse_key)].append(doc)

    merged: List[Document] = []

    for (chapter, verse), group in sorted(verse_groups.items()):
        parts: List[str] = []
        sanskrit = ""
        transliteration = ""
        sources_used: List[str] = []

        for doc in group:
            source = doc.metadata.get("source", "unknown")
            label = doc.metadata.get("source_label", source)
            sources_used.append(source)

            # strip redundant "Chapter N, Verse V\n\n" header from content
            content = doc.page_content
            for hdr in (
                f"Chapter {chapter}, Verse {verse}\n\n",
                f"Chapter {chapter}, Verse {int(verse):02d}\n\n",
            ):
                if content.startswith(hdr):
                    content = content[len(hdr):]
                    break

            parts.append(f"**{label}**\n{content.strip()}")

            if not sanskrit:
                sanskrit = doc.metadata.get("sanskrit", "") or ""
            if not transliteration:
                transliteration = doc.metadata.get("transliteration", "") or ""

        page_content = (
            f"Chapter {chapter}, Verse {verse}\n\n"
            + "\n\n---\n\n".join(parts)
        )
        merged.append(
            Document(
                page_content=page_content,
                metadata={
                    "chapter": chapter,
                    "verse": verse,
                    "source": "merged",
                    "source_label": f"Merged ({len(set(sources_used))} sources)",
                    "sources": ",".join(sorted(set(sources_used))),
                    "sanskrit": sanskrit[:200],
                    "transliteration": transliteration[:200],
                },
            )
        )

    total_input = sum(len(g) for g in verse_groups.values())
    print(
        f"  Merged {total_input} verse docs → {len(merged)} unique verses  "
        f"| {len(unlocated)} Q&A/unlocated docs kept as-is"
    )
    return merged + unlocated


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
    "jdhruv14_dataset": {
        "label": "JDhruv14/Bhagavad-Gita_Dataset (HuggingFace)",
        "url": "https://huggingface.co/datasets/JDhruv14/Bhagavad-Gita_Dataset",
        "type": "Structured verse & Q&A dataset",
        "description": (
            "Verse-level structured dataset by JDhruv14 covering all 18 chapters. "
            "Includes chapter/verse metadata, Sanskrit text, transliteration, "
            "meanings, and optionally Q&A pairs — broadens retrieval recall."
        ),
        "fields": "chapter_number, verse_number, text, transliteration, meaning/answer (flexible)",
        "index_key": "meaning or answer text → embedded per entry",
    },
}


def load_all_documents(force: bool = False) -> List[Document]:
    """
    Download all seven datasets, parse them, and return combined Documents.

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
    print("\n[Dataset 1/7] gita/gita GitHub YAML repo …")
    gita_dest = RAW_DATA_DIR / "gita"
    try:
        _clone_or_pull(GITA_REPO_URL, gita_dest)
        all_docs.extend(_parse_gita_yaml_dir(gita_dest))
    except Exception as e:
        print(f"  [WARN] gita repo loading failed: {e}")

    # ── 2: SatyaSanatan alpaca JSON ────────────────────────────────────────
    print("\n[Dataset 2/7] SatyaSanatan alpaca JSON (HuggingFace) …")
    alpaca_dest = RAW_DATA_DIR / "Shrimad-bhagvad-gita.json"
    try:
        _download_file(ALPACA_JSON_URL, alpaca_dest)
        all_docs.extend(_parse_alpaca_json(alpaca_dest))
    except Exception as e:
        print(f"  [WARN] alpaca JSON loading failed: {e}")

    # ── 3: praneshp1org verse.json ─────────────────────────────────────────
    print("\n[Dataset 3/7] praneshp1org/Bhagavad-Gita-JSON-data (GitHub) …")
    pranesh_dest = RAW_DATA_DIR / "pranesh_verse.json"
    try:
        _download_file(PRANESH_JSON_URL, pranesh_dest)
        all_docs.extend(_parse_pranesh_json(pranesh_dest))
    except Exception as e:
        print(f"  [WARN] pranesh JSON loading failed: {e}")

    # ── 4: JDhruv14/Bhagavad-Gita-QA ──────────────────────────────────────
    print("\n[Dataset 4/7] JDhruv14/Bhagavad-Gita-QA (HuggingFace) …")
    try:
        all_docs.extend(_parse_jdhruv14_qa(RAW_DATA_DIR))
    except Exception as e:
        print(f"  [WARN] JDhruv14 QA loading failed: {e}")

    # ── 5: utkarshpophli/bhagwat_gita ─────────────────────────────────────
    print("\n[Dataset 5/7] utkarshpophli/bhagwat_gita (HuggingFace) …")
    try:
        all_docs.extend(_parse_utkarsh_gita(RAW_DATA_DIR))
    except Exception as e:
        print(f"  [WARN] utkarshpophli gita loading failed: {e}")

    # ── 6: Modotte/Bhagwat-Gita-Infinity ──────────────────────────────────
    print("\n[Dataset 6/7] Modotte/Bhagwat-Gita-Infinity (HuggingFace) …")
    try:
        all_docs.extend(_parse_modotte_infinity(RAW_DATA_DIR))
    except Exception as e:
        print(f"  [WARN] Modotte infinity loading failed: {e}")

    # ── 7: JDhruv14/Bhagavad-Gita_Dataset ────────────────────────────────
    print("\n[Dataset 7/7] JDhruv14/Bhagavad-Gita_Dataset (HuggingFace) …")
    try:
        all_docs.extend(_parse_jdhruv14_dataset(RAW_DATA_DIR))
    except Exception as e:
        print(f"  [WARN] JDhruv14 Dataset loading failed: {e}")

    if not all_docs:
        raise RuntimeError(
            "No documents loaded from any dataset. "
            "Check your network connection and try again."
        )

    # ── Merge all sources by (chapter, verse) ─────────────────────────────
    print("\nMerging documents by chapter + verse …")
    all_docs = _merge_by_verse(all_docs)

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
