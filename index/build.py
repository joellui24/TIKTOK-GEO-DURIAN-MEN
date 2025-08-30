#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a Chroma vector index for 'laws' with sentence-level, obligation-focused chunks.

- Reads source texts from: data/laws/*.md (also .txt supported)
- Extracts metadata (jurisdiction, law, source_url) from file content or filename
- Splits into sentences and **keeps only sentences that look like legal obligations**
  (shall|must|required|prohibited|consent|verify|report|age verification|curfew)
- Adds each sentence to Chroma with metadata and a stable chunk_id

This index is used by:
  scripts/build_sonar_payloads.py  (retrieval for payload context cards)
  llm_classifier.py                (retrieval-augmented classification)

Usage:
  python index/build.py
  (or it will be called programmatically via build_index())
"""

from __future__ import annotations
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from fastembed import TextEmbedding

# ---------------------- Config ----------------------

ROOT = Path(__file__).resolve().parents[1]
LAWS_DIR = ROOT / "data" / "laws"          # source documents
INDEX_DIR = ROOT / "data" / "index" / "chroma"
COLLECTION_NAME = "laws"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Obligation signal (case-insensitive)
OBLIGATION_REGEX = re.compile(
    r"\b(shall|must|required|prohibited|prohibit|consent|verify|verification|"
    r"report|reporting|age\s*verification|curfew|restrict|restriction)\b",
    re.IGNORECASE,
)

# Soft guard to avoid over-long chunks
MAX_SENT_LEN = 600  # chars
MIN_SENT_LEN = 20   # chars (skip too-short fragments)

# ---------------------- Utilities ----------------------

def read_sources() -> List[Tuple[Path, str]]:
    """Load .md/.txt files from LAWS_DIR."""
    if not LAWS_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {LAWS_DIR}")
    paths = sorted(list(LAWS_DIR.glob("*.md")) + list(LAWS_DIR.glob("*.txt")))
    out: List[Tuple[Path, str]] = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            text = p.read_text(errors="ignore")
        out.append((p, text))
    if not out:
        raise FileNotFoundError(f"No .md/.txt files found in {LAWS_DIR}")
    return out


def first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def extract_frontmatter(text: str) -> Dict[str, str]:
    """
    Super-lightweight frontmatter/inline metadata parser.
    Looks for lines like:
      **Jurisdiction:** Utah, United States
      **Law ID:** Utah Code §13-2c-301
      **Source:** https://...
    Also tolerates 'Jurisdiction:' without bold.
    """
    meta = {"jurisdiction": "", "law": "", "source_url": ""}
    for line in text.splitlines()[:80]:  # only scan header-ish part
        s = line.strip()
        # Normalize markdown bold markers
        s = s.replace("**", "")
        if s.lower().startswith("jurisdiction:"):
            meta["jurisdiction"] = s.split(":", 1)[1].strip()
        elif s.lower().startswith("law id:"):
            meta["law"] = s.split(":", 1)[1].strip()
        elif s.lower().startswith("source:"):
            meta["source_url"] = s.split(":", 1)[1].strip()
        # Fallbacks for common headings
        elif s.startswith("# "):
            # H1 as law name if nothing else
            title = s[2:].strip()
            if not meta["law"]:
                meta["law"] = title
    return meta


def infer_meta_from_filename(path: Path) -> Dict[str, str]:
    """
    Heuristic: filename like 'US-UT-Social-Media-Act-2024.md'
      -> jurisdiction: 'Utah, United States'
      -> law: 'US-UT-Social-Media-Act-2024'
    """
    stem = path.stem
    juris = ""
    law = stem.replace("_", " ").replace("-", " ").strip()
    # Simple US state mapping if prefixed like 'US-UT-...'
    m = re.match(r"([A-Za-z]{2})-([A-Za-z]{2})-", stem)
    if m and m.group(1).upper() == "US":
        # crude map of US states; keep it minimal to avoid hard deps
        STATES = {"UT": "Utah", "CA": "California", "FL": "Florida", "NY": "New York", "TX": "Texas"}
        st = STATES.get(m.group(2).upper())
        if st:
            juris = f"{st}, United States"
    return {"jurisdiction": juris, "law": law, "source_url": ""}


def normalize_meta(file_meta: Dict[str, str], name_meta: Dict[str, str]) -> Dict[str, str]:
    juris = (file_meta.get("jurisdiction") or name_meta.get("jurisdiction") or "").strip() or "UNKNOWN"
    law = (file_meta.get("law") or name_meta.get("law") or "").strip() or "UNKNOWN LAW"
    src = (file_meta.get("source_url") or name_meta.get("source_url") or "").strip()
    return {"jurisdiction": juris, "law": law, "source_url": src}


def split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter (regex-based).
    Good enough for legal prose without adding heavy NLP deps.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.strip())
    # Split on ., !, ? followed by space and capital OR end of text
    # Keep the delimiter
    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\"(])", text)
    # Also split on '•' or '-' list bullets into separate "sentences"
    out: List[str] = []
    for p in parts:
        # Break bullet-style lists further
        bullets = re.split(r"(?:^|\s)[\-\u2022]\s+", p)
        for b in bullets:
            s = b.strip()
            if s:
                out.append(s)
    return out


def extract_obligation_sentences(text: str) -> List[str]:
    """
    Return only sentences that match obligation keywords.
    """
    sents = split_sentences(text)
    kept: List[str] = []
    for s in sents:
        if len(s) > MAX_SENT_LEN or len(s) < MIN_SENT_LEN:
            continue
        if OBLIGATION_REGEX.search(s):
            kept.append(s)
    return kept


def ensure_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    # If collection exists, drop and recreate to avoid dupes/stale data
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.create_collection(name=COLLECTION_NAME)


# ---------------------- Build ----------------------

def build_index() -> None:
    """
    Build (or rebuild) the 'laws' collection with obligation-focused sentence chunks.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    collection = ensure_collection()
    embedding = TextEmbedding(model_name=EMBED_MODEL)

    sources = read_sources()

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict] = []
    embs: List[List[float]] = []

    total_added = 0

    for path, text in sources:
        file_meta = extract_frontmatter(text)
        name_meta = infer_meta_from_filename(path)
        meta_base = normalize_meta(file_meta, name_meta)

        # Use only obligation-style sentences to tighten retrieval
        sentences = extract_obligation_sentences(text)

        # If none found, fall back to first ~2 sentences so doc is still represented
        if not sentences:
            sents = split_sentences(text)[:2]
            sentences = [s for s in sents if MIN_SENT_LEN <= len(s) <= MAX_SENT_LEN]

        for i, sent in enumerate(sentences, start=1):
            chunk_id = f"{path.stem}-s{i:03d}"
            ids.append(chunk_id)
            docs.append(sent)
            metas.append({
                "chunk_id": chunk_id,
                "jurisdiction": meta_base["jurisdiction"],
                "law": meta_base["law"],
                "source_url": meta_base["source_url"],
                "file": str(path.relative_to(ROOT)),
            })

            # embed
            vec = list(embedding.embed([sent]))[0]
            embs.append(vec)

            # Flush in batches to keep memory steady
            if len(ids) >= 512:
                collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                total_added += len(ids)
                ids, docs, metas, embs = [], [], [], []

        # Optional: also index a short "law summary" header if present and reasonably informative
        h1 = first_line(text)
        if h1.startswith("# "):
            summary = h1[2:].strip()
            if summary and len(summary) >= MIN_SENT_LEN:
                chunk_id = f"{path.stem}-hdr"
                ids.append(chunk_id)
                docs.append(summary[:MAX_SENT_LEN])
                metas.append({
                    "chunk_id": chunk_id,
                    "jurisdiction": meta_base["jurisdiction"],
                    "law": meta_base["law"],
                    "source_url": meta_base["source_url"],
                    "file": str(path.relative_to(ROOT)),
                    "is_header": True
                })
                vec = list(embedding.embed([docs[-1]]))[0]
                embs.append(vec)

        # Batch flush per file too
        if ids:
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            total_added += len(ids)
            ids, docs, metas, embs = [], [], [], []

    print(f"✅ Indexed {total_added} chunks into collection '{COLLECTION_NAME}' at {INDEX_DIR}")


# ---------------------- CLI ----------------------

def main():
    build_index()

if __name__ == "__main__":
    main()
