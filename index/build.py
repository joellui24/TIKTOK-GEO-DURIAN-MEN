#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a Chroma vector index for 'laws' with sentence-level, obligation-focused chunks.

- Reads source texts from: data/laws/*.md (also .txt supported)
- Extracts metadata (jurisdiction, law, source_url) from file content or filename
- Splits into sentences and **keeps only sentences that look like legal obligations**
  (shall|must|required|prohibited|consent|verify|report|age verification|curfew|restrict)
- Adds each sentence to Chroma with metadata and a stable chunk_id
- Stores normalized geo fields to enable jurisdiction-first retrieval

Usage:
  python3 index/build.py
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from fastembed import TextEmbedding

# ---------------------- Config ----------------------

ROOT = Path(__file__).resolve().parents[1]
LAWS_DIR = ROOT / "data" / "laws"           # source documents
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

# ---------------------- Geo helpers ----------------------

_US_STATES = {
    "UT": "Utah", "CA": "California", "FL": "Florida", "NY": "New York", "TX": "Texas",
    # extend as needed
}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def parse_jurisdiction(j: str) -> Dict[str, str]:
    """
    Returns normalized fields: juris_norm, country, state, region.
    e.g., "Utah, United States" -> country="united states", state="utah", region="us"
          "European Union" -> region="eu"
    """
    jn = _norm(j)
    country = state = region = ""
    if "european union" in jn or jn == "eu":
        region = "eu"
    if "united states" in jn or jn in {"usa", "us"}:
        country = "united states"; region = "us"
    for abbr, name in _US_STATES.items():
        if name.lower() in jn or f"us-{abbr.lower()}" in jn:
            state = name.lower()
            if not country:
                country = "united states"; region = "us"
            break
    return {"juris_norm": jn, "country": country, "state": state, "region": region}

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
        s = line.strip().replace("**", "")
        low = s.lower()
        if low.startswith("jurisdiction:"):
            meta["jurisdiction"] = s.split(":", 1)[1].strip()
        elif low.startswith("law id:"):
            meta["law"] = s.split(":", 1)[1].strip()
        elif low.startswith("source:"):
            meta["source_url"] = s.split(":", 1)[1].strip()
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
    m = re.match(r"([A-Za-z]{2})-([A-Za-z]{2})-", stem)
    if m and m.group(1).upper() == "US":
        st = _US_STATES.get(m.group(2).upper())
        if st:
            juris = f"{st}, United States"
    return {"jurisdiction": juris, "law": law, "source_url": ""}


def normalize_meta(file_meta: Dict[str, str], name_meta: Dict[str, str]) -> Dict[str, str]:
    juris = (file_meta.get("jurisdiction") or name_meta.get("jurisdiction") or "").strip() or "UNKNOWN"
    law = (file_meta.get("law") or name_meta.get("law") or "").strip() or "UNKNOWN LAW"
    src = (file_meta.get("source_url") or name_meta.get("source_url") or "").strip()
    return {"jurisdiction": juris, "law": law, "source_url": src}


def split_sentences(text: str) -> List[str]:
    """Regex sentence splitter (lightweight)."""
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\"(])", text)
    out: List[str] = []
    for p in parts:
        bullets = re.split(r"(?:^|\s)[\-\u2022]\s+", p)  # split bullet lists
        for b in bullets:
            s = b.strip()
            if s:
                out.append(s)
    return out


def extract_obligation_sentences(text: str) -> List[str]:
    """Return only sentences that match obligation keywords."""
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
    # fresh rebuild to avoid dupes/stale data
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.create_collection(name=COLLECTION_NAME)

# ---------------------- Build ----------------------

def build_index() -> None:
    """Build (or rebuild) the 'laws' collection with obligation-focused sentence chunks."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    collection = ensure_collection()

    # Load embedder once
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    # warmup (first call may trigger download)
    _ = list(embedder.embed(["warmup"]))

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
        geo = parse_jurisdiction(meta_base["jurisdiction"])

        # obligation-only sentences (fallback to first 2 if none)
        sentences = extract_obligation_sentences(text)
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
                "is_obligation": bool(OBLIGATION_REGEX.search(sent)),
                **geo
            })

            # embed
            embs.append(list(embedder.embed([sent]))[0])

            # flush in batches
            if len(ids) >= 512:
                collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                total_added += len(ids)
                ids, docs, metas, embs = [], [], [], []

        # Optional header summary
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
                    "is_header": True,
                    "is_obligation": bool(OBLIGATION_REGEX.search(summary)),
                    **geo
                })
                embs.append(list(embedder.embed([docs[-1]]))[0])

        # flush per-file too
        if ids:
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            total_added += len(ids)
            ids, docs, metas, embs = [], [], [], []

    print(f"✅ Indexed {total_added} chunks into collection '{COLLECTION_NAME}' at {INDEX_DIR}")

def main():
    build_index()

if __name__ == "__main__":
    main()
