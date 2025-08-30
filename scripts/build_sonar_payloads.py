#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Sonar payloads from a JSON file of features, with retrieval.

Usage:
  python -m scripts.build_sonar_payloads \
    --features data/triaged_15_features2.json \
    --outdir outputs/sonar_payloads \
    --k 4 \
    --fewshots prompts/fewshots.md

What it does:
- Loads features (list of {feature_id, feature_name, feature_description}).
- Uses FastEmbed + Chroma (collection="laws") to retrieve top-k legal passages per feature.
- Formats passages into one-line context cards: [ctx_N] JURIS — LAW — "obligation sentence..."
- Assembles Sonar request payloads (system + optional few-shots + user task + schema).
- Writes {feature_id}.json (API-ready) and {feature_id}.chat.txt (human-readable).
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import chromadb
from fastembed import TextEmbedding

# ---------- Paths & config ----------
ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index" / "chroma"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "laws"

# ---------- Prompts (align with llm_classifier.py) ----------
SYSTEM_POLICY = """You are a compliance triage assistant for geo-specific regulation.
Return ONE JSON object that matches the provided JSON Schema. No extra text, no markdown.

Decision policy:
- “Geo-specific” = any legal obligation scoped to a jurisdictional subset (e.g., EU users, US federal scope, Utah minors, California teens). Harmonized regional laws count.
- Say "yes" ONLY if at least one context item states a concrete legal obligation containing one of:
  shall|must|required|prohibited|consent|verify|report|age verification|curfew.
- Prefer citations whose jurisdiction matches the feature’s stated geo. Tie-breaker:
  exact match > regional (EU/US federal) > adjacent/analog US state > unrelated.
- Say "no" for business/UX/experiments/analytics/monetization with no legal trigger.
- Say "uncertain" if no obligation sentence is present, or info is insufficient/conflicting.
- Do not invent laws. Cite only provided context IDs in “citations”.
- "reasoning" ≤ 80 words; refer only to context IDs (e.g., ctx_1) or law IDs.

Risk levels:
- critical: shipping without control likely unlawful or penalized by statute.
- high: clear obligation, penalties unclear, high enforcement risk.
- medium: probable obligation but scope/implementation unclear.
- low: no trigger found or purely non-legal change.

Output rules:
- Exactly one JSON object; no prose before/after.
- Citations must be context IDs: ^ctx_[0-9]+$ only.
- If needs_geo_compliance="yes": citations.length ≥ 1.
- If no qualifying obligation sentence exists: needs_geo_compliance="uncertain".
- If you cannot produce valid JSON per schema, return exactly: {"needs_geo_compliance":"uncertain","citations":[]}
"""

USER_TASK_TMPL = """Task: Classify whether this feature needs geo-specific compliance logic according to the provided legal context.

Feature:
- title: {title}
- description: {description}

Retrieved legal context (top {k}):
{context_cards}
# Each card is ONE line: [ctx_N] {{JURISDICTION}} — {{LAW ID}} — "obligation sentence..."

Labels:
- needs_geo_compliance ∈ {{yes, no, uncertain}}

Rubric recap:
- yes: at least one obligation sentence and the feature applies to a jurisdictional subset.
- no: no legal trigger in context (pure UX/analytics/monetization/experiments).
- uncertain: no obligation sentence or conflicting/insufficient info.

Return JSON only, matching this schema.
"""

RESPONSE_SCHEMA_INLINE = {
    "type": "object",
    "properties": {
        "needs_geo_compliance": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "risk_level": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
        "citations": {"type": "array", "items": {"type": "string"}, "minItems": 0},
        "reasoning": {"type": "string", "maxLength": 80}
    },
    "required": ["needs_geo_compliance", "citations"],
    "additionalProperties": False
}

# ---------- Retrieval helpers ----------

def ensure_index() -> chromadb.Collection:
    """
    Connect to Chroma persistent index and return the 'laws' collection.
    If it doesn't exist, attempt to build it by importing index/build.py.
    """
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as e:
        if "does not exist" in str(e).lower() or "not found" in str(e).lower():
            print("🔧 Vector index not found. Building automatically...")
            try:
                import sys
                sys.path.append(str(ROOT / "index"))
                from build import build_index
                build_index()
                print("✅ Vector index built successfully!")
                return client.get_collection(COLLECTION_NAME)
            except Exception as be:
                raise RuntimeError(f"Failed to build vector index: {be}") from be
        raise

def embed_query(embedding_model: TextEmbedding, text: str) -> List[float]:
    vec = list(embedding_model.embed([text]))[0]
    return np.asarray(vec, dtype=np.float32).tolist()

def retrieve_passages(collection: chromadb.Collection, query_vec: List[float], k: int) -> List[Dict]:
    res = collection.query(
        query_embeddings=[query_vec],
        n_results=max(k, 1),
        include=["documents", "metadatas", "distances"]
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out: List[Dict] = []
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        out.append({
            "text": docs[i],
            "jurisdiction": meta.get("jurisdiction", "") or "UNKNOWN",
            "law": meta.get("law", "") or "UNKNOWN LAW",
            "chunk_id": meta.get("chunk_id", f"chunk_{i}"),
            "source_url": meta.get("source_url", ""),
            "distance": float(dists[i]) if i < len(dists) else None,
            "relevance_score": (1.0 - float(dists[i])) if i < len(dists) else None
        })
    return out

def to_context_cards(passages: List[Dict]) -> List[str]:
    """
    Convert retrieved passages into one-sentence context cards.
    Keep first sentence; trim to ~260 chars to reduce drift.
    """
    cards: List[str] = []
    for idx, p in enumerate(passages, 1):
        text = (p.get("text") or "").strip()
        # Heuristic: first sentence
        sent = text.split(".")[0].strip()
        if len(sent) > 260:
            sent = sent[:257] + "..."
        juris = (p.get("jurisdiction") or "UNKNOWN").strip()
        law = (p.get("law") or "UNKNOWN LAW").strip()
        cards.append(f'[ctx_{idx}] {juris} — {law} — "{sent}"')
    return cards

# ---------- Few-shots loader ----------

def load_fewshots(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Few-shots file not found: {path}")
    # accept md/txt/json; if json is list or {"blocks":[...]} join with blank lines
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        blocks = data["blocks"] if isinstance(data, dict) and "blocks" in data else data
        if not isinstance(blocks, list):
            raise ValueError("Invalid few-shots JSON: expected list or {'blocks': [...]}.")
        return "\n\n".join(str(b).strip() for b in blocks if str(b).strip())
    return p.read_text().strip()

# ---------- Payload assembly ----------

def build_payload(feature: Dict, cards: List[str], fewshots: str) -> Tuple[Dict, List[Dict], str]:
    user_msg = USER_TASK_TMPL.format(
        title=feature["feature_name"],
        description=feature["feature_description"],
        k=len(cards),
        context_cards="\n".join(cards)
    )

    messages = [{"role": "system", "content": SYSTEM_POLICY}]
    if fewshots:
        messages.append({"role": "user", "content": fewshots})
    messages.append({"role": "user", "content": user_msg})

    payload = {
        "model": "sonar-pro",
        "temperature": 0,
        "top_p": 0,
        "disable_search": True,
        "max_tokens": 400,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": RESPONSE_SCHEMA_INLINE}
        }
    }
    # For convenience, also return a compact “debug” structure of raw passages (not sent to Sonar)
    debug_passages = [{"jurisdiction": p["jurisdiction"], "law": p["law"],
                       "chunk_id": p["chunk_id"], "relevance": p["relevance_score"]} for p in cards_raw]
    return payload, debug_passages, user_msg  # cards_raw is defined in main before calling

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to features JSON file")
    ap.add_argument("--outdir", default="outputs/sonar_payloads", help="Output directory")
    ap.add_argument("--fewshots", default="", help="Optional path to few-shots (md/txt/json)")
    ap.add_argument("--k", type=int, default=4, help="Top-K passages to include per feature")
    args = ap.parse_args()

    features = json.loads(Path(args.features).read_text())
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fewshots_text = load_fewshots(args.fewshots)

    # Init retrieval stack
    embedding = TextEmbedding(model_name=EMBED_MODEL)
    collection = ensure_index()

    for feat in features:
        feat_id = feat.get("feature_id") or "FEAT"
        title = feat.get("feature_name", "")
        desc = feat.get("feature_description", "")
        query_text = f"{title}\n\n{desc}".strip()

        # Embed + retrieve
        qvec = embed_query(embedding, query_text)
        passages = retrieve_passages(collection, qvec, k=args.k)

        # Build cards
        cards = to_context_cards(passages)[:args.k]

        # Build payload
        # NOTE: to include a minimal debug map of passages back out, we capture passages as cards_raw here
        global cards_raw
        cards_raw = passages  # used inside build_payload for the debug list
        payload, debug_passages, chat_txt = build_payload(feat, cards, fewshots_text)

        # Write artifacts
        out_json = outdir / f"{feat_id}.json"
        out_chat = outdir / f"{feat_id}.chat.txt"
        out_ctx  = outdir / f"{feat_id}.context.json"

        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        with out_chat.open("w", encoding="utf-8") as f:
            for m in payload["messages"]:
                f.write(f"[{m['role'].upper()}]\n{m['content']}\n\n")
        out_ctx.write_text(json.dumps({"cards": cards, "passages": passages}, ensure_ascii=False, indent=2))

        print(f"✅ wrote {out_json.name}, {out_chat.name}, {out_ctx.name} -> {outdir}")

    print("\nAll payloads generated.")

if __name__ == "__main__":
    main()
