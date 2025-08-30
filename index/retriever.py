#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jurisdiction-first retriever for law passages.

Priority tiers (all obligation-filtered):
  1) exact juris_norm
  2) same state (US)
  3) same country
  4) same region (eu/us)
  5) global fallback (optional)

Returns top-k unique chunks ranked by vector similarity with a small jurisdiction bonus.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re

import chromadb
from fastembed import TextEmbedding

# --- Config ---
ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index" / "chroma"
COLLECTION_NAME = "laws"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

OBLIGATION_REGEX = re.compile(
    r"\b(shall|must|required|prohibited|prohibit|consent|verify|verification|"
    r"report|reporting|age\s*verification|curfew|restrict|restriction)\b",
    re.IGNORECASE,
)

# --- Helpers ---
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def _parse_jurisdiction(j: str) -> Dict[str, str]:
    jn = _norm(j)
    country = state = region = ""
    if "european union" in jn or jn == "eu":
        region = "eu"
    if "united states" in jn or jn in {"usa", "us"}:
        country = "united states"; region = "us"
    for nm in ["utah", "california", "florida", "new york", "texas"]:
        if nm in jn:
            state = nm
            if not country:
                country = "united states"; region = "us"
            break
    return {"juris_norm": jn, "country": country, "state": state, "region": region}

def _where_eq_map(d: Dict[str, object]) -> Dict[str, object]:
    """
    Build a Chroma 0.5-compliant filter:
      {"$and": [{"field": {"$eq": val}}, ...]}
    Falls back to a single {"field": {"$eq": val}} when only one item.
    """
    items = [{k: {"$eq": v}} for k, v in d.items()]
    if len(items) == 1:
        return items[0]
    return {"$and": items}

# --- Retriever class ---
class LawRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(INDEX_DIR))
        self.coll = self.client.get_collection(COLLECTION_NAME)
        self.embedder = TextEmbedding(model_name=EMBED_MODEL)
        # warmup (first call may trigger model load)
        _ = list(self.embedder.embed(["warmup"]))

    def _embed(self, text: str) -> List[float]:
        return list(self.embedder.embed([text]))[0]

    def _query(self, qvec: List[float], n_results: int, where: Dict[str, Any]) -> Dict[str, Any]:
        return self.coll.query(
            query_embeddings=[qvec],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],  # no "ids" in Chroma 0.5 include
        )

    def search(
        self,
        feature_title: str,
        feature_description: str,
        feature_jurisdiction: str,
        k: int = 8,
        allow_global_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Jurisdiction-first retrieval with obligation filtering and small geo bonuses.

        If feature_jurisdiction is empty/UNKNOWN and allow_global_fallback=False,
        returns [] (so no law can be quoted).
        """
        qtext = f"{feature_title}\n\n{feature_description}"
        qvec = self._embed(qtext)

        # Handle unknown/blank jurisdiction
        if not feature_jurisdiction or feature_jurisdiction.strip().upper() == "UNKNOWN":
            if not allow_global_fallback:
                return []  # strict: no context when jurisdiction is unknown
            geo = {"juris_norm": "", "state": "", "country": "", "region": ""}
            tiers: List[Dict[str, Any]] = [{"is_obligation": True}]  # global only
        else:
            geo = _parse_jurisdiction(feature_jurisdiction)
            tiers: List[Dict[str, Any]] = []
            if geo["juris_norm"]:
                tiers.append({"juris_norm": geo["juris_norm"], "is_obligation": True})
            if geo["state"]:
                tiers.append({"state": geo["state"], "is_obligation": True})
            if geo["country"]:
                tiers.append({"country": geo["country"], "is_obligation": True})
            if geo["region"]:
                tiers.append({"region": geo["region"], "is_obligation": True})
            # global tier last if allowed
            if allow_global_fallback:
                tiers.append({"is_obligation": True})

        seen = set()
        bag: List[Tuple[float, Dict[str, Any]]] = []
        per_tier_fetch = max(k * 2, 16)

        for filt in tiers:
            res = self._query(qvec, per_tier_fetch, where=_where_eq_map(filt))
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]

            for idx in range(min(len(docs), len(metas), len(dists))):
                _doc = docs[idx]
                _meta = metas[idx]
                _dist = dists[idx]
                _id = _meta.get("chunk_id") or f"row_{idx}"  # stable-ish fallback

                if _id in seen:
                    continue
                if not OBLIGATION_REGEX.search(_doc or ""):
                    continue

                bonus = 0.0
                if geo.get("juris_norm") and _norm(_meta.get("juris_norm", "")) == geo["juris_norm"]:
                    bonus -= 0.03
                elif geo.get("state") and _norm(_meta.get("state", "")) == geo["state"]:
                    bonus -= 0.02
                elif geo.get("country") and _norm(_meta.get("country", "")) == geo["country"]:
                    bonus -= 0.01
                elif geo.get("region") and _norm(_meta.get("region", "")) == geo["region"]:
                    bonus -= 0.005

                seen.add(_id)
                bag.append((
                    float(_dist) + bonus,
                    {
                        "id": _id,
                        "text": _doc,
                        "distance": float(_dist),
                        "jurisdiction": _meta.get("jurisdiction", ""),
                        "law": _meta.get("law", ""),
                        "source_url": _meta.get("source_url", ""),
                        "file": _meta.get("file", ""),
                    }
                ))

            if len(bag) >= k:
                break

        bag.sort(key=lambda x: x[0])
        return [item for _, item in bag[:k]]

# --- CLI smoke test ---
if __name__ == "__main__":
    r = LawRetriever()
    results = r.search(
        feature_title="Curfew login blocker",
        feature_description="Block minor logins during night hours as required by state law.",
        feature_jurisdiction="Utah, United States",
        k=5
    )
    for i, c in enumerate(results, 1):
        print(f'[ctx_{i}] {c["jurisdiction"]} — {c["law"]} — "{c["text"]}"  (dist={c["distance"]:.3f})')
