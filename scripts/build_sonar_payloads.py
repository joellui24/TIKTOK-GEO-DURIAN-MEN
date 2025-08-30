#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Sonar payloads from a JSON file of features, with jurisdiction auto-detect + retrieval.

Usage:
  python -m scripts.build_sonar_payloads \
    --features data/triaged_15_features2.json \
    --outdir outputs/sonar_payloads \
    --k 4 \
    [--fewshots prompts/fewshots.md] \
    [--no-auto-jurisdiction]

What it does:
- Loads features (list of {feature_id, feature_name, feature_description, [jurisdiction]?}).
- If no jurisdiction is provided, auto-detects from title/description (Utah/California/Florida/US/EU/UK/CA/AU, etc.).
- Uses index/retriever.py to retrieve top-k *obligation* sentences, prioritizing the inferred jurisdiction.
- Formats context cards and assembles a Sonar-ready JSON payload + human-readable chat + debug context.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import re

# ---------- Retrieval (jurisdiction-first) ----------
from index.retriever import LawRetriever  # make sure index/retriever.py exists

# ---------- Prompts ----------
SYSTEM_POLICY = """You are a compliance triage assistant for geo-specific regulation.
Return ONE JSON object that matches the provided JSON Schema. No extra text, no markdown.

Decision policy:
- “Geo-specific” = any legal obligation scoped to a jurisdictional subset (e.g., EU users, US federal scope, Utah minors, California teens). Harmonized regional laws count.
- Say "yes" ONLY if at least one context item states a concrete legal obligation containing one of:
  shall|must|required|prohibited|consent|verify|report|age verification|curfew|restrict.
- Prefer citations whose jurisdiction matches the feature’s stated geo. Tie-breaker:
  exact match > regional (EU/US federal) > adjacent/analog state > unrelated.
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

# ---------- Jurisdiction inference ----------
STATE_WORDS = {
    "utah": "Utah, United States",
    "california": "California, United States",
    "florida": "Florida, United States",
    "new york": "New York, United States",
    "texas": "Texas, United States",
}

COUNTRY_WORDS = {
    "canada": "Canada",
    "australia": "Australia",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "united states": "United States",
    "us federal": "United States",
    "usa": "United States",
    "us ": "United States",
}

REGION_WORDS = {
    "european union": "European Union",
    " eu ": "European Union",
    "eu dsa": "European Union",
    "digital services act": "European Union",
    "gdpr": "European Union",
}

LAW_CUES = [
    # California
    (r"\bsb\s*976\b", "California, United States"),
    (r"\bcalifornia\b", "California, United States"),
    # Utah
    (r"\butah\b", "Utah, United States"),
    (r"\bsocial\s+media\s+regulation\s+act\b", "Utah, United States"),
    # Florida
    (r"\bflorida\b", "Florida, United States"),
    # US federal / child safety
    (r"\bncmec\b", "United States"),
    (r"\bcopp?a\b", "United States"),
    (r"\bcsam\b", "United States"),
    (r"\bus federal\b", "United States"),
    # EU
    (r"\bdigital\s+services\s+act\b", "European Union"),
    (r"\beu\s*dsa\b", "European Union"),
    (r"\bgdpr\b", "European Union"),
    (r"\beuropean\s+union\b", "European Union"),
    # Canada
    (r"\bcanada\b", "Canada"),
    # UK
    (r"\bofcom\b", "United Kingdom"),
    (r"\buk\b", "United Kingdom"),
    # Australia
    (r"\baustralia\b", "Australia"),
]

def infer_jurisdiction(title: str, desc: str) -> str:
    """Heuristic inference from free text; returns a single best jurisdiction string."""
    text = f" {title} {desc} ".lower()

    # 1) High-precision law cues
    for patt, juris in LAW_CUES:
        if re.search(patt, text):
            return juris

    # 2) Explicit state names
    for token, juris in STATE_WORDS.items():
        if token in text:
            return juris

    # 3) Explicit region cues (EU first)
    for token, juris in REGION_WORDS.items():
        if token in text:
            return juris

    # 4) Country cues
    for token, juris in COUNTRY_WORDS.items():
        if token in text:
            return juris

    # 5) Ambiguous "CA": prefer Canada ONLY if "Canada" appears elsewhere; else California when 'SB' present
    if " ca " in text or text.strip().endswith(" ca"):
        if "canada" in text:
            return "Canada"
        if re.search(r"\bsb\s*\d{2,4}\b", text):
            return "California, United States"

    # Default fallback
    return "European Union"

# ---------- Formatting ----------
def to_context_cards(passages: List[Dict]) -> List[str]:
    cards: List[str] = []
    for idx, p in enumerate(passages, 1):
        text = (p.get("text") or "").strip()
        sent = text.split(".")[0].strip()
        if len(sent) > 260:
            sent = sent[:257] + "..."
        juris = (p.get("jurisdiction") or "UNKNOWN").strip()
        law = (p.get("law") or "UNKNOWN LAW").strip()
        cards.append(f'[ctx_{idx}] {juris} — {law} — "{sent}"')
    return cards

# ---------- Few-shots ----------
def load_fewshots(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Few-shots file not found: {path}")
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        blocks = data["blocks"] if isinstance(data, dict) and "blocks" in data else data
        if not isinstance(blocks, list):
            raise ValueError("Invalid few-shots JSON: expected list or {'blocks': [...]}.")
        return "\n\n".join(str(b).strip() for b in blocks if str(b).strip())
    return p.read_text().strip()

# ---------- Payload assembly ----------
def build_payload(feature: Dict, cards: List[str], fewshots: str) -> Tuple[Dict, str]:
    user_msg = USER_TASK_TMPL.format(
        title=feature.get("feature_name", ""),
        description=feature.get("feature_description", ""),
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
    return payload, user_msg

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to features JSON file")
    ap.add_argument("--outdir", default="outputs/sonar_payloads", help="Output directory")
    ap.add_argument("--fewshots", default="", help="Optional path to few-shots (md/txt/json)")
    ap.add_argument("--k", type=int, default=4, help="Top-K passages to include per feature")
    ap.add_argument("--no-auto-jurisdiction", action="store_true",
                    help="Disable auto-detection; only use feature['jurisdiction'] or fallback to EU")
    ap.add_argument("--fallback-jurisdiction", default="European Union",
                    help="Used if neither explicit nor inferred jurisdiction is available")
    args = ap.parse_args()

    features_obj = json.loads(Path(args.features).read_text())
    features: List[Dict] = features_obj["features"] if isinstance(features_obj, dict) and "features" in features_obj else features_obj

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fewshots_text = load_fewshots(args.fewshots)

    retriever = LawRetriever()

    for feat in features:
        feat_id = feat.get("feature_id") or feat.get("id") or "FEAT"
        title = feat.get("feature_name", "")
        desc = feat.get("feature_description", "")

        explicit = (feat.get("jurisdiction") or "").strip()
        inferred = ""
        if not args.no_auto_jurisdiction and not explicit:
            inferred = infer_jurisdiction(title, desc)
        juris = explicit or inferred or args.fallback_jurisdiction

        print(f"🔎 [{feat_id}] jurisdiction -> explicit='{explicit or '-'}' | inferred='{inferred or '-'}' | using='{juris}'")

        passages = retriever.search(
            feature_title=title,
            feature_description=desc,
            feature_jurisdiction=juris,
            k=args.k
        )

        cards = to_context_cards(passages)[:args.k]
        payload, chat_txt = build_payload(feat, cards, fewshots_text)

        out_json = outdir / f"{feat_id}.json"
        out_chat = outdir / f"{feat_id}.chat.txt"
        out_ctx  = outdir / f"{feat_id}.context.json"

        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        with out_chat.open("w", encoding="utf-8") as f:
            for m in payload["messages"]:
                f.write(f"[{m['role'].upper()}]\n{m['content']}\n\n")
        out_ctx.write_text(json.dumps({
            "used_jurisdiction": juris,
            "explicit_jurisdiction": explicit,
            "inferred_jurisdiction": inferred,
            "cards": cards,
            "passages": passages
        }, ensure_ascii=False, indent=2))

        print(f"✅ wrote {out_json.name}, {out_chat.name}, {out_ctx.name} -> {outdir}")

    print("\nAll payloads generated.")

if __name__ == "__main__":
    main()

