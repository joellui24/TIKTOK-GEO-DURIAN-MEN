#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from typing import List, Dict

# Import your classifier (uses your own Chroma index; no external API calls)
from compliance.simple_classifier import SimpleComplianceClassifier, ROOT  # adjust if your path differs


PROMPTS_DIR = ROOT / "prompts" / "sonar_v1"
SYSTEM_PATH = PROMPTS_DIR / "system.txt"
USER_TPL_PATH = PROMPTS_DIR / "user_template.txt"
SCHEMA_PATH = PROMPTS_DIR / "schema.json"

OUT_DIR = ROOT / "outputs" / "sonar_payloads"


def _load(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _truncate_snippet(text: str, limit: int = 600) -> str:
    t = " ".join(text.split())  # collapse whitespace/newlines
    return t[:limit] + ("…" if len(t) > limit else "")


def _format_context(passages: List[Dict]) -> str:
    """
    Build the [ctx_i] blocks Sonar will see/cite.
    Each block: [ctx_i] doc_id, section, chunk_index, short snippet.
    """
    lines = []
    for i, p in enumerate(passages, start=1):
        doc_id = p.get("doc_id", "") or f"{p.get('jurisdiction','').strip()} {p.get('law','').strip()}".strip()
        section = p.get("section_title", "")
        chunk_idx = p.get("chunk_index", 0)
        score = p.get("relevance_score", 0.0)
        snippet = _truncate_snippet(p.get("text", ""), 600)
        lines.append(f"[ctx_{i}] doc_id={doc_id}, section=\"{section}\", chunk_index={chunk_idx}, score={score:.3f}\n{snippet}")
    return "\n".join(lines) if lines else "(no context found)"


def build_payload(model: str,
                  system_prompt: str,
                  user_template: str,
                  schema_obj: dict,
                  feature_id: str,
                  title: str,
                  description: str,
                  passages: List[Dict],
                  k: int = 5,
                  disable_search: bool = True) -> dict:
    """
    Render a Sonar-compatible chat.completions request body (OpenAI-compatible shape).
    """
    context_blocks = _format_context(passages[:k])
    user_msg = user_template.format(
        title=title.replace('"', '\\"'),
        description=description.replace('"', '\\"'),
        k=len(passages[:k]),
        context_blocks=context_blocks
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0,
        # default for your RAG flow: keep Sonar from browsing the web
        "disable_search": disable_search,
        # enforce JSON output schema
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": schema_obj
            }
        },
        "max_tokens": 400
    }
    return payload


def load_features_from_json(path: Path) -> List[Dict]:
    """
    Expect a list like:
    [{"feature_id":"A1","feature_name":"...","feature_description":"..."}, ...]
    """
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Build Sonar request payloads without calling the API.")
    parser.add_argument("--model", default="sonar-pro", help="Perplexity model name (e.g., sonar or sonar-pro).")
    parser.add_argument("--features", type=str, default="", help="Path to a JSON file of features.")
    parser.add_argument("--k", type=int, default=5, help="Top-k context passages.")
    parser.add_argument("--enable-search", action="store_true", help="If set, do NOT disable Sonar search.")
    args = parser.parse_args()

    # Load prompt pack
    system_prompt = _load(SYSTEM_PATH)
    user_template = _load(USER_TPL_PATH)
    schema_obj = json.loads(_load(SCHEMA_PATH))

    # Get features
    if args.features:
        features = load_features_from_json(Path(args.features))
    else:
        # small smoke-set if none provided
        features = [
            {
                "feature_id": "TEST-001",
                "feature_name": "Curfew login blocker with ASL and GH for Utah minors",
                "feature_description": (
                    "To comply with the Utah Social Media Regulation Act, we implement a curfew restriction "
                    "for users under 18. ASL detects minor accounts; GH applies logic only within Utah."
                ),
            },
            {
                "feature_id": "TEST-002",
                "feature_name": "Universal PF deactivation on guest mode",
                "feature_description": "By default, PF will be turned off for all users browsing in guest mode.",
            },
            {
                "feature_id": "TEST-003",
                "feature_name": "Child abuse content scanner using T5 and CDS triggers",
                "feature_description": (
                    "In line with US federal law to report child sexual abuse material to NCMEC, this scans uploads "
                    "and flags suspected materials tagged as T5."
                ),
            },
        ]

    # Build retrieval once (no external API)
    clf = SimpleComplianceClassifier()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for f in features:
        fid = f.get("feature_id", "NA")
        title = f.get("feature_name", "")
        desc = f.get("feature_description", "")
        # Use your own retrieval to keep payloads realistic
        text_for_retrieval = f"{title}\n\n{desc}".strip()
        passages = clf._retrieve_relevant_laws(text_for_retrieval, top_k=args.k)  # uses your Chroma index

        payload = build_payload(
            model=args.model,
            system_prompt=system_prompt,
            user_template=user_template,
            schema_obj=schema_obj,
            feature_id=fid,
            title=title,
            description=desc,
            passages=passages,
            k=args.k,
            disable_search=(not args.enable_search)
        )

        out_path = OUT_DIR / f"{fid}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ wrote {out_path}")

    print(f"\nAll payloads written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
