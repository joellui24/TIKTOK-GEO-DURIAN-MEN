#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
import numpy as np
from fastembed import TextEmbedding


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index" / "chroma"
TERMS_PATH = ROOT / "ingest" / "terms.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ComplianceResult:
    """Result of compliance analysis."""
    feature_id: str
    title: str
    needs_compliance: str              # "yes", "no", "uncertain"
    confidence_score: float            # 0.0 to 1.0
    reasoning: str
    applicable_regulations: List[str]
    risk_level: str                    # "critical", "high", "medium", "low"
    retrieved_passages: List[Dict]

    # Human-review fields
    needs_review: bool = False
    review_reason: str = ""
    suggested_questions: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────────────────────────────────────
class SimpleComplianceClassifier:
    """Rule-based compliance classifier using semantic retrieval."""

    def __init__(self) -> None:
        self.embedding_model = TextEmbedding(model_name=EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=str(INDEX_DIR))
        self.collection = self._ensure_index_exists()
        self.terms = self._load_terms()

        # ── Compliance indicators (regex, case-insensitive) ──
        self.compliance_patterns: Dict[str, List[str]] = {
            "yes": [
                r"comply\s+with|compliance\s+with|in\s+line\s+with",
                r"legal\s+requirement|regulatory\s+requirement|law\s+requiring",
                r"utah\s+social\s+media|california.*sb976|florida.*online\s+protections",
                r"digital\s+services\s+act|dsa|gdpr|coppa",
                r"ncmec|child\s+abuse\s+reporting|18\s+usc\s+2258a",
                r"age\s+verification|parental\s+consent|minor.*restriction",
                r"geo.*enforce|region.*restrict|jurisdiction.*specific",
                r"asl.*gh|cds.*trigger|lcp.*parameter|redline.*status",
            ],
            "business": [
                r"market\s+testing|a/b\s+test|experiment|rollout\s+strategy",
                r"user\s+experience|engagement|analytics|metrics",
                r"performance|optimization|efficiency",
                r"leaderboard|creator\s+fund|monetization",
                r"ui\s+overhaul|interface|design\s+update",
            ],
            "uncertain": [
                r"compliance\s+concerns|for\s+compliance|compliance\s+context",
                r"privacy\s+context|safety.*implied|policy.*gate",
            ],
        }

        self.risk_patterns: Dict[str, List[str]] = {
            "critical": [r"csam|child\s+abuse|t5.*content|ncmec", r"immediate.*legal|critical.*violation"],
            "high": [r"age.*verification|minor.*protection|parental.*consent", r"utah|california|florida.*law"],
            "medium": [r"dsa|transparency|notice.*action|content.*moderation"],
            "low": [r"data.*retention|logging|audit.*trail"],
        }

        self.regulation_patterns: Dict[str, List[str]] = {
            "Utah Social Media Regulation Act §13-2c-301": [r"utah|curfew.*minor|asl.*gh.*utah|night.*hour.*restrict"],
            "California SB976 §22675": [r"california|sb976|pf.*teen|personalized.*feed.*teen|default.*disable"],
            "Florida HB3 §501.2044": [r"florida|jellybean|parental.*notification|parental.*control"],
            "EU DSA Article 24": [r"eu.*dsa|digital.*services.*act|notice.*action|illegal.*content.*eu"],
            "EU DSA Article 28": [r"risk.*assessment|systemic.*risk|algorithmic.*amplif"],
            "18 USC §2258A": [r"ncmec|child.*abuse|t5.*content|2258a|csam|child.*sexual.*abuse"],
        }

        # Detailed requirements to enrich reasoning / risk overrides
        self.legal_requirements: Dict[str, Dict] = {
            "Utah Social Media Regulation Act §13-2c-301": {
                "requirement": "Curfew restrictions for users under 18 between 10:30 PM and 6:00 AM in Utah.",
                "triggers": ["curfew", "night hour", "time restriction", "minor access control", "ASL", "Utah"],
                "geo_specific": True,
                "risk": "high",
            },
            "California SB976 §22675": {
                "requirement": "Personalized feeds disabled by default for users under 18; parental opt-in required.",
                "triggers": ["PF", "personalized feed", "default disable", "teen", "California", "parental opt-in"],
                "geo_specific": True,
                "risk": "high",
            },
            "Florida HB3 §501.2044": {
                "requirement": "Parental controls and notifications for minors' social media activity.",
                "triggers": ["Jellybean", "parental control", "parental notification", "Florida", "minor account"],
                "geo_specific": True,
                "risk": "medium",
            },
            "EU DSA Article 24": {
                "requirement": "Notice-and-action mechanisms for illegal content reporting in the EU.",
                "triggers": ["notice action", "illegal content", "EU", "DSA", "transparency"],
                "geo_specific": True,
                "risk": "medium",
            },
            "EU DSA Article 28": {
                "requirement": "Risk assessment for algorithmic amplification systems (VLOPs).",
                "triggers": ["risk assessment", "algorithmic", "amplification", "systemic risk"],
                "geo_specific": True,
                "risk": "high",
            },
            "18 USC §2258A": {
                "requirement": "Providers must report known child sexual abuse material to NCMEC (US federal).",
                "triggers": ["NCMEC", "child abuse", "T5", "CSAM", "child sexual abuse", "federal law"],
                "geo_specific": False,
                "risk": "critical",
            },
        }

    # ── Setup helpers ──────────────────────────────────────────────────────────
    def _ensure_index_exists(self):
        """Ensure Chroma 'laws' collection exists; build it if missing."""
        try:
            return self.client.get_collection("laws")
        except Exception as e:
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                print("🔧 Vector index not found. Building automatically…")
                try:
                    import sys
                    sys.path.append(str(ROOT / "index"))
                    from build import build_index  # type: ignore
                    build_index()
                    print("✅ Vector index built.")
                    return self.client.get_collection("laws")
                except Exception as build_error:
                    print(f"❌ Failed to build vector index: {build_error}")
                    print("Fallback:")
                    print("  1) Run: python3 index/build.py")
                    print("  2) Ensure laws exist under data/laws/*.md")
                    print("  3) Check requirements are installed")
                    raise
            raise

    def _load_terms(self) -> Dict[str, str]:
        """Load acronym/jargon expansions."""
        try:
            if TERMS_PATH.exists():
                return json.loads(TERMS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    # ── Text utilities ────────────────────────────────────────────────────────
    def _expand_terms(self, text: str) -> str:
        """Expand acronyms and internal jargon based on terms.json."""
        expanded_text = text
        for term, expansions in self.terms.items():
            if term.lower() in text.lower():
                first_expansion = expansions.split("|")[0]
                expanded_text = re.sub(
                    rf"\b{re.escape(term)}\b",
                    f"{term} ({first_expansion})",
                    expanded_text,
                    flags=re.IGNORECASE,
                )
        return expanded_text

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def _retrieve_relevant_laws(self, query: str, top_k: int = 5) -> List[Dict]:
        """Embed query and return top-k law passages from Chroma."""
        query_embedding = list(self.embedding_model.embed([query]))[0]
        query_embedding = np.asarray(query_embedding, dtype=np.float32).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        passages: List[Dict] = []
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        dists = results.get("distances") or [[]]

        if not docs or not docs[0]:
            return passages

        for i in range(len(docs[0])):
            doc = docs[0][i]
            meta = metas[0][i]
            distance = float(dists[0][i])

            passages.append({
                "doc_id": meta.get("doc_id", ""),
                "chunk_id": meta.get("chunk_id", f"chunk_{i}"),
                "chunk_index": meta.get("chunk_index", i),
                "jurisdiction": meta.get("jurisdiction", ""),
                "law": meta.get("law", ""),
                "section_title": meta.get("section_title", ""),
                "source_url": meta.get("source_url", ""),
                "text": doc,
                "distance": distance,
                "relevance_score": 1.0 - distance,
            })
        return passages

    # ── Heuristics ────────────────────────────────────────────────────────────
    def _identify_regulations(self, text: str) -> List[str]:
        """Identify likely applicable regulations by text patterns."""
        regs: List[str] = []
        for regulation, patterns in self.regulation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    regs.append(regulation)
                    break
        return regs

    def _classify_compliance_need(self, text: str, identified_regulations: List[str]) -> Tuple[str, float, str]:
        """Rule-based classification of compliance need with specific legal reasoning."""
        yes_score = 0
        business_score = 0
        uncertain_score = 0
        compliance_triggers: List[str] = []
        business_triggers: List[str] = []
        uncertain_triggers: List[str] = []

        for pattern in self.compliance_patterns["yes"]:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                yes_score += len(matches) * 2
                compliance_triggers.extend(matches)

        for pattern in self.compliance_patterns["business"]:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                business_score += len(matches)
                business_triggers.extend(matches)

        for pattern in self.compliance_patterns["uncertain"]:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                uncertain_score += len(matches) * 1.5
                uncertain_triggers.extend(matches)

        # Build specific reasoning based on identified regulations
        specific_reasoning: List[str] = []
        if identified_regulations:
            lower = text.lower()
            for regulation in identified_regulations:
                if regulation in self.legal_requirements:
                    req_info = self.legal_requirements[regulation]
                    matched_triggers = [t for t in req_info["triggers"] if t.lower() in lower]
                    if matched_triggers:
                        specific_reasoning.append(
                            f"{regulation}: {req_info['requirement']} (Triggers: {', '.join(matched_triggers)})."
                        )

        # Decision logic
        if identified_regulations and yes_score > 0:
            confidence = min(0.95, 0.70 + (yes_score / 10.0))
            reasoning_parts = [
                "COMPLIANCE REQUIRED – explicit legal requirements referenced.",
                f"Indicators: {', '.join(sorted(set(compliance_triggers))) if compliance_triggers else 'regulatory language detected'}."
            ]
            reasoning_parts.extend(specific_reasoning)
            reasoning_parts.append(f"Geo-specific logic for: {', '.join(identified_regulations)}.")
            return "yes", confidence, " ".join(reasoning_parts)

        if business_score > 0 and yes_score == 0 and not identified_regulations:
            confidence = min(0.90, 0.60 + (business_score / 10.0))
            reasoning = (
                "NO COMPLIANCE REQUIRED – product/business change. "
                f"Indicators: {', '.join(sorted(set(business_triggers)))}. "
                "No legal requirements detected."
            )
            return "no", confidence, reasoning

        if uncertain_score > 0 or (identified_regulations and yes_score == 0):
            confidence = 0.30 + min(0.40, uncertain_score / 10.0)
            parts = ["UNCERTAIN – ambiguous compliance context."]
            if uncertain_triggers:
                parts.append(f"Uncertain indicators: {', '.join(sorted(set(uncertain_triggers)))}.")
            if identified_regulations:
                parts.append(f"Potential regulations: {', '.join(identified_regulations)}.")
            parts.append("Human legal review recommended.")
            return "uncertain", confidence, " ".join(parts)

        # Default uncertain
        return "uncertain", 0.20, (
            "UNCERTAIN – insufficient context to determine compliance requirements. "
            "Feature lacks clear legal or business indicators. Human review recommended."
        )

    def _determine_risk_level(self, text: str, classification: str) -> str:
        """Determine risk level based on content and matched patterns."""
        if classification == "no":
            return "low"
        for risk, patterns in self.risk_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    return risk
        return "medium"  # default for compliance features

    # ── Review helpers ────────────────────────────────────────────────────────
    def _suggest_questions(
        self, text: str, classification: str, regulations: List[str], passages: List[Dict]
    ) -> List[str]:
        """Suggest clarifying questions to speed up human review."""
        qs: List[str] = []
        lower = text.lower()
        top = passages[0] if passages else {}
        law_lower = f"{top.get('jurisdiction','')} {top.get('law','')}".lower().strip()

        if classification == "uncertain":
            qs += [
                "Which jurisdictions will this feature launch in first?",
                "Does this involve users under 13 or under 18?",
                "Is age verification or parental consent required in any target region?",
                "Is the feature available in the EU (DSA) or Utah/California/Florida (minor protections)?",
            ]
        if any(k in lower for k in ["minor", "under 13", "under 18", "teen"]):
            qs.append("Please confirm exact age thresholds and regions (e.g., under 13 vs under 18).")
        if any(k in lower for k in ["csam", "child abuse", "ncmec", "t5"]):
            qs.append("Confirm reporting workflow to NCMEC and data retention details.")
        if "digital services act" in lower or "dsa" in lower or "digital services act" in law_lower or "dsa" in law_lower:
            qs.append("Will you provide notice-and-action and transparency reporting for the EU DSA?")
        if any(k in law_lower for k in ["utah", "california", "florida"]):
            qs.append("Do you need geo-restricted logic (e.g., curfews/parental defaults) for those states?")

        out: List[str] = []
        for q in qs:
            if q not in out:
                out.append(q)
        return out[:5]

    # ── Public API ────────────────────────────────────────────────────────────
    def analyze_feature(self, feature_id: str, title: str, description: str) -> ComplianceResult:
        """Analyze a single feature for compliance needs and decide if human review is required."""
        # Step 1: Expand terminology
        full_text = f"{title}\n\n{description}".strip()
        expanded_description = self._expand_terms(full_text)

        # Step 2: Identify likely regulations from text
        regulations = self._identify_regulations(expanded_description)

        # Step 3: Retrieve relevant legal passages (semantic search)
        passages = self._retrieve_relevant_laws(expanded_description, top_k=5)

        # Step 4: Rule-based classification with specific legal reasoning
        classification, confidence, reasoning = self._classify_compliance_need(
            expanded_description, regulations
        )

        # Step 5: Determine risk level (override with highest regulation-specific risk if present)
        risk_level = self._determine_risk_level(expanded_description, classification)
        if regulations:
            risk_priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            reg_risks = [
                self.legal_requirements[reg]["risk"]
                for reg in regulations
                if reg in self.legal_requirements
            ]
            if reg_risks:
                risk_level = max(reg_risks, key=lambda r: risk_priority.get(r, 0))

        # Step 6: Enrich reasoning with top law context if relevant
        if passages and classification in ("yes", "uncertain"):
            top = max(passages, key=lambda p: p.get("relevance_score", 0.0))
            if top.get("relevance_score", 0.0) >= 0.40:
                reasoning += (
                    f" Evidence: {top.get('jurisdiction','').strip()} "
                    f"{top.get('law','').strip()} (ctx score {top.get('relevance_score',0.0):.3f})."
                )

        # Step 7: Human-review policy (flag items needing escalation)
        LOW_CONF_YES_THRESHOLD = 0.75
        STRONG_CONTEXT_FOR_NO = 0.65
        best_rel = max((p.get("relevance_score", 0.0) for p in passages), default=0.0)

        needs_review = False
        review_reason = ""
        if classification == "uncertain":
            needs_review = True
            review_reason = "Model uncertain: ambiguous or insufficient legal signals."
        elif classification == "yes" and confidence < LOW_CONF_YES_THRESHOLD:
            needs_review = True
            review_reason = f"Low confidence on 'yes' ({confidence:.2f} < {LOW_CONF_YES_THRESHOLD})."
        elif classification == "no" and best_rel > STRONG_CONTEXT_FOR_NO:
            needs_review = True
            review_reason = f"Strong legal context retrieved (score {best_rel:.2f}) despite 'no'."

        suggested_qs = self._suggest_questions(expanded_description, classification, regulations, passages)

        return ComplianceResult(
            feature_id=feature_id,
            title=title,
            needs_compliance=classification,
            confidence_score=confidence,
            reasoning=reasoning,
            applicable_regulations=regulations,
            risk_level=risk_level,
            retrieved_passages=passages,
            needs_review=needs_review,
            review_reason=review_reason,
            suggested_questions=suggested_qs,
        )

    def analyze_batch(self, features: List[Dict]) -> List[ComplianceResult]:
        """Analyze multiple features."""
        results: List[ComplianceResult] = []
        for feature in features:
            result = self.analyze_feature(
                feature_id=feature.get("feature_id", ""),
                title=feature.get("feature_name", ""),
                description=feature.get("feature_description", ""),
            )
            results.append(result)
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def write_review_queue(results: List[ComplianceResult], path: Path) -> None:
    """Append flagged items to a review queue CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "feature_id", "title", "needs_compliance", "confidence", "risk_level",
                "review_reason", "top_jurisdiction", "top_law", "top_relevance", "suggested_questions"
            ])
        for r in results:
            if r.needs_review:
                top = r.retrieved_passages[0] if r.retrieved_passages else {}
                w.writerow([
                    r.feature_id,
                    r.title,
                    r.needs_compliance,
                    f"{r.confidence_score:.2f}",
                    r.risk_level,
                    r.review_reason,
                    top.get("jurisdiction", ""),
                    top.get("law", ""),
                    f"{top.get('relevance_score', 0.0):.3f}",
                    " | ".join(r.suggested_questions),
                ])


# ──────────────────────────────────────────────────────────────────────────────
# Demo / Smoke test
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """Test the simple compliance classifier with a few examples."""
    test_features = [
        {
            "feature_id": "TEST-001",
            "feature_name": "Curfew login blocker with ASL and GH for Utah minors",
            "feature_description": (
                "To comply with the Utah Social Media Regulation Act, we are implementing a "
                "curfew-based login restriction for users under 18. The system uses ASL to detect "
                "minor accounts and routes enforcement through GH to apply only within Utah boundaries."
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
                "In line with the US federal law requiring providers to report child sexual abuse material "
                "to NCMEC, this feature scans uploads and flags suspected materials tagged as T5."
            ),
        },
    ]

    classifier = SimpleComplianceClassifier()
    results = classifier.analyze_batch(test_features)

    for result in results:
        print(f"\n{'=' * 60}")
        print(f"Feature: {result.title}")
        print(f"Classification: {result.needs_compliance}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Risk Level: {result.risk_level}")
        print(f"Reasoning: {result.reasoning}")
        regs = ", ".join(result.applicable_regulations) if result.applicable_regulations else "None"
        print(f"Regulations: {regs}")
        if result.needs_review:
            print(f"⚠️  Needs review: {result.review_reason}")
            if result.suggested_questions:
                print("Suggested questions:")
                for q in result.suggested_questions:
                    print(f" - {q}")

    out = ROOT / "outputs" / "review_queue.csv"
    write_review_queue(results, out)
    print(f"\nReview queue written to: {out}")


if __name__ == "__main__":
    main()
