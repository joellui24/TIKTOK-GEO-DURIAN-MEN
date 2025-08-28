#!/usr/bin/env python3

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import chromadb
from fastembed import TextEmbedding

# Configuration
ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index" / "chroma"
TERMS_PATH = ROOT / "ingest" / "terms.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

@dataclass
class ComplianceResult:
    """Result of compliance analysis"""
    feature_id: str
    title: str
    needs_compliance: str  # "yes", "no", "uncertain"
    confidence_score: float  # 0.0 to 1.0
    reasoning: str
    applicable_regulations: List[str]
    risk_level: str  # "critical", "high", "medium", "low"
    retrieved_passages: List[Dict]

class SimpleComplianceClassifier:
    """Rule-based compliance classifier using patterns and retrieval"""
    
    def __init__(self):
        self.embedding_model = TextEmbedding(model_name=EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=str(INDEX_DIR))
        self.collection = self.client.get_collection("laws")
        self.terms = self._load_terms()
        
        # Compliance indicators
        self.compliance_patterns = {
            "yes": [
                r"comply\s+with|compliance\s+with|in\s+line\s+with",
                r"legal\s+requirement|regulatory\s+requirement|law\s+requiring",
                r"Utah\s+Social\s+Media|California.*SB976|Florida.*Online\s+Protections",
                r"Digital\s+Services\s+Act|DSA|GDPR|COPPA",
                r"NCMEC|child\s+abuse\s+reporting|18\s+USC\s+2258A",
                r"age\s+verification|parental\s+consent|minor.*restriction",
                r"geo.*enforce|region.*restrict|jurisdiction.*specific",
                r"ASL.*GH|CDS.*trigger|LCP.*parameter|Redline.*status"
            ],
            "business": [
                r"market\s+testing|A/B\s+test|experiment|rollout\s+strategy",
                r"user\s+experience|engagement|analytics|metrics",
                r"performance|optimization|efficiency", 
                r"leaderboard|creator\s+fund|monetization",
                r"UI\s+overhaul|interface|design\s+update"
            ],
            "uncertain": [
                r"compliance\s+concerns|for\s+compliance|compliance\s+context",
                r"privacy\s+context|safety.*implied|policy.*gate"
            ]
        }
        
        self.risk_patterns = {
            "critical": [r"CSAM|child\s+abuse|T5.*content|NCMEC", r"immediate.*legal|critical.*violation"],
            "high": [r"age.*verification|minor.*protection|parental.*consent", r"Utah|California|Florida.*law"],
            "medium": [r"DSA|transparency|notice.*action|content.*moderation"],
            "low": [r"data.*retention|logging|audit.*trail"]
        }
        
        self.regulation_patterns = {
            "Utah Social Media Regulation Act §13-2c-301": [r"Utah|curfew.*minor|ASL.*GH.*Utah|night.*hour.*restrict"],
            "California SB976 §22675": [r"California|SB976|PF.*teen|personalized.*feed.*teen|default.*disable"],
            "Florida HB3 §501.2044": [r"Florida|Jellybean|parental.*notification|parental.*control"],
            "EU DSA Article 24": [r"EU.*DSA|Digital.*Services.*Act|notice.*action|illegal.*content.*EU"],
            "EU DSA Article 28": [r"risk.*assessment|systemic.*risk|algorithmic.*amplif"],
            "18 USC §2258A": [r"NCMEC|child.*abuse|T5.*content|2258A|CSAM|child.*sexual.*abuse"]
        }
        
        # Detailed legal requirements for precise reasoning
        self.legal_requirements = {
            "Utah Social Media Regulation Act §13-2c-301": {
                "requirement": "Requires social media platforms to implement curfew restrictions for users under 18 between 10:30 PM and 6:00 AM",
                "triggers": ["curfew", "night hour", "time restriction", "minor access control", "ASL", "Utah"],
                "geo_specific": True,
                "risk": "high"
            },
            "California SB976 §22675": {
                "requirement": "Mandates personalized feeds be disabled by default for users under 18, requiring explicit parental opt-in",
                "triggers": ["PF", "personalized feed", "default disable", "teen", "California", "parental opt-in"],
                "geo_specific": True,
                "risk": "high"
            },
            "Florida HB3 §501.2044": {
                "requirement": "Requires robust parental controls and notifications for minors' social media activity",
                "triggers": ["Jellybean", "parental control", "parental notification", "Florida", "minor account"],
                "geo_specific": True,
                "risk": "medium"
            },
            "EU DSA Article 24": {
                "requirement": "Mandates notice-and-action mechanisms for illegal content reporting in EU",
                "triggers": ["notice action", "illegal content", "EU", "DSA", "transparency"],
                "geo_specific": True,
                "risk": "medium"
            },
            "EU DSA Article 28": {
                "requirement": "Requires risk assessment of algorithmic amplification systems for VLOPs",
                "triggers": ["risk assessment", "algorithmic", "amplification", "systemic risk"],
                "geo_specific": True,
                "risk": "high"
            },
            "18 USC §2258A": {
                "requirement": "Federal law requiring all providers to report known child sexual abuse material to NCMEC",
                "triggers": ["NCMEC", "child abuse", "T5", "CSAM", "child sexual abuse", "federal law"],
                "geo_specific": False,  # US federal law
                "risk": "critical"
            }
        }
    
    def _load_terms(self) -> Dict[str, str]:
        """Load terminology dictionary"""
        if TERMS_PATH.exists():
            return json.loads(TERMS_PATH.read_text())
        return {}
    
    def _expand_terms(self, text: str) -> str:
        """Expand acronyms and internal jargon"""
        expanded_text = text
        for term, expansions in self.terms.items():
            if term.lower() in text.lower():
                # Replace acronym with first expansion
                first_expansion = expansions.split("|")[0]
                expanded_text = re.sub(rf'\b{term}\b', f"{term} ({first_expansion})", expanded_text, flags=re.IGNORECASE)
        return expanded_text
    
    def _retrieve_relevant_laws(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant law passages"""
        query_embedding = list(self.embedding_model.embed([query]))[0]
        query_embedding = np.asarray(query_embedding, dtype=np.float32).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        passages = []
        for i in range(len(results["documents"][0])):
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            passages.append({
                "chunk_id": meta.get("chunk_id", f"chunk_{i}"),
                "jurisdiction": meta.get("jurisdiction", ""),
                "law": meta.get("law", ""),
                "source_url": meta.get("source_url", ""),
                "text": doc,
                "relevance_score": 1.0 - distance,
                "distance": distance
            })
        
        return passages
    
    def _classify_compliance_need(self, text: str, identified_regulations: List[str]) -> Tuple[str, float, str]:
        """Rule-based classification of compliance need with specific legal reasoning"""
        text_lower = text.lower()
        
        # Check for compliance indicators
        yes_score = 0
        compliance_triggers = []
        for pattern in self.compliance_patterns["yes"]:
            matches = re.findall(pattern, text_lower)
            if matches:
                yes_score += len(matches) * 2
                compliance_triggers.extend(matches)
        
        business_score = 0
        business_triggers = []
        for pattern in self.compliance_patterns["business"]:
            matches = re.findall(pattern, text_lower)
            if matches:
                business_score += len(matches)
                business_triggers.extend(matches)
        
        uncertain_score = 0
        uncertain_triggers = []
        for pattern in self.compliance_patterns["uncertain"]:
            matches = re.findall(pattern, text_lower)
            if matches:
                uncertain_score += len(matches) * 1.5
                uncertain_triggers.extend(matches)
        
        # Build specific reasoning based on identified regulations
        specific_reasoning = []
        if identified_regulations:
            for regulation in identified_regulations:
                if regulation in self.legal_requirements:
                    req_info = self.legal_requirements[regulation]
                    matched_triggers = []
                    for trigger in req_info["triggers"]:
                        if trigger.lower() in text_lower:
                            matched_triggers.append(trigger)
                    
                    if matched_triggers:
                        specific_reasoning.append(
                            f"{regulation}: {req_info['requirement']}. "
                            f"Triggered by: {', '.join(matched_triggers)}."
                        )
        
        # Determine classification
        total_signals = yes_score + business_score + uncertain_score
        
        if identified_regulations and yes_score > 0:
            # Strong compliance case with specific regulations
            confidence = min(0.95, 0.7 + (yes_score / 10))
            reasoning_parts = [
                f"COMPLIANCE REQUIRED - Feature explicitly references legal requirements.",
                f"Compliance indicators: {', '.join(set(compliance_triggers)) if compliance_triggers else 'regulatory language detected'}."
            ]
            reasoning_parts.extend(specific_reasoning)
            reasoning_parts.append(f"Geo-specific logic required for: {', '.join(identified_regulations)}.")
            return "yes", confidence, " ".join(reasoning_parts)
            
        elif business_score > 0 and yes_score == 0 and not identified_regulations:
            # Pure business feature
            confidence = min(0.9, 0.6 + (business_score / 10))
            reasoning = (
                f"NO COMPLIANCE REQUIRED - Pure business/product feature. "
                f"Business indicators: {', '.join(set(business_triggers))}. "
                f"No regulatory language or legal requirements detected."
            )
            return "no", confidence, reasoning
            
        elif uncertain_score > 0 or (identified_regulations and yes_score == 0):
            # Uncertain case
            confidence = 0.3 + min(0.4, uncertain_score / 10)
            reasoning_parts = [
                f"UNCERTAIN - Ambiguous compliance context detected."
            ]
            if uncertain_triggers:
                reasoning_parts.append(f"Uncertain indicators: {', '.join(set(uncertain_triggers))}.")
            if identified_regulations:
                reasoning_parts.append(f"Potential regulations: {', '.join(identified_regulations)}.")
            reasoning_parts.append("Human legal review recommended to clarify compliance requirements.")
            return "uncertain", confidence, " ".join(reasoning_parts)
            
        else:
            # Default uncertain case
            confidence = 0.2
            reasoning = (
                f"UNCERTAIN - Insufficient context to determine compliance requirements. "
                f"Feature description lacks clear legal or business indicators. "
                f"Human review recommended."
            )
            return "uncertain", confidence, reasoning
    
    def _determine_risk_level(self, text: str, classification: str) -> str:
        """Determine risk level based on content"""
        if classification == "no":
            return "low"
        
        text_lower = text.lower()
        
        for risk, patterns in self.risk_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return risk
        
        return "medium"  # Default for compliance features
    
    def _identify_regulations(self, text: str) -> List[str]:
        """Identify applicable regulations"""
        regulations = []
        text_lower = text.lower()
        
        for regulation, patterns in self.regulation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    regulations.append(regulation)
                    break  # Avoid duplicates
        
        return regulations
    
    def analyze_feature(self, feature_id: str, title: str, description: str) -> ComplianceResult:
        """Analyze a single feature for compliance needs"""
        
        # Step 1: Expand terminology
        full_text = f"{title}\n\n{description}"
        expanded_description = self._expand_terms(full_text)
        
        # Step 2: Identify applicable regulations first
        regulations = self._identify_regulations(expanded_description)
        
        # Step 3: Retrieve relevant legal passages
        passages = self._retrieve_relevant_laws(expanded_description, top_k=5)
        
        # Step 4: Rule-based classification with specific legal reasoning
        classification, confidence, reasoning = self._classify_compliance_need(expanded_description, regulations)
        
        # Step 5: Determine risk level (override with regulation-specific risk if available)
        risk_level = self._determine_risk_level(expanded_description, classification)
        if regulations:
            # Use the highest risk from identified regulations
            regulation_risks = []
            for reg in regulations:
                if reg in self.legal_requirements:
                    regulation_risks.append(self.legal_requirements[reg]["risk"])
            if regulation_risks:
                risk_priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}
                highest_risk = max(regulation_risks, key=lambda x: risk_priority.get(x, 0))
                risk_level = highest_risk
        
        # Step 6: Enhance reasoning with legal document context if relevant
        if passages and classification in ["yes", "uncertain"]:
            top_law = passages[0]
            if top_law['relevance_score'] > 0.4:  # Only add if highly relevant
                reasoning += f" Legal database contains relevant provisions in {top_law['jurisdiction']} {top_law['law']} (relevance: {top_law['relevance_score']:.3f})."
        
        return ComplianceResult(
            feature_id=feature_id,
            title=title,
            needs_compliance=classification,
            confidence_score=confidence,
            reasoning=reasoning,
            applicable_regulations=regulations,
            risk_level=risk_level,
            retrieved_passages=passages
        )

    def analyze_batch(self, features: List[Dict]) -> List[ComplianceResult]:
        """Analyze multiple features"""
        results = []
        for feature in features:
            result = self.analyze_feature(
                feature_id=feature.get("feature_id", ""),
                title=feature.get("feature_name", ""),
                description=feature.get("feature_description", "")
            )
            results.append(result)
        return results

def main():
    """Test the simple compliance classifier"""
    
    test_features = [
        {
            "feature_id": "TEST-001",
            "feature_name": "Curfew login blocker with ASL and GH for Utah minors",
            "feature_description": "To comply with the Utah Social Media Regulation Act, we are implementing a curfew-based login restriction for users under 18. The system uses ASL to detect minor accounts and routes enforcement through GH to apply only within Utah boundaries."
        },
        {
            "feature_id": "TEST-002", 
            "feature_name": "Universal PF deactivation on guest mode",
            "feature_description": "By default, PF will be turned off for all users browsing in guest mode."
        },
        {
            "feature_id": "TEST-003",
            "feature_name": "Child abuse content scanner using T5 and CDS triggers", 
            "feature_description": "In line with the US federal law requiring providers to report child sexual abuse content to NCMEC, this feature scans uploads and flags suspected materials tagged as T5."
        }
    ]
    
    classifier = SimpleComplianceClassifier()
    results = classifier.analyze_batch(test_features)
    
    for result in results:
        print(f"\n{'='*60}")
        print(f"Feature: {result.title}")
        print(f"Classification: {result.needs_compliance}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Risk Level: {result.risk_level}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Regulations: {', '.join(result.applicable_regulations) if result.applicable_regulations else 'None'}")

if __name__ == "__main__":
    main()