#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import chromadb
from fastembed import TextEmbedding
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fix tokenizers warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

class LLMComplianceClassifier:
    """LLM-enhanced compliance classifier using semantic retrieval"""
    
    def __init__(self):
        self.embedding_model = TextEmbedding(model_name=EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=str(INDEX_DIR))
        self.collection = self._ensure_index_exists()
        self.terms = self._load_terms()
        self.llm_client = self._setup_llm()
    
    def _ensure_index_exists(self):
        """Ensure vector index exists, build if necessary"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection("laws")
            return collection
        except Exception as e:
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                print("🔧 Vector index not found. Building automatically...")
                print("⏱️  This is a one-time setup that may take 1-2 minutes...")
                try:
                    # Import and run the build function
                    import sys
                    sys.path.append(str(ROOT / "index"))
                    from build import build_index
                    build_index()
                    print("✅ Vector index built successfully!")
                    # Now get the collection
                    collection = self.client.get_collection("laws")
                    return collection
                except Exception as build_error:
                    print(f"❌ Failed to build vector index: {build_error}")
                    print("\nFallback options:")
                    print("1. Run manually: python3 index/build.py")
                    print("2. Check that data/laws/*.md files exist")
                    print("3. Ensure all dependencies are installed")
                    raise Exception(f"Vector index build failed: {build_error}")
            else:
                # Some other ChromaDB error
                raise e
        
    def _load_terms(self) -> Dict[str, str]:
        """Load terminology dictionary"""
        if TERMS_PATH.exists():
            return json.loads(TERMS_PATH.read_text())
        return {}
    
    def _setup_llm(self):
        """Setup LLM client based on configuration"""
        provider = os.getenv("LLM_PROVIDER", "perplexity").lower()
        
        if provider == "perplexity":
            import openai  # Perplexity uses OpenAI-compatible API
            api_key = os.getenv("PERPLEXITY_API_KEY")
            if not api_key:
                raise ValueError("PERPLEXITY_API_KEY not found in environment variables")
            return openai.OpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai"
            )
        
        elif provider == "anthropic":
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            return anthropic.Anthropic(api_key=api_key)
        
        elif provider == "openai":
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            return openai.OpenAI(api_key=api_key)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _expand_terms(self, text: str) -> str:
        """Expand acronyms and internal jargon"""
        expanded_text = text
        for term, expansions in self.terms.items():
            if term.lower() in text.lower():
                # Replace acronym with first expansion
                first_expansion = expansions.split("|")[0]
                expanded_text = expanded_text.replace(term, f"{term} ({first_expansion})")
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
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with the given prompt"""
        provider = os.getenv("LLM_PROVIDER", "perplexity").lower()
        
        try:
            if provider == "perplexity":
                model = os.getenv("PERPLEXITY_MODEL", "sonar")
                response = self.llm_client.chat.completions.create(
                    model=model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            
            elif provider == "anthropic":
                model = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
                response = self.llm_client.messages.create(
                    model=model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif provider == "openai":
                model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
                response = self.llm_client.chat.completions.create(
                    model=model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")
    
    def _analyze_with_llm(self, feature_text: str, legal_passages: List[Dict]) -> Tuple[str, float, str, List[str], str]:
        """Use LLM to analyze compliance requirements"""
        
        # Format legal context
        legal_context = ""
        for i, passage in enumerate(legal_passages[:3], 1):
            legal_context += f"[Legal Source {i}] {passage['jurisdiction']} {passage['law']} (relevance: {passage['relevance_score']:.3f})\n"
            legal_context += f"{passage['text'][:300]}...\n\n"
        
        prompt = f"""You are a legal compliance expert analyzing whether a social media feature requires geo-specific compliance logic.

FEATURE TO ANALYZE:
{feature_text}

RELEVANT LEGAL CONTEXT:
{legal_context}

TASK: Analyze if this feature needs geo-specific compliance logic based on legal requirements.

Consider:
1. Does the feature interact with regulations that vary by geographic jurisdiction?
2. Are there specific legal obligations that require different implementation by location?
3. Do laws require geo-targeted enforcement, age verification, or content controls?

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "needs_compliance": "yes|no|uncertain",
    "confidence_score": 0.0-1.0,
    "reasoning": "Detailed explanation of your analysis",
    "applicable_regulations": ["regulation1", "regulation2"],
    "risk_level": "critical|high|medium|low"
}}

GUIDELINES:
- "yes": Clear legal requirements that vary by jurisdiction
- "no": Pure business/product feature with no geo-specific legal obligations  
- "uncertain": Ambiguous context requiring human legal review
- Risk levels: critical (immediate legal violation risk), high (significant compliance impact), medium (moderate legal considerations), low (minimal compliance concerns)
- Be specific about which regulations apply and why
- Consider TikTok's global operations and regulatory complexity"""

        try:
            llm_response = self._call_llm(prompt)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return (
                    result.get("needs_compliance", "uncertain"),
                    float(result.get("confidence_score", 0.5)),
                    result.get("reasoning", "LLM analysis failed to provide reasoning"),
                    result.get("applicable_regulations", []),
                    result.get("risk_level", "medium")
                )
            else:
                # Fallback parsing
                return self._fallback_parse(llm_response)
                
        except Exception as e:
            return "uncertain", 0.3, f"LLM analysis failed: {str(e)}", [], "medium"
    
    def _fallback_parse(self, response: str) -> Tuple[str, float, str, List[str], str]:
        """Fallback parsing if JSON parsing fails"""
        response_lower = response.lower()
        
        if "needs_compliance" in response_lower and ("yes" in response_lower or "compliance required" in response_lower):
            needs = "yes"
            confidence = 0.7
        elif "no compliance" in response_lower or "business feature" in response_lower:
            needs = "no" 
            confidence = 0.7
        else:
            needs = "uncertain"
            confidence = 0.4
            
        # Extract risk level
        if "critical" in response_lower:
            risk = "critical"
        elif "high" in response_lower:
            risk = "high"
        elif "low" in response_lower:
            risk = "low"
        else:
            risk = "medium"
            
        return needs, confidence, response[:500], [], risk
    
    def analyze_feature(self, feature_id: str, title: str, description: str) -> ComplianceResult:
        """Analyze a single feature for compliance needs using LLM"""
        
        # Step 1: Expand terminology
        full_text = f"{title}\n\n{description}"
        expanded_description = self._expand_terms(full_text)
        
        # Step 2: Retrieve relevant legal passages
        passages = self._retrieve_relevant_laws(expanded_description, top_k=5)
        
        # Step 3: Use LLM for analysis
        classification, confidence, reasoning, regulations, risk_level = self._analyze_with_llm(
            expanded_description, passages
        )
        
        # Step 4: Enhance reasoning with retrieval context
        if passages and classification in ["yes", "uncertain"]:
            top_law = passages[0]
            if top_law['relevance_score'] > 0.3:
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
    """Test the LLM compliance classifier"""
    
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
    
    try:
        classifier = LLMComplianceClassifier()
        results = classifier.analyze_batch(test_features)
        
        for result in results:
            print(f"\n{'='*60}")
            print(f"Feature: {result.title}")
            print(f"Classification: {result.needs_compliance}")
            print(f"Confidence: {result.confidence_score:.2f}")
            print(f"Risk Level: {result.risk_level}")
            print(f"Reasoning: {result.reasoning}")
            print(f"Regulations: {', '.join(result.applicable_regulations) if result.applicable_regulations else 'None'}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Copy .env.example to .env and add your API keys")
        print("2. Set LLM_PROVIDER to 'perplexity', 'anthropic', or 'openai'")

if __name__ == "__main__":
    main()