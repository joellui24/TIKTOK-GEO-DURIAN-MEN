#!/usr/bin/env python3
"""
Simple function to generate Sonar JSON for any feature
"""
import json
import numpy as np
import chromadb
from fastembed import TextEmbedding
from pathlib import Path

# Configuration
ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data" / "index" / "chroma"
TERMS_PATH = ROOT / "ingest" / "terms.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def load_terms():
    """Load TikTok jargon terms from terms.json"""
    if TERMS_PATH.exists():
        return json.loads(TERMS_PATH.read_text())
    return {}

def expand_terms(text: str, terms: dict) -> str:
    """Expand TikTok jargon using terms dictionary"""
    expanded_parts = []
    for term, expansions in terms.items():
        if term.lower() in text.lower():
            primary_expansion = expansions.split("|")[0]
            expanded_parts.append(f"{term}: {primary_expansion}")
    
    if expanded_parts:
        return text + "\n\nTerm expansions: " + " ; ".join(expanded_parts)
    return text

def extract_article_context(doc: str) -> str:
    """Extract specific article/section references and context when main extraction fails"""
    
    lines = doc.split('\n')
    article_info = []
    current_article = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for article/section headers
        if line.startswith("### ") and any(word in line.lower() for word in ["section", "article"]):
            current_article = line.replace("###", "").strip()
            continue
        elif line.startswith("## ") and any(word in line.lower() for word in ["article", "section"]):
            current_article = line.replace("##", "").strip()
            continue
            
        # Capture key requirements under articles
        if current_article and any(keyword in line.lower() for keyword in ["must", "shall", "required", "obligation", "compliance"]):
            article_ref = current_article.split("-")[0].strip()  # Get just "Article 24" or "Section 13-2c-301"
            short_req = line[:100] + ("..." if len(line) > 100 else "")
            article_info.append(f"{article_ref}: {short_req}")
            break  # Take first relevant requirement found
    
    if article_info:
        return article_info[0]
    
    # Last resort - find any article/section reference
    for line in lines:
        if "article" in line.lower() or "section" in line.lower():
            if len(line.strip()) < 150:  # Reasonable length
                return f"Context: {line.strip()}"
            else:
                return f"Context: {line.strip()[:100]}..."
    
    return "Legal provisions available - review full document for specific requirements"

def create_concise_summary(doc: str) -> str:
    """Create 1-2 line summary focused on geo-compliance decision factors"""
    
    # Extract key elements
    jurisdiction = ""
    law_name = ""
    key_requirements = []
    geographic_scope = ""
    
    lines = doc.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Extract jurisdiction and law
        if "**Jurisdiction:**" in line:
            jurisdiction = line.split("**Jurisdiction:**")[1].strip().split("**")[0].strip()
        elif "# " in line and any(word in line.lower() for word in ["act", "law", "regulation", "dsa"]):
            law_name = line.replace("#", "").strip().split("**")[0].strip()
        
        # Key requirements (most important)
        if "must restrict access" in line.lower():
            key_requirements.append("curfew restrictions")
        elif "must disable" in line.lower() and "feed" in line.lower():
            key_requirements.append("disable personalized feeds by default")
        elif "age verification" in line.lower() and ("required" in line.lower() or "must" in line.lower()):
            key_requirements.append("age verification required")
        elif "geo-location detection" in line.lower() and "required" in line.lower():
            key_requirements.append("geo-location detection required")
        elif "parental consent" in line.lower() and ("required" in line.lower() or "mandatory" in line.lower()):
            key_requirements.append("parental consent mandatory")
        elif "put in place mechanisms" in line.lower():
            key_requirements.append("illegal content notification mechanisms")
            
        # Geographic enforcement specifics
        if "utah residents only" in line.lower():
            geographic_scope = "Utah residents only"
        elif "california" in line.lower() and ("users" in line.lower() or "residents" in line.lower()):
            geographic_scope = "California users"
        elif "eu region" in line.lower() or ("european union" in line.lower() and "applies" in line.lower()):
            geographic_scope = "EU region"
        elif "geo-location detection" in line.lower() and any(place in line.lower() for place in ["utah", "california"]):
            if "utah" in line.lower():
                geographic_scope = "Utah geo-detection required"
            elif "california" in line.lower():
                geographic_scope = "California geo-detection required"
    
    # Build concise summary
    parts = []
    
    # Law identification
    if law_name and jurisdiction:
        law_short = law_name.split("(")[0].split("-")[0].strip()
        if "utah" in jurisdiction.lower():
            parts.append(f"{law_short} (Utah)")
        elif "california" in jurisdiction.lower():
            parts.append(f"{law_short} (California)")
        elif "eu" in jurisdiction.lower() or "european" in jurisdiction.lower():
            parts.append("EU Digital Services Act")
        else:
            parts.append(f"{law_short}")
    
    # Requirements and geographic scope in one line
    req_text = ""
    if key_requirements:
        main_req = key_requirements[0]  # Take most important
        if "curfew" in main_req:
            req_text = "REQUIRES: Curfew restrictions 10:30PM-6:00AM for under-18 users"
        elif "disable personalized feeds" in main_req:
            req_text = "REQUIRES: Disable personalized feeds by default for minors"
        elif "age verification" in main_req:
            req_text = "REQUIRES: Age verification for all users"
        elif "notification mechanisms" in main_req:
            req_text = "REQUIRES: Illegal content notification mechanisms"
        else:
            req_text = f"REQUIRES: {main_req.title()}"
    
    if geographic_scope:
        if req_text:
            req_text += f" - GEOGRAPHIC: {geographic_scope}"
        else:
            req_text = f"GEOGRAPHIC: {geographic_scope}"
    
    if req_text:
        parts.append(req_text)
    
    # Join into 1-2 lines max
    if parts:
        return " - ".join(parts)
    else:
        # Improved fallback with specific article references
        return extract_article_context(doc)

def query_chromadb_for_feature(title: str, description: str, top_k: int = 3):
    """Query ChromaDB and return results in the format needed for JSON template"""
    
    # Setup
    embedding_model = TextEmbedding(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = client.get_collection("laws")
    terms = load_terms()
    
    # Create query text with term expansion
    query_text = f"{title}\n\n{description}"
    expanded_query = expand_terms(query_text, terms)
    
    # Get embedding
    query_embedding = list(embedding_model.embed([expanded_query]))[0]
    query_embedding = np.asarray(query_embedding, dtype=np.float32).tolist()
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    context_blocks = []
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        
        # Extract relevant metadata - use existing doc_id from ChromaDB
        stored_doc_id = meta.get("doc_id", "")
        jurisdiction = meta.get("jurisdiction", "")
        law = meta.get("law", "")
        chunk_id = meta.get("chunk_id", f"chunk_{i}")
        
        # Use the stored doc_id if available, otherwise construct one
        if stored_doc_id and stored_doc_id.strip():
            doc_id = stored_doc_id.replace(" ", "_")
        elif law and law.strip() and jurisdiction and jurisdiction.strip():
            doc_id = f"{jurisdiction}_{law}".replace(" ", "_")
        elif law and law.strip():
            doc_id = law.replace(" ", "_")
        elif jurisdiction and jurisdiction.strip():
            doc_id = jurisdiction.replace(" ", "_")
        else:
            doc_id = f"document_{i}"
            
        section = meta.get("section", chunk_id)
        chunk_index = chunk_id
        
        # Create concise 1-2 line summary for Sonar
        snippet = create_concise_summary(doc)
        
        context_blocks.append({
            "doc_id": doc_id,
            "section": section,
            "chunk_index": chunk_index,
            "snippet": snippet
        })
    
    return {
        "top_k": top_k,
        "context_blocks": context_blocks
    }

def generate_sonar_json(feature_title: str, feature_description: str, top_k: int = 3) -> dict:
    """
    Generate Sonar JSON for any feature title and description
    
    Args:
        feature_title: The name/title of the feature
        feature_description: Detailed description of what the feature does
        top_k: Number of legal context chunks to retrieve (default: 3)
        
    Returns:
        dict: Complete JSON template ready for Sonar
    """
    
    # Load terms for jargon expansion
    terms = load_terms()
    
    # Get expanded jargon for this feature
    feature_text = f"{feature_title}\n\n{feature_description}"
    jargon_expansions = {}
    for term, expansions in terms.items():
        if term.lower() in feature_text.lower():
            jargon_expansions[term] = expansions.split("|")[0]  # Use primary expansion
    
    # Query ChromaDB
    retrieved_context = query_chromadb_for_feature(feature_title, feature_description, top_k)
    
    # Create enhanced JSON template with reasoning requirements
    template = {
        "task": "Classify whether the feature needs geo-specific compliance logic according to the provided legal context.",
        "schema": {
            "type": "object",
            "properties": {
                "needs_geo_compliance": {
                    "type": "string",
                    "enum": ["yes", "no", "uncertain"]
                },
                "reasoning": {
                    "type": "string",
                    "description": "Detailed explanation of your decision citing specific legal provisions"
                },
                "cited_articles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific articles/sections referenced in your analysis"
                }
            },
            "required": ["needs_geo_compliance", "reasoning", "cited_articles"]
        },
        "requirements": [
            "Return STRICT JSON only, matching the schema.",
            "MUST include detailed reasoning explaining your decision.",
            "MUST cite specific articles/sections from the legal context.",
            "If answering 'yes', identify which laws require geo-specific implementation.",
            "If answering 'uncertain', specify which articles need further review.",
            "If context mentions specific geographic requirements (Utah, California, EU), strongly consider 'yes'.",
            "Do not include extra text outside the JSON object."
        ],
        "input_data": {
            "feature": {
                "title": feature_title,
                "description": feature_description,
                "tiktok_jargon_expansions": jargon_expansions
            },
            "retrieved_legal_context": retrieved_context
        },
        "rubric": {
            "yes": "Explicit legal obligation requiring geographic enforcement (e.g., Utah curfew laws, California feed restrictions, EU content controls). Must cite specific article/section.",
            "no": "Business, UX, experiments, analytics, monetization with no clear legal trigger requiring geo-specific compliance.",
            "uncertain": "Geographic intent present but need to review specific articles for clarity. List articles requiring further investigation."
        },
        "guidance_for_uncertain": "If uncertain, suggest reviewing these resources: 1) Full text of cited articles, 2) Regulatory guidance documents, 3) Similar implementations by other platforms, 4) Legal counsel consultation for complex cross-jurisdictional issues."
    }
    
    return template

def main():
    """Main function to handle command line arguments"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python3 generate_sonar_json.py \"<feature_title>\" \"<feature_description>\"")
        print("\nExample:")
        print('python3 generate_sonar_json.py "Night mode blocker" "Disable notifications during sleep hours"')
        print("\nThis will:")
        print("1. Query ChromaDB for relevant legal articles")
        print("2. Expand TikTok jargon automatically") 
        print("3. Generate complete Sonar JSON with legal context")
        sys.exit(1)
    
    feature_title = sys.argv[1]
    feature_description = sys.argv[2]
    
    print(f"Generating Sonar JSON for: {feature_title}")
    print(f"Description: {feature_description}")
    print(f"Querying ChromaDB for relevant legal articles...\n")
    
    try:
        result = generate_sonar_json(feature_title, feature_description)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error generating JSON: {e}")
        print("Make sure ChromaDB index exists. Run: python3 index/build.py")
        sys.exit(1)

if __name__ == "__main__":
    main()