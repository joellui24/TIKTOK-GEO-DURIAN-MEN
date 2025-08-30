#!/usr/bin/env python3
"""
Generate JSON for Sonar compliance checking.
Takes feature title and description, retrieves top 5 relevant articles,
expands TikTok jargon, and formats them into the required JSON template.

Usage: python generate_sonar_compliance_json.py "Feature Title" "Feature Description"
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import chromadb
from fastembed import TextEmbedding

# Configuration
ROOT = Path(__file__).resolve().parents[0]
INDEX_DIR = ROOT / "data" / "index" / "chroma"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TIKTOK_JARGONS_FILE = ROOT / "ingest" / "tiktok_jargons.json"

def load_tiktok_jargons() -> Dict[str, str]:
    """Load TikTok jargons from tiktok_jargons.json file"""
    try:
        with open(TIKTOK_JARGONS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Warning: {TIKTOK_JARGONS_FILE} not found. No jargon expansion available.")
        return {}
    except json.JSONDecodeError as e:
        print(f"⚠️  Warning: Error parsing {TIKTOK_JARGONS_FILE}: {e}")
        return {}

def extract_tiktok_jargons(text: str) -> Dict[str, str]:
    """Extract TikTok jargons found in the text"""
    found_jargons = {}
    jargons = load_tiktok_jargons()
    
    if not jargons:
        return found_jargons
    
    # Create a case-insensitive search for jargons
    text_upper = text.upper()
    
    for jargon, meaning in jargons.items():
        # Look for exact word matches (not substrings)
        pattern = r'\b' + re.escape(jargon.upper()) + r'\b'
        if re.search(pattern, text_upper):
            found_jargons[jargon] = meaning
    
    return found_jargons

def search_compliance_articles(feature_title: str, feature_description: str, top_k: int = 5) -> List[Dict]:
    """Search for relevant compliance articles"""
    # Initialize embedding model
    embedding_model = TextEmbedding(model_name=EMBED_MODEL)
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        collection = client.get_collection("laws")
    except Exception as e:
        raise RuntimeError(f"Failed to load collection: {e}. Run 'python index/build.py' first.")
    
    # Generate query embedding
    query = f"{feature_title} {feature_description}"
    query_embedding = list(embedding_model.embed([query]))[0]
    query_embedding = np.asarray(query_embedding, dtype=np.float32).tolist()
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results with new structure
    articles = []
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        
        articles.append({
            "Law": f"{meta.get('jurisdiction', '')} {meta.get('law', '')}".strip() or "N/A",
            "Article": meta.get('article_title', 'N/A'),
            "Section": meta.get('article_number', 'N/A'), 
            "Content": doc,
            "distance": results["distances"][0][i],
            "metadata": meta
        })
    
    return articles

def generate_sonar_json(feature_title: str, feature_description: str) -> Tuple[Dict, List[Dict]]:
    """Generate the JSON structure for Sonar"""
    
    # Extract TikTok jargons from feature title and description
    full_text = f"{feature_title} {feature_description}"
    jargons_found = extract_tiktok_jargons(full_text)
    
    # Retrieve relevant articles
    articles = search_compliance_articles(feature_title, feature_description)
    
    # Format articles for the new JSON structure
    formatted_articles = []
    for article in articles:
        formatted_articles.append({
            "Law": article["Law"],
            "Article": article["Article"], 
            "Section": article["Section"],
            "Content": article["Content"]
        })
    
    # Create the JSON structure
    sonar_json = {
        "task": "Based solely on the provided articles, determine whether the described feature requires geo-specific compliance.",
        "feature_title": feature_title,
        "feature_description": feature_description,
        "Expanded_tiktok_jargons": jargons_found,
        "articles": formatted_articles,
        "output_format_json": {
            "verdict": "yes | no | uncertain",
            "reasoning": "Concise, article-based justification, maximum 500 characters",
            "references": ["List of relevant laws and sections "]
        },
        "constraints": [
            "Use only the provided articles as legal authority.",
            "Do not use any external knowledge, assumptions, or general legal principles.",
            "Keep the reasoning as short as possible.",
            "Cite specific article IDs for every factual claim."
        ]
    }
    
    return sonar_json, articles

def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_sonar_compliance_json.py \"Feature Title\" \"Feature Description\"")
        print("\nExample:")
        print('python generate_sonar_compliance_json.py "Curfew login blocker with ASL and GH" "System using ASL to detect minors and GH for Utah boundaries"')
        sys.exit(1)
    
    feature_title = sys.argv[1]
    feature_description = sys.argv[2]
    
    try:
        print(f"🔍 Searching for compliance articles related to: {feature_title}")
        print(f"📝 Feature description: {feature_description[:100]}...")
        print()
        
        # Generate the JSON
        sonar_json, articles = generate_sonar_json(feature_title, feature_description)
        
        # Show TikTok jargons found
        jargons = sonar_json.get("Expanded_tiktok_jargons", {})
        if jargons:
            print("🏷️  TikTok Jargons Found:")
            print("-" * 60)
            for jargon, meaning in jargons.items():
                print(f"   {jargon}: {meaning}")
            print()
        
        # Print article metadata for reference
        print("📚 Retrieved Articles:")
        print("-" * 60)
        for i, article in enumerate(articles):
            print(f"[{i+1}] {article['Law']}")
            print(f"    Article: {article['Article']}")
            print(f"    Section: {article['Section']}")
            print(f"    Distance: {article['distance']:.4f}")
            print(f"    Content Length: {len(article['Content'])} chars")
            print()
        
        print("=" * 80)
        print("🚀 SONAR COMPLIANCE JSON")
        print("=" * 80)
        
        # Output the formatted JSON
        print(json.dumps(sonar_json, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
