#!/usr/bin/env python3

import sys
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import chromadb
from fastembed import TextEmbedding

# Configuration
ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index" / "chroma"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def load_index_metadata():
    """Load index metadata"""
    metadata_path = INDEX_DIR / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    return None

def search_index(query: str, top_k: int = 5) -> List[Dict]:
    """Search the vector index and return results"""
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"Index not found at {INDEX_DIR}. Run 'python index/build.py' first.")
    
    # Initialize embedding model
    embedding_model = TextEmbedding(model_name=EMBED_MODEL)
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        collection = client.get_collection("laws")
    except Exception as e:
        raise RuntimeError(f"Failed to load collection: {e}. Run 'python index/build.py' first.")
    
    # Generate query embedding
    query_embedding = list(embedding_model.embed([query]))[0]
    query_embedding = np.asarray(query_embedding, dtype=np.float32).tolist()
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    formatted_results = []
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        chunk_id = meta.get("chunk_id", f"chunk_{i}")
        distance = results["distances"][0][i]
        
        formatted_results.append({
            "rank": i + 1,
            "chunk_id": chunk_id,
            "distance": distance,
            "jurisdiction": meta.get("jurisdiction", ""),
            "law": meta.get("law", ""),
            "source_url": meta.get("source_url", ""),
            "snippet": doc[:200] + ("..." if len(doc) > 200 else ""),
            "full_text": doc
        })
    
    return formatted_results

def interactive_mode():
    """Interactive search mode"""
    print("=== Legal Compliance Index - Interactive Search ===")
    
    # Check index status
    metadata = load_index_metadata()
    if metadata:
        print(f"Index version: {metadata.get('version', 'unknown')}")
        print(f"Chunks indexed: {metadata.get('chunk_count', 'unknown')}")
        print(f"Documents: {', '.join(metadata.get('documents', []))}")
        print()
    else:
        print("Warning: Index metadata not found")
    
    print("Enter search queries (type 'quit' to exit):")
    print()
    
    while True:
        try:
            query = input("> ").strip()
            
            if query.lower() in ('quit', 'exit', 'q'):
                break
                
            if not query:
                continue
                
            print(f"\nSearching for: '{query}'\n")
            
            results = search_index(query)
            
            if not results:
                print("No results found.")
                continue
            
            for result in results:
                print(f"[{result['rank']}] {result['chunk_id']} (distance: {result['distance']:.4f})")
                print(f"    Law: {result['jurisdiction']} {result['law']}")
                print(f"    Snippet: {result['snippet']}")
                if result['source_url']:
                    print(f"    Source: {result['source_url']}")
                print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()

def single_query_mode(query: str, top_k: int = 5):
    """Single query mode for command line usage"""
    try:
        results = search_index(query, top_k)
        
        print(f"Search results for: '{query}'")
        print("=" * 50)
        
        if not results:
            print("No results found.")
            return
        
        for result in results:
            print(f"\n[{result['rank']}] {result['chunk_id']} (distance: {result['distance']:.4f})")
            print(f"Law: {result['jurisdiction']} {result['law']}")
            print(f"Text: {result['snippet']}")
            if result['source_url']:
                print(f"Source: {result['source_url']}")
        
        print(f"\nFound {len(results)} results")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line query
        query = " ".join(sys.argv[1:])
        single_query_mode(query)
    else:
        # Interactive mode
        interactive_mode()