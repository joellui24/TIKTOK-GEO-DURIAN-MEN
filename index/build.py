#!/usr/bin/env python3

import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import chromadb
from fastembed import TextEmbedding

# Configuration
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "laws"
INDEX_DIR = ROOT / "data" / "index" / "chroma"
MANIFEST_PATH = ROOT / "ingest" / "manifest.json"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TARGET_WORDS = 220  # target chunk size in words
OVERLAP_WORDS = 40  # overlap between chunks
MIN_WORDS = 50      # minimum chunk size

def load_manifest() -> Dict[str, Any]:
    """Load the document manifest"""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}

def extract_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """Extract YAML frontmatter and return metadata dict + remaining content"""
    if not content.startswith('---\n'):
        return {}, content
    
    try:
        # Find the end of frontmatter
        end_idx = content.find('\n---\n', 4)
        if end_idx == -1:
            return {}, content
        
        # Parse frontmatter (simple key: value parsing)
        frontmatter_text = content[4:end_idx]
        metadata = {}
        for line in frontmatter_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
        
        # Return metadata and remaining content
        remaining_content = content[end_idx + 5:].strip()
        return metadata, remaining_content
    
    except Exception as e:
        print(f"Warning: Failed to parse frontmatter: {e}")
        return {}, content

def split_by_headings(text: str) -> List[str]:
    """Split text by legal headings (Article, Section, etc.) while preserving heading context"""
    # Legal heading patterns
    heading_patterns = [
        r'^## (Article \d+.*?)$',
        r'^## (Section \d+.*?)$',
        r'^## \([a-z]\) .*?$',
        r'^# .*?$'
    ]
    
    sections = []
    current_section = []
    current_heading = ""
    
    lines = text.split('\n')
    
    for line in lines:
        is_heading = False
        for pattern in heading_patterns:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                # Save previous section if it exists
                if current_section:
                    section_text = (current_heading + "\n\n" + "\n".join(current_section)).strip()
                    if section_text:
                        sections.append(section_text)
                
                # Start new section
                current_heading = line.strip()
                current_section = []
                is_heading = True
                break
        
        if not is_heading and line.strip():
            current_section.append(line)
    
    # Add final section
    if current_section:
        section_text = (current_heading + "\n\n" + "\n".join(current_section)).strip()
        if section_text:
            sections.append(section_text)
    
    return sections if sections else [text]

def sliding_window_chunks(text: str, target_words: int, overlap_words: int) -> List[str]:
    """Create overlapping chunks using sliding window approach"""
    words = text.split()
    if len(words) <= target_words:
        return [text] if len(words) >= MIN_WORDS else []
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + target_words, len(words))
        chunk_words = words[start:end]
        
        if len(chunk_words) >= MIN_WORDS:
            chunks.append(' '.join(chunk_words))
        
        # Move window forward
        start += target_words - overlap_words
        
        # Break if we're at the end
        if end >= len(words):
            break
    
    return chunks

def process_document(file_path: Path, doc_id: str, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process a single document into chunks with metadata"""
    print(f"Processing {file_path.name}...")
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Try with a different encoding if utf-8 fails
        content = file_path.read_text(encoding='utf-8', errors='replace')
    metadata, body = extract_frontmatter(content)
    
    # Split by legal headings first
    sections = split_by_headings(body)
    
    chunks = []
    chunk_id = 0
    
    for section in sections:
        # Apply sliding window to each section
        section_chunks = sliding_window_chunks(section, TARGET_WORDS, OVERLAP_WORDS)
        
        for chunk_text in section_chunks:
            chunk_metadata = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}-{chunk_id:03d}",
                "jurisdiction": metadata.get("jurisdiction", ""),
                "law": metadata.get("law", ""),
                "source_url": metadata.get("source_url", ""),
                "effective_start": metadata.get("effective_start", ""),
                "word_count": len(chunk_text.split())
            }
            
            chunks.append({
                "id": chunk_metadata["chunk_id"],
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            
            chunk_id += 1
    
    return chunks

def build_index():
    """Build the vector index from all law documents"""
    print("Building vector index...")
    
    # Initialize embedding model
    embedding_model = TextEmbedding(model_name=EMBED_MODEL)
    
    # Load manifest
    manifest = load_manifest()
    if not manifest:
        raise ValueError("No manifest found. Create ingest/manifest.json first.")
    
    # Process all documents
    all_chunks = []
    for filename, doc_info in manifest.items():
        file_path = DATA_DIR / filename
        if not file_path.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue
        
        doc_chunks = process_document(file_path, doc_info["doc_id"], manifest)
        all_chunks.extend(doc_chunks)
    
    if not all_chunks:
        raise ValueError("No chunks generated. Check your document files.")
    
    print(f"Generated {len(all_chunks)} chunks from {len(manifest)} documents")
    
    # Create ChromaDB collection
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    
    # Delete existing collection if it exists
    try:
        client.delete_collection("laws")
    except:
        pass
    
    collection = client.create_collection(
        name="laws",
        metadata={"description": "Legal compliance document chunks"}
    )
    
    # Prepare data for insertion
    ids = [chunk["id"] for chunk in all_chunks]
    documents = [chunk["text"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings_generator = embedding_model.embed(documents)
    embeddings = []
    for i, embedding in enumerate(embeddings_generator):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(documents)} embeddings...")
        embeddings.append(embedding.tolist())
    
    # Insert into ChromaDB
    print("Inserting into database...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    
    # Generate index version hash
    content_hash = hashlib.md5()
    for chunk in all_chunks:
        content_hash.update(chunk["text"].encode())
    index_version = content_hash.hexdigest()[:12]
    
    # Save index metadata
    index_metadata = {
        "version": index_version,
        "embedding_model": EMBED_MODEL,
        "chunk_count": len(all_chunks),
        "documents": list(manifest.keys()),
        "target_words": TARGET_WORDS,
        "overlap_words": OVERLAP_WORDS,
        "created_at": str(Path(__file__).stat().st_mtime)
    }
    
    (INDEX_DIR / "metadata.json").write_text(json.dumps(index_metadata, indent=2))
    
    print(f" Index built successfully!")
    print(f"   Version: {index_version}")
    print(f"   Chunks: {len(all_chunks)}")
    print(f"   Model: {EMBED_MODEL}")
    print(f"   Location: {INDEX_DIR}")

if __name__ == "__main__":
    build_index()