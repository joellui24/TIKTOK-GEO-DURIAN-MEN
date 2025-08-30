#!/usr/bin/env python3

import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import chromadb
from fastembed import TextEmbedding
import sys

# Add parent directory to path for provence_integration import
sys.path.append(str(Path(__file__).resolve().parents[1]))
from provence_integration import ProvenceProcessor

# Configuration
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "laws"
INDEX_DIR = ROOT / "data" / "index" / "chroma"
MANIFEST_PATH = ROOT / "ingest" / "manifest.json"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TARGET_WORDS = 220  # target chunk size in words
OVERLAP_WORDS = 40  # overlap between chunks
MIN_WORDS = 50      # minimum chunk size

# Compliance extraction configuration
COMPLIANCE_QUERY = "What are the requirements for platforms or providers to comply"
MIN_COMPLIANCE_WORDS = 20  # minimum words in pruned compliance content

def extract_compliance_content(text: str, provence_processor: ProvenceProcessor) -> Dict[str, Any]:
    """
    Extract compliance-relevant content using Provence
    
    Args:
        text: Full legal document text
        provence_processor: Initialized Provence processor
        
    Returns:
        Dict with compliance content and metadata
    """
    try:
        result = provence_processor.prune_context(
            question=COMPLIANCE_QUERY,
            context=text,
            custom_threshold=0.2  # Moderate pruning for compliance content
        )
        
        # Check if we got meaningful compliance content
        pruned_words = len(result['pruned_context'].split())
        
        if pruned_words >= MIN_COMPLIANCE_WORDS:
            return {
                'compliance_content': result['pruned_context'],
                'compliance_score': result['reranking_score'],
                'compression_ratio': result['compression_ratio'],
                'has_compliance': True,
                'original_words': result['original_length'],
                'compliance_words': pruned_words
            }
        else:
            # Try more conservative threshold if initial pruning was too aggressive
            conservative_result = provence_processor.prune_context(
                question=COMPLIANCE_QUERY,
                context=text,
                custom_threshold=0.05  # Very conservative
            )
            
            conservative_words = len(conservative_result['pruned_context'].split())
            
            if conservative_words >= MIN_COMPLIANCE_WORDS:
                return {
                    'compliance_content': conservative_result['pruned_context'],
                    'compliance_score': conservative_result['reranking_score'],
                    'compression_ratio': conservative_result['compression_ratio'],
                    'has_compliance': True,
                    'original_words': conservative_result['original_length'],
                    'compliance_words': conservative_words
                }
            else:
                return {
                    'compliance_content': '',
                    'compliance_score': 0.0,
                    'compression_ratio': 0.0,
                    'has_compliance': False,
                    'original_words': len(text.split()),
                    'compliance_words': 0
                }
                
    except Exception as e:
        print(f"Warning: Provence processing failed: {e}")
        return {
            'compliance_content': text,  # Fallback to full text
            'compliance_score': 0.0,
            'compression_ratio': 1.0,
            'has_compliance': True,  # Include as fallback
            'original_words': len(text.split()),
            'compliance_words': len(text.split()),
            'provence_error': str(e)
        }

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

def article_based_chunks(text: str, doc_metadata: dict) -> List[dict]:
    """
    Chunk by complete articles with rich contextual metadata.
    Based on MVP approach that preserves semantic boundaries.
    """
    chunks = []
    current_section = "Document Content"
    current_section_description = ""
    
    lines = text.split('\n')
    current_article = None
    current_content = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Detect section headers (## format)
        if match := re.match(r'^##\s+(.+)$', line_stripped):
            section_text = match.group(1).strip()
            # Extract section name and description if present
            if ' - ' in section_text:
                current_section, current_section_description = section_text.split(' - ', 1)
            else:
                current_section = section_text
                current_section_description = ""
            continue
            
        # Detect article headers (various formats)
        article_match = None
        
        # Format 1: ### _Article X - Title_ (italic)
        if match := re.match(r'^#+\s*_Article\s+(\d+[a-zA-Z]?)\s*[-–]\s*(.+)_$', line_stripped):
            article_match = (match.group(1), match.group(2).strip())
        
        # Format 2: ### _Section X - Title_ (italic, for US docs, including hyphenated like 13-2c-301)
        elif match := re.match(r'^#+\s*_Section\s+([\d\-a-zA-Z]+(?:\.[\d\-a-zA-Z]+)*)\s*[-–]\s*(.+)_$', line_stripped):
            article_match = (f"Section {match.group(1)}", match.group(2).strip())
            
        # Format 3: ### _(a) Title_ - US Code italic subsections  
        elif match := re.match(r'^#+\s*_\(([a-z])\)\s+(.+)_$', line_stripped):
            article_match = (f"({match.group(1)})", match.group(2).strip())
            
        # Format 4: **Bold Article Headers**
        elif match := re.match(r'^\*\*([^*]+)\*\*:?\s*$', line_stripped):
            header_text = match.group(1).strip()
            if 'Article' in header_text or 'Section' in header_text:
                article_match = (header_text, "")
        
        if article_match:
            # Save previous article
            if current_article and current_content:
                content_text = '\n'.join(current_content).strip()
                if len(content_text.split()) >= MIN_WORDS:
                    chunks.append({
                        'content': content_text,
                        'metadata': {
                            **doc_metadata,
                            'article_number': current_article['number'],
                            'article_title': current_article['title'],
                            'parent_section': current_section,
                            'section_description': current_section_description,
                            'chunk_type': 'complete_article',
                            'chunk_words': len(content_text.split())
                        }
                    })
            
            # Start new article
            current_article = {
                'number': article_match[0],
                'title': article_match[1]
            }
            current_content = [line]  # Include article header
            
        elif current_article:
            current_content.append(line)
        
        # If no current article, still collect content for potential fallback chunking
        elif line_stripped and not current_article:
            if not current_content:
                current_content = []
            current_content.append(line)
    
    # Add final article
    if current_article and current_content:
        content_text = '\n'.join(current_content).strip()
        if len(content_text.split()) >= MIN_WORDS:
            chunks.append({
                'content': content_text,
                'metadata': {
                    **doc_metadata,
                    'article_number': current_article['number'],
                    'article_title': current_article['title'],
                    'parent_section': current_section,
                    'section_description': current_section_description,
                    'chunk_type': 'complete_article',
                    'chunk_words': len(content_text.split())
                }
            })
    
    # Fallback: if no articles found, use sliding window on remaining content
    if not chunks and current_content:
        fallback_text = '\n'.join(current_content).strip()
        if len(fallback_text.split()) >= MIN_WORDS:
            fallback_chunks = sliding_window_chunks(fallback_text, TARGET_WORDS, OVERLAP_WORDS)
            for i, chunk_text in enumerate(fallback_chunks):
                chunks.append({
                    'content': chunk_text,
                    'metadata': {
                        **doc_metadata,
                        'article_number': f"chunk_{i+1}",
                        'article_title': "Fallback chunk",
                        'parent_section': current_section,
                        'section_description': "No article structure detected",
                        'chunk_type': 'fallback_sliding_window',
                        'chunk_words': len(chunk_text.split())
                    }
                })
    
    return chunks

def process_document(file_path: Path, doc_id: str, manifest: Dict[str, Any], provence_processor: ProvenceProcessor) -> List[Dict[str, Any]]:
    """Process a single document into chunks with metadata"""
    print(f"Processing {file_path.name}...")
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Try with a different encoding if utf-8 fails
        content = file_path.read_text(encoding='utf-8', errors='replace')
    metadata, body = extract_frontmatter(content)
    
    # Base document metadata
    doc_metadata = {
        "doc_id": doc_id,
        "jurisdiction": metadata.get("jurisdiction", ""),
        "law": metadata.get("law", ""),
        "source_url": metadata.get("source_url", ""),
        "effective_start": metadata.get("effective_start", "")
    }
    
    # Try article-based chunking first
    print(f"  Attempting article-based chunking...")
    article_chunks = article_based_chunks(body, doc_metadata)
    
    if article_chunks:
        print(f"  ✅ Found {len(article_chunks)} articles")
        print(f"  🔍 Extracting compliance content with Provence...")
        
        chunks = []
        compliance_chunks_count = 0
        
        for i, chunk_data in enumerate(article_chunks):
            # Extract compliance-relevant content
            compliance_result = extract_compliance_content(
                chunk_data['content'], 
                provence_processor
            )
            
            # Skip chunks with no compliance content
            if not compliance_result['has_compliance']:
                print(f"    ⏭️  Skipping {chunk_data['metadata']['article_number']} (no compliance content)")
                continue
            
            compliance_chunks_count += 1
            
            # Enhanced chunk metadata with compliance info
            enhanced_metadata = {
                **chunk_data['metadata'],
                "chunk_id": f"{doc_id}-{i:03d}",
                "word_count": compliance_result['compliance_words'],
                "original_word_count": compliance_result['original_words'],
                "compliance_score": float(compliance_result['compliance_score']),  # Convert to standard float
                "compression_ratio": float(compliance_result['compression_ratio']),  # Convert to standard float
                "provence_applied": True
            }
            
            # Add provence error info if it exists
            if 'provence_error' in compliance_result:
                enhanced_metadata['provence_error'] = compliance_result['provence_error']
            
            chunks.append({
                "text": compliance_result['compliance_content'],  # Use compliance content for embedding
                "metadata": enhanced_metadata
            })
        
        print(f"  ✅ Generated {compliance_chunks_count} compliance-focused chunks")
        return chunks
    
    # Fallback to original sliding window approach
    print(f"  ⚠️ No articles detected, falling back to sliding window...")
    sections = split_by_headings(body)
    
    chunks = []
    chunk_id = 0
    
    for section in sections:
        # Apply sliding window to each section
        section_chunks = sliding_window_chunks(section, TARGET_WORDS, OVERLAP_WORDS)
        
        for chunk_text in section_chunks:
            chunk_metadata = {
                **doc_metadata,
                "chunk_id": f"{doc_id}-{chunk_id:03d}",
                "article_number": f"legacy_chunk_{chunk_id}",
                "article_title": "Legacy sliding window chunk",
                "parent_section": "Unknown Section",
                "section_description": "",
                "chunk_type": "legacy_sliding_window",
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
    print("Building compliance-focused vector index...")
    
    # Initialize embedding model
    embedding_model = TextEmbedding(model_name=EMBED_MODEL)
    
    # Initialize Provence processor
    print("Initializing Provence processor for compliance extraction...")
    provence_processor = ProvenceProcessor(threshold=0.2)
    
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
        
        doc_chunks = process_document(file_path, doc_info["doc_id"], manifest, provence_processor)
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
    ids = [chunk["metadata"]["chunk_id"] for chunk in all_chunks]
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