#!/usr/bin/env python3
"""
Provence Integration Module for Legal Compliance System

This module integrates Provence (context pruning model) with the existing
ChromaDB-based legal document retrieval system to provide more precise
and question-aware context extraction.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from transformers import AutoModel
import nltk
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProvenceProcessor:
    """
    Wrapper class for Provence context pruning model
    """
    
    def __init__(self, 
                 model_name: str = "naver/provence-reranker-debertav3-v1",
                 cache_dir: Optional[str] = None,
                 threshold: float = 0.1,
                 always_select_title: bool = True):
        """
        Initialize Provence processor
        
        Args:
            model_name: HuggingFace model identifier for Provence
            cache_dir: Directory to cache the model (optional)
            threshold: Pruning threshold (0.1 conservative, 0.5 aggressive)
            always_select_title: Whether to always keep first sentence as title
        """
        self.model_name = model_name
        self.threshold = threshold
        self.always_select_title = always_select_title
        self.model = None
        self.cache_dir = cache_dir
        
        # Ensure NLTK punkt is available
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt_tab')
    
    def _load_model(self):
        """Lazy load the Provence model"""
        if self.model is None:
            logger.info(f"Loading Provence model: {self.model_name}")
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True,
                    cache_dir=self.cache_dir
                )
                logger.info("Provence model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Provence model: {e}")
                raise
    
    def prune_context(self, 
                     question: str, 
                     context: str,
                     custom_threshold: Optional[float] = None,
                     custom_always_select_title: Optional[bool] = None) -> Dict[str, Any]:
        """
        Prune context using Provence model
        
        Args:
            question: User question or feature description
            context: Document context to be pruned
            custom_threshold: Override default threshold for this call
            custom_always_select_title: Override title selection for this call
            
        Returns:
            Dict with pruned_context and reranking_score
        """
        self._load_model()
        
        if not context.strip():
            return {
                'pruned_context': '',
                'reranking_score': 0.0,
                'original_length': 0,
                'pruned_length': 0,
                'compression_ratio': 0.0
            }
        
        # Use custom parameters if provided
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        always_select_title = custom_always_select_title if custom_always_select_title is not None else self.always_select_title
        
        try:
            # Process with Provence
            result = self.model.process(
                question=question,
                context=context,
                threshold=threshold,
                always_select_title=always_select_title
            )
            
            # Calculate metrics
            original_length = len(context.split())
            pruned_length = len(result['pruned_context'].split())
            compression_ratio = pruned_length / original_length if original_length > 0 else 0.0
            
            return {
                'pruned_context': result['pruned_context'],
                'reranking_score': result['reranking_score'],
                'original_length': original_length,
                'pruned_length': pruned_length,
                'compression_ratio': compression_ratio
            }
            
        except Exception as e:
            logger.error(f"Error during Provence processing: {e}")
            # Fallback to original context if Provence fails
            return {
                'pruned_context': context,
                'reranking_score': 0.0,
                'original_length': len(context.split()),
                'pruned_length': len(context.split()),
                'compression_ratio': 1.0,
                'error': str(e)
            }
    
    def batch_prune_contexts(self,
                           questions: List[str],
                           contexts: List[List[str]],
                           custom_threshold: Optional[float] = None,
                           reorder: bool = False,
                           top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Batch process multiple questions with multiple contexts each
        
        Args:
            questions: List of questions
            contexts: List of lists of contexts (one list per question)
            custom_threshold: Override default threshold
            reorder: Whether to reorder contexts by relevance score
            top_k: Number of top contexts to keep if reordering
            
        Returns:
            List of lists of pruned results
        """
        self._load_model()
        
        if len(questions) != len(contexts):
            raise ValueError("Number of questions must match number of context lists")
        
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        
        try:
            # Process batch with Provence
            batch_results = self.model.process(
                question=questions,
                context=contexts,
                threshold=threshold,
                always_select_title=self.always_select_title,
                reorder=reorder,
                top_k=top_k
            )
            
            # Format results
            formatted_results = []
            for i, question_results in enumerate(batch_results):
                question_formatted = []
                for j, result in enumerate(question_results):
                    original_context = contexts[i][j] if j < len(contexts[i]) else ""
                    original_length = len(original_context.split())
                    pruned_length = len(result['pruned_context'].split())
                    
                    question_formatted.append({
                        'pruned_context': result['pruned_context'],
                        'reranking_score': result['reranking_score'],
                        'original_length': original_length,
                        'pruned_length': pruned_length,
                        'compression_ratio': pruned_length / original_length if original_length > 0 else 0.0
                    })
                
                formatted_results.append(question_formatted)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during batch Provence processing: {e}")
            # Fallback to original contexts
            fallback_results = []
            for i, question_contexts in enumerate(contexts):
                question_fallback = []
                for context in question_contexts:
                    length = len(context.split())
                    question_fallback.append({
                        'pruned_context': context,
                        'reranking_score': 0.0,
                        'original_length': length,
                        'pruned_length': length,
                        'compression_ratio': 1.0,
                        'error': str(e)
                    })
                fallback_results.append(question_fallback)
            return fallback_results


class LegalContextProcessor:
    """
    High-level processor that combines Provence with legal document analysis
    """
    
    def __init__(self, 
                 provence_threshold: float = 0.3,  # More aggressive default
                 min_context_words: int = 15,
                 max_context_words: int = 300):    # Shorter max to avoid splitting
        """
        Initialize legal context processor
        
        Args:
            provence_threshold: Default Provence pruning threshold
            min_context_words: Minimum words to keep after pruning
            max_context_words: Maximum words to process (for performance)
        """
        self.provence = ProvenceProcessor(threshold=provence_threshold)
        self.min_context_words = min_context_words
        self.max_context_words = max_context_words
    
    def process_legal_snippet(self, 
                            feature_query: str, 
                            legal_document: str,
                            metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a legal document snippet for a specific feature query
        
        Args:
            feature_query: Feature title and description combined
            legal_document: Raw legal document text
            metadata: Document metadata (jurisdiction, law, etc.)
            
        Returns:
            Dict with processed snippet and analysis
        """
        if not legal_document.strip():
            return {
                'processed_snippet': '',
                'relevance_score': 0.0,
                'processing_method': 'empty_input',
                'metadata': metadata or {}
            }
        
        # Truncate if too long
        words = legal_document.split()
        if len(words) > self.max_context_words:
            truncated_doc = ' '.join(words[:self.max_context_words])
            logger.warning(f"Truncated document from {len(words)} to {self.max_context_words} words")
        else:
            truncated_doc = legal_document
        
        # Apply Provence pruning
        provence_result = self.provence.prune_context(feature_query, truncated_doc)
        
        # Ensure minimum context length
        if provence_result['pruned_length'] < self.min_context_words and len(words) > self.min_context_words:
            logger.info(f"Provence pruned too aggressively ({provence_result['pruned_length']} words), using fallback")
            # Use more conservative threshold
            provence_result = self.provence.prune_context(
                feature_query, 
                truncated_doc, 
                custom_threshold=0.05  # More conservative
            )
        
        return {
            'processed_snippet': provence_result['pruned_context'],
            'relevance_score': provence_result['reranking_score'],
            'compression_ratio': provence_result['compression_ratio'],
            'original_length': provence_result['original_length'],
            'pruned_length': provence_result['pruned_length'],
            'processing_method': 'provence_pruning',
            'metadata': metadata or {},
            'provence_threshold': self.provence.threshold
        }
    
    def create_enhanced_snippet(self,
                              feature_query: str,
                              legal_document: str,
                              metadata: Dict[str, Any] = None) -> str:
        """
        Create an enhanced snippet that combines Provence pruning with legal structure
        
        Args:
            feature_query: Feature title and description
            legal_document: Raw legal document text
            metadata: Document metadata
            
        Returns:
            Formatted legal snippet string
        """
        result = self.process_legal_snippet(feature_query, legal_document, metadata)
        
        if not result['processed_snippet'].strip():
            return "No relevant legal content found."
        
        # Extract basic legal info from metadata
        law_name = ""
        jurisdiction = ""
        if metadata:
            law_name = metadata.get('law', '').upper()
            jurisdiction = metadata.get('jurisdiction', '').upper()
        
        # Build formatted snippet
        snippet_parts = []
        
        if jurisdiction and law_name:
            snippet_parts.append(f"LAW: {jurisdiction} {law_name}")
        elif law_name:
            snippet_parts.append(f"LAW: {law_name}")
        
        # Add relevance indicator
        relevance = result['relevance_score']
        if relevance > 2.0:
            relevance_indicator = "HIGH RELEVANCE"
        elif relevance > 1.0:
            relevance_indicator = "MODERATE RELEVANCE"
        else:
            relevance_indicator = "LOW RELEVANCE"
        
        snippet_parts.append(f"RELEVANCE: {relevance_indicator} (Score: {relevance:.2f})")
        snippet_parts.append("CONTENT:")
        snippet_parts.append(result['processed_snippet'])
        
        # Add compression info for debugging
        compression = result['compression_ratio']
        if compression < 0.8:
            snippet_parts.append(f"[Pruned to {compression:.0%} of original length using AI context filtering]")
        
        return '\n'.join(snippet_parts)


# Global instances for easy import
default_provence = ProvenceProcessor()
default_legal_processor = LegalContextProcessor()

# Convenience functions
def prune_legal_context(question: str, context: str, threshold: float = 0.1) -> str:
    """
    Simple function to prune legal context using Provence
    
    Args:
        question: Feature query or question
        context: Legal document text
        threshold: Pruning threshold (0.1 conservative, 0.5 aggressive)
        
    Returns:
        Pruned context string
    """
    result = default_provence.prune_context(question, context, custom_threshold=threshold)
    return result['pruned_context']

def create_legal_snippet(feature_query: str, legal_document: str, metadata: Dict = None) -> str:
    """
    Create enhanced legal snippet with Provence pruning
    
    Args:
        feature_query: Feature title and description
        legal_document: Raw legal document text
        metadata: Document metadata
        
    Returns:
        Formatted legal snippet
    """
    return default_legal_processor.create_enhanced_snippet(feature_query, legal_document, metadata)