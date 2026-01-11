"""
Paragraph-aware semantic chunker for narrative consistency RAG
No dependency parsing - simple and robust chunking strategy
"""

import logging
from typing import List
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Chunk narrative text by paragraphs first, then sentence merging.
    Guarantees semantic integrity without complex dependency parsing.
    """

    def __init__(self, max_chunk_size: int = 300):
        """
        Initialize chunker.
        
        Args:
            max_chunk_size: Maximum words per chunk
        """
        self.max_chunk_size = max_chunk_size
        logger.info(f"SemanticChunker initialized (max_words={max_chunk_size})")

    def chunk_text(self, text: str, overlap_ratio: float = 0.1) -> List[str]:
        """
        Chunk text by paragraphs + sentence merging.
        
        Strategy:
        1. Split by paragraphs (double newlines)
        2. For each paragraph, split into sentences
        3. Merge sentences until approaching max_chunk_size
        4. Add small overlap between chunks
        
        Args:
            text: Raw text to chunk
            overlap_ratio: Fraction of previous chunk to repeat (for context)
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Split into paragraphs
        paragraphs = [
            p.strip() 
            for p in text.split('\n\n') 
            if p.strip()
        ]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            # Split paragraph into sentences
            try:
                sentences = sent_tokenize(para)
            except:
                # Fallback: simple period-based split
                sentences = [s.strip() for s in para.split('.') if s.strip()]
            
            for sent in sentences:
                sent_words = len(sent.split())
                
                # If adding this sentence exceeds limit and we have content, flush
                if current_size + sent_words > self.max_chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Add overlap (last few sentences of previous chunk)
                    overlap_count = max(1, int(len(current_chunk) * overlap_ratio))
                    current_chunk = current_chunk[-overlap_count:] if overlap_count > 0 else []
                    current_size = sum(len(s.split()) for s in current_chunk)
                
                # Add sentence to current chunk
                current_chunk.append(sent)
                current_size += sent_words
        
        # Flush remaining
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Chunked text into {len(chunks)} segments (avg size: {current_size / len(chunks) if chunks else 0:.0f} words)")
        return chunks