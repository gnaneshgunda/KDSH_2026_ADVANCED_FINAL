"""
Pathway-based real-time document indexing and retrieval
"""

import logging
import pathway as pw
from pathlib import Path
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


class PathwayIndexer:
    """
    Real-time document indexing using Pathway for live updates.
    Monitors book directory and automatically reindexes on changes.
    """
    
    def __init__(self, books_dir: Path, chunker, context_builder, client):
        self.books_dir = books_dir
        self.chunker = chunker
        self.context_builder = context_builder
        self.client = client
        logger.info(f"PathwayIndexer initialized for {books_dir}")
    
    def create_live_index(self):
        """
        Create a live index that updates automatically when files change.
        """
        # Define input connector for text files
        input_table = pw.io.fs.read(
            str(self.books_dir),
            format="plaintext",
            mode="streaming",
            with_metadata=True
        )
        
        # Process documents: chunk + embed + metadata
        processed = input_table.select(
            path=pw.this.path,
            content=pw.this.data,
            chunks=pw.apply(self._process_document, pw.this.data)
        )
        
        return processed
    
    def _process_document(self, text: str) -> List[Dict]:
        """Process document into chunks with embeddings and metadata."""
        chunk_dicts = self.chunker.chunk_text(text)
        
        # Extract texts for batch embedding
        chunk_texts = [c['text'] for c in chunk_dicts]
        embeddings = self.client.embed(chunk_texts)
        
        # Combine chunks with embeddings
        processed_chunks = []
        for i, chunk_dict in enumerate(chunk_dicts):
            processed_chunks.append({
                'text': chunk_dict['text'],
                'embedding': embeddings[i],
                'metadata': chunk_dict['metadata'],
                'start_pos': chunk_dict['start_pos'],
                'end_pos': chunk_dict['end_pos']
            })
        
        return processed_chunks
    
    def query_live_index(self, processed_table, query: str, character: str = None, top_k: int = 5):
        """
        Query the live index with real-time results.
        """
        query_emb = self.client.embed([query])[0]
        
        # Filter by character if specified
        if character:
            filtered = processed_table.filter(
                pw.apply(lambda chunks: self._has_character(chunks, character), pw.this.chunks)
            )
        else:
            filtered = processed_table
        
        # Compute similarities and rank
        ranked = filtered.select(
            chunks=pw.this.chunks,
            scores=pw.apply(lambda chunks: self._compute_scores(chunks, query_emb), pw.this.chunks)
        )
        
        return ranked
    
    def _has_character(self, chunks: List[Dict], character: str) -> bool:
        """Check if any chunk mentions the character."""
        char_lower = character.lower()
        for chunk in chunks:
            entities = chunk['metadata'].get('characters', [])
            if any(char_lower in e.lower() for e in entities):
                return True
        return False
    
    def _compute_scores(self, chunks: List[Dict], query_emb: np.ndarray) -> List[float]:
        """Compute similarity scores for chunks."""
        scores = []
        for chunk in chunks:
            emb = np.array(chunk['embedding'])
            score = float(np.dot(query_emb, emb))
            scores.append(score)
        return scores


def enable_pathway_mode(books_dir: Path, chunker, context_builder, client):
    """
    Enable Pathway real-time indexing mode.
    Returns PathwayIndexer instance.
    """
    try:
        indexer = PathwayIndexer(books_dir, chunker, context_builder, client)
        logger.info("Pathway real-time indexing enabled")
        return indexer
    except Exception as e:
        logger.warning(f"Pathway mode failed: {e}. Falling back to static indexing.")
        return None
