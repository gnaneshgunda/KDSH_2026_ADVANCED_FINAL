"""
Index manager for LangGraph RAG system
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List
import numpy as np

from config import nlp, DEFAULT_CHUNK_SIZE
from models import ChunkMetadata

logger = logging.getLogger(__name__)


class LangGraphIndexManager:
    """Manages corpus indexing and caching for LangGraph RAG"""
    
    def __init__(self, client, books_dir: Path, index_path: Path):
        """
        Initialize index manager
        
        Args:
            client: LangChainNVIDIAClient instance
            books_dir: Directory containing book text files
            index_path: Path to cache file
        """
        self.client = client
        self.books_dir = Path(books_dir)
        self.index_path = Path(index_path)
        self.corpus: Dict[str, List[ChunkMetadata]] = {}
        
        logger.info(f"Index manager initialized | Books: {self.books_dir}")
    
    def load_or_build(self):
        """Load cached index or build new one"""
        if self.index_path.exists():
            logger.info(f"Loading cached index from {self.index_path}")
            try:
                with open(self.index_path, "rb") as f:
                    self.corpus = pickle.load(f)
                logger.info(f"Loaded {len(self.corpus)} books from cache")
                return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}. Building new index...")
        
        self._build_corpus()
    
    def _build_corpus(self):
        """Build corpus from book files"""
        logger.info("Building corpus index...")
        
        if not self.books_dir.exists():
            logger.error(f"Books directory not found: {self.books_dir}")
            return
        
        book_files = list(self.books_dir.glob("*.txt"))
        logger.info(f"Found {len(book_files)} book files")
        
        for book_file in book_files:
            book_key = book_file.stem.lower()
            logger.info(f"Processing: {book_key}")
            
            try:
                with open(book_file, "r", encoding="utf-8") as f:
                    text = f.read()
                
                chunks = self._chunk_text(text, book_key)
                self.corpus[book_key] = chunks
                logger.info(f"  -> {len(chunks)} chunks created")
            except Exception as e:
                logger.error(f"Failed to process {book_key}: {e}")
        
        # Save cache
        self._save_cache()
    
    def _chunk_text(self, text: str, book_key: str) -> List[ChunkMetadata]:
        """Chunk text and create metadata"""
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        chunks = []
        current_chunk = []
        current_words = 0
        start_pos = 0
        
        for sent in sentences:
            words = len(sent.split())
            if current_words + words > DEFAULT_CHUNK_SIZE and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start_pos": start_pos,
                    "end_pos": start_pos + len(chunk_text)
                })
                current_chunk = []
                current_words = 0
                start_pos += len(chunk_text)
            
            current_chunk.append(sent)
            current_words += words
        
        # Last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "start_pos": start_pos,
                "end_pos": start_pos + len(chunk_text)
            })
        
        # Generate embeddings
        logger.info(f"  Generating embeddings for {len(chunks)} chunks...")
        chunk_texts = [c["text"] for c in chunks]
        
        # Batch embed
        batch_size = 20
        all_embeddings = []
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i+batch_size]
            embeddings = self.client.embed_texts(batch)
            all_embeddings.extend(embeddings)
        
        # Create ChunkMetadata objects
        chunk_objects = []
        for idx, (chunk_data, embedding) in enumerate(zip(chunks, all_embeddings)):
            # Extract entities
            chunk_doc = nlp(chunk_data["text"][:1000])  # Limit for performance
            entities = [ent.text for ent in chunk_doc.ents]
            
            # Simple sentiment
            sentiment = 0.0
            if any(word in chunk_data["text"].lower() for word in ["happy", "joy", "love"]):
                sentiment = 0.5
            if any(word in chunk_data["text"].lower() for word in ["sad", "angry", "hate"]):
                sentiment = -0.5
            
            # Temporal markers
            temporal = [token.text for token in chunk_doc if token.pos_ == "NUM" or 
                       token.text.lower() in ["yesterday", "today", "tomorrow", "before", "after"]]
            
            # Causal indicators
            causal = [token.text for token in chunk_doc if 
                     token.text.lower() in ["because", "since", "therefore", "thus", "so"]]
            
            chunk_obj = ChunkMetadata(
                text=chunk_data["text"],
                embedding=embedding,
                context_vector=embedding,  # Simplified
                chunk_id=f"{book_key}_chunk_{idx}",
                start_pos=chunk_data["start_pos"],
                end_pos=chunk_data["end_pos"],
                entities=entities[:10],
                sentiment=sentiment,
                temporal_markers=temporal[:5],
                causal_indicators=causal[:5]
            )
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    def _save_cache(self):
        """Save corpus to cache file"""
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump(self.corpus, f)
            logger.info(f"Corpus cached to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def get_corpus(self) -> Dict[str, List[ChunkMetadata]]:
        """Get the corpus dictionary"""
        return self.corpus
