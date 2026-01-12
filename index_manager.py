"""
Index manager for building and caching narrative chunks per book
Creates separate pkl files in ./db directory for each book
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from models import ChunkMetadata

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Builds, caches, and manages narrative chunk indices.
    Creates separate .pkl file for each book in ./db directory.
    """

    def __init__(self, chunker, context_builder, client, books_dir, db_path):
        """
        Initialize IndexManager.
        
        Args:
            chunker: SemanticChunker instance
            context_builder: ContextVectorBuilder instance
            client: NVIDIAClient instance
            books_dir: Directory containing .txt files
            db_path: Directory to store .pkl files (one per book)
        """
        self.chunker = chunker
        self.context_builder = context_builder
        self.client = client
        self.books_dir = Path(books_dir)
        self.db_path = Path(db_path)
        self.corpus = {}  # book_name -> List[ChunkMetadata]
        
        # Ensure db_path exists
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"IndexManager initialized (books_dir={books_dir}, db={db_path})")

    def load_or_build(self) -> Dict[str, List[ChunkMetadata]]:
        """
        Load cached indices or build new ones.
        For each book, checks if .pkl exists in db_path.
        If not, builds from .txt file.
        
        Returns:
            Dictionary mapping book_name -> List[ChunkMetadata]
        """
        if not self.books_dir.exists():
            logger.error(f"Books directory not found: {self.books_dir}")
            return {}

        txt_files = list(self.books_dir.glob("*.txt"))
        logger.info(f"Found {len(txt_files)} text files in {self.books_dir}")

        for book_file in txt_files:
            try:
                book_key = book_file.stem.lower()
                logger.info(f"Processing book: {book_key}")
                
                # Check if pkl exists
                pkl_path = self.db_path / f"{book_key}.pkl"
                
                if pkl_path.exists():
                    logger.info(f"  Loading cached index: {pkl_path}")
                    self._load_book_index(book_key, pkl_path)
                else:
                    logger.info(f"  Building index from text: {book_file}")
                    self._build_book_index(book_key, book_file, pkl_path)
                    
            except Exception as e:
                logger.error(f"Error processing {book_file}: {e}")

        logger.info(f"Index loading complete. Books loaded: {list(self.corpus.keys())}")
        return self.corpus

    def _load_book_index(self, book_key: str, pkl_path: Path):
        """Load cached index for a specific book"""
        try:
            with open(pkl_path, "rb") as f:
                chunks = pickle.load(f)
            self.corpus[book_key] = chunks
            logger.info(f"  ✓ Loaded {len(chunks)} chunks from cache")
        except Exception as e:
            logger.error(f"Failed to load cache {pkl_path}: {e}")
            raise

    def _build_book_index(self, book_key, book_file, pkl_path):
        try:
            text = book_file.read_text(encoding="utf-8")
            chunk_dicts = self.chunker.chunk_text(text)  # Now returns list of dicts
            
            # Extract just the text for embedding
            chunk_texts = [c['text'] for c in chunk_dicts]
            embeddings = self.client.embed(chunk_texts)
            
            book_chunks = []
            for i, chunk_dict in enumerate(chunk_dicts):
                chunk_text = chunk_dict['text']
                metadata = chunk_dict['metadata']
                
                # Use metadata from chunker, enrich with context builder
                entities = metadata.get('characters', [])
                temporal = metadata.get('temporal', [])
                locations = metadata.get('locations', [])
                
                # Additional analysis
                causal = self.context_builder.extract_causal_indicators(chunk_text)
                sentiment = self.context_builder.analyze_sentiment(chunk_text)
                
                meta = ChunkMetadata(
                    text=chunk_text,
                    embedding=np.array(embeddings[i]),
                    chunk_id=f"{book_key}_{i}",
                    entities=entities,
                    temporal_markers=temporal,
                    causal_indicators=causal,
                    sentiment=sentiment,
                    start_pos=chunk_dict['start_pos'],
                    end_pos=chunk_dict['end_pos'],
                    locations=locations,
                    has_dialogue=metadata.get('has_dialogue', False)
                )
                book_chunks.append(meta)
                
            # Save to pkl
            with open(pkl_path, "wb") as f:
                pickle.dump(book_chunks, f)
            self.corpus[book_key] = book_chunks
            logger.info(f"  ✓ Built {len(book_chunks)} chunks and cached to {pkl_path}")

        except Exception as e:
            logger.error(f"Failed to build index for {book_key}: {e}")
            raise

    def _save_book_index(self, book_key: str, pkl_path: Path, chunks: List[ChunkMetadata]):
        """Save book index to pkl file"""
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(chunks, f)
            logger.debug(f"  Cached {len(chunks)} chunks to {pkl_path}")
        except Exception as e:
            logger.error(f"Failed to save index to {pkl_path}: {e}")
            raise

    def get_corpus(self) -> Dict[str, List[ChunkMetadata]]:
        """Get the loaded corpus"""
        return self.corpus

    def get_book_chunks(self, book_key: str) -> List[ChunkMetadata]:
        """Get chunks for a specific book"""
        return self.corpus.get(book_key, [])

    def list_cached_books(self) -> List[str]:
        """List all cached books in db directory"""
        pkl_files = list(self.db_path.glob("*.pkl"))
        return [f.stem for f in pkl_files]