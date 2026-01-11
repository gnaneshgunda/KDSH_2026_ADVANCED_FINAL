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

    def _build_book_index(self, book_key: str, book_file: Path, pkl_path: Path):
        """Build index for a specific book and save to pkl"""
        try:
            text = book_file.read_text(encoding="utf-8")
            logger.info(f"  Text size: {len(text)} characters")

            # Chunk the text
            chunks = self.chunker.chunk_text(text)
            logger.info(f"  Created {len(chunks)} chunks")

            # Embed chunks and build metadata
            chunk_texts = [c for c in chunks]
            embeddings = self.client.embed(chunk_texts)
            logger.info(f"  Embedded {len(embeddings)} chunks")

            book_chunks = []
            char_pos = 0

            for i, chunk_text in enumerate(chunk_texts):
                embedding = np.array(embeddings[i])
                context_vec = self.context_builder.build_context_vector(
                    chunk_text, embedding
                )

                # Extract entities, sentiment, temporal/causal markers
                try:
                    import spacy
                    nlp = spacy.load("en_core_web_sm")
                    doc = nlp(chunk_text[:1000])  # Limit for performance
                    entities = [ent.text for ent in doc.ents]
                except Exception as e:
                    logger.debug(f"Failed to extract entities: {e}")
                    entities = []

                sentiment = self.context_builder.analyze_sentiment(chunk_text)
                temporal = self.context_builder.extract_temporal_markers(chunk_text)
                causal = self.context_builder.extract_causal_indicators(chunk_text)

                meta = ChunkMetadata(
                    text=chunk_text,
                    embedding=embedding,
                    context_vector=context_vec,
                    chunk_id=f"{book_key}_{i}",
                    start_pos=char_pos,
                    end_pos=char_pos + len(chunk_text),
                    entities=entities,
                    sentiment=sentiment,
                    temporal_markers=temporal,
                    causal_indicators=causal
                )
                book_chunks.append(meta)
                char_pos += len(chunk_text) + 1

            self.corpus[book_key] = book_chunks
            
            # Save to pkl file
            self._save_book_index(book_key, pkl_path, book_chunks)
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