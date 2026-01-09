"""
Index building and management for narrative corpus
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from config import DEFAULT_BOOKS_DIR, DEFAULT_INDEX_PATH, nlp
from models import ChunkMetadata
from chunker import DependencyChunker
from context_builder import ContextVectorBuilder
from graph_rag import GraphRAG

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages building, loading, and saving narrative indices"""

    def __init__(self, chunker: DependencyChunker, context_builder: ContextVectorBuilder,
                 client, books_dir: Path = DEFAULT_BOOKS_DIR, 
                 index_path: Path = DEFAULT_INDEX_PATH):
        """
        Initialize index manager
        
        Args:
            chunker: DependencyChunker instance
            context_builder: ContextVectorBuilder instance
            client: NVIDIAClient instance
            books_dir: Directory containing book texts
            index_path: Path to save/load pickled index
        """
        self.chunker = chunker
        self.context_builder = context_builder
        self.client = client
        self.books_dir = books_dir
        self.index_path = index_path
        
        self.corpus: Dict[str, List[ChunkMetadata]] = {}
        self.graph_rag: Dict[str, GraphRAG] = {}
        
        logger.info(f"IndexManager initialized for {books_dir}")

    def load_or_build(self):
        """Load cached index or build from scratch"""
        if self.index_path.exists():
            self._load_from_cache()
        else:
            self._build_index()

    def _load_from_cache(self):
        """Load index from pickle file"""
        try:
            with open(self.index_path, "rb") as f:
                self.corpus, self.graph_rag = pickle.load(f)
            logger.info(f"Loaded cached index from {self.index_path}")
            logger.info(f"  Corpus contains {len(self.corpus)} books")
            logger.info(f"  Total chunks indexed: {sum(len(c) for c in self.corpus.values())}")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            logger.info("Building fresh index...")
            self._build_index()

    def _build_index(self):
        """Build index from book files"""
        logger.info("Building advanced narrative index (this may take a while)...")
        
        if not self.books_dir.exists():
            logger.warning(f"Books directory not found: {self.books_dir}")
            return

        for book_file in self.books_dir.glob("*.txt"):
            self._index_book(book_file)

        self._save_to_cache()

    def _index_book(self, book_file: Path):
        """Index a single book file"""
        book_key = book_file.stem.lower()
        text = book_file.read_text(encoding="utf-8", errors="ignore")

        logger.info(f"Processing {book_key}...")

        # Chunk with dependency parsing
        chunks = self.chunker.chunk_text(text)
        chunk_texts = [c[0] for c in chunks]

        # Generate embeddings
        try:
            embeddings = self.client.embed(chunk_texts)
        except Exception as e:
            logger.error(f"Failed to embed chunks for {book_key}: {e}")
            return

        # Create chunk metadata objects
        chunk_objects = []
        for i, (chunk_text, dep_graph, entities) in enumerate(chunks):
            chunk_id = f"{book_key}_chunk_{i}"

            # Build context vector
            context_vec = self.context_builder.build_context_vector(chunk_text, embeddings[i])

            chunk_obj = ChunkMetadata(
                text=chunk_text,
                embedding=embeddings[i],
                context_vector=context_vec,
                chunk_id=chunk_id,
                start_pos=0,  # Simplified
                end_pos=len(chunk_text),
                dependency_graph=dep_graph,
                entities=entities,
                sentiment=self.context_builder.analyze_sentiment(chunk_text),
                temporal_markers=self.context_builder.extract_temporal_markers(chunk_text),
                causal_indicators=self.context_builder.extract_causal_indicators(chunk_text)
            )
            chunk_objects.append(chunk_obj)

        self.corpus[book_key] = chunk_objects

        # Build Graph-RAG
        self.graph_rag[book_key] = GraphRAG(chunk_objects)

        logger.info(f"  Indexed {len(chunk_objects)} chunks")

    def _save_to_cache(self):
        """Save index to pickle file"""
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump((self.corpus, self.graph_rag), f)
            logger.info(f"Index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def get_corpus(self) -> Dict[str, List[ChunkMetadata]]:
        """Get indexed corpus"""
        return self.corpus

    def get_graph_rag(self) -> Dict[str, GraphRAG]:
        """Get Graph-RAG instances by book"""
        return self.graph_rag

    def get_chunks_for_book(self, book_key: str) -> List[ChunkMetadata]:
        """Get all chunks for a specific book"""
        return self.corpus.get(book_key, [])
