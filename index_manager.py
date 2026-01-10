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
        """Load index from per-book pickle files"""
        try:
            books_loaded = 0
            for book_file in self.books_dir.glob("*.txt"):
                book_key = book_file.stem.lower()
                per_book_index = self.index_path.parent / f"advanced_index_{book_key}.pkl"
                
                if per_book_index.exists():
                    with open(per_book_index, "rb") as f:
                        chunks, graph = pickle.load(f)
                        self.corpus[book_key] = chunks
                        self.graph_rag[book_key] = graph
                    books_loaded += 1
                    logger.info(f"  Loaded {book_key}: {len(chunks)} chunks")
            
            if books_loaded == 0:
                logger.info("No cached indices found. Building fresh index...")
                self._build_index()
            else:
                logger.info(f"Loaded cached index for {books_loaded} books")
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
        # Create chunk metadata objects
        chunk_objects = []
        current_pos = 0
        total_len = len(text)
        
        for i, (chunk_text, dep_graph, entities) in enumerate(chunks):
            chunk_id = f"{book_key}_chunk_{i}"
            
            # Find exact position (simplified approach: assuming ordered chunks reconstruct text)
            # In a real scenario, we'd use exact span indices from the tokenizer/splitter
            start_pos = text.find(chunk_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos # Fallback
            
            end_pos = start_pos + len(chunk_text)
            current_pos = end_pos

            # Build context vector
            context_vec = self.context_builder.build_context_vector(chunk_text, embeddings[i])

            chunk_obj = ChunkMetadata(
                text=chunk_text,
                embedding=embeddings[i],
                context_vector=context_vec,
                chunk_id=chunk_id,
                start_pos=start_pos,
                end_pos=end_pos,
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
        """Save index to per-book pickle files"""
        try:
            for book_key, chunks in self.corpus.items():
                graph = self.graph_rag.get(book_key)
                per_book_index = self.index_path.parent / f"advanced_index_{book_key}.pkl"
                
                with open(per_book_index, "wb") as f:
                    pickle.dump((chunks, graph), f)
                logger.info(f"Saved index for {book_key} to {per_book_index}")
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
