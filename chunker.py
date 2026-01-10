"""
Chunking module with dependency parsing for intelligent text segmentation
"""

import logging
from typing import List, Tuple
import networkx as nx
from config import nlp, DEFAULT_CHUNK_SIZE, DEFAULT_MIN_EDGE_DENSITY
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


class DependencyChunker:
    """Intelligently chunk text using dependency parsing and sentence boundaries"""

    def __init__(self, max_chunk_size: int = DEFAULT_CHUNK_SIZE, 
                 min_edge_density: float = DEFAULT_MIN_EDGE_DENSITY):
        self.max_chunk_size = max_chunk_size
        self.min_edge_density = min_edge_density
        logger.info(f"Chunker initialized with max_size={max_chunk_size}")

    def build_dependency_graph(self, sent: str) -> Tuple[nx.DiGraph, List[str]]:
        """
        Build dependency graph for a sentence
        
        Args:
            sent: Input sentence
            
        Returns:
            Tuple of (dependency graph, named entities)
        """
        doc = nlp(sent)
        graph = nx.DiGraph()
        entities = []

        # Add nodes from tokens
        for token in doc:
            graph.add_node(token.i, word=token.text, pos=token.pos_, dep=token.dep_)
            if token.head.i != token.i:
                graph.add_edge(token.head.i, token.i, dep=token.dep_)

        # Extract named entities
        for ent in doc.ents:
            entities.append(ent.text)

        return graph, entities

    def chunk_text(self, text: str) -> List[Tuple[str, nx.DiGraph, List[str]]]:
        """
        Intelligently chunk text using edge density and dependency boundaries
        
        Strategy:
        1. Tokenize into sentences
        2. For each sentence, build dependency graph
        3. Build chunks respecting sentence boundaries and max size
        4. Use edge density to identify strong semantic clusters
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of (chunk_text, dependency_graph, entities) tuples
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_words = 0
        current_graph = nx.DiGraph()
        current_ents = []

        logger.debug(f"Chunking text into {len(sentences)} sentences")

        for sent in sentences:
            graph, entities = self.build_dependency_graph(sent)
            sent_words = len(sent.split())

            # Calculate edge density of potential chunk
            test_graph = nx.compose(current_graph, graph)
            edge_density = self._calculate_edge_density(test_graph)

            # Check if adding this sentence exceeds limits or breaks coherence
            size_exceeded = current_words + sent_words > self.max_chunk_size
            density_broken = edge_density < self.min_edge_density and current_chunk
            
            if (size_exceeded or density_broken) and current_chunk:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append((chunk_text, current_graph, current_ents))
                
                # Start new chunk
                current_chunk = [sent]
                current_words = sent_words
                current_graph = graph
                current_ents = entities[:]
            else:
                # Add to current chunk
                current_chunk.append(sent)
                current_words += sent_words
                current_graph = test_graph
                current_ents.extend(entities)

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, current_graph, current_ents))

        logger.info(f"Text chunked into {len(chunks)} segments (edge density strategy)")
        return chunks

    def _calculate_edge_density(self, graph: nx.DiGraph) -> float:
        """Calculate edge density of graph (edges / max_possible_edges)"""
        n = graph.number_of_nodes()
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1)  # Directed graph
        actual_edges = graph.number_of_edges()
        return actual_edges / max_edges if max_edges > 0 else 0.0
