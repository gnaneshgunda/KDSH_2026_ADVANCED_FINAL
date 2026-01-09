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
        Intelligently chunk text while respecting dependency boundaries
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of (chunk_text, dependency_graph, entities) tuples
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_words = 0

        logger.debug(f"Chunking text into {len(sentences)} sentences")

        for sent in sentences:
            graph, entities = self.build_dependency_graph(sent)
            sent_words = len(sent.split())

            # Check if adding this sentence would exceed size limit
            if current_words + sent_words > self.max_chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk)
                combined_graph = nx.DiGraph()
                combined_ents = []
                
                for s in current_chunk:
                    g, e = self.build_dependency_graph(s)
                    combined_graph = nx.compose(combined_graph, g)
                    combined_ents.extend(e)

                chunks.append((chunk_text, combined_graph, combined_ents))
                current_chunk = [sent]
                current_words = sent_words
            else:
                current_chunk.append(sent)
                current_words += sent_words

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            combined_graph = nx.DiGraph()
            combined_ents = []
            
            for s in current_chunk:
                g, e = self.build_dependency_graph(s)
                combined_graph = nx.compose(combined_graph, g)
                combined_ents.extend(e)
            
            chunks.append((chunk_text, combined_graph, combined_ents))

        logger.info(f"Text chunked into {len(chunks)} segments")
        return chunks
