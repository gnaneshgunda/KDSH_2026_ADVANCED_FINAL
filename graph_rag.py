"""
Graph-RAG module for multi-hop reasoning over narrative graph
"""

import logging
from typing import List
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from models import ChunkMetadata
from config import SIMILARITY_THRESHOLD, MULTI_HOP_DEPTH

logger = logging.getLogger(__name__)


class GraphRAG:
    """Multi-hop reasoning over narrative chunk graph"""

    def __init__(self, chunks: List[ChunkMetadata]):
        """
        Initialize GraphRAG with chunks and build similarity graph
        
        Args:
            chunks: List of ChunkMetadata objects
        """
        self.chunks = chunks
        self.graph = self._build_narrative_graph()
        logger.info(f"GraphRAG initialized with {len(chunks)} chunks and {self.graph.number_of_edges()} edges")

    def _build_narrative_graph(self) -> nx.DiGraph:
        """
        Build graph with chunks as nodes, edges based on semantic similarity
        
        Returns:
            Directed graph with weighted edges
        """
        graph = nx.DiGraph()

        # Add chunks as nodes
        for chunk in self.chunks:
            graph.add_node(chunk.chunk_id, metadata=chunk)

        # Add edges based on semantic similarity (co-reference, causality)
        embeddings = np.array([c.embedding for c in self.chunks])
        similarities = cosine_similarity(embeddings, embeddings)

        for i in range(len(self.chunks)):
            for j in range(len(self.chunks)):
                if i != j and similarities[i][j] > SIMILARITY_THRESHOLD:
                    weight = similarities[i][j]
                    graph.add_edge(self.chunks[i].chunk_id, self.chunks[j].chunk_id, weight=weight)

        logger.debug(f"Built narrative graph with {graph.number_of_nodes()} nodes")
        return graph

    def multi_hop_search(self, query_embedding: np.ndarray, start_chunk_id: str,
                        hops: int = MULTI_HOP_DEPTH) -> List[str]:
        """
        Find related chunks via multi-hop graph traversal
        
        Args:
            query_embedding: Query vector (unused currently, for future enhancement)
            start_chunk_id: Starting chunk ID for traversal
            hops: Number of hops to explore
            
        Returns:
            List of chunk IDs found via multi-hop search
        """
        visited = set()
        results = []

        def dfs(node_id, depth):
            """Depth-first search for multi-hop discovery"""
            if depth == 0 or node_id in visited:
                return
            visited.add(node_id)
            results.append(node_id)

            if node_id in self.graph:
                for neighbor in self.graph.neighbors(node_id):
                    dfs(neighbor, depth - 1)

        dfs(start_chunk_id, hops)
        logger.debug(f"Multi-hop search found {len(results)} related chunks")
        return results

    def find_reasoning_path(self, source_id: str, target_id: str) -> List[str]:
        """
        Find shortest reasoning path between two chunks
        
        Args:
            source_id: Starting chunk ID
            target_id: Target chunk ID
            
        Returns:
            List of chunk IDs forming the path
        """
        try:
            if source_id in self.graph and target_id in self.graph:
                path = nx.shortest_path(self.graph, source=source_id, target=target_id)
                logger.debug(f"Found reasoning path of length {len(path)}")
                return path
        except nx.NetworkXNoPath:
            logger.debug(f"No path found between {source_id} and {target_id}")
        
        return []
