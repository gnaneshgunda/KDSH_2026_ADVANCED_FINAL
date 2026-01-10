"""
Data models and structures for Advanced Narrative Consistency RAG
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import networkx as nx


@dataclass
class ChunkMetadata:
    """Rich metadata for each narrative chunk"""
    text: str
    embedding: np.ndarray
    context_vector: np.ndarray  # Temporal + Emotional + Causal
    chunk_id: str
    start_pos: int
    end_pos: int
    dependency_graph: nx.DiGraph
    entities: List[str]
    sentiment: float  # -1 (negative) to 1 (positive)
    temporal_markers: List[str]
    causal_indicators: List[str]


@dataclass
class BackstoryClaim:
    """Extracted and structured backstory claim"""
    claim_id: str
    text: str
    embedding: np.ndarray
    context_vector: np.ndarray
    claim_type: str  # "event", "belief", "motivation", "fear", "trait"
    entities: List[str]
    importance: float  # 0-1 relevance score


@dataclass
class ConsistencyAnalysis:
    """Full consistency analysis result for a backstory"""
    backstory_id: str
    prediction: int  # 0: Contradict, 1: Consistent
    confidence: float
    supporting_chunks: List[Tuple[str, float]]  # (chunk_id, similarity)
    opposing_chunks: List[Tuple[str, float]]  # (chunk_id, neg_similarity)
    reasoning: str
    graph_path: List[str]  # Multi-hop reasoning chain
