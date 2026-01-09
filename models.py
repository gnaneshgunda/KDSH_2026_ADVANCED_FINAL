"""
State models for LangGraph-based RAG system
"""

from typing import TypedDict, List, Dict, Optional, Annotated
from dataclasses import dataclass
import numpy as np
import operator

# LangGraph State
class GraphState(TypedDict):
    """State passed between LangGraph nodes"""
    # Input
    record_id: str
    book_key: str
    character: str
    backstory_text: str
    
    # Corpus
    corpus: Dict
    
    # Processing
    chunks: List
    claims: List
    claim_embeddings: List[np.ndarray]
    
    # Retrieval
    supporting_chunks: Annotated[List, operator.add]
    opposing_chunks: Annotated[List, operator.add]
    
    # Analysis
    prediction: Optional[int]
    confidence: Optional[float]
    reasoning: Optional[str]
    
    # Control
    error: Optional[str]
    iteration: int


@dataclass
class ChunkMetadata:
    """Rich metadata for each narrative chunk"""
    text: str
    embedding: np.ndarray
    context_vector: np.ndarray
    chunk_id: str
    start_pos: int
    end_pos: int
    entities: List[str]
    sentiment: float
    temporal_markers: List[str]
    causal_indicators: List[str]


@dataclass
class BackstoryClaim:
    """Extracted and structured backstory claim"""
    claim_id: str
    text: str
    embedding: np.ndarray
    context_vector: np.ndarray
    claim_type: str
    entities: List[str]
    importance: float


@dataclass
class ConsistencyAnalysis:
    """Full consistency analysis result"""
    backstory_id: str
    prediction: int
    confidence: float
    supporting_chunks: List
    opposing_chunks: List
    reasoning: str
    graph_path: List[str]
