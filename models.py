"""
Models for Advanced Narrative Consistency RAG
"""

from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict


@dataclass
class ChunkMetadata:
    """Metadata for a narrative chunk"""
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
    """A single claim extracted from backstory"""
    text: str
    embedding: np.ndarray
    claim_type: str  # "event", "belief", "motivation", "fear", "trait"
    claim_id: str = ""
    importance: float = 0.8


@dataclass
class ClaimVerificationResult:
    """Result of verifying a single claim"""
    claim: str
    verdict: str  # supported, contradicted, not_mentioned, unknown
    explanation: str
    confidence: float = 0.5
    supporting_chunks: List[str] = field(default_factory=list)
    opposing_chunks: List[str] = field(default_factory=list)


@dataclass
class ConsistencyAnalysis:
    """Overall consistency analysis result"""
    backstory_id: str
    verdict: str  # consistent, contradicted, unknown
    confidence: float
    rationale: str
    claim_results: List[Dict] = field(default_factory=list)
    num_supported_claims: int = 0
    num_contradicted_claims: int = 0
    num_unknown_claims: int = 0