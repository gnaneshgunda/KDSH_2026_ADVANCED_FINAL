"""
Context vector building module for temporal, emotional, and causal information
"""

import logging
from typing import List
import numpy as np
from config import nlp, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class ContextVectorBuilder:
    """Build rich context vectors combining semantic embeddings with contextual signals"""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        
        self.temporal_keywords = {
            "past": ["ago", "previously", "before", "then", "once", "years earlier"],
            "present": ["now", "currently", "today", "these days", "at present"],
            "future": ["will", "shall", "going to", "later", "tomorrow", "ahead"]
        }
        
        self.emotional_keywords = {
            "positive": ["happy", "joy", "love", "admire", "proud", "grateful"],
            "negative": ["sad", "hate", "angry", "afraid", "ashamed", "bitter"],
            "neutral": ["think", "know", "understand", "realize", "discover"]
        }
        
        self.causal_keywords = [
            "because", "cause", "due to", "lead to", "result", "if", "then", "therefore"
        ]
        
        logger.info("ContextVectorBuilder initialized")

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment polarity of text
        
        Args:
            text: Input text
            
        Returns:
            Sentiment score from -1.0 (negative) to 1.0 (positive)
        """
        doc = nlp(text)
        pos_score = sum(
            1 for token in doc 
            if token.text.lower() in self.emotional_keywords["positive"]
        )
        neg_score = sum(
            1 for token in doc 
            if token.text.lower() in self.emotional_keywords["negative"]
        )
        total = pos_score + neg_score
        sentiment = (pos_score - neg_score) / max(total, 1)
        return float(sentiment)

    def extract_temporal_markers(self, text: str) -> List[str]:
        """
        Extract temporal indicators from text
        
        Args:
            text: Input text
            
        Returns:
            List of detected temporal markers
        """
        markers = []
        text_lower = text.lower()
        for tense, keywords in self.temporal_keywords.items():
            if any(kw in text_lower for kw in keywords):
                markers.append(tense)
        return markers

    def extract_causal_indicators(self, text: str) -> List[str]:
        """
        Extract causal relationships and indicators
        
        Args:
            text: Input text
            
        Returns:
            List of causal indicators
        """
        doc = nlp(text)
        indicators = []
        for token in doc:
            if token.text.lower() in self.causal_keywords:
                # Extract dependent clauses
                for child in token.children:
                    indicators.append(child.text)
        return indicators[:3]  # Top 3

    def build_context_vector(self, text: str, base_embedding: np.ndarray) -> np.ndarray:
        """
        Build augmented context vector combining base embedding with contextual signals
        
        Args:
            text: Input text
            base_embedding: Pre-computed semantic embedding
            
        Returns:
            Normalized context vector with temporal, emotional, and causal components
        """
        # Extract signals
        sentiment = self.analyze_sentiment(text)
        temporal = self.extract_temporal_markers(text)
        causal = self.extract_causal_indicators(text)

        # Create component vectors
        sentiment_vec = np.array([sentiment] * 32)
        temporal_score = len(temporal) / 3.0  # Normalize
        temporal_vec = np.array([temporal_score] * 32)
        causal_score = len(causal) / 3.0
        causal_vec = np.array([causal_score] * 32)

        # Concatenate: prioritize base embedding
        augmented = np.concatenate([
            base_embedding[:900],  # Original embedding (most important)
            sentiment_vec,
            temporal_vec,
            causal_vec
        ])

        # Normalize
        norm = np.linalg.norm(augmented) + 1e-8
        augmented = augmented / norm
        
        return augmented
