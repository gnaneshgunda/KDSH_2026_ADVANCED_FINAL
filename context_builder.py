"""
Context vector builder for temporal, emotional, and causal information
"""

import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class ContextVectorBuilder:
    """
    Build context vectors enriching semantic embeddings with:
    - Sentiment analysis
    - Temporal marker extraction
    - Causal indicator extraction
    """

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize ContextVectorBuilder.
        
        Args:
            embedding_dim: Dimension of semantic embeddings
        """
        self.embedding_dim = embedding_dim
        
        # Temporal keywords
        self.temporal_keywords = {
            "past": [
                "was", "were", "had", "did", "ago", "before", "previously",
                "once", "earlier", "formerly", "back then", "in the past"
            ],
            "present": [
                "is", "are", "has", "have", "does", "do", "now", "currently",
                "these days", "at present", "right now"
            ],
            "future": [
                "will", "would", "shall", "going to", "going", "plan", "expect",
                "tomorrow", "later", "soon", "eventually", "in the future"
            ]
        }
        
        # Causal indicators
        self.causal_keywords = [
            "because", "since", "as", "caused", "caused by", "due to",
            "result in", "results in", "led to", "lead to", "therefore",
            "thus", "hence", "consequently", "as a result", "so that",
            "effect", "impact", "influence"
        ]
        
        # Sentiment words
        self.positive_words = {
            "happy", "good", "great", "excellent", "wonderful", "beautiful",
            "love", "enjoy", "proud", "successful", "amazing", "brilliant",
            "delighted", "grateful", "hopeful", "peaceful", "strong", "brave"
        }
        
        self.negative_words = {
            "sad", "bad", "terrible", "awful", "horrible", "ugly",
            "hate", "fear", "ashamed", "failed", "tragic", "broken",
            "devastated", "angry", "hopeless", "weak", "afraid", "lost"
        }
        
        logger.info(f"ContextVectorBuilder initialized (embedding_dim={embedding_dim})")

    def build_context_vector(self, text: str, embedding: np.ndarray) -> np.ndarray:
        """
        Build enriched context vector combining semantic and contextual signals.
        
        Args:
            text: Text content
            embedding: Semantic embedding (numpy array)
            
        Returns:
            Context vector (enriched embedding)
        """
        # For now, return embedding as-is. Can be extended to concatenate
        # contextual features as additional dimensions.
        return embedding

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        score = (positive_count - negative_count) / total
        return float(np.clip(score, -1.0, 1.0))

    def extract_temporal_markers(self, text: str) -> List[str]:
        """
        Extract temporal markers (past, present, future references).
        
        Args:
            text: Input text
            
        Returns:
            List of detected temporal markers
        """
        markers = []
        text_lower = text.lower()
        
        for time_type, keywords in self.temporal_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    markers.append(time_type)
                    break
        
        return list(set(markers))  # Remove duplicates

    def extract_causal_indicators(self, text: str) -> List[str]:
        """
        Extract causal relationships and indicators.
        
        Args:
            text: Input text
            
        Returns:
            List of causal indicators found
        """
        indicators = []
        text_lower = text.lower()
        
        for keyword in self.causal_keywords:
            if keyword in text_lower:
                indicators.append(keyword)
        
        return indicators[:5]  # Return top 5

    def analyze_entities(self, text: str) -> List[str]:
        """
        Extract named entities (simplified without spaCy).
        
        Args:
            text: Input text
            
        Returns:
            List of potential entities (capitalized words)
        """
        words = text.split()
        entities = [
            word.rstrip('.,!?;:')
            for word in words
            if word[0].isupper() and len(word) > 2
        ]
        return list(set(entities))[:10]  # Top 10 unique